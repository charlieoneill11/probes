import os
import json
import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import torch
import yaml
from tqdm import tqdm
import transformer_lens as tl

from probe import ProbeTrainer, ProbeConfig
from data import ConceptExampleGenerator


@dataclass
class Config:
    """Main configuration for the probe training process."""
    # Basic settings
    openai_api_key: str = ""  # Will be loaded from config.yaml
    model_name: str = "gemma-2-2b"
    layer: int = 22
    
    # Directory structure - standardized paths
    concepts_file: str = "concepts.json"
    examples_dir: str = "examples"
    probes_dir: str = "probes"
    
    # Example generation settings
    examples_per_concept: int = 200
    batch_size: int = 50
    skip_generation: bool = False
    force_generation: bool = False
    domain: Optional[str] = None
    domain_description: Optional[str] = None
    example_length: str = "long"
    
    # Training settings
    cross_val: bool = True
    cv_folds: int = 5
    retrain_after_cv: bool = True
    
    # Probe hyperparameters
    num_epochs: int = 200
    weight_decay: float = 0.01
    learning_rate: float = 0.003
    
    # General negative examples
    use_general_negatives: bool = True
    general_negatives_count: int = 200
    general_negatives_file: str = "inputs/general_negative_examples.json"


def load_yaml_config(file_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def load_concepts(config: Config) -> List[str]:
    """Load concepts and domain description from the concepts file."""
    try:
        with open(config.concepts_file, "r") as f:
            data = json.load(f)
        
        # Support both old (list) and new (dict with metadata) formats
        if isinstance(data, dict) and "concepts" in data:
            concepts = data["concepts"]
            domain_description = data.get("domain_description")
            print(f"Loaded {len(concepts)} concepts with domain description")
        elif isinstance(data, list):
            concepts = data
            domain_description = None
            print(f"Loaded {len(concepts)} concepts (old format)")
        else:
            raise ValueError(f"Unexpected format in {config.concepts_file}")
        
        # Use domain from config if provided
        if config.domain and not config.domain_description:
            config.domain_description = f"Examples should be in the domain of {config.domain}"
        # Use domain_description from file if provided and not in config
        elif domain_description and not config.domain_description:
            config.domain_description = domain_description
            
        return concepts
    except Exception as e:
        raise ValueError(f"Error loading concepts file: {e}")


def _generate_general_negatives(config: Config, generator: ConceptExampleGenerator, 
                               concepts: List[str], output_path: str) -> None:
    """Generate general negative examples unrelated to any of the concepts."""
    print(f"Generating {config.general_negatives_count} general negative examples")
    
    examples = generator.generate_unrelated_examples_batch(
        concepts=concepts,
        k=config.general_negatives_count,
        batch_size=config.batch_size,
        domain=config.domain,
        domain_description=config.domain_description,
        example_length=config.example_length
    )
    
    print(f"Generated {len(examples)} general negative examples")
    
    # Save to file
    with open(output_path, "w") as f:
        json.dump({
            "examples": examples,
            "concepts_avoided": concepts,
            "domain_description": config.domain_description,
            "timestamp": str(datetime.datetime.now())
        }, f, indent=2)
    
    print(f"Saved general negative examples to {output_path}")


def _generate_examples_for_concept(config: Config, generator: ConceptExampleGenerator, 
                                  concept: str, output_path: str) -> None:
    """Generate examples for a specific concept."""
    print(f"Generating examples for concept: {concept}")
    
    examples = generator.generate_examples_batch(
        concept=concept,
        k=config.examples_per_concept,
        batch_size=config.batch_size,
        domain=config.domain,
        domain_description=config.domain_description,
        example_length=config.example_length
    )
    
    print(f"Generated {len(examples)} examples for '{concept}'")
    
    # Save examples to file
    generator.save_examples_to_file(
        examples=examples,
        file_path=output_path,
        concept=concept,
        domain_description=config.domain_description
    )
    
    print(f"Saved examples to {output_path}")


def generate_examples(config: Config, generator: Optional[ConceptExampleGenerator], concepts: List[str]) -> None:
    """Generate examples for all concepts if needed."""
    if config.skip_generation:
        print("Skipping example generation as requested")
        return
        
    if generator is None:
        raise ValueError("No generator available for example generation")
    
    # Create examples directory if it doesn't exist
    os.makedirs(config.examples_dir, exist_ok=True)
    
    # Handle general negative examples if needed
    if config.use_general_negatives:
        general_negatives_path = os.path.join(config.examples_dir, config.general_negatives_file)
        
        # Check if general negatives exist and load them
        if os.path.exists(general_negatives_path) and not config.force_generation:
            print(f"Loading existing general negative examples from {general_negatives_path}")
            try:
                with open(general_negatives_path, "r") as f:
                    data = json.load(f)
                    if "examples" in data and len(data["examples"]) >= config.general_negatives_count:
                        print(f"Loaded {len(data['examples'])} general negative examples")
                    else:
                        print("Not enough general negative examples found. Generating new ones...")
                        _generate_general_negatives(config, generator, concepts, general_negatives_path)
            except Exception as e:
                print(f"Error loading general negative examples: {e}. Generating new ones...")
                _generate_general_negatives(config, generator, concepts, general_negatives_path)
        else:
            # Generate new general negative examples
            if config.force_generation and os.path.exists(general_negatives_path):
                print("Forcing regeneration of general negative examples")
            else:
                print("General negative examples file not found. Generating new examples.")
            
            _generate_general_negatives(config, generator, concepts, general_negatives_path)
    
    # Generate examples for each concept if needed
    for concept in tqdm(concepts, desc="Processing concepts"):
        concept_key = concept.replace(" ", "_")
        examples_path = os.path.join(config.examples_dir, f"{concept_key}_examples.json")
        
        # Check if examples already exist and if we should use them
        if not config.force_generation and os.path.exists(examples_path):
            print(f"Examples file already exists for {concept}. Skipping generation.")
            continue
            
        # Generate new examples
        if config.force_generation and os.path.exists(examples_path):
            print(f"Forcing regeneration of examples for {concept}")
        else:
            print(f"Examples file not found for {concept}. Generating new examples.")
            
        _generate_examples_for_concept(config, generator, concept, examples_path)


def train_probes(config: Config, concepts: List[str]) -> Dict:
    """Train probes for all concepts."""
    print(f"Loading model {config.model_name} for all concepts...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tl.HookedTransformer.from_pretrained(config.model_name, device=device)
    
    # Create probes directory if it doesn't exist
    os.makedirs(config.probes_dir, exist_ok=True)
    
    # Load general negative examples if needed
    general_negative_examples = None
    if config.use_general_negatives:
        general_negatives_path = os.path.join(config.examples_dir, config.general_negatives_file)
        if os.path.exists(general_negatives_path):
            try:
                with open(general_negatives_path, "r") as f:
                    data = json.load(f)
                    if "examples" in data:
                        general_negative_examples = data["examples"][:config.general_negatives_count]
                        print(f"Loaded {len(general_negative_examples)} general negative examples")
                    else:
                        print("Warning: No valid general negative examples found")
            except Exception as e:
                print(f"Error loading general negative examples: {e}")
    
    # Process each concept
    results = {}
    for concept in tqdm(concepts, desc="Training probes"):
        concept_key = concept.replace(" ", "_")
        examples_path = os.path.join(config.examples_dir, f"{concept_key}_examples.json")
        
        # Check if examples file exists
        if not os.path.exists(examples_path):
            print(f"Warning: Examples file not found for {concept}: {examples_path}")
            print(f"Skipping {concept}")
            continue
        
        # Train probe
        try:
            # Create probe configuration
            probe_config = ProbeConfig(
                concept=concept,
                model_name=config.model_name,
                layer=config.layer,
                save_dir=config.probes_dir,
                do_cross_val=config.cross_val,
                cv_folds=config.cv_folds,
                retrain_after_cv=config.retrain_after_cv,
                num_epochs=config.num_epochs,
                weight_decay=config.weight_decay,
                learning_rate=config.learning_rate
            )
            
            trainer = ProbeTrainer(probe_config, model=model)
            trainer.load_examples_from_file(examples_path)
            
            # Add general negative examples if available
            if general_negative_examples:
                trainer.add_general_negative_examples(general_negative_examples)
            
            # Train and analyze the probe
            _, save_path = trainer.train_and_analyze()
            
            # Load results
            with open(os.path.join(save_path, "config.json"), "r") as f:
                probe_results = json.load(f)
            
            # Build results dict
            concept_results = {"save_path": save_path}
            
            # Add training metrics
            if "train_metrics" in probe_results and "accuracy" in probe_results["train_metrics"]:
                concept_results["train_accuracy"] = probe_results["train_metrics"]["accuracy"]
                if "f1_score" in probe_results["train_metrics"]:
                    concept_results["train_f1"] = probe_results["train_metrics"]["f1_score"]
            
            # Add validation metrics
            if "val_metrics" in probe_results and probe_results["val_metrics"]:
                if "accuracy" in probe_results["val_metrics"]:
                    concept_results["val_accuracy"] = probe_results["val_metrics"]["accuracy"]
                if "f1_score" in probe_results["val_metrics"]:
                    concept_results["val_f1"] = probe_results["val_metrics"]["f1_score"]
                if "separation" in probe_results["val_metrics"]:
                    concept_results["separation"] = probe_results["val_metrics"]["separation"]
            
            # Add cross-validation metrics
            if "cv_metrics" in probe_results:
                concept_results["cv_avg_val_accuracy"] = probe_results["cv_metrics"]["avg_val_accuracy"]
                concept_results["cv_std_val_accuracy"] = probe_results["cv_metrics"]["std_val_accuracy"]
                if "avg_val_f1" in probe_results["cv_metrics"]:
                    concept_results["cv_avg_val_f1"] = probe_results["cv_metrics"]["avg_val_f1"]
                    concept_results["cv_std_val_f1"] = probe_results["cv_metrics"]["std_val_f1"]
            
            results[concept_key] = concept_results
            
            # Log success message
            if config.cross_val and "cv_avg_val_accuracy" in concept_results:
                cv_acc = concept_results['cv_avg_val_accuracy']
                cv_std = concept_results['cv_std_val_accuracy']
                print(f"Successfully trained probe for '{concept}' with CV accuracy: {cv_acc:.4f} (±{cv_std:.4f})")
            elif "val_accuracy" in concept_results:
                val_acc = concept_results['val_accuracy']
                print(f"Successfully trained probe for '{concept}' with validation accuracy: {val_acc:.4f}")
            else:
                print(f"Successfully trained probe for '{concept}'")
                
        except Exception as e:
            print(f"Error training probe for '{concept}': {e}")
            results[concept_key] = {"error": str(e)}
    
    # Save summary
    summary_path = os.path.join(config.probes_dir, "summary.json")
    with open(summary_path, "w") as f:
        summary = {
            "model": config.model_name,
            "layer": config.layer,
            "examples_per_concept": config.examples_per_concept,
            "domain": config.domain,
            "domain_description": config.domain_description,
            "cross_validation": config.cross_val,
            "cv_folds": config.cv_folds if config.cross_val else None,
            "retrain_after_cv": config.retrain_after_cv if config.cross_val else None,
            "results": results
        }
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to {summary_path}")
    return results


def report_results(config: Config, results: Dict) -> None:
    """Generate and display summary of results."""
    print("\nResults summary:")
    
    if config.cross_val:
        print("Using cross-validation")
        print("-" * 120)
        header = f"{'Concept':<25} {'CV Accuracy':<20} {'CV F1 Score':<20} {'Final Accuracy':<15} {'Final F1':<15} {'Separation':<15}"
        print(header)
        print("-" * 120)
        
        # Collect values for calculating mean
        val_accuracies = []
        val_f1_scores = []
        cv_accuracies = []
        cv_f1_scores = []
        
        for concept, result in results.items():
            if "error" not in result:
                # Handle CV accuracy
                cv_value = result.get('cv_avg_val_accuracy', 'N/A')
                if isinstance(cv_value, (float, int)):
                    cv_acc = f"{cv_value:.4f}"
                    if 'cv_std_val_accuracy' in result:
                        cv_acc += f" (±{result['cv_std_val_accuracy']:.4f})"
                    cv_accuracies.append(cv_value)
                else:
                    cv_acc = str(cv_value)
                
                # Handle CV F1 score
                cv_f1_value = result.get('cv_avg_val_f1', 'N/A')
                if isinstance(cv_f1_value, (float, int)):
                    cv_f1 = f"{cv_f1_value:.4f}"
                    if 'cv_std_val_f1' in result:
                        cv_f1 += f" (±{result['cv_std_val_f1']:.4f})"
                    cv_f1_scores.append(cv_f1_value)
                else:
                    cv_f1 = str(cv_f1_value)
                
                # Get final accuracy value
                if 'val_accuracy' in result:
                    acc_value = result['val_accuracy']
                    final_acc = f"{acc_value:.4f}" if isinstance(acc_value, (float, int)) else str(acc_value)
                    if isinstance(acc_value, (float, int)):
                        val_accuracies.append(acc_value)
                else:
                    acc_value = result.get('train_accuracy', 'N/A')
                    final_acc = f"{acc_value:.4f}" if isinstance(acc_value, (float, int)) else str(acc_value)
                
                # Get final F1 score
                if 'val_f1' in result:
                    f1_value = result['val_f1']
                    final_f1 = f"{f1_value:.4f}" if isinstance(f1_value, (float, int)) else str(f1_value)
                    if isinstance(f1_value, (float, int)):
                        val_f1_scores.append(f1_value)
                else:
                    f1_value = result.get('train_f1', 'N/A')
                    final_f1 = f"{f1_value:.4f}" if isinstance(f1_value, (float, int)) else str(f1_value)
                
                # Handle separation value
                sep_value = result.get('separation', 'N/A')
                sep = f"{sep_value:.4f}" if isinstance(sep_value, (float, int)) else str(sep_value)
                
                print(f"{concept.replace('_', ' '):<25} {cv_acc:<20} {cv_f1:<20} {final_acc:<15} {final_f1:<15} {sep:<15}")
            else:
                print(f"{concept.replace('_', ' '):<25} {'ERROR':<20} {'N/A':<20} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
        
        # Print mean CV accuracy and F1 score
        print("-" * 120)
        if cv_accuracies:
            mean_cv_acc = sum(cv_accuracies) / len(cv_accuracies)
            print(f"Mean CV accuracy across all concepts: {mean_cv_acc:.4f}")
        if cv_f1_scores:
            mean_cv_f1 = sum(cv_f1_scores) / len(cv_f1_scores)
            print(f"Mean CV F1 score across all concepts: {mean_cv_f1:.4f}")
    else:
        print("Not using cross-validation")
        print("-" * 80)
        print(f"{'Concept':<30} {'Val Accuracy':<15} {'Val F1 Score':<15} {'Separation':<15}")
        print("-" * 80)
        
        # Collect values for calculating mean
        val_accuracies = []
        val_f1_scores = []
        
        for concept, result in results.items():
            if "error" not in result:
                # Handle validation accuracy
                val_value = result.get('val_accuracy', 'N/A')
                val_acc = f"{val_value:.4f}" if isinstance(val_value, (float, int)) else str(val_value)
                
                # Add to list for mean calculation
                if isinstance(val_value, (float, int)):
                    val_accuracies.append(val_value)
                
                # Handle validation F1 score
                val_f1_value = result.get('val_f1', 'N/A')
                val_f1 = f"{val_f1_value:.4f}" if isinstance(val_f1_value, (float, int)) else str(val_f1_value)
                
                # Add to list for mean calculation
                if isinstance(val_f1_value, (float, int)):
                    val_f1_scores.append(val_f1_value)
                
                # Handle separation value
                sep_value = result.get('separation', 'N/A')
                sep = f"{sep_value:.4f}" if isinstance(sep_value, (float, int)) else str(sep_value)
                
                print(f"{concept.replace('_', ' '):<30} {val_acc:<15} {val_f1:<15} {sep:<15}")
            else:
                print(f"{concept.replace('_', ' '):<30} {'ERROR':<15} {'N/A':<15} {'N/A':<15}")
        
        # Print mean validation accuracy and F1 score
        print("-" * 80)
        if val_accuracies:
            mean_val_acc = sum(val_accuracies) / len(val_accuracies)
            print(f"Mean validation accuracy across all concepts: {mean_val_acc:.4f}")
        if val_f1_scores:
            mean_val_f1 = sum(val_f1_scores) / len(val_f1_scores)
            print(f"Mean validation F1 score across all concepts: {mean_val_f1:.4f}")
    
    print("-" * (120 if config.cross_val else 80))


def main():
    """Main function to run the training pipeline."""
    # ============================================================
    # Configuration - Modify these settings as needed
    # ============================================================
    config = Config(
        # Model settings
        model_name="gemma-2-2b",
        layer=22,
        
        # Paths
        concepts_file="inputs/concepts.json",
        examples_dir="examples",
        probes_dir="probes",
        
        # Example generation settings
        examples_per_concept=200,
        batch_size=50,
        skip_generation=True,  # Set to True to skip example generation
        force_generation=False, # Set to True to regenerate examples even if they exist
        example_length="long",  # Options: "short", "medium", "long"
        
        # Training settings
        cross_val=True,
        cv_folds=5,
        retrain_after_cv=True,
        
        # Hyperparameters
        num_epochs=100,
        weight_decay=0.01,
        learning_rate=0.003,
        
        # General negative examples settings
        use_general_negatives=True,
        general_negatives_count=200
    )
    
    # Load OpenAI API key from config.yaml
    yaml_config = load_yaml_config("config.yaml")
    config.openai_api_key = yaml_config.get("openai_key", "")
    
    if not config.openai_api_key and not config.skip_generation:
        raise ValueError("OpenAI API key not found in config.yaml and generation is enabled")
    
    # Create necessary directories
    os.makedirs(config.examples_dir, exist_ok=True)
    os.makedirs(config.probes_dir, exist_ok=True)
    
    # Load concepts
    concepts = load_concepts(config)
    print(f"Processing {len(concepts)} concepts: {', '.join(concepts)}")
    
    # Initialize generator if needed
    generator = None
    if not config.skip_generation:
        generator = ConceptExampleGenerator(config.openai_api_key)
    
    # Generate examples
    generate_examples(config, generator, concepts)
    
    # Train probes
    results = train_probes(config, concepts)
    
    # Report results
    report_results(config, results)


if __name__ == "__main__":
    main()