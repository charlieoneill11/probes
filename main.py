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
from data_mining import NegativeExampleMiner


@dataclass
class Config:
    """Main configuration for the probe training process."""
    # Basic settings
    openai_api_key: str = ""  # Will be loaded from config.yaml
    model_name: str = "gemma-2-2b"
    layer: int = 22
    
    # Directory structure - standardized paths
    concepts_file: str = "inputs/concepts_copy.json"
    examples_dir: str = "examples"
    probes_dir: str = "probes_trials/v1" # changed path for Chi Chi's trials
    
    # Example generation settings
    examples_per_concept: int = 20 #200
    batch_size: int = 50
    skip_generation: bool = True
    force_generation: bool = False
    domain: Optional[str] = None
    domain_description: Optional[str] = None
    example_length: str = "medium"
    
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
    
    # General positive examples (new)
    use_general_positives: bool = True
    general_positives_count: int = 200
    
    # Mined negative examples
    use_mined_negatives: bool = True
    mined_negatives_count: int = 200
    mined_negatives_file: str = "inputs/mined_negative_examples.json"
    mined_max_length: int = 128
    mined_min_length: int = 20
    mined_source: str = "pile"


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


def _mine_negative_examples(config: Config, output_path: str) -> None:
    """Mine negative examples from the specified source"""
    print(f"Mining {config.mined_negatives_count} negative examples from {config.mined_source}")
    
    # Create miner
    miner = NegativeExampleMiner(cache_dir=config.examples_dir)
    
    # Mine examples
    examples = miner.mine_examples(
        source=config.mined_source,
        num_examples=config.mined_negatives_count,
        max_length=config.mined_max_length,
        min_length=config.mined_min_length,
        cache_file=output_path
    )
    
    print(f"Mined {len(examples)} negative examples")
    
    # Save is handled by the miner


def _create_general_positive_examples(concept_examples: List[str], general_negative_examples: List[str], count: int) -> List[str]:
    """Create general positive examples by combining concept examples with general negative examples.
    
    Each general positive example is a concatenation of a concept-specific positive example
    and a general negative example, preserving the concept-specific information while adding
    general context.
    """
    import random
    
    # Ensure we have enough examples to work with
    if not concept_examples or not general_negative_examples:
        print("Warning: Cannot create general positive examples - missing source examples")
        return []
    
    # Limit count to the minimum available examples
    count = min(count, len(concept_examples), len(general_negative_examples))
    
    # Create combined examples
    general_positive_examples = []
    for i in range(count):
        # Sample randomly with replacement if we need more examples than available
        concept_example = concept_examples[i % len(concept_examples)]
        # Randomly select a general negative example
        general_example = random.choice(general_negative_examples)
        
        # Combine the examples (concept example first, then general example)
        combined = f"{concept_example} {general_example}"
        general_positive_examples.append(combined)
    
    print(f"Created {len(general_positive_examples)} general positive examples")
    # Print a sample to verify
    if general_positive_examples:
        sample = general_positive_examples[0][:100] + "..." if len(general_positive_examples[0]) > 100 else general_positive_examples[0]
        print(f"Sample general positive example: {sample}")
    
    return general_positive_examples


def generate_examples(config: Config, generator: Optional[ConceptExampleGenerator], concepts: List[str]) -> None:
    """Generate examples for all concepts if needed."""
    if config.skip_generation:
        print("Skipping example generation as requested")
        return
        
    # Create examples directory if it doesn't exist
    os.makedirs(config.examples_dir, exist_ok=True)
    
    # Handle mined negative examples if needed
    if config.use_mined_negatives:
        mined_negatives_path = os.path.join(config.examples_dir, config.mined_negatives_file)
        
        # Check if mined negatives exist and load them
        if os.path.exists(mined_negatives_path) and not config.force_generation:
            print(f"Mined negative examples file exists at {mined_negatives_path}")
            try:
                with open(mined_negatives_path, "r") as f:
                    data = json.load(f)
                    if "examples" in data and len(data["examples"]) >= config.mined_negatives_count:
                        print(f"Loaded {len(data['examples'])} mined negative examples")
                    else:
                        print("Not enough mined negative examples found. Mining new ones...")
                        _mine_negative_examples(config, mined_negatives_path)
            except Exception as e:
                print(f"Error loading mined negative examples: {e}. Mining new ones...")
                _mine_negative_examples(config, mined_negatives_path)
        else:
            # Mine new negative examples
            if config.force_generation and os.path.exists(mined_negatives_path):
                print("Forcing regeneration of mined negative examples")
            else:
                print("Mined negative examples file not found. Mining new examples.")
            
            _mine_negative_examples(config, mined_negatives_path)
    
    # Handle general negative examples if needed
    if generator is not None and config.use_general_negatives:
        general_negatives_path = config.general_negatives_file
        if not os.path.isabs(general_negatives_path):
            general_negatives_path = os.path.join(config.examples_dir, general_negatives_path)
        
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
    if generator is not None:
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
        # Use the full path directly
        general_negatives_path = config.general_negatives_file
        print(f"Looking for general negative examples at: {general_negatives_path}")
        
        if os.path.exists(general_negatives_path):
            try:
                with open(general_negatives_path, "r") as f:
                    data = json.load(f)
                    if "examples" in data:
                        general_negative_examples = data["examples"][:config.general_negatives_count]
                        print(f"Loaded {len(general_negative_examples)} general negative examples")
                    else:
                        print(f"Warning: No 'examples' key found in general negative examples file")
            except Exception as e:
                print(f"Error loading general negative examples: {e}")
        else:
            print(f"Warning: General negative examples file not found at {general_negatives_path}")
    
    # Load mined negative examples if needed
    mined_negative_examples = None
    if config.use_mined_negatives:
        # Use the full path directly
        mined_negatives_path = config.mined_negatives_file
        print(f"Looking for mined negative examples at: {mined_negatives_path}")
        
        if os.path.exists(mined_negatives_path):
            try:
                with open(mined_negatives_path, "r") as f:
                    data = json.load(f)
                    if "examples" in data:
                        mined_negative_examples = data["examples"][:config.mined_negatives_count]
                        print(f"Loaded {len(mined_negative_examples)} mined negative examples")
                        # Print a sample to verify content
                        if mined_negative_examples:
                            sample = mined_negative_examples[0][:100] + "..." if len(mined_negative_examples[0]) > 100 else mined_negative_examples[0]
                            print(f"Sample mined example: {sample}")
                    else:
                        print(f"Warning: No 'examples' key found in mined negative examples file")
            except Exception as e:
                print(f"Error loading mined negative examples: {e}")
        else:
            print(f"Warning: Mined negative examples file not found at {mined_negatives_path}")
    
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
                print(f"Added {len(general_negative_examples)} general negative examples for {concept}")
            else:
                print(f"No general negative examples available for {concept}")
            
            # Create and add general positive examples if enabled
            if config.use_general_positives and general_negative_examples:
                # Extract positive examples from loaded file
                try:
                    with open(examples_path, "r") as f:
                        loaded_examples = json.load(f)["examples"]
                        concept_positive_examples = [ex["positive"] for ex in loaded_examples]
                        
                        # Create general positive examples
                        general_positive_examples = _create_general_positive_examples(
                            concept_positive_examples,
                            general_negative_examples,
                            config.general_positives_count
                        )
                        
                        # Add to trainer
                        if general_positive_examples:
                            trainer.add_general_positive_examples(general_positive_examples)
                            print(f"Added {len(general_positive_examples)} general positive examples for {concept}")
                except Exception as e:
                    print(f"Error creating general positive examples: {e}")
            
            # Add mined negative examples if available
            if mined_negative_examples:
                trainer.add_mined_negative_examples(mined_negative_examples)
                print(f"Added {len(mined_negative_examples)} mined negative examples for {concept}")
            else:
                print(f"No mined negative examples available for {concept}")
            
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
                if "roc_auc" in probe_results["train_metrics"]:
                    concept_results["train_roc_auc"] = probe_results["train_metrics"]["roc_auc"]
            
            # Add validation metrics
            if "val_metrics" in probe_results and probe_results["val_metrics"]:
                if "accuracy" in probe_results["val_metrics"]:
                    concept_results["val_accuracy"] = probe_results["val_metrics"]["accuracy"]
                if "f1_score" in probe_results["val_metrics"]:
                    concept_results["val_f1"] = probe_results["val_metrics"]["f1_score"]
                if "roc_auc" in probe_results["val_metrics"]:
                    concept_results["val_roc_auc"] = probe_results["val_metrics"]["roc_auc"]
                if "separation" in probe_results["val_metrics"]:
                    concept_results["separation"] = probe_results["val_metrics"]["separation"]
            
            # Add cross-validation metrics
            if "cv_metrics" in probe_results:
                concept_results["cv_avg_val_accuracy"] = probe_results["cv_metrics"]["avg_val_accuracy"]
                concept_results["cv_std_val_accuracy"] = probe_results["cv_metrics"]["std_val_accuracy"]
                if "avg_val_f1" in probe_results["cv_metrics"]:
                    concept_results["cv_avg_val_f1"] = probe_results["cv_metrics"]["avg_val_f1"]
                    concept_results["cv_std_val_f1"] = probe_results["cv_metrics"]["std_val_f1"]
                if "avg_val_roc_auc" in probe_results["cv_metrics"]:
                    concept_results["cv_avg_val_roc_auc"] = probe_results["cv_metrics"]["avg_val_roc_auc"]
                    concept_results["cv_std_val_roc_auc"] = probe_results["cv_metrics"]["std_val_roc_auc"]
            
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
            "use_general_negatives": config.use_general_negatives,
            "general_negatives_count": len(general_negative_examples) if general_negative_examples else 0,
            "use_mined_negatives": config.use_mined_negatives,
            "mined_negatives_count": len(mined_negative_examples) if mined_negative_examples else 0,
            "mined_source": config.mined_source if config.use_mined_negatives else None,
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
        print("-" * 140)  # Increased width to accommodate ROC AUC
        header = f"{'Concept':<25} {'CV Accuracy':<20} {'CV F1 Score':<20} {'CV ROC AUC':<20} {'Final Accuracy':<15} {'Final F1':<15} {'ROC AUC':<15}"
        print(header)
        print("-" * 140)  # Increased width
        
        # Collect values for calculating mean
        val_accuracies = []
        val_f1_scores = []
        val_roc_aucs = []
        cv_accuracies = []
        cv_f1_scores = []
        cv_roc_aucs = []
        
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
                
                # Handle CV ROC AUC
                cv_roc_auc_value = result.get('cv_avg_val_roc_auc', 'N/A')
                if isinstance(cv_roc_auc_value, (float, int)):
                    cv_roc_auc = f"{cv_roc_auc_value:.4f}"
                    if 'cv_std_val_roc_auc' in result:
                        cv_roc_auc += f" (±{result['cv_std_val_roc_auc']:.4f})"
                    cv_roc_aucs.append(cv_roc_auc_value)
                else:
                    cv_roc_auc = str(cv_roc_auc_value)
                
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
                
                # Get final ROC AUC
                if 'val_roc_auc' in result:
                    roc_auc_value = result['val_roc_auc']
                    final_roc_auc = f"{roc_auc_value:.4f}" if isinstance(roc_auc_value, (float, int)) else str(roc_auc_value)
                    if isinstance(roc_auc_value, (float, int)):
                        val_roc_aucs.append(roc_auc_value)
                else:
                    roc_auc_value = result.get('train_roc_auc', 'N/A')
                    final_roc_auc = f"{roc_auc_value:.4f}" if isinstance(roc_auc_value, (float, int)) else str(roc_auc_value)
                
                print(f"{concept.replace('_', ' '):<25} {cv_acc:<20} {cv_f1:<20} {cv_roc_auc:<20} {final_acc:<15} {final_f1:<15} {final_roc_auc:<15}")
            else:
                print(f"{concept.replace('_', ' '):<25} {'ERROR':<20} {'N/A':<20} {'N/A':<20} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
        
        # Print mean CV accuracy, F1 score, and ROC AUC
        print("-" * 140)
        if cv_accuracies:
            mean_cv_acc = sum(cv_accuracies) / len(cv_accuracies)
            print(f"Mean CV accuracy across all concepts: {mean_cv_acc:.4f}")
        if cv_f1_scores:
            mean_cv_f1 = sum(cv_f1_scores) / len(cv_f1_scores)
            print(f"Mean CV F1 score across all concepts: {mean_cv_f1:.4f}")
        if cv_roc_aucs:
            mean_cv_roc_auc = sum(cv_roc_aucs) / len(cv_roc_aucs)
            print(f"Mean CV ROC AUC across all concepts: {mean_cv_roc_auc:.4f}")
    else:
        print("Not using cross-validation")
        print("-" * 100)  # Increased width to accommodate ROC AUC
        print(f"{'Concept':<30} {'Val Accuracy':<15} {'Val F1 Score':<15} {'Val ROC AUC':<15} {'Separation':<15}")
        print("-" * 100)
        
        # Collect values for calculating mean
        val_accuracies = []
        val_f1_scores = []
        val_roc_aucs = []
        
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
                
                # Handle validation ROC AUC
                val_roc_auc_value = result.get('val_roc_auc', 'N/A')
                val_roc_auc = f"{val_roc_auc_value:.4f}" if isinstance(val_roc_auc_value, (float, int)) else str(val_roc_auc_value)
                
                # Add to list for mean calculation
                if isinstance(val_roc_auc_value, (float, int)):
                    val_roc_aucs.append(val_roc_auc_value)
                
                # Handle separation value
                sep_value = result.get('separation', 'N/A')
                sep = f"{sep_value:.4f}" if isinstance(sep_value, (float, int)) else str(sep_value)
                
                print(f"{concept.replace('_', ' '):<30} {val_acc:<15} {val_f1:<15} {val_roc_auc:<15} {sep:<15}")
            else:
                print(f"{concept.replace('_', ' '):<30} {'ERROR':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
        
        # Print mean validation accuracy, F1 score, and ROC AUC
        print("-" * 100)
        if val_accuracies:
            mean_val_acc = sum(val_accuracies) / len(val_accuracies)
            print(f"Mean validation accuracy across all concepts: {mean_val_acc:.4f}")
        if val_f1_scores:
            mean_val_f1 = sum(val_f1_scores) / len(val_f1_scores)
            print(f"Mean validation F1 score across all concepts: {mean_val_f1:.4f}")
        if val_roc_aucs:
            mean_val_roc_auc = sum(val_roc_aucs) / len(val_roc_aucs)
            print(f"Mean validation ROC AUC across all concepts: {mean_val_roc_auc:.4f}")
    
    print("-" * (140 if config.cross_val else 100))


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
        skip_generation=False,  # Set to False to generate examples for specific concepts
        force_generation=False,
        example_length="medium",
        
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
        general_negatives_count=200,
        general_negatives_file="inputs/general_negative_examples.json",
        
        # General positive examples settings (new)
        use_general_positives=True,
        general_positives_count=200,
        
        # Mined negative examples settings
        use_mined_negatives=True,
        mined_negatives_count=100, #200
        mined_negatives_file="inputs/mined_negative_examples.json",
        mined_max_length=128,
        mined_min_length=20,
        mined_source="pile"
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
    
    # Generate examples - only for specific concepts, assuming general/mined negatives exist
    if not config.skip_generation:
        # Skip generating general and mined negatives, assume they exist
        print("Assuming general and mined negative examples already exist")
        
        # Only generate examples for each concept
        for concept in tqdm(concepts, desc="Generating examples for concepts"):
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
    else:
        # Skip all generation
        print("Skipping example generation as requested")
    
    # Train probes
    results = train_probes(config, concepts)
    
    # Report results
    report_results(config, results)


if __name__ == "__main__":
    main()