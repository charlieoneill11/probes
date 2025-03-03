import os
import json
import argparse
import yaml
from typing import List, Dict, Optional
from tqdm import tqdm
import torch
import datetime

from probe import ProbeTrainer, ProbeConfig
from data import ConceptExampleGenerator

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def generate_examples_for_concept(
    generator: ConceptExampleGenerator,
    concept: str,
    k: int = 200,
    domain: Optional[str] = None,
    domain_description: Optional[str] = None,
    batch_size: int = 25
) -> List[Dict]:
    """Generate examples for a specific concept"""
    print(f"Generating examples for concept: {concept}")
    
    # Generate a batch of examples
    examples = generator.generate_examples_batch(
        concept=concept,
        k=k,
        batch_size=batch_size,
        domain=domain,
        domain_description=domain_description
    )
    
    print(f"Generated {len(examples)} examples for '{concept}'")
    
    # Save examples to file
    examples_dir = "examples"
    os.makedirs(examples_dir, exist_ok=True)
    
    file_path = os.path.join(examples_dir, f"{concept.replace(' ', '_')}_examples.json")
    generator.save_examples_to_file(
        examples=examples,
        file_path=file_path,
        concept=concept,
        domain_description=domain_description
    )
    
    print(f"Saved examples to {file_path}")
    
    return examples

def generate_general_negative_examples(
    generator: ConceptExampleGenerator,
    concepts: List[str],
    k: int = 200,
    domain: Optional[str] = None,
    domain_description: Optional[str] = None,
    batch_size: int = 25
) -> List[str]:
    """Generate general negative examples unrelated to any of the concepts"""
    print(f"Generating {k} general negative examples unrelated to {len(concepts)} concepts")
    
    # Check if general negative examples already exist
    #examples_dir = "examples"
    file_path = "general_negative_examples.json" #os.path.join(examples_dir, "general_negative_examples.json")
    
    if os.path.exists(file_path):
        print(f"Loading existing general negative examples from {file_path}")
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                if "examples" in data and len(data["examples"]) >= k:
                    print(f"Loaded {len(data['examples'])} general negative examples")
                    return data["examples"]
                else:
                    print(f"Found only {len(data.get('examples', []))} examples, need {k}. Regenerating...")
        except Exception as e:
            print(f"Error loading general negative examples: {e}. Regenerating...")
    
    # Generate examples unrelated to any of the concepts
    examples = generator.generate_unrelated_examples_batch(
        concepts=concepts,
        k=k,
        batch_size=batch_size,
        domain=domain,
        domain_description=domain_description
    )
    
    print(f"Generated {len(examples)} general negative examples")
    
    # Save examples to file
    #os.makedirs(examples_dir, exist_ok=True)
    
    with open(file_path, "w") as f:
        json.dump({
            "examples": examples,
            "concepts_avoided": concepts,
            "domain_description": domain_description,
            "timestamp": str(datetime.datetime.now())
        }, f, indent=2)
    
    print(f"Saved general negative examples to {file_path}")
    
    return examples

def load_general_negative_examples(file_path: str, k: int = 200) -> List[str]:
    """Load general negative examples from a JSON file"""
    print(f"Loading general negative examples from {file_path}")
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            
        if "examples" in data:
            examples = data["examples"]
            if len(examples) >= k:
                print(f"Loaded {len(examples)} general negative examples")
                return examples[:k]  # Return only k examples if more are available
            else:
                print(f"Warning: Found only {len(examples)} examples, needed {k}")
                return examples
        else:
            print(f"Warning: No 'examples' key found in {file_path}")
            return []
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading general negative examples: {e}")
        return []

def load_examples_from_file(file_path: str) -> Dict:
    """Load examples from a JSON file"""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            
        # Check different possible structures of the file
        if "examples" in data:
            examples = data["examples"]
            domain_description = data.get("domain_description")
            print(f"Loaded {len(examples)} examples from {file_path}")
            return {
                "examples": examples,
                "domain_description": domain_description
            }
        else:
            print(f"Warning: No 'examples' key found in {file_path}")
            return {"examples": [], "domain_description": None}
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading examples from {file_path}: {e}")
        return {"examples": [], "domain_description": None}

def train_probe_for_concept(
    concept: str,
    model_name: str,
    layer: int,
    examples_path: Optional[str] = None,
    examples: Optional[List[Dict]] = None,
    save_dir: str = "probes",
    do_cross_val: bool = True,
    cv_folds: int = 5,
    retrain_after_cv: bool = True,
    model = None,
    general_negative_examples: Optional[List[str]] = None
) -> str:
    """Train a probe for a specific concept"""
    print(f"Training probe for concept: {concept}")

    num_epochs = 200
    weight_decay = 0.01       # L2 regularisation strength
    learning_rate = 0.003
    
    # Create configuration
    config = ProbeConfig(
        concept=concept,
        model_name=model_name,
        layer=layer,
        save_dir=save_dir,
        do_cross_val=do_cross_val,
        cv_folds=cv_folds,
        retrain_after_cv=retrain_after_cv,
        num_epochs=num_epochs,
        weight_decay=weight_decay,
        learning_rate=learning_rate
    )
    
    # Initialize trainer with pre-loaded model if provided
    trainer = ProbeTrainer(config, model=model)
    
    # Load examples if provided path
    if examples_path:
        trainer.load_examples_from_file(examples_path)
    # Otherwise, use provided examples or raise error
    elif examples:
        trainer.pos_examples = [ex["positive"] for ex in examples]
        trainer.neg_examples = [ex["negative"] for ex in examples]
        print(f"Using {len(examples)} provided examples")
    else:
        raise ValueError("Either examples_path or examples must be provided")
    
    # Add general negative examples if provided
    if general_negative_examples:
        trainer.add_general_negative_examples(general_negative_examples)
    
    # Train and analyze the probe
    _, save_path = trainer.train_and_analyze()
    
    return save_path

def main():
    """Main function to parse arguments and run the training pipeline"""
    parser = argparse.ArgumentParser(description="Train probes for multiple concepts")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--concepts_file", type=str, help="Path to JSON file with list of concepts")
    parser.add_argument("--concepts", nargs="+", help="List of concepts to process")
    parser.add_argument("--model", type=str, default="gemma-2-2b", help="Model name")
    parser.add_argument("--layer", type=int, default=22, help="Layer to extract representations from")
    parser.add_argument("--examples_per_concept", type=int, default=200, help="Number of examples per concept")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for example generation")
    parser.add_argument("--skip_generation", action="store_true", help="Skip example generation entirely")
    parser.add_argument("--force_generation", action="store_true", help="Force regeneration of examples even if they exist")
    parser.add_argument("--save_dir", type=str, default="probes", help="Directory to save probes")
    parser.add_argument("--domain", type=str, help="Domain for examples (e.g. 'clinical medicine')")
    parser.add_argument("--cross_val", action="store_true", help="Use cross-validation for training")
    parser.add_argument("--no_cross_val", action="store_true", help="Disable cross-validation for training")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--no_retrain", action="store_true", help="Do not retrain on all data after cross-validation")
    parser.add_argument("--no_general_negatives", action="store_true", help="Disable using general negative examples")
    parser.add_argument("--general_negatives_count", type=int, default=200, help="Number of general negative examples to generate")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine list of concepts and domain description
    concepts = []
    domain_description = None
    
    if args.concepts_file:
        # Try to load the new format first
        try:
            with open(args.concepts_file, "r") as f:
                data = json.load(f)
            
            # Check if it's the new format with domain_description
            if isinstance(data, dict) and "concepts" in data:
                concepts = data["concepts"]
                domain_description = data.get("domain_description")
                print(f"Loaded {len(concepts)} concepts with domain description from {args.concepts_file}")
            # Otherwise, assume it's the old format (just a list)
            elif isinstance(data, list):
                concepts = data
                print(f"Loaded {len(concepts)} concepts from {args.concepts_file} (old format)")
            else:
                print(f"Warning: Unexpected format in {args.concepts_file}")
                if args.concepts:
                    concepts = args.concepts
        except Exception as e:
            print(f"Error loading concepts file: {e}")
            if args.concepts:
                concepts = args.concepts
    elif args.concepts:
        concepts = args.concepts
    else:
        raise ValueError("Either --concepts_file or --concepts must be provided")
    
    # Override domain description with command line argument if provided
    if args.domain:
        domain_description = f"Examples should be in the domain of {args.domain}"
    
    print(f"Processing {len(concepts)} concepts: {', '.join(concepts)}")
    if domain_description:
        print(f"Using domain description: {domain_description}")
    
    # Create directories
    os.makedirs("examples", exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize generator if needed
    generator = None
    if not args.skip_generation:
        api_key = config.get("openai_key")
        if not api_key:
            raise ValueError("OpenAI API key not found in config")
        
        generator = ConceptExampleGenerator(api_key)

    # Handle general negative examples - use by default unless explicitly disabled
    general_negative_examples = None
    if not args.no_general_negatives:
        general_negatives_file = "general_negative_examples.json"
        
        # Check if the file exists
        if os.path.exists(general_negatives_file):
            general_negative_examples = load_general_negative_examples(
                general_negatives_file,
                k=args.general_negatives_count
            )
            print(f"Loaded {len(general_negative_examples)} general negative examples from {general_negatives_file}")
        # Generate if needed
        elif not args.skip_generation:
            if generator is None:
                print("Warning: Cannot generate general negative examples without a generator")
                api_key = config.get("openai_key")
                if api_key:
                    generator = ConceptExampleGenerator(api_key)
            
            if generator:
                general_negative_examples = generate_general_negative_examples(
                    generator=generator,
                    concepts=concepts,
                    k=args.general_negatives_count,
                    domain=args.domain,
                    domain_description=domain_description,
                    batch_size=args.batch_size
                )
        
        if general_negative_examples:
            print(f"Using {len(general_negative_examples)} general negative examples for all concepts")
        else:
            print("Warning: No general negative examples available")
    else:
        print("General negative examples disabled with --no_general_negatives flag")

    # Load the model once
    print(f"Loading model {args.model} once for all concepts...")
    import transformer_lens as tl
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tl.HookedTransformer.from_pretrained(args.model, device=device)
    
    # Process each concept
    results = {}
    for concept in tqdm(concepts, desc="Processing concepts"):
        concept_key = concept.replace(" ", "_")
        results[concept_key] = {}
        
        # Determine examples path
        examples_path = f"examples/{concept_key}_examples.json"
        
        # Determine if examples need to be generated
        examples_data = {"examples": None, "domain_description": None}
        
        if args.skip_generation:
            # Skip generation entirely - just check if examples file exists
            if not os.path.exists(examples_path):
                print(f"Warning: Examples file not found for {concept}: {examples_path}")
                print(f"Skipping {concept}")
                continue
        else:
            # Check if examples already exist and if force regeneration is disabled
            if not args.force_generation and os.path.exists(examples_path):
                print(f"Examples file already exists for {concept}. Loading from {examples_path}")
                examples_data = load_examples_from_file(examples_path)
                
                if not examples_data["examples"]:
                    if generator is None:
                        print(f"Failed to load examples and no generator available. Skipping {concept}")
                        continue
                    print(f"Failed to load valid examples for {concept}. Will generate new ones.")
                    examples_data["examples"] = generate_examples_for_concept(
                        generator=generator,
                        concept=concept,
                        k=args.examples_per_concept,
                        domain=args.domain,
                        domain_description=domain_description,
                        batch_size=args.batch_size
                    )
            else:
                # Generate new examples if they don't exist or if forced
                if args.force_generation and os.path.exists(examples_path):
                    print(f"Forcing regeneration of examples for {concept}")
                else:
                    print(f"Examples file not found for {concept}. Generating new examples.")
                
                if generator is None:
                    print(f"No generator available. Skipping {concept}")
                    continue
                
                examples_data["examples"] = generate_examples_for_concept(
                    generator=generator,
                    concept=concept,
                    k=args.examples_per_concept,
                    domain=args.domain, 
                    domain_description=domain_description,
                    batch_size=args.batch_size
                )
                examples_data["domain_description"] = domain_description
        
        # Train probe
        try:
            save_path = train_probe_for_concept(
                concept=concept,
                model_name=args.model,
                layer=args.layer,
                examples_path=examples_path,
                examples=examples_data["examples"],
                save_dir=args.save_dir,
                do_cross_val=args.cross_val,
                cv_folds=args.cv_folds,
                retrain_after_cv=not args.no_retrain,
                model=model,  # Pass the pre-loaded model
                general_negative_examples=general_negative_examples  # Pass general negative examples
            )
            
            # Load results
            with open(os.path.join(save_path, "config.json"), "r") as f:
                probe_results = json.load(f)
            
            # Build results dict based on available metrics
            concept_results = {
                "save_path": save_path,
            }
            
            # Add training metrics
            if "train_metrics" in probe_results and "accuracy" in probe_results["train_metrics"]:
                concept_results["train_accuracy"] = probe_results["train_metrics"]["accuracy"]
                # Add F1 score if available
                if "f1_score" in probe_results["train_metrics"]:
                    concept_results["train_f1"] = probe_results["train_metrics"]["f1_score"]
            
            # Add validation metrics if available
            if "val_metrics" in probe_results and probe_results["val_metrics"]:
                if "accuracy" in probe_results["val_metrics"]:
                    concept_results["val_accuracy"] = probe_results["val_metrics"]["accuracy"]
                if "f1_score" in probe_results["val_metrics"]:
                    concept_results["val_f1"] = probe_results["val_metrics"]["f1_score"]
                if "separation" in probe_results["val_metrics"]:
                    concept_results["separation"] = probe_results["val_metrics"]["separation"]
            
            # Add cross-validation metrics if available
            if "cv_metrics" in probe_results:
                concept_results["cv_avg_val_accuracy"] = probe_results["cv_metrics"]["avg_val_accuracy"]
                concept_results["cv_std_val_accuracy"] = probe_results["cv_metrics"]["std_val_accuracy"]
                # Add F1 score metrics if available
                if "avg_val_f1" in probe_results["cv_metrics"]:
                    concept_results["cv_avg_val_f1"] = probe_results["cv_metrics"]["avg_val_f1"]
                    concept_results["cv_std_val_f1"] = probe_results["cv_metrics"]["std_val_f1"]
            
            results[concept_key] = concept_results
            
            # Determine which metrics to display in the success message
            if args.cross_val and "cv_avg_val_accuracy" in concept_results:
                cv_acc = concept_results['cv_avg_val_accuracy']
                cv_std = concept_results['cv_std_val_accuracy']
                cv_f1 = concept_results.get('cv_avg_val_f1', 'N/A')
                cv_f1_std = concept_results.get('cv_std_val_f1', 'N/A')
                
                if isinstance(cv_acc, (float, int)) and isinstance(cv_std, (float, int)):
                    print(f"Successfully trained probe for '{concept}' with CV accuracy: {cv_acc:.4f} (±{cv_std:.4f})")
                    if isinstance(cv_f1, (float, int)) and isinstance(cv_f1_std, (float, int)):
                        print(f"CV F1 score: {cv_f1:.4f} (±{cv_f1_std:.4f})")
                else:
                    print(f"Successfully trained probe for '{concept}' with CV accuracy: {cv_acc}")
            elif "val_accuracy" in concept_results:
                val_acc = concept_results['val_accuracy']
                val_f1 = concept_results.get('val_f1', 'N/A')
                if isinstance(val_acc, (float, int)):
                    print(f"Successfully trained probe for '{concept}' with validation accuracy: {val_acc:.4f}")
                    if isinstance(val_f1, (float, int)):
                        print(f"Validation F1 score: {val_f1:.4f}")
                else:
                    print(f"Successfully trained probe for '{concept}' with validation accuracy: {val_acc}")
            else:
                print(f"Successfully trained probe for '{concept}'")
                
        except Exception as e:
            print(f"Error training probe for '{concept}': {e}")
            results[concept_key]["error"] = str(e)
    
    # Save summary
    summary_path = os.path.join(args.save_dir, "summary.json")
    with open(summary_path, "w") as f:
        summary = {
            "model": args.model,
            "layer": args.layer,
            "examples_per_concept": args.examples_per_concept,
            "domain": args.domain,
            "domain_description": domain_description,
            "cross_validation": args.cross_val,
            "cv_folds": args.cv_folds if args.cross_val else None,
            "retrain_after_cv": not args.no_retrain if args.cross_val else None,
            "results": results
        }
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to {summary_path}")
    
    # Print results
    print("\nResults summary:")
    
    if args.cross_val:
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
    
    print("-" * (120 if args.cross_val else 80))

if __name__ == "__main__":
    main()