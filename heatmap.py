import os
import json
import torch
import numpy as np
import transformer_lens as tl
from typing import Dict, List
import joblib
import pickle

def load_all_probes(probes_dir: str) -> Dict:
    """Load all scikit-learn probes from the probes directory"""
    probes = {}
    concepts = [d for d in os.listdir(probes_dir) if os.path.isdir(os.path.join(probes_dir, d))]
    
    print(f"Found {len(concepts)} potential concept directories")
    
    for concept in concepts:
        concept_dir = os.path.join(probes_dir, concept)
        config_path = os.path.join(concept_dir, "config.json")
        joblib_path = os.path.join(concept_dir, "probe.joblib")
        pkl_path = os.path.join(concept_dir, "probe.pkl")
        
        # Check if config exists and either joblib or pkl file exists
        if os.path.exists(config_path) and (os.path.exists(joblib_path) or os.path.exists(pkl_path)):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                
                # Try to load the model
                try:
                    if os.path.exists(joblib_path):
                        probe = joblib.load(joblib_path)
                        model_path = joblib_path
                    else:
                        with open(pkl_path, 'rb') as f:
                            probe = pickle.load(f)
                        model_path = pkl_path
                except Exception as e:
                    print(f"Error loading model for {concept}: {e}")
                    continue
                
                # Add to probes dict
                probes[concept] = {
                    "probe": probe,
                    "config": config
                }
                print(f"Successfully loaded probe for concept: {concept} from {model_path}")
            except Exception as e:
                print(f"Error loading probe for {concept}: {e}")
    
    return probes

def main():
    # Parameters
    probes_dir = "probes"
    model_name = "gemma-2-2b"  # Change to your model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sample text - replace with your own
    positive_text = "example.txt"
    negative_text = "unrelated.txt"
    positive = False
    example_path = positive_text if positive else negative_text
    with open(example_path, "r") as f:
        medical_text = f.read()
    print(medical_text)
    
    # Load all probes
    print(f"Using device: {device}")
    print("Loading probes...")
    all_probes = load_all_probes(probes_dir)
    print(f"Successfully loaded {len(all_probes)} probes")
    
    # Load language model
    print(f"Loading model: {model_name}")
    model = tl.HookedTransformer.from_pretrained(model_name, device=device)
    
    # Tokenize the text
    tokens = model.to_tokens(medical_text)
    token_strs = model.to_str_tokens(tokens[0])
    seq_len = tokens.shape[1]
    print(f"Text tokenized to {seq_len} tokens")
    
    # Run the model and cache activations
    print("Running model and caching activations...")
    _, cache = model.run_with_cache(tokens)
    
    # Initialize results matrix (n_probes × seq_len)
    concepts = list(all_probes.keys())
    results_matrix = np.zeros((len(concepts), seq_len))
    
    # Apply each probe to each token position
    print("Applying probes to all token positions...")
    for i, concept in enumerate(concepts):
        probe_data = all_probes[concept]
        
        # Get the hook name and layer from config
        layer = probe_data["config"]["layer"]
        hook_name = probe_data["config"].get("hook_name", f"blocks.{layer}.hook_resid_post")
        
        # Get activations at this layer for all tokens
        activations = cache[hook_name]  # Shape: [batch_size, seq_len, hidden_dim]
        
        # Apply probe to all positions
        probe = probe_data["probe"]
        
        # Process all positions
        batch_size, seq_len, hidden_dim = activations.shape
        
        # For each position, apply the probe
        for pos in range(seq_len):
            # Get activations for this position
            pos_activations = activations[0, pos].cpu().numpy().reshape(1, -1)
            
            # Apply sklearn probe (get probability of positive class)
            probe_output = probe.predict_proba(pos_activations)[0, 1]  # Get probability of class 1
            
            # Store in results matrix
            results_matrix[i, pos] = probe_output
    
    # Save results to file
    output = {
        "tokens": token_strs,
        "concepts": concepts,
        "probe_outputs_matrix": results_matrix.tolist()
    }

    # Plotly save heatmap
    import plotly.express as px
    
    # Normalize each row (concept) in the matrix
    normalized_matrix = np.zeros_like(results_matrix)
    for i in range(results_matrix.shape[0]):
        row = results_matrix[i, :]
        row_min, row_max = row.min(), row.max()
        if row_max > row_min:  # Avoid division by zero
            normalized_matrix[i, :] = (row - row_min) / (row_max - row_min)
        else:
            normalized_matrix[i, :] = np.zeros_like(row)

    # Create a heatmap using Plotly with normalized values
    fig = px.imshow(normalized_matrix,
                    x=token_strs,
                    y=concepts,
                    color_continuous_scale='Viridis')
    fig.update_layout(
        title="Normalized Probe Outputs Heatmap",
        xaxis_title="Tokens",
        yaxis_title="Concepts"
    )
    fig.write_image("heatmap.png")
    
    output_file = "probe_analysis.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to {output_file}")
    
    # Print a sample of the results
    print("\nSample of results (first 5 tokens, first 3 concepts):")
    for i, concept in enumerate(concepts[:min(3, len(concepts))]):
        print(f"{concept}:")
        for j, token in enumerate(token_strs[:min(5, len(token_strs))]):
            print(f"  {token}: {results_matrix[i, j]:.4f}")

    # Add an additional analysis for a global view
    print("\nConcept activation summary (average across all tokens):")
    for i, concept in enumerate(concepts):
        avg_activation = np.mean(results_matrix[i, :])
        max_activation = np.max(results_matrix[i, :])
        max_token_idx = np.argmax(results_matrix[i, :])
        max_token = token_strs[max_token_idx] if max_token_idx < len(token_strs) else "Unknown"
        
        print(f"{concept}:")
        print(f"  Average activation: {avg_activation:.4f}")
        print(f"  Max activation: {max_activation:.4f} at token '{max_token}'")

if __name__ == "__main__":
    main()