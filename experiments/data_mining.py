import os
import json
import random
from typing import List, Optional, Dict, Any
from datasets import load_dataset
from tqdm import tqdm
import torch
from transformer_lens import HookedTransformer

class NegativeExampleMiner:
    """Class for mining negative examples from datasets"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the miner with optional cache directory"""
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        # Load the Gemma-2-2b tokenizer
        print("Loading Gemma-2-2b tokenizer...")
        model = HookedTransformer.from_pretrained("google/gemma-2-2b", device="cpu")
        self.tokenizer = model.tokenizer
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.padding_side = 'right'
        # Delete model to free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("Tokenizer loaded and model freed from memory")
    
    def mine_from_pile(
        self, 
        num_examples: int = 200, 
        token_length: int = 256,
        seed: int = 42,
        cache_file: Optional[str] = None
    ) -> tuple[List[str], List[str]]:
        """
        Mine examples from the Pile dataset (monology/pile-uncopyrighted)
        
        Args:
            num_examples: Number of examples to mine
            token_length: Exact length of examples in tokens
            seed: Random seed for reproducibility
            cache_file: Path to cache file to save/load examples
        
        Returns:
            Tuple of (truncated examples, tokenized examples)
        """
        # Check if cache file exists and load from it if requested
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'examples' in data and len(data['examples']) >= num_examples:
                        print(f"Loaded {len(data['examples'])} mined examples from cache: {cache_file}")
                        return data['examples'][:num_examples], data['tokenized_examples'][:num_examples]
                    else:
                        print(f"Cache file exists but doesn't contain enough examples. Mining new examples.")
            except Exception as e:
                print(f"Error loading from cache: {e}. Mining new examples.")
        
        # Set random seed for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
        
        print(f"Mining {num_examples} examples from monology/pile-uncopyrighted...")
        
        # Load dataset
        dataset = load_dataset(
            "monology/pile-uncopyrighted", 
            streaming=True,
            split="train"
        )
        
        # Mine examples
        truncated_examples = []
        tokenized_examples = []
        for item in tqdm(dataset, total=num_examples*5):  # Sample more than needed to account for filtering
            text = item['text'].strip()
            if not text:
                continue
                
            # Tokenize the text
            tokens = self.tokenizer.encode(text)
            
            # If text is long enough, truncate or continue searching
            if len(tokens) >= token_length:
                # Truncate to exact token length and decode back to text
                truncated_tokens = tokens[:token_length]
                truncated_text = self.tokenizer.decode(truncated_tokens)
                truncated_examples.append(truncated_text)
                tokenized_examples.append(truncated_text)
            
            # Check if we have enough examples
            if len(truncated_examples) >= num_examples:
                break
        
        # Shuffle both lists with the same order
        combined = list(zip(truncated_examples, tokenized_examples))
        random.shuffle(combined)
        truncated_examples, tokenized_examples = zip(*combined)
        
        # Trim to requested number
        truncated_examples = list(truncated_examples[:num_examples])
        tokenized_examples = list(tokenized_examples[:num_examples])
        
        print(f"Mined {len(truncated_examples)} examples from Pile dataset")
        
        # Save to cache if requested
        if cache_file:
            cache_dir = os.path.dirname(cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            
            with open(cache_file, 'w') as f:
                json.dump({
                    'examples': truncated_examples,
                    'tokenized_examples': tokenized_examples,
                    'source': 'monology/pile-uncopyrighted',
                    'token_length': token_length,
                    'seed': seed
                }, f, indent=2)
            print(f"Saved mined examples to cache: {cache_file}")
        
        return truncated_examples, tokenized_examples
    
    def mine_examples(
        self,
        source: str = "pile",
        num_examples: int = 200,
        token_length: int = 256,
        seed: int = 42,
        cache_file: Optional[str] = None,
        **kwargs
    ) -> tuple[List[str], List[str]]:
        """
        Mine examples from the specified source
        
        Args:
            source: Source dataset ("pile" for now, can be extended)
            num_examples: Number of examples to mine
            token_length: Exact length of examples in tokens
            seed: Random seed for reproducibility
            cache_file: Path to cache file to save/load examples
            **kwargs: Additional arguments for specific miners
        
        Returns:
            Tuple of (original examples, tokenized examples)
        """
        if source.lower() == "pile":
            return self.mine_from_pile(
                num_examples=num_examples,
                token_length=token_length,
                seed=seed,
                cache_file=cache_file
            )
        else:
            raise ValueError(f"Unknown source: {source}. Supported sources: 'pile'")

def main():
    """
    Main function to mine negative examples and save them in the correct format.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Mine negative examples from datasets")
    parser.add_argument("--num_examples", type=int, default=200, help="Number of examples to mine")
    parser.add_argument("--output_file", type=str, default="inputs/mined_negative_examples.json", 
                        help="Path to save the mined examples")
    parser.add_argument("--token_length", type=int, default=256, help="Exact length of examples in tokens")
    parser.add_argument("--source", type=str, default="pile", help="Source dataset to mine from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create miner
    miner = NegativeExampleMiner()
    
    # Mine examples
    print(f"Mining {args.num_examples} examples from {args.source}...")
    truncated_examples, tokenized_examples = miner.mine_examples(
        source=args.source,
        num_examples=args.num_examples,
        token_length=args.token_length,
        seed=args.seed
    )

    print(f"Number of examples: {len(truncated_examples)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Save examples in the same format as general negative examples
    with open(args.output_file, 'w') as f:
        # Create the data dictionary
        data = {
            "examples": truncated_examples,
            "tokenized_examples": tokenized_examples,
            "source": args.source,
            "token_length": args.token_length,
            "seed": args.seed,
            "timestamp": str(datetime.datetime.now())
        }
        
        # Custom JSON serialization with newlines after commas in lists
        f.write("{\n")
        f.write('  "examples": [\n')
        for i, example in enumerate(truncated_examples):
            json_str = json.dumps(example)
            if i < len(truncated_examples) - 1:
                f.write(f"    {json_str},\n")
            else:
                f.write(f"    {json_str}\n")
        f.write("  ],\n")
        
        f.write('  "tokenized_examples": [\n')
        for i, example in enumerate(tokenized_examples):
            json_str = json.dumps(example)
            if i < len(tokenized_examples) - 1:
                f.write(f"    {json_str},\n")
            else:
                f.write(f"    {json_str}\n")
        f.write("  ],\n")
        
        # Write the remaining fields normally
        f.write(f'  "source": {json.dumps(args.source)},\n')
        f.write(f'  "token_length": {args.token_length},\n')
        f.write(f'  "seed": {args.seed},\n')
        f.write(f'  "timestamp": {json.dumps(str(datetime.datetime.now()))}\n')
        f.write("}")
    
    print(f"Saved {len(truncated_examples)} mined examples to {args.output_file}")
    
    # Print a sample
    if truncated_examples:
        sample_orig = truncated_examples[0][:100] + "..." if len(truncated_examples[0]) > 100 else truncated_examples[0]
        sample_tokenized = tokenized_examples[0][:100] + "..." if len(tokenized_examples[0]) > 100 else tokenized_examples[0]
        print(f"Sample original example: {sample_orig}")
        print(f"Sample tokenized example: {sample_tokenized}")


if __name__ == "__main__":
    import datetime
    main()
