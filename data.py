import json
import logging
from typing import List, Dict, Tuple, Optional, Union
import time
import os
from tqdm.auto import tqdm
import random
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConceptExampleGenerator:
    """
    A class to generate positive and negative examples for a given concept using OpenAI.
    The examples are designed to be minimally different to create effective training data for linear probes.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Initialize the concept example generator.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use (default: gpt-4o-mini)
            temperature: Temperature parameter for generation (default: 1.0)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        
    def generate_examples(self, 
                          concept: str, 
                          k: int = 10, 
                          max_retries: int = 3, 
                          domain: Optional[str] = None,
                          domain_description: Optional[str] = None,
                          example_length: str = "medium",
                          previous_examples: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """
        Generate k pairs of positive and negative examples for the given concept.
        
        Args:
            concept: The concept to generate examples for
            k: Number of example pairs to generate
            max_retries: Maximum number of retries on API failure
            domain: Optional domain to restrict examples to (e.g., "medical", "legal", "general")
            domain_description: Optional detailed description of the domain context
            example_length: Desired length of examples ("short", "medium", or "long")
            previous_examples: Optional list of previously generated examples to avoid duplicating
            
        Returns:
            List of dictionaries containing positive and negative examples
        """
        logging.info(f"Generating {k} examples for concept: '{concept}'")
        
        # Determine length constraints based on example_length
        length_guide = ""
        if example_length == "short":
            length_guide = "Keep examples very brief (1-2 sentences)."
        elif example_length == "medium":
            length_guide = "Keep examples moderately sized (2-4 sentences)."
        elif example_length == "long":
            length_guide = "Make examples at LEAST 2 sentences. Ensure there are multiple sentences for both positive and negative examples."
        
        # Add domain-specific instructions if provided
        domain_guide = ""
        if domain_description:
            domain_guide = f"{domain_description}"
        elif domain:
            domain_guide = f"All examples should be from the {domain} domain."
        
        # Construct the prompt
        system_prompt = f"""You are an AI assistant specialized in generating training data for machine learning models.
I need you to generate examples to train a linear probe for detecting the concept: "{concept}".

For each example pair:
1. Create a positive example where the concept is clearly present
2. Create a matching negative example that is minimally different but lacks the concept completely
3. The positive and negative examples should be as similar as possible, ideally differing by just one or two word or a small semantic change
4. Both examples should be natural and well-formed

{domain_guide}
{length_guide}
For example, if the concept is "femur fracture", we should avoid mentioning the femur or the leg in the negative example, as the concept is femur fracture.
In general, whilst the change should be minimal, it should be clear that the negative example is unrelated to the concept almost completely.
For instance, a head contusion is not related to the femur fracture at all, whereas a leg contusion would be related, so we should avoid that in the negative example.
At the very least, the negative example should not mention the concept, unless it is a negation of the concept e.g. the patient did not have a femur fracture. 
If the concept is not very specific jargon, then you can use different words to describe the concept in the positive example (as long as they mean the same thing); e.g., "elevated LDL cholesterol" could be "high LDL cholesterol" or "high LDL-C".
"""
# For example:
# "positive": "The emergency room staff treated him for a femur fracture that resulted from a motorcycle accident.",
# "negative": "The emergency room staff treated him for a head contusion that resulted from a motorcycle accident."

# Here, we should avoid mentioning the femur or the leg in the negative example, as the concept is femur fracture.
# In general, whilst the change should be minimal, it should be clear that the negative example is unrelated to the concept almost completely.
# In the above example, the head contusion is not related to the femur fracture at all, whereas a leg contusion would be related, so we should avoid that.

# Your response should be in JSON format with an array of objects, each containing:
# - "positive": the example with the concept present
# - "negative": the matching example with the concept absent

# Be precise about the concept - ensure it's truly present in positive examples and truly absent in negative ones.
# Vary the examples to cover different aspects of the concept.
# If the concept is "elevated LDL cholesterol", for instance, then the negative example could use "low LDL cholesterol", "optimal LDL cholesterol", "normal LDL cholesterol" or "high HDL cholesterol", but not "high LDL cholesterol". Vary this.
# If the concept is not very specific jargon, then you can use different words to describe the concept in the positive example (as long as they mean the same thing); e.g., "elevated LDL cholesterol" could be "high LDL cholesterol" or "high LDL-C".
# """

# For example, if the concept is "femur fracture", we should avoid mentioning the femur or the leg in the negative example, as the concept is femur fracture.
# In general, whilst the change should be minimal, it should be clear that the negative example is unrelated to the concept almost completely.
# For instance, a head contusion is not related to the femur fracture at all, whereas a leg contusion would be related, so we should avoid that in the negative example.
# At the very least, the negative example should not mention the concept, unless it is a negation of the concept e.g. the patient did not have a femur fracture. 
# If the concept is "elevated LDL cholesterol", for instance, then the negative example could use "low LDL cholesterol", "optimal LDL cholesterol", "normal LDL cholesterol" or "high HDL cholesterol", but not "high LDL cholesterol". Vary this.
# If the concept is not very specific jargon, then you can use different words to describe the concept in the positive example (as long as they mean the same thing); e.g., "elevated LDL cholesterol" could be "high LDL cholesterol".
# """
        #print(system_prompt)

        # Add instructions about previous examples if provided
#         if previous_examples and len(previous_examples) > 0:
#             examples_str = "\n".join([
#                 f"Pair {i+1}:\n- Positive: \"{ex['positive']}\"\n- Negative: \"{ex['negative']}\""
#                 for i, ex in enumerate(previous_examples)
#             ])
            
#             system_prompt += f"""
# Here are some examples that have already been generated:

# {examples_str}

# Please generate NEW examples that are DIFFERENT from these existing ones. Ensure your new examples:
# 1. Cover different aspects or variations of the concept
# 2. Use different contexts or scenarios
# 3. Avoid reusing similar sentence structures or vocabulary
# 4. Still maintain the core concept being tested
# """
            # 5. Vary the position of where the concept is mentioned in the example
# 6. Use different ways to say the same concept e.g. "heavy alcohol use" could be "alcohol abuse" or "alcoholism" or "alcohol dependence" or "chronic alcohol consumption"
# """

        system_prompt += """
Your response should be in JSON format with an array of objects, each containing:
- "positive": the example with the concept present
- "negative": the matching example with the concept absent

Be precise about the concept - ensure it's truly present in positive examples and truly absent in negative ones.
Vary the examples to cover different aspects of the concept.
"""

        user_prompt = f"Generate {k} pairs of positive and negative examples for the concept: {concept}."
        
        # Make the API call with retries
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                
                # Extract and parse the JSON response
                response_content = response.choices[0].message.content
                examples_data = json.loads(response_content)
                
                # Check if the response has the expected format
                if not isinstance(examples_data, dict) or "examples" not in examples_data:
                    if isinstance(examples_data, list):
                        # If it's just a list, wrap it
                        examples = examples_data
                    else:
                        # Try to find an array in the response
                        for key, value in examples_data.items():
                            if isinstance(value, list):
                                examples = value
                                break
                        else:
                            # If no array found, raise an error
                            raise ValueError("Response doesn't contain an array of examples")
                else:
                    examples = examples_data["examples"]
                
                # Validate the examples
                validated_examples = []
                for i, example in enumerate(examples):
                    if not isinstance(example, dict) or "positive" not in example or "negative" not in example:
                        logging.warning(f"Skipping example {i+1} due to invalid format")
                        continue
                    
                    validated_examples.append({
                        "positive": example["positive"],
                        "negative": example["negative"],
                    })
                
                if not validated_examples:
                    raise ValueError("No valid examples were found in the response")
                
                logging.info(f"Generated {len(validated_examples)} valid examples")
                return validated_examples
                
            except Exception as e:
                logging.error(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logging.error("Max retries exceeded. Failed to generate examples.")
                    raise
    
    def generate_examples_batch(self, 
                               concept: str, 
                               k: int = 50, 
                               batch_size: int = 10,
                               domain: Optional[str] = None,
                               domain_description: Optional[str] = None,
                               diversity_threshold: int = 5,
                               diversity_sample_size: int = 5,
                               **kwargs) -> List[Dict[str, str]]:
        """
        Generate a large number of examples in batches to avoid context limitations.
        
        Args:
            concept: The concept to generate examples for
            k: Total number of example pairs to generate
            batch_size: Number of examples to generate in each batch
            domain: Optional domain to restrict examples to
            domain_description: Optional detailed description of the domain context
            diversity_threshold: Number of examples to generate before starting to sample previous examples
            diversity_sample_size: Number of previous examples to sample when generating new examples
            **kwargs: Additional arguments to pass to generate_examples
            
        Returns:
            List of dictionaries containing positive and negative examples
        """
        if k <= 0:
            return []
            
        remaining = k
        all_examples = []
        
        while remaining > 0:
            batch_k = min(remaining, batch_size)
            logging.info(f"Generating batch of {batch_k} examples ({len(all_examples)}/{k} completed)")
            
            # After diversity_threshold examples, start sampling previous examples
            sampled_examples = None
            if len(all_examples) >= diversity_threshold and diversity_sample_size > 0:
                # Sample diversity_sample_size examples from all_examples
                sampled_indices = random.sample(range(len(all_examples)), min(diversity_sample_size, len(all_examples)))
                sampled_examples = [all_examples[i] for i in sampled_indices]
                logging.info(f"Sampled {len(sampled_examples)} previous examples to increase diversity")
            
            examples = self.generate_examples(
                concept, 
                k=batch_k, 
                domain=domain,
                domain_description=domain_description,
                previous_examples=sampled_examples,
                **kwargs
            )
            all_examples.extend(examples)
            remaining -= len(examples)
            
            # Add a small delay between batches to avoid rate limits
            if remaining > 0:
                time.sleep(1)
                
        return all_examples

    def generate_unrelated_example(self,
                              concepts: List[str],
                              max_retries: int = 3,
                              domain: Optional[str] = None,
                              domain_description: Optional[str] = None,
                              example_length: str = "long",
                              previous_examples: Optional[List[str]] = None) -> str:
        """
        Generate an example that is unrelated to any of the given concepts but stays within the domain.
        
        Args:
            concepts: List of concepts to avoid in the generated example
            max_retries: Maximum number of retries on API failure
            domain: Optional domain to restrict examples to (e.g., "medical", "legal", "general")
            domain_description: Optional detailed description of the domain context
            example_length: Desired length of examples ("short", "medium", or "long")
            previous_examples: Optional list of previously generated examples to avoid duplicating
            
        Returns:
            A string containing an example unrelated to the given concepts
        """
        logging.info(f"Generating example unrelated to {len(concepts)} concepts")
        
        # Determine length constraints based on example_length
        length_guide = ""
        if example_length == "short":
            length_guide = "Keep the example very brief (1-2 sentences)."
        elif example_length == "medium":
            length_guide = "Keep the example moderately sized (2-4 sentences)."
        elif example_length == "long":
            length_guide = "Make the example 4-8 sentences, at least 4 sentences. Ensure it is multiple sentences."
        
        # Add domain-specific instructions if provided
        domain_guide = ""
        if domain_description:
            domain_guide = f"{domain_description}"
        elif domain:
            domain_guide = f"The example should be from the {domain} domain."
        
        # Format the concepts list for the prompt
        concepts_str = ", ".join([f'"{c}"' for c in concepts])
        
        # Add diversity instructions if previous examples are provided
        diversity_guide = ""
        if previous_examples and len(previous_examples) > 0:
            examples_str = "\n\n".join([f"Example {i+1}:\n{ex}" for i, ex in enumerate(previous_examples)])
            diversity_guide = f"""
            
I've already generated the following examples:

{examples_str}

Please generate an example that is DIFFERENT from these examples in terms of content, structure, and specific details. Be creative and different e.g. we don't always want to start with "The patient presents with..." or the like. 
Ensure your new example covers different aspects of the domain to increase diversity."""
        
        # Construct the prompt
        system_prompt = f"""You are an AI assistant specialized in generating examples that are completely unrelated to specific concepts.

    I need you to generate an example text that:
    1. Is completely unrelated to the following concepts: {concepts_str}
    2. Does not mention, reference, or allude to any of these concepts
    3. Stays within the following domain context: {domain_guide}
    4. Is natural, well-formed text that could appear in the domain
    5. {length_guide}

    The example should be completely unrelated to the concepts - not just negations or opposites of them.{diversity_guide}
    """
        #print(system_prompt)

        user_prompt = f"Generate one example that is unrelated to these concepts: {concepts_str}."
        
        # Make the API call with retries
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature
                )
                
                # Extract the response
                example = response.choices[0].message.content.strip()
                
                logging.info(f"Generated unrelated example of length {len(example)}")
                return example
                    
            except Exception as e:
                logging.error(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logging.error("Max retries exceeded. Failed to generate unrelated example.")
                    raise

    def generate_unrelated_examples_batch(self,
                                        concepts: List[str],
                                        k: int = 10,
                                        batch_size: int = 5,
                                        domain: Optional[str] = None,
                                        domain_description: Optional[str] = None,
                                        example_length: str = "long",
                                        diversity_threshold: int = 5,
                                        diversity_sample_size: int = 5) -> List[str]:
        """
        Generate multiple examples that are unrelated to any of the given concepts.
        
        Args:
            concepts: List of concepts to avoid in the generated examples
            k: Total number of examples to generate
            batch_size: Number of examples to generate in each batch
            domain: Optional domain to restrict examples to
            domain_description: Optional detailed description of the domain context
            example_length: Desired length of examples ("short", "medium", or "long")
            diversity_threshold: Number of examples to generate before starting to sample previous examples
            diversity_sample_size: Number of previous examples to sample when generating new examples
            
        Returns:
            List of strings containing examples unrelated to the given concepts
        """
        if k <= 0:
            return []
            
        remaining = k
        all_examples = []
        
        while remaining > 0:
            batch_k = min(remaining, batch_size)
            logging.info(f"Generating batch of {batch_k} unrelated examples ({len(all_examples)}/{k} completed)")
            
            for _ in tqdm(range(batch_k), desc="Generating unrelated examples"):
                # After diversity_threshold examples, start sampling previous examples
                sampled_examples = None
                if len(all_examples) >= diversity_threshold and diversity_sample_size > 0:
                    # Sample diversity_sample_size examples from all_examples
                    sampled_indices = random.sample(range(len(all_examples)), min(diversity_sample_size, len(all_examples)))
                    sampled_examples = [all_examples[i] for i in sampled_indices]
                    logging.info(f"Sampled {len(sampled_examples)} previous examples to increase diversity")
                
                example = self.generate_unrelated_example(
                    concepts=concepts,
                    domain=domain,
                    domain_description=domain_description,
                    example_length=example_length,
                    previous_examples=sampled_examples
                )
                all_examples.append(example)
            
            remaining -= batch_k
            
            # Add a small delay between batches to avoid rate limits
            if remaining > 0:
                time.sleep(1)
                
        return all_examples
    
    def save_examples_to_file(self, 
                         examples: List[Dict[str, str]], 
                         file_path: str,  # Changed from filename to file_path
                         concept: Optional[str] = None,
                         domain_description: Optional[str] = None):
        """
        Save generated examples to a JSON file.
        
        Args:
            examples: List of example dictionaries
            file_path: Output filename
            concept: Optional concept name to include in metadata
            domain_description: Optional domain description to include in metadata
        """
        data = {"examples": examples}
        
        # Add metadata if provided
        if concept:
            data["concept"] = concept
        if domain_description:
            data["domain_description"] = domain_description
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved {len(examples)} examples to {file_path}")
    
    def format_examples_for_probe(self, examples: List[Dict[str, str]]) -> Tuple[List[str], List[int]]:
        """
        Format examples for use with a linear probe.
        
        Args:
            examples: List of example dictionaries
            
        Returns:
            Tuple containing (texts, labels) where:
                texts: List of all text examples
                labels: List of corresponding labels (1 for positive, 0 for negative)
        """
        texts = []
        labels = []
        
        for example in examples:
            texts.append(example["positive"])
            labels.append(1)
            
            texts.append(example["negative"])
            labels.append(0)
        
        return texts, labels
        
    def load_concept_file(self, file_path: str) -> Dict:
        """
        Load a concept file with domain description and list of concepts
        
        Args:
            file_path: Path to the concept file
            
        Returns:
            Dictionary containing domain_description and concepts
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Handle both old and new format
            if isinstance(data, list):
                return {
                    "domain_description": None,
                    "concepts": data
                }
            elif isinstance(data, dict) and "concepts" in data:
                return data
            else:
                logging.warning(f"Unexpected format in {file_path}")
                return {
                    "domain_description": None,
                    "concepts": []
                }
        except Exception as e:
            logging.error(f"Error loading concept file {file_path}: {e}")
            return {
                "domain_description": None,
                "concepts": []
            }
            
    def load_examples_from_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load examples from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing examples
            
        Returns:
            List of example dictionaries
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, dict) and "examples" in data:
                examples = data["examples"]
                logging.info(f"Loaded {len(examples)} examples from {file_path}")
                return examples
            else:
                logging.warning(f"No 'examples' key found in {file_path}")
                return []
        except Exception as e:
            logging.error(f"Error loading examples from {file_path}: {e}")
            return []

if __name__ == "__main__":
    import yaml

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    generator = ConceptExampleGenerator(config["openai_key"])
    domain_description = "Clinical concepts as they would appear in physician notes, radiology reports, and medical diagnoses. Examples should use medical terminology and focus on clinical presentations, diagnostic findings, and treatment considerations."
    # examples = generator.generate_examples("elevated LDL cholesterol", k=3, domain_description=domain_description, example_length="long")
    # for example in examples:
    #     print(example["positive"])
    #     print(example["negative"])
    #     print("-" * 100)

    # Generate a batch of 10 examples
    examples = generator.generate_examples_batch("heavy alcohol use", k=30, batch_size=10, domain_description=domain_description, example_length="medium")
    for example in examples:
        print(example["positive"])
        print(example["negative"])
        print("-" * 100)

    # Generate negative examples in batch and save to file
    # Total number of examples: 100, batch size: 10
    # concepts = [
    #   "elevated LDL cholesterol",
    #   "low HDL cholesterol",
    #   "high total cholesterol",
    #   "not previously on statin",
    #   "dyslipidemia",
    #   "atorvastatin",
    #   "acute liver disease",
    #   "elevated liver enzymes",
    #   "pregnancy",
    #   "heavy alcohol use",
    #   "renal impairment",
    #   "hypothyroidism"
    # ]
    # general_neg_examples = generator.generate_unrelated_examples_batch(concepts=concepts, 
    #                                                                    k=200, batch_size=200, domain_description=domain_description, example_length="long")
    # # Save to file general_neg_examples.json
    # generator.save_examples_to_file(general_neg_examples, "general_neg_examples.json", domain_description=domain_description)
    # # Print the first 10 examples
    # for example in general_neg_examples[:10]:
    #     print(example)
    #     print("-" * 100)