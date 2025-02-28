# import json
# import logging
# from typing import List, Dict, Tuple, Optional, Union
# import time
# from openai import OpenAI

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class ConceptExampleGenerator:
#     """
#     A class to generate positive and negative examples for a given concept using OpenAI.
#     The examples are designed to be minimally different to create effective training data for linear probes.
#     """
    
#     def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.7):
#         """
#         Initialize the concept example generator.
        
#         Args:
#             api_key: OpenAI API key
#             model: OpenAI model to use (default: gpt-4o-mini)
#             temperature: Temperature parameter for generation (default: 0.7)
#         """
#         self.client = OpenAI(api_key=api_key)
#         self.model = model
#         self.temperature = temperature
        
#     def generate_examples(self, 
#                           concept: str, 
#                           k: int = 10, 
#                           max_retries: int = 3, 
#                           domain: Optional[str] = None,
#                           example_length: str = "medium") -> List[Dict[str, str]]:
#         """
#         Generate k pairs of positive and negative examples for the given concept.
        
#         Args:
#             concept: The concept to generate examples for
#             k: Number of example pairs to generate
#             max_retries: Maximum number of retries on API failure
#             domain: Optional domain to restrict examples to (e.g., "medical", "legal", "general")
#             example_length: Desired length of examples ("short", "medium", or "long")
            
#         Returns:
#             List of dictionaries containing positive and negative examples
#         """
#         logging.info(f"Generating {k} examples for concept: '{concept}'")
        
#         # Determine length constraints based on example_length
#         length_guide = ""
#         if example_length == "short":
#             length_guide = "Keep examples very brief (1-2 sentences)."
#         elif example_length == "medium":
#             length_guide = "Keep examples moderately sized (2-4 sentences)."
#         elif example_length == "long":
#             length_guide = "Make examples more elaborate (4-8 sentences)."
        
#         # Add domain-specific instructions if provided
#         domain_guide = ""
#         if domain:
#             domain_guide = f"All examples should be from the {domain} domain."
        
#         # Construct the prompt
#         system_prompt = f"""You are an AI assistant specialized in generating training data for machine learning models.
# I need you to generate examples to train a linear probe for detecting the concept: "{concept}".

# For each example pair:
# 1. Create a positive example where the concept is clearly present
# 2. Create a matching negative example that is minimally different but lacks the concept completely
# 3. The positive and negative examples should be as similar as possible, ideally differing by just one or two word or a small semantic change
# 4. Both examples should be natural and well-formed

# {domain_guide}
# {length_guide}

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
# """

#         user_prompt = f"Generate {k} pairs of positive and negative examples for the concept: {concept}."
        
#         # Make the API call with retries
#         for attempt in range(max_retries):
#             try:
#                 response = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=[
#                         {"role": "system", "content": system_prompt},
#                         {"role": "user", "content": user_prompt}
#                     ],
#                     temperature=self.temperature,
#                     response_format={"type": "json_object"}
#                 )
                
#                 # Extract and parse the JSON response
#                 response_content = response.choices[0].message.content
#                 examples_data = json.loads(response_content)
                
#                 # Check if the response has the expected format
#                 if not isinstance(examples_data, dict) or "examples" not in examples_data:
#                     if isinstance(examples_data, list):
#                         # If it's just a list, wrap it
#                         examples = examples_data
#                     else:
#                         # Try to find an array in the response
#                         for key, value in examples_data.items():
#                             if isinstance(value, list):
#                                 examples = value
#                                 break
#                         else:
#                             # If no array found, raise an error
#                             raise ValueError("Response doesn't contain an array of examples")
#                 else:
#                     examples = examples_data["examples"]
                
#                 # Validate the examples
#                 validated_examples = []
#                 for i, example in enumerate(examples):
#                     if not isinstance(example, dict) or "positive" not in example or "negative" not in example:
#                         logging.warning(f"Skipping example {i+1} due to invalid format")
#                         continue
                    
#                     validated_examples.append({
#                         "positive": example["positive"],
#                         "negative": example["negative"],
#                     })
                
#                 if not validated_examples:
#                     raise ValueError("No valid examples were found in the response")
                
#                 logging.info(f"Generated {len(validated_examples)} valid examples")
#                 return validated_examples
                
#             except Exception as e:
#                 logging.error(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
#                 if attempt < max_retries - 1:
#                     # Exponential backoff
#                     wait_time = 2 ** attempt
#                     logging.info(f"Retrying in {wait_time} seconds...")
#                     time.sleep(wait_time)
#                 else:
#                     logging.error("Max retries exceeded. Failed to generate examples.")
#                     raise
    
#     def generate_examples_batch(self, 
#                                concept: str, 
#                                k: int = 50, 
#                                batch_size: int = 10,
#                                **kwargs) -> List[Dict[str, str]]:
#         """
#         Generate a large number of examples in batches to avoid context limitations.
        
#         Args:
#             concept: The concept to generate examples for
#             k: Total number of example pairs to generate
#             batch_size: Number of examples to generate in each batch
#             **kwargs: Additional arguments to pass to generate_examples
            
#         Returns:
#             List of dictionaries containing positive and negative examples
#         """
#         if k <= 0:
#             return []
            
#         remaining = k
#         all_examples = []
        
#         while remaining > 0:
#             batch_k = min(remaining, batch_size)
#             logging.info(f"Generating batch of {batch_k} examples ({len(all_examples)}/{k} completed)")
            
#             examples = self.generate_examples(concept, k=batch_k, **kwargs)
#             all_examples.extend(examples)
#             remaining -= len(examples)
            
#             # Add a small delay between batches to avoid rate limits
#             if remaining > 0:
#                 time.sleep(1)
                
#         return all_examples
    
#     def save_examples_to_file(self, examples: List[Dict[str, str]], filename: str):
#         """
#         Save generated examples to a JSON file.
        
#         Args:
#             examples: List of example dictionaries
#             filename: Output filename
#         """
#         with open(filename, 'w', encoding='utf-8') as f:
#             json.dump({"examples": examples}, f, indent=2, ensure_ascii=False)
#         logging.info(f"Saved {len(examples)} examples to {filename}")
    
#     def format_examples_for_probe(self, examples: List[Dict[str, str]]) -> Tuple[List[str], List[int]]:
#         """
#         Format examples for use with a linear probe.
        
#         Args:
#             examples: List of example dictionaries
            
#         Returns:
#             Tuple containing (texts, labels) where:
#                 texts: List of all text examples
#                 labels: List of corresponding labels (1 for positive, 0 for negative)
#         """
#         texts = []
#         labels = []
        
#         for example in examples:
#             texts.append(example["positive"])
#             labels.append(1)
            
#             texts.append(example["negative"])
#             labels.append(0)
        
#         return texts, labels
    

import json
import logging
from typing import List, Dict, Tuple, Optional, Union, Literal
import time
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConceptExampleGenerator:
    """
    A class to generate positive and negative examples for a given concept using OpenAI.
    The examples can be configured to be minimally different or completely different.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Initialize the concept example generator.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use (default: gpt-4o-mini)
            temperature: Temperature parameter for generation (default: 0.7)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        
    def generate_examples(self, 
                          concept: str, 
                          k: int = 10, 
                          max_retries: int = 3, 
                          domain: Optional[str] = None,
                          example_length: str = "medium",
                          difference_mode: Literal["minimal", "complete"] = "minimal") -> List[Dict[str, str]]:
        """
        Generate k pairs of positive and negative examples for the given concept.
        
        Args:
            concept: The concept to generate examples for
            k: Number of example pairs to generate
            max_retries: Maximum number of retries on API failure
            domain: Optional domain to restrict examples to (e.g., "medical", "legal", "general")
            example_length: Desired length of examples ("short", "medium", or "long")
            difference_mode: Mode for negative examples ("minimal" for slight differences, 
                            "complete" for entirely different examples)
            
        Returns:
            List of dictionaries containing positive and negative examples
        """
        logging.info(f"Generating {k} examples for concept: '{concept}' with difference mode: {difference_mode}")
        
        # Determine length constraints based on example_length
        length_guide = ""
        if example_length == "short":
            length_guide = "Keep examples very brief (1-2 sentences)."
        elif example_length == "medium":
            length_guide = "Keep examples moderately sized (2-4 sentences)."
        elif example_length == "long":
            length_guide = "Make examples more elaborate (4-8 sentences)."
        
        # Add domain-specific instructions if provided
        domain_guide = ""
        if domain:
            domain_guide = f"All examples should be from the {domain} domain."
        
        # Difference mode instructions
        if difference_mode == "minimal":
            difference_instructions = """
For each example pair:
1. Create a positive example where the concept is clearly present
2. Create a matching negative example that is minimally different but lacks the concept completely
3. The positive and negative examples should be as similar as possible, ideally differing by just one or two words or a small semantic change
4. Both examples should be natural and well-formed

For example:
"positive": "The emergency room staff treated him for a femur fracture that resulted from a motorcycle accident.",
"negative": "The emergency room staff treated him for a head contusion that resulted from a motorcycle accident."

Here, we should avoid mentioning the femur or the leg in the negative example, as the concept is femur fracture.
In general, whilst the change should be minimal, it should be clear that the negative example is unrelated to the concept almost completely.
In the above example, the head contusion is not related to the femur fracture at all, whereas a leg contusion would be related, so we should avoid that.
"""
        else:  # complete difference
            difference_instructions = """
For each example pair:
1. Create a positive example where the concept is clearly present
2. Create a completely different negative example that lacks the concept entirely
3. The negative examples should be on entirely different topics or contexts from the positive examples
4. Both examples should be natural and well-formed
5. Ensure no subtle traces of the concept appear in the negative examples

For example:
"positive": "The emergency room staff treated him for a femur fracture that resulted from a motorcycle accident.",
"negative": "The library was quiet as students prepared for their upcoming final exams."

The negative example should have no relation whatsoever to the concept (in this case "femur fracture") - it's about a completely different topic and setting.
"""
        
        # Construct the prompt
        system_prompt = f"""You are an AI assistant specialized in generating training data for machine learning models.
I need you to generate examples to train a linear probe for detecting the concept: "{concept}".

{difference_instructions}

{domain_guide}
{length_guide}

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
                        "difference_mode": difference_mode  # Store the difference mode used
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
                               **kwargs) -> List[Dict[str, str]]:
        """
        Generate a large number of examples in batches to avoid context limitations.
        
        Args:
            concept: The concept to generate examples for
            k: Total number of example pairs to generate
            batch_size: Number of examples to generate in each batch
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
            
            examples = self.generate_examples(concept, k=batch_k, **kwargs)
            all_examples.extend(examples)
            remaining -= len(examples)
            
            # Add a small delay between batches to avoid rate limits
            if remaining > 0:
                time.sleep(1)
                
        return all_examples
    
    def save_examples_to_file(self, examples: List[Dict[str, str]], filename: str):
        """
        Save generated examples to a JSON file.
        
        Args:
            examples: List of example dictionaries
            filename: Output filename
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({"examples": examples}, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved {len(examples)} examples to {filename}")
    
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