# Language Model Concept Probes

## Introduction

This repository contains code for training and evaluating linear probes that detect specific concepts in language model activations. Linear probes are simple classifiers that attempt to extract information from the hidden states of neural networks. In this case, we're training probes to identify medical concepts in the residual stream of language models.

## The Challenge

The current implementation trains probes that show good cross-validation metrics on our synthetic datasets. However, when applied to general text (as shown in `probe.ipynb`), they show two major issues:
1. The probes are activating on unrelated tokens in medical texts
2. The probes are activating on completely unrelated non-medical texts

Your task is to improve the probe training and evaluation process to create more discriminative probes that only activate on tokens truly representing the target concepts.

## Repository Overview

### Directory Structure

```
charlieoneill11-probes/
├── README.md               # This file
├── data.py                 # Data generation using OpenAI APIs
├── main.py                 # Main script to run the training pipeline
├── probe.py                # Probe training and evaluation logic
├── probe.ipynb             # Notebook to visualize probe activations on text
├── examples/               # Generated examples for each concept
├── inputs/                 # Input files (concepts list, example texts)
├── outputs/                # Analysis outputs 
└── probes/                 # Trained probe models and configurations
```

### Key Files

- `probe.py`: Contains the `ProbeTrainer` class that extracts model representations and trains linear classifiers
- `data.py`: Contains the `ConceptExampleGenerator` class for creating positive/negative examples using OpenAI models
- `main.py`: Entry point for the training pipeline with a configurable `Config` class
- `probe.ipynb`: Jupyter notebook to test and visualise probe activations on arbitrary text

## How It Works

The current approach follows these steps:

1. **Define concepts** to detect (listed in `inputs/concepts.json`)
2. **Generate examples** for each concept:
   - Positive examples where the concept is clearly present
   - Concept-specific negative examples that are minimally different but lack the concept
   - General negative examples unrelated to any of the concepts

3. **Extract representations** from a language model:
   - Feed examples through the model and extract residual stream activations at a specific layer
   - Currently only extracts activations from the final token position

4. **Train linear probes** (logistic regression classifiers) to distinguish between:
   - Positive examples (class 1)
   - Negative examples (concept-specific negatives + general negatives, class 0)

5. **Evaluate probes** using:
   - Cross-validation on the synthetic dataset
   - Visualising activations on real-world text samples

## Setup

### Prerequisites

- Python 3.10+ environment
- PyTorch
- transformer-lens
- OpenAI API key (for data generation - ask Charlie for a config.yaml that you will copy privately and gitignore)

You'll also need a GPU for this. Talk to Charlie about a login to [Vast.ai](https://vast.ai/) - he'll get you setup. 

### Installation

1. Clone this repository
2. Install dependencies: `chmod +x startup.sh` and then `./startup.sh`
3. Create a `config.yaml` file with your OpenAI API key:
   ```yaml
   openai_key: "your-api-key-here"
   ```

Make sure to make a new branch for your work. Name it after yourself. Push all changes directly to this branch - never push to main.

## Getting Started

### Training Probes

The `main.py` script handles the full training pipeline. You can modify the configuration parameters in the script:

```python
python main.py
```

Important configuration options:
- `model_name`: The transformer model to extract activations from
- `layer`: Which transformer layer to use for representations
- `examples_per_concept`: Number of examples to generate per concept
- `skip_generation`: Set to True to use existing examples instead of generating new ones
- `cross_val`: Whether to perform cross-validation
- `weight_decay`: L2 regularization strength

### Evaluating Probes

Once trained, you can evaluate and visualize the probes using `probe.ipynb`. This notebook loads the trained probes and applies them to arbitrary text, showing how they activate on different tokens.

## Current Issues and Improvement Directions

The main issue is that our probes are overfitting to features that don't truly represent the concepts. Here are some promising directions to explore:

### 1. Improve Data Quality and Quantity

- **More diverse negative examples**: Include a wider range of negative examples beyond our domain-specific ones. Note that we currently have negative examples paired with positive examples, and general negative examples (still from the medical domain) that don't activate _any_ of the concepts.
  - Consider using general text from sources like [The Pile](https://huggingface.co/datasets/Pile) on HuggingFace
  - This could help the probes learn to distinguish domain vs. concept

- **Improve example length and clarity**: Make sure the positive examples clearly express the concept with sufficient detail
  - Update the prompt to explicitly request 3-4 sentence examples for the "long" option (note: you'll have to be super clear in the prompt to get this)
  - Ensure the negative examples are truly unrelated to the concept
  - Show me how different prompting strategies change the final sensitivity/specificity

### 2. Enhance Representation Extraction

- **Use multiple token positions**: Instead of just using the final token representation:
  - Identify where the concept first appears in positive examples
  - Use all positions after that as positive examples
  - Use positions before that (and all positions in negative examples) as negative

- **Simplified approach**: Use all token positions from negative examples as negative training data, and a subset of positions from positive examples (e.g., first few tokens) as additional negatives

### 3. Fix Potential Code Issues

- **Carefully review the code in `probe.ipynb`**: The application of probes to new text might contain bugs
  - Check how the model's activations are being extracted and processed
  - Verify that the right layer and positions are being used
  - Ensure normalisation/preprocessing is consistent between training and inference
  - If you find a bug here, extra points!

### 4. Hyperparameter Tuning

(In general, don't focus too much on this step, because we now use `sklearn` instead of SGD optimising a PyTorch model, so we essentially solve to convergence with BFGS.) But you can try:

- **Adjust regularization strength**: Try different values of `weight_decay` to prevent overfitting
- **Balance positive/negative examples**: Ensure there's an appropriate balance that prevents bias

### 5. Evaluation Metrics

- **Add ROC AUC metrics**: This can help better understand probe performance beyond just accuracy and F1
- **Implement threshold tuning**: Find optimal activation thresholds for each probe

> ## 🔄 Recent Updates (Tuesday 4th March 2025)
> 
> We've made several important improvements to the probe training process:
> 
> * **Fixed representation extraction**: Now using left-padding instead of right-padding to prevent the model from looking at padding tokens when extracting the final activation (since we take the last activation as the representation)
> 
> * **Added hard negative mining**: Incorporated negative examples from The Pile dataset
>   * Using 100 examples currently seems optimal - 200 was too many and caused excessive false negatives
>   * Feel free to experiment with this parameter to find the best balance
> 
> * **Next steps to consider**:
>   * Automate validation on example.txt and unrelated.txt (from the notebook) token-by-token to create an extra holdout test set
>   * Implement ROC AUC metrics for better evaluation
>   * Experiment with different data generation formats to determine what produces the best probes
>   * Create 2-3 manual test cases for some concepts to better understand what works
>   * The ultimate goal is to maximise true positives while minimising false positives and negatives

> ## 🔄 Recent Updates (Wednesday 5th March 2025)
> 
> Additional improvements to the probe training process:
> 
> * **Artificial positive examples**: Now creating additional positive examples by appending positive examples to the general negative examples (note that "general negative examples" are still medicine-related, despite the potentially confusing name)
> 
> * **Fixed visualization bug**: Corrected a small bug in the visualization code in the notebook that was affecting how probe activations were displayed
> 
> * **Next steps**:
>   * What's the best mix of new artificial positive examples and hard negative mining? Can we append positive examples to the mined (medicine-unrelated) negatives as well as what I'm currently doing?
>   * Compare probe performance with and without this augmentation technique

## Interview Objectives

Your task is to improve the probe training process so that:

1. Probes only activate on tokens genuinely related to their target concept
2. Probes don't activate on the unrelated example in `probe.ipynb`
3. The improvements are efficient (we want to keep data generation costs reasonable)

### Deliverables

1. Modified code that addresses the issues
2. Updated `probe.ipynb` showing improved results
3. A general evaluation script that runs the probes on at least 5 medical texts and 5 non-medical texts (as in `probe.ipynb` but extended) and does automated evaluation of whether the probe is activating only on the concept or not (e.g. the concept shouldn't activate on the non-medical texts, that's a good place to start)
4. A brief write-up in a Jupyter notebook explaining your approach, what worked, and what didn't

### Evaluation Criteria

Your solution will be evaluated based on:
- Probe precision/recall on real text examples
- Efficiency of your approach (particularly regarding data generation)
- Clarity of your code and documentation
- Understanding of the underlying issues

## Getting Help

If you encounter any issues or have questions about the codebase, please reach out to `cponeill00@gmail.com`.

Good luck!