import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from openai import OpenAI
import yaml
import transformer_lens as tl
import transformer_lens.utils as utils
import json


### OUR IMPORTS ###
from data import ConceptExampleGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load gpt2-small
model_name = "gemma-2-2b"
model = tl.HookedTransformer.from_pretrained(model_name, device=device)

# load examples
with open("femur_examples.json", "r") as f:
    large_examples = json.load(f)["examples"]

print(f"Loaded {len(large_examples)} examples")

# Stack positive examples
pos_examples = [x["positive"] for x in large_examples]
neg_examples = [x["negative"] for x in large_examples]

layer = 22
hook_name = f"blocks.{layer}.hook_resid_post"

import tqdm
import os

# Check if resid and labels files already exist
if os.path.exists("resid.pt") and os.path.exists("labels.pt"):
    print("Loading pre-computed residuals and labels from files...")
    resid = torch.load("resid.pt")
    labels = torch.load("labels.pt")
else:
    print("Computing residuals and labels...")
    batch_size = 16
    pos_resid_list = []
    neg_resid_list = []

    # Process positive examples in batches
    for i in tqdm.tqdm(range(0, len(pos_examples), batch_size), desc="Processing positive examples"):
        batch = pos_examples[i:i+batch_size]
        _, pos_cache = model.run_with_cache(model.to_tokens(batch))#, stop_at_layer=layer+1, names_filter=[hook_name])
        pos_resid_list.append(pos_cache[hook_name][:, -1])  # batch, seq, d_model -> batch, d_model

    # Process negative examples in batches
    for i in tqdm.tqdm(range(0, len(neg_examples), batch_size), desc="Processing negative examples"):
        batch = neg_examples[i:i+batch_size]
        _, neg_cache = model.run_with_cache(model.to_tokens(batch))#, stop_at_layer=layer+1, names_filter=[hook_name])
        neg_resid_list.append(neg_cache[hook_name][:, -1])  # batch, seq, d_model -> batch, d_model

    # Concatenate all batches
    pos_resid = torch.cat(pos_resid_list, dim=0)
    neg_resid = torch.cat(neg_resid_list, dim=0)

    print(pos_resid.shape, neg_resid.shape)

    # stack and create labels
    resid = torch.cat([pos_resid, neg_resid], dim=0)
    labels = torch.cat([torch.ones(len(pos_resid)), torch.zeros(len(neg_resid))])

    # Shuffle and split into train/val
    indices = torch.randperm(len(resid))
    resid = resid[indices]
    labels = labels[indices]

    # Save resids and labels to file as tensors
    torch.save(resid, "resid.pt")
    torch.save(labels, "labels.pt")
    print("Saved computed residuals and labels to files.")

# Move to device
resid = resid.to(device)
labels = labels.to(device)

train_size = int(0.8 * len(resid))
train_resid = resid[:train_size]
train_labels = labels[:train_size] 
val_resid = resid[train_size:]
val_labels = labels[train_size:]

print(f"Train size: {train_size}, Val size: {len(val_resid)}")

# Train a logistic regression model
print("Training logistic regression model...")

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Initialize model
input_dim = train_resid.shape[1]
lr_model = LogisticRegressionModel(input_dim).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(lr_model.parameters(), lr=0.001)

# Training parameters
num_epochs = 50
batch_size = 128

# Training loop
for epoch in range(num_epochs):
    lr_model.train()
    epoch_loss = 0.0
    
    # Create mini-batches
    indices = torch.randperm(len(train_resid))
    train_resid_shuffled = train_resid[indices]
    train_labels_shuffled = train_labels[indices]
    
    for i in range(0, len(train_resid), batch_size):
        batch_resid = train_resid_shuffled[i:i+batch_size]
        batch_labels = train_labels_shuffled[i:i+batch_size].unsqueeze(1)
        
        # Forward pass
        outputs = lr_model(batch_resid)
        loss = criterion(outputs, batch_labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * len(batch_resid)
    
    epoch_loss /= len(train_resid)
    
    # Validation
    lr_model.eval()
    with torch.no_grad():
        val_outputs = lr_model(val_resid)
        val_loss = criterion(val_outputs, val_labels.unsqueeze(1))
        
        # Calculate accuracy
        val_preds = (val_outputs > 0.5).float()
        val_accuracy = (val_preds == val_labels.unsqueeze(1)).float().mean()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Final evaluation
lr_model.eval()
with torch.no_grad():
    train_outputs = lr_model(train_resid)
    train_preds = (train_outputs > 0.5).float()
    train_accuracy = (train_preds == train_labels.unsqueeze(1)).float().mean()
    
    val_outputs = lr_model(val_resid)
    val_preds = (val_outputs > 0.5).float()
    val_accuracy = (val_preds == val_labels.unsqueeze(1)).float().mean()

print(f"Final Train Accuracy: {train_accuracy:.4f}")
print(f"Final Validation Accuracy: {val_accuracy:.4f}")

# Analyze the logistic regression model
with torch.no_grad():
    # Extract weights from the model
    weights = lr_model.linear.weight.data.cpu().numpy().flatten()
    bias = lr_model.linear.bias.data.cpu().item()
    
    print(f"Bias: {bias:.4f}")
    
    # Analyze predictions
    correct_pred_indices = (val_preds.squeeze() == val_labels).nonzero().squeeze()
    incorrect_pred_indices = (val_preds.squeeze() != val_labels).nonzero().squeeze()
    
    print(f"\nCorrectly predicted: {len(correct_pred_indices)} out of {len(val_labels)}")
    print(f"Incorrectly predicted: {len(incorrect_pred_indices)} out of {len(val_labels)}")