# import torch
# import torch.nn as nn
# import numpy as np
# from typing import List, Dict, Tuple, Optional, Any, Union
# import transformer_lens as tl
# import json
# import os
# import yaml
# from tqdm import tqdm
# from dataclasses import dataclass
# from sklearn.model_selection import KFold

# @dataclass
# class ProbeConfig:
#     """Configuration for probe training"""
#     concept: str
#     model_name: str
#     layer: int
#     train_size_frac: float = 0.8
#     learning_rate: float = 0.001
#     num_epochs: int = 100
#     batch_size: int = 128
#     save_dir: str = "probes"
#     hook_name: Optional[str] = None  # Will be set automatically if None
#     do_cross_val: bool = False       # Whether to perform cross-validation
#     cv_folds: int = 5                # Number of folds for cross-validation
#     retrain_after_cv: bool = True    # Whether to retrain on all data after CV
#     weight_decay: float = 0.01       # L2 regularisation strength

# class LogisticRegressionProbe(nn.Module):
#     """Simple logistic regression probe model"""
#     def __init__(self, input_dim):
#         super(LogisticRegressionProbe, self).__init__()
#         self.linear = nn.Linear(input_dim, 1)
        
#     def forward(self, x):
#         return torch.sigmoid(self.linear(x))

# class ProbeTrainer:
#     """Class for training linear probes on transformer representations"""
    
#     def __init__(self, config: ProbeConfig, model=None):
#         """Initialize the probe trainer with a configuration and optional pre-loaded model"""
#         self.config = config
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Using device: {self.device}")
        
#         # Set hook name if not specified
#         if self.config.hook_name is None:
#             self.config.hook_name = f"blocks.{self.config.layer}.hook_resid_post"
        
#         # Make sure save directory exists
#         os.makedirs(self.config.save_dir, exist_ok=True)
        
#         # Use provided model or load a new one
#         if model is not None:
#             self.model = model
#             print(f"Using provided pre-loaded model: {self.config.model_name}")
#         else:
#             # Initialize model
#             self.model = tl.HookedTransformer.from_pretrained(
#                 self.config.model_name, 
#                 device=self.device
#             )
#             print(f"Loaded model: {self.config.model_name}")
        
#         # Initialize with empty data
#         self.pos_examples = []
#         self.neg_examples = []
#         self.probe = None
#         self.train_metrics = {}
#         self.val_metrics = {}
#         self.cv_metrics = {} 
        
#         # Add a new attribute for general negative examples
#         self.general_neg_examples = []
        
#     def load_examples_from_file(self, file_path: str):
#         """Load examples from a JSON file"""
#         with open(file_path, "r") as f:
#             examples = json.load(f)["examples"]
        
#         self.pos_examples = [x["positive"] for x in examples]
#         self.neg_examples = [x["negative"] for x in examples]
        
#         print(f"Loaded {len(examples)} examples from {file_path}")
#         return examples
    
#     @torch.no_grad()
#     def extract_representations(self, batch_size: int = 16):
#         """Extract representations from the model for all examples"""
#         print("Extracting representations...")

#         # Set PyTorch CUDA memory allocation configuration
#         os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
#         pos_resid_list = []
#         neg_resid_list = []
#         general_neg_resid_list = []
        
#         # Process positive examples in batches
#         for i in tqdm(range(0, len(self.pos_examples), batch_size), desc="Processing positive examples"):
#             batch = self.pos_examples[i:i+batch_size]
#             _, pos_cache = self.model.run_with_cache(self.model.to_tokens(batch))
#             pos_resid_list.append(pos_cache[self.config.hook_name][:, -1].to('cpu'))
#             del pos_cache, batch
        
#         # Process concept-specific negative examples in batches
#         for i in tqdm(range(0, len(self.neg_examples), batch_size), desc="Processing concept-specific negative examples"):
#             batch = self.neg_examples[i:i+batch_size]
#             _, neg_cache = self.model.run_with_cache(self.model.to_tokens(batch))
#             neg_resid_list.append(neg_cache[self.config.hook_name][:, -1].to('cpu'))
#             del neg_cache, batch
        
#         # Process general negative examples in batches if available
#         general_batch_size = 4
#         if self.general_neg_examples:
#             for i in tqdm(range(0, len(self.general_neg_examples), general_batch_size), desc="Processing general negative examples"):
#                 batch = self.general_neg_examples[i:i+general_batch_size]
#                 _, general_neg_cache = self.model.run_with_cache(self.model.to_tokens(batch))
#                 general_neg_resid_list.append(general_neg_cache[self.config.hook_name][:, -1].to('cpu'))
#                 del general_neg_cache, batch
        
#         # Concatenate all batches
#         pos_resid = torch.cat(pos_resid_list, dim=0)
#         neg_resid = torch.cat(neg_resid_list, dim=0)
        
#         # Include general negative examples if available
#         if general_neg_resid_list:
#             general_neg_resid = torch.cat(general_neg_resid_list, dim=0)
#             # Combine concept-specific and general negative examples
#             all_neg_resid = torch.cat([neg_resid, general_neg_resid], dim=0)
#             print(f"Extracted representations: {pos_resid.shape}, {neg_resid.shape} (concept-specific), {general_neg_resid.shape} (general)")
#         else:
#             all_neg_resid = neg_resid
#             print(f"Extracted representations: {pos_resid.shape}, {neg_resid.shape}")
        
#         # Stack and create labels
#         resid = torch.cat([pos_resid, all_neg_resid], dim=0)
#         labels = torch.cat([torch.ones(len(pos_resid)), torch.zeros(len(all_neg_resid))])
        
#         # Shuffle data
#         indices = torch.randperm(len(resid))
#         resid = resid[indices]
#         labels = labels[indices]
        
#         # Move to device
#         resid = resid.to(self.device)
#         labels = labels.to(self.device)
        
#         # If not doing cross-validation, split into train/val
#         if not self.config.do_cross_val:
#             train_size = int(self.config.train_size_frac * len(resid))
#             train_resid = resid[:train_size]
#             train_labels = labels[:train_size]
            
#             val_resid = resid[train_size:]
#             val_labels = labels[train_size:]
            
#             print(f"Train size: {train_size}, Val size: {len(val_resid)}")
#         else:
#             # For cross-validation, we'll use the full dataset later
#             train_resid = resid
#             train_labels = labels
#             val_resid = None
#             val_labels = None
            
#             print(f"Total examples for cross-validation: {len(resid)}")
        
#         return {
#             'train_resid': train_resid,
#             'train_labels': train_labels,
#             'val_resid': val_resid,
#             'val_labels': val_labels,
#             'full_resid': resid,
#             'full_labels': labels,
#             'pos_resid': pos_resid,
#             'neg_resid': all_neg_resid  # Use combined negative examples
#         }
    
#     def _train_single_model(self, train_resid, train_labels, val_resid=None, val_labels=None, verbose=True):
#         """Internal method to train a single model with the given data"""
#         # Initialize model
#         input_dim = train_resid.shape[1]
#         probe = LogisticRegressionProbe(input_dim).to(self.device)
#         criterion = nn.BCELoss()
#         optimizer = torch.optim.Adam(probe.parameters(), 
#                                      lr=self.config.learning_rate,
#                                      weight_decay=self.config.weight_decay)
        
#         # Training loop
#         for epoch in range(self.config.num_epochs):
#             probe.train()
#             epoch_loss = 0.0
            
#             # Create mini-batches
#             indices = torch.randperm(len(train_resid))
#             train_resid_shuffled = train_resid[indices]
#             train_labels_shuffled = train_labels[indices]
            
#             for i in range(0, len(train_resid), self.config.batch_size):
#                 batch_resid = train_resid_shuffled[i:i+self.config.batch_size].to(self.device)
#                 batch_labels = train_labels_shuffled[i:i+self.config.batch_size].unsqueeze(1).to(self.device)
                
#                 # Forward pass
#                 outputs = probe(batch_resid)
#                 loss = criterion(outputs, batch_labels)
                
#                 # Backward and optimize
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
                
#                 epoch_loss += loss.item() * len(batch_resid)
            
#             epoch_loss /= len(train_resid)
            
#             # Validation (if data provided)
#             if val_resid is not None and val_labels is not None:
#                 probe.eval()
#                 with torch.no_grad():
#                     val_outputs = probe(val_resid)
#                     val_loss = criterion(val_outputs, val_labels.unsqueeze(1))
                    
#                     # Calculate accuracy
#                     val_preds = (val_outputs > 0.5).float()
#                     val_accuracy = (val_preds == val_labels.unsqueeze(1)).float().mean()
                
#                 if verbose and (epoch + 1) % 10 == 0:
#                     print(f"Epoch {epoch+1}/{self.config.num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
#             elif verbose and (epoch + 1) % 10 == 0:
#                 print(f"Epoch {epoch+1}/{self.config.num_epochs}, Train Loss: {epoch_loss:.4f}")
        
#         # Final evaluation
#         probe.eval()
#         metrics = {}
        
#         with torch.no_grad():
#             train_outputs = probe(train_resid)
#             train_preds = (train_outputs > 0.5).float()
#             train_accuracy = (train_preds == train_labels.unsqueeze(1)).float().mean()
            
#             # Calculate F1 score for training data
#             train_tp = ((train_preds == 1) & (train_labels.unsqueeze(1) == 1)).sum().float()
#             train_fp = ((train_preds == 1) & (train_labels.unsqueeze(1) == 0)).sum().float()
#             train_fn = ((train_preds == 0) & (train_labels.unsqueeze(1) == 1)).sum().float()
            
#             train_precision = train_tp / (train_tp + train_fp + 1e-10)
#             train_recall = train_tp / (train_tp + train_fn + 1e-10)
#             train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall + 1e-10)
            
#             metrics['train_accuracy'] = train_accuracy.item()
#             metrics['train_f1_score'] = train_f1.item()
#             metrics['train_precision'] = train_precision.item()
#             metrics['train_recall'] = train_recall.item()
#             metrics['train_size'] = len(train_resid)
            
#             if val_resid is not None and val_labels is not None:
#                 val_outputs = probe(val_resid)
#                 val_preds = (val_outputs > 0.5).float()
#                 val_accuracy = (val_preds == val_labels.unsqueeze(1)).float().mean()
                
#                 # Calculate F1 score for validation data
#                 val_tp = ((val_preds == 1) & (val_labels.unsqueeze(1) == 1)).sum().float()
#                 val_fp = ((val_preds == 1) & (val_labels.unsqueeze(1) == 0)).sum().float()
#                 val_fn = ((val_preds == 0) & (val_labels.unsqueeze(1) == 1)).sum().float()
                
#                 val_precision = val_tp / (val_tp + val_fp + 1e-10)
#                 val_recall = val_tp / (val_tp + val_fn + 1e-10)
#                 val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-10)
                
#                 metrics['val_accuracy'] = val_accuracy.item()
#                 metrics['val_f1_score'] = val_f1.item()
#                 metrics['val_precision'] = val_precision.item()
#                 metrics['val_recall'] = val_recall.item()
#                 metrics['val_size'] = len(val_resid)
                
#                 if verbose:
#                     print(f"Final Train Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
#                     print(f"Final Validation Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
#             elif verbose:
#                 print(f"Final Train Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
        
#         return probe, metrics
    
#     def train_probe(self, data_dict: Dict[str, torch.Tensor]):
#         """Train a logistic regression probe on the extracted representations"""
#         print("Training logistic regression probe...")
        
#         train_resid = data_dict['train_resid']
#         train_labels = data_dict['train_labels']
#         val_resid = data_dict['val_resid']
#         val_labels = data_dict['val_labels']
        
#         self.probe, metrics = self._train_single_model(train_resid, train_labels, val_resid, val_labels)
        
#         # Store metrics
#         self.train_metrics = {
#             'accuracy': metrics['train_accuracy'],
#             'f1_score': metrics['train_f1_score'],
#             'precision': metrics['train_precision'],
#             'recall': metrics['train_recall'],
#             'size': metrics['train_size']
#         }
        
#         if 'val_accuracy' in metrics:
#             self.val_metrics = {
#                 'accuracy': metrics['val_accuracy'],
#                 'f1_score': metrics['val_f1_score'],
#                 'precision': metrics['val_precision'],
#                 'recall': metrics['val_recall'],
#                 'size': metrics['val_size']
#             }
        
#         return self.probe
    
#     def cross_validate(self, data_dict: Dict[str, torch.Tensor]):
#         """Perform k-fold cross-validation"""
#         print(f"Performing {self.config.cv_folds}-fold cross-validation...")
        
#         full_resid = data_dict['full_resid']
#         full_labels = data_dict['full_labels']
        
#         # Initialize cross-validation
#         kf = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        
#         # Store metrics for each fold
#         fold_metrics = []
        
#         # Perform cross-validation
#         for fold, (train_idx, val_idx) in enumerate(kf.split(full_resid)):
#             print(f"\nTraining fold {fold+1}/{self.config.cv_folds}")
            
#             # Split data
#             fold_train_resid = full_resid[train_idx]
#             fold_train_labels = full_labels[train_idx]
#             fold_val_resid = full_resid[val_idx]
#             fold_val_labels = full_labels[val_idx]
            
#             # Train model for this fold
#             _, fold_metrics_dict = self._train_single_model(
#                 fold_train_resid, 
#                 fold_train_labels,
#                 fold_val_resid,
#                 fold_val_labels,
#                 verbose=False  # Less verbose during CV
#             )
            
#             print(f"Fold {fold+1} - Train Accuracy: {fold_metrics_dict['train_accuracy']:.4f}, F1: {fold_metrics_dict['train_f1_score']:.4f}")
#             print(f"Fold {fold+1} - Val Accuracy: {fold_metrics_dict['val_accuracy']:.4f}, F1: {fold_metrics_dict['val_f1_score']:.4f}")
            
#             fold_metrics.append(fold_metrics_dict)
        
#         # Calculate average metrics
#         avg_train_acc = np.mean([m['train_accuracy'] for m in fold_metrics])
#         avg_val_acc = np.mean([m['val_accuracy'] for m in fold_metrics])
#         std_train_acc = np.std([m['train_accuracy'] for m in fold_metrics])
#         std_val_acc = np.std([m['val_accuracy'] for m in fold_metrics])
        
#         # Calculate average F1 scores
#         avg_train_f1 = np.mean([m['train_f1_score'] for m in fold_metrics])
#         avg_val_f1 = np.mean([m['val_f1_score'] for m in fold_metrics])
#         std_train_f1 = np.std([m['train_f1_score'] for m in fold_metrics])
#         std_val_f1 = np.std([m['val_f1_score'] for m in fold_metrics])
        
#         print("\nCross-validation results:")
#         print(f"Average Train Accuracy: {avg_train_acc:.4f} (±{std_train_acc:.4f})")
#         print(f"Average Validation Accuracy: {avg_val_acc:.4f} (±{std_val_acc:.4f})")
#         print(f"Average Train F1 Score: {avg_train_f1:.4f} (±{std_train_f1:.4f})")
#         print(f"Average Validation F1 Score: {avg_val_f1:.4f} (±{std_val_f1:.4f})")
        
#         # Store cross-validation metrics
#         self.cv_metrics = {
#             'folds': self.config.cv_folds,
#             'fold_results': fold_metrics,
#             'avg_train_accuracy': float(avg_train_acc),
#             'std_train_accuracy': float(std_train_acc),
#             'avg_val_accuracy': float(avg_val_acc),
#             'std_val_accuracy': float(std_val_acc),
#             'avg_train_f1': float(avg_train_f1),
#             'std_train_f1': float(std_train_f1),
#             'avg_val_f1': float(avg_val_f1),
#             'std_val_f1': float(std_val_f1)
#         }
        
#         # Optionally retrain on all data
#         if self.config.retrain_after_cv:
#             print("\nRetraining final model on all data...")
#             original_epochs = self.config.num_epochs
#             self.config.num_epochs = original_epochs * 1  # Same number of epochs for final training
#             self.probe, final_metrics = self._train_single_model(full_resid, full_labels, verbose=True)
#             # Restore original epochs setting
#             self.config.num_epochs = original_epochs
            
#             # Store metrics for the final model
#             self.train_metrics = {
#                 'accuracy': final_metrics['train_accuracy'],
#                 'f1_score': final_metrics['train_f1_score'],
#                 'precision': final_metrics['train_precision'],
#                 'recall': final_metrics['train_recall'],
#                 'size': len(full_resid)
#             }
            
#             # No validation metrics for final model as we used all data
#             self.val_metrics = {}
        
#         return self.probe
    
#     def analyze_probe(self, data_dict: Dict[str, torch.Tensor]):
#         """Analyze the trained probe"""
#         if self.probe is None:
#             raise ValueError("Probe must be trained before analysis")
        
#         pos_resid = data_dict['pos_resid']
#         neg_resid = data_dict['neg_resid']
        
#         # For cross-validation, we use the full dataset for final analysis
#         if self.config.do_cross_val:
#             val_resid = data_dict['full_resid']
#             val_labels = data_dict['full_labels']
#         else:
#             val_resid = data_dict['val_resid']
#             val_labels = data_dict['val_labels']
        
#         with torch.no_grad():
#             # Extract weights from the model
#             weights = self.probe.linear.weight.data.cpu().numpy().flatten()
#             bias = self.probe.linear.bias.data.cpu().item()
            
#             print(f"Bias: {bias:.4f}")
            
#             # Analyze predictions
#             val_outputs = self.probe(val_resid)
#             val_preds = (val_outputs > 0.5).float()
            
#             correct_pred_indices = (val_preds.squeeze() == val_labels).nonzero().squeeze()
#             incorrect_pred_indices = (val_preds.squeeze() != val_labels).nonzero().squeeze()
            
#             correct_count = len(correct_pred_indices) if len(correct_pred_indices.shape) > 0 else 1
#             incorrect_count = len(incorrect_pred_indices) if len(incorrect_pred_indices.shape) > 0 else 1
            
#             print(f"\nCorrectly predicted: {correct_count} out of {len(val_labels)}")
#             print(f"Incorrectly predicted: {incorrect_count} out of {len(val_labels)}")
#             print(f"Accuracy: {correct_count / len(val_labels):.4f}")
            
#             # Interpretability: Find the direction in the residual space
#             direction = self.probe.linear.weight.data[0].cpu()
#             direction = direction / torch.norm(direction)  # Normalize the direction vector
            
#             # Project all examples onto this direction
#             pos_proj = torch.matmul(pos_resid.to('cpu'), direction)
#             neg_proj = torch.matmul(neg_resid.to('cpu'), direction)
            
#             # Calculate mean and std of projections
#             pos_mean, pos_std = pos_proj.mean().item(), pos_proj.std().item()
#             neg_mean, neg_std = neg_proj.mean().item(), neg_proj.std().item()
            
#             print(f"\nProjection analysis:")
#             print(f"Positive examples - Mean: {pos_mean:.4f}, Std: {pos_std:.4f}")
#             print(f"Negative examples - Mean: {neg_mean:.4f}, Std: {neg_std:.4f}")
#             print(f"Separation (mean difference): {abs(pos_mean - neg_mean):.4f}")
            
#             # Store additional metrics
#             additional_metrics = {
#                 'direction_norm': torch.norm(direction).item(),
#                 'pos_proj_mean': pos_mean,
#                 'pos_proj_std': pos_std,
#                 'neg_proj_mean': neg_mean,
#                 'neg_proj_std': neg_std,
#                 'separation': abs(pos_mean - neg_mean),
#                 'bias': bias
#             }
            
#             # Update metrics
#             if self.config.do_cross_val and self.config.retrain_after_cv:
#                 # For cross-validation with final retraining, add to train metrics
#                 self.train_metrics.update(additional_metrics)
#             else:
#                 # For regular training, add to validation metrics
#                 self.val_metrics.update(additional_metrics)
            
#             return additional_metrics
    
#     def save_probe(self):
#         """Save the trained probe and its configuration"""
#         if self.probe is None:
#             raise ValueError("Probe must be trained before saving")
        
#         # Create concept-specific directory
#         concept_dir = os.path.join(self.config.save_dir, self.config.concept.replace(" ", "_"))
#         os.makedirs(concept_dir, exist_ok=True)
        
#         # Save model
#         model_path = os.path.join(concept_dir, "probe.pt")
#         torch.save(self.probe.state_dict(), model_path)
        
#         # Save config and metrics
#         config_dict = {
#             'concept': self.config.concept,
#             'model_name': self.config.model_name,
#             'layer': self.config.layer,
#             'hook_name': self.config.hook_name,
#             'train_metrics': self.train_metrics,
#             'val_metrics': self.val_metrics,
#             'input_dim': self.probe.linear.weight.shape[1],
#             'do_cross_val': self.config.do_cross_val,
#         }
        
#         # Add cross-validation metrics if available
#         if self.config.do_cross_val and self.cv_metrics:
#             config_dict['cv_metrics'] = self.cv_metrics
        
#         config_path = os.path.join(concept_dir, "config.json")
#         with open(config_path, "w") as f:
#             json.dump(config_dict, f, indent=2)
        
#         print(f"Saved probe and config to {concept_dir}")
        
#         return concept_dir
    
#     def visualize_data(self, data_dict: Dict[str, torch.Tensor], save_dir: str):
#         """Visualize the data using PCA and histograms"""
#         try:
#             from sklearn.decomposition import PCA
#             import plotly.graph_objects as go
#             import plotly.io as pio
            
#             # Make sure to move tensors to CPU before converting to numpy
#             full_resid = data_dict['full_resid'].cpu()
#             full_labels = data_dict['full_labels'].cpu()
#             pos_resid = data_dict['pos_resid'].cpu()
#             neg_resid = data_dict['neg_resid'].cpu()
            
#             # Apply PCA to reduce dimensions for visualization
#             pca = PCA(n_components=2)
#             pca_result = pca.fit_transform(full_resid.numpy())
            
#             # Create plotly figure for PCA
#             fig = go.Figure()
            
#             # Add scatter traces for positive and negative examples
#             fig.add_trace(go.Scatter(
#                 x=pca_result[full_labels.numpy() == 1, 0],
#                 y=pca_result[full_labels.numpy() == 1, 1],
#                 mode='markers',
#                 marker=dict(color='blue', opacity=0.5),
#                 name='Positive Examples'
#             ))
            
#             fig.add_trace(go.Scatter(
#                 x=pca_result[full_labels.numpy() == 0, 0],
#                 y=pca_result[full_labels.numpy() == 0, 1],
#                 mode='markers',
#                 marker=dict(color='red', opacity=0.5),
#                 name='Negative Examples'
#             ))
            
#             # Update layout
#             fig.update_layout(
#                 title=f'PCA of Residual Representations for {self.config.concept}',
#                 xaxis_title='PC1',
#                 yaxis_title='PC2',
#                 legend=dict(x=0.02, y=0.98),
#                 width=800,
#                 height=600
#             )
            
#             # Save the figure
#             pca_path = os.path.join(save_dir, "pca_visualization.png")
#             pio.write_image(fig, pca_path)
#             print(f"PCA visualization saved to '{pca_path}'")
            
#             # Extract the direction from the trained probe
#             if self.probe is not None:
#                 direction = self.probe.linear.weight.data[0].cpu()
#                 direction = direction / torch.norm(direction)
                
#                 # Project all examples onto this direction
#                 pos_proj = torch.matmul(pos_resid, direction)
#                 neg_proj = torch.matmul(neg_resid, direction)
                
#                 # Create plotly figure for histogram
#                 fig_hist = go.Figure()
                
#                 # Add histogram traces
#                 fig_hist.add_trace(go.Histogram(
#                     x=pos_proj.numpy(),
#                     nbinsx=30,
#                     opacity=0.5,
#                     name='Positive Examples'
#                 ))
                
#                 fig_hist.add_trace(go.Histogram(
#                     x=neg_proj.numpy(),
#                     nbinsx=30,
#                     opacity=0.5,
#                     name='Negative Examples'
#                 ))
                
#                 # Update layout
#                 fig_hist.update_layout(
#                     title=f'Projection onto Probe Direction for {self.config.concept}',
#                     xaxis_title='Projection Value',
#                     yaxis_title='Frequency',
#                     barmode='overlay',
#                     width=800,
#                     height=600
#                 )
                
#                 # Save the figure
#                 hist_path = os.path.join(save_dir, "projection_histogram.png")
#                 pio.write_image(fig_hist, hist_path)
#                 print(f"Projection histogram saved to '{hist_path}'")
                
#                 return True
#         except ImportError:
#             print("Skipping visualizations - required libraries not available")
#             return False
#         except Exception as e:
#             print(f"Visualization error: {e}")
#             return False
        
#     def train_and_analyze(self):
#         """Convenience function to run the full training and analysis pipeline"""
#         # Extract representations
#         data_dict = self.extract_representations()
        
#         # Train probe with or without cross-validation
#         if self.config.do_cross_val:
#             self.cross_validate(data_dict)
#         else:
#             self.train_probe(data_dict)
        
#         # Analyze probe
#         self.analyze_probe(data_dict)
        
#         # Save probe
#         save_dir = self.save_probe()
        
#         # Visualize data (optional)
#         try:
#             self.visualize_data(data_dict, save_dir)
#         except Exception as e:
#             print(f"Visualization failed: {e}")
        
#         return self.probe, save_dir

#     def add_general_negative_examples(self, examples: List[str]):
#         """Add general negative examples that are unrelated to the concept"""
#         self.general_neg_examples = examples
#         print(f"Added {len(examples)} general negative examples")
    
#     def prepare_data(self):
#         """Prepare data for training by extracting representations"""
#         if not self.pos_examples or not self.neg_examples:
#             raise ValueError("No examples loaded. Call load_examples_from_file() first.")
        
#         print(f"Preparing data with {len(self.pos_examples)} positive examples, "
#               f"{len(self.neg_examples)} concept-specific negative examples, and "
#               f"{len(self.general_neg_examples)} general negative examples")
        
#         # Combine all negative examples
#         all_neg_examples = self.neg_examples + self.general_neg_examples
        
#         # Extract representations for positive examples
#         pos_reps = []
#         for example in tqdm(self.pos_examples, desc="Processing positive examples"):
#             rep = self.get_representation(example)
#             pos_reps.append(rep)
        
#         # Extract representations for negative examples
#         neg_reps = []
#         for example in tqdm(all_neg_examples, desc="Processing negative examples"):
#             rep = self.get_representation(example)
#             neg_reps.append(rep)
        
#         # Convert to tensors
#         self.X_pos = torch.stack(pos_reps)
#         self.X_neg = torch.stack(neg_reps)
        
#         # Create labels
#         self.y_pos = torch.ones(len(self.X_pos))
#         self.y_neg = torch.zeros(len(self.X_neg))
        
#         # Combine data
#         self.X = torch.cat([self.X_pos, self.X_neg])
#         self.y = torch.cat([self.y_pos, self.y_neg])
        
#         print(f"Prepared {len(self.X)} examples ({len(self.X_pos)} positive, {len(self.X_neg)} negative)")
        
#         # Save the data dimensions in the config
#         self.config.data_stats = {
#             "num_pos_examples": len(self.X_pos),
#             "num_concept_neg_examples": len(self.neg_examples),
#             "num_general_neg_examples": len(self.general_neg_examples),
#             "total_examples": len(self.X)
#         }

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import transformer_lens as tl
import json
import os
import yaml
from tqdm import tqdm
from dataclasses import dataclass
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

@dataclass
class ProbeConfig:
    """Configuration for probe training"""
    concept: str
    model_name: str
    layer: int
    train_size_frac: float = 0.8
    learning_rate: float = 0.001  # Used only for learning rate scheduler if applicable
    num_epochs: int = 100         # Used for iteration count in sklearn
    batch_size: int = 128         # Not directly used in sklearn but kept for compatibility
    save_dir: str = "probes"
    hook_name: Optional[str] = None  # Will be set automatically if None
    do_cross_val: bool = False       # Whether to perform cross-validation
    cv_folds: int = 5                # Number of folds for cross-validation
    retrain_after_cv: bool = True    # Whether to retrain on all data after CV
    weight_decay: float = 0.01       # L2 regularisation strength (maps to C=1/weight_decay)

class ProbeTrainer:
    """Class for training linear probes on transformer representations using sklearn"""
    
    def __init__(self, config: ProbeConfig, model=None):
        """Initialize the probe trainer with a configuration and optional pre-loaded model"""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Set hook name if not specified
        if self.config.hook_name is None:
            self.config.hook_name = f"blocks.{self.config.layer}.hook_resid_post"
        
        # Make sure save directory exists
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # Use provided model or load a new one
        if model is not None:
            self.model = model
            print(f"Using provided pre-loaded model: {self.config.model_name}")
        else:
            # Initialize model
            self.model = tl.HookedTransformer.from_pretrained(
                self.config.model_name, 
                device=self.device
            )
            print(f"Loaded model: {self.config.model_name}")
        
        # Initialize with empty data
        self.pos_examples = []
        self.neg_examples = []
        self.probe = None
        self.train_metrics = {}
        self.val_metrics = {}
        self.cv_metrics = {} 
        
        # Add a new attribute for general negative examples
        self.general_neg_examples = []
        
    def load_examples_from_file(self, file_path: str):
        """Load examples from a JSON file"""
        with open(file_path, "r") as f:
            examples = json.load(f)["examples"]
        
        self.pos_examples = [x["positive"] for x in examples]
        self.neg_examples = [x["negative"] for x in examples]
        
        print(f"Loaded {len(examples)} examples from {file_path}")
        return examples
    
    @torch.no_grad()
    def extract_representations(self, batch_size: int = 16):
        """Extract representations from the model for all examples"""
        print("Extracting representations...")

        # Set PyTorch CUDA memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        pos_resid_list = []
        neg_resid_list = []
        general_neg_resid_list = []
        
        # Process positive examples in batches
        for i in tqdm(range(0, len(self.pos_examples), batch_size), desc="Processing positive examples"):
            batch = self.pos_examples[i:i+batch_size]
            _, pos_cache = self.model.run_with_cache(self.model.to_tokens(batch))
            pos_resid_list.append(pos_cache[self.config.hook_name][:, -1].to('cpu'))
            del pos_cache, batch
        
        # Process concept-specific negative examples in batches
        for i in tqdm(range(0, len(self.neg_examples), batch_size), desc="Processing concept-specific negative examples"):
            batch = self.neg_examples[i:i+batch_size]
            _, neg_cache = self.model.run_with_cache(self.model.to_tokens(batch))
            neg_resid_list.append(neg_cache[self.config.hook_name][:, -1].to('cpu'))
            del neg_cache, batch
        
        # Process general negative examples in batches if available
        general_batch_size = 4
        if self.general_neg_examples:
            for i in tqdm(range(0, len(self.general_neg_examples), general_batch_size), desc="Processing general negative examples"):
                batch = self.general_neg_examples[i:i+general_batch_size]
                _, general_neg_cache = self.model.run_with_cache(self.model.to_tokens(batch))
                general_neg_resid_list.append(general_neg_cache[self.config.hook_name][:, -1].to('cpu'))
                del general_neg_cache, batch
        
        # Concatenate all batches
        pos_resid = torch.cat(pos_resid_list, dim=0)
        neg_resid = torch.cat(neg_resid_list, dim=0)
        
        # Include general negative examples if available
        if general_neg_resid_list:
            general_neg_resid = torch.cat(general_neg_resid_list, dim=0)
            # Combine concept-specific and general negative examples
            all_neg_resid = torch.cat([neg_resid, general_neg_resid], dim=0)
            print(f"Extracted representations: {pos_resid.shape}, {neg_resid.shape} (concept-specific), {general_neg_resid.shape} (general)")
        else:
            all_neg_resid = neg_resid
            print(f"Extracted representations: {pos_resid.shape}, {neg_resid.shape}")
        
        # Stack and create labels
        resid = torch.cat([pos_resid, all_neg_resid], dim=0)
        labels = torch.cat([torch.ones(len(pos_resid)), torch.zeros(len(all_neg_resid))])
        
        # Shuffle data
        indices = torch.randperm(len(resid))
        resid = resid[indices]
        labels = labels[indices]
        
        # Convert to numpy for sklearn
        resid_np = resid.numpy()
        labels_np = labels.numpy()
        
        # If not doing cross-validation, split into train/val
        if not self.config.do_cross_val:
            train_size = int(self.config.train_size_frac * len(resid_np))
            train_resid = resid_np[:train_size]
            train_labels = labels_np[:train_size]
            
            val_resid = resid_np[train_size:]
            val_labels = labels_np[train_size:]
            
            print(f"Train size: {train_size}, Val size: {len(val_resid)}")
        else:
            # For cross-validation, we'll use the full dataset later
            train_resid = resid_np
            train_labels = labels_np
            val_resid = None
            val_labels = None
            
            print(f"Total examples for cross-validation: {len(resid_np)}")
        
        return {
            'train_resid': train_resid,
            'train_labels': train_labels,
            'val_resid': val_resid,
            'val_labels': val_labels,
            'full_resid': resid_np,
            'full_labels': labels_np,
            'pos_resid': pos_resid.numpy(),
            'neg_resid': all_neg_resid.numpy(),
            'torch_pos_resid': pos_resid,  # Keep torch versions for directional analysis
            'torch_neg_resid': all_neg_resid
        }
    
    def _train_single_model(self, train_resid, train_labels, val_resid=None, val_labels=None, verbose=True):
        """Internal method to train a single model with the given data using sklearn"""
        # Calculate C parameter for sklearn (C = 1/weight_decay)
        C = 1.0 / self.config.weight_decay
        
        # Initialize scikit-learn logistic regression with L2 penalty
        probe = LogisticRegression(
            #penalty='l2',
            C=C,
            #solver='lbfgs',  # Efficient for small datasets, supports L2
            #max_iter=self.config.num_epochs,
            verbose=1 if verbose else 0,
            random_state=42,
            n_jobs=-1  # Use all available CPUs
        )
        
        # Train the model
        probe.fit(train_resid, train_labels)
        
        # Calculate training metrics
        train_preds = probe.predict(train_resid)
        train_probs = probe.predict_proba(train_resid)[:, 1]  # Probability of positive class
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)
        train_precision = precision_score(train_labels, train_preds)
        train_recall = recall_score(train_labels, train_preds)
        
        metrics = {
            'train_accuracy': train_accuracy,
            'train_f1_score': train_f1,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_size': len(train_resid)
        }
        
        # Calculate validation metrics if provided
        if val_resid is not None and val_labels is not None:
            val_preds = probe.predict(val_resid)
            val_probs = probe.predict_proba(val_resid)[:, 1]
            val_accuracy = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds)
            val_precision = precision_score(val_labels, val_preds)
            val_recall = recall_score(val_labels, val_preds)
            
            metrics.update({
                'val_accuracy': val_accuracy,
                'val_f1_score': val_f1,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_size': len(val_resid)
            })
            
            if verbose:
                print(f"Train Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
                print(f"Validation Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
        elif verbose:
            print(f"Train Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
        
        return probe, metrics
    
    def train_probe(self, data_dict: Dict[str, np.ndarray]):
        """Train a logistic regression probe on the extracted representations"""
        print("Training logistic regression probe...")
        
        train_resid = data_dict['train_resid']
        train_labels = data_dict['train_labels']
        val_resid = data_dict['val_resid']
        val_labels = data_dict['val_labels']
        
        self.probe, metrics = self._train_single_model(train_resid, train_labels, val_resid, val_labels)
        
        # Store metrics
        self.train_metrics = {
            'accuracy': metrics['train_accuracy'],
            'f1_score': metrics['train_f1_score'],
            'precision': metrics['train_precision'],
            'recall': metrics['train_recall'],
            'size': metrics['train_size']
        }
        
        if 'val_accuracy' in metrics:
            self.val_metrics = {
                'accuracy': metrics['val_accuracy'],
                'f1_score': metrics['val_f1_score'],
                'precision': metrics['val_precision'],
                'recall': metrics['val_recall'],
                'size': metrics['val_size']
            }
        
        return self.probe
    
    def cross_validate(self, data_dict: Dict[str, np.ndarray]):
        """Perform k-fold cross-validation"""
        print(f"Performing {self.config.cv_folds}-fold cross-validation...")
        
        full_resid = data_dict['full_resid']
        full_labels = data_dict['full_labels']
        
        # Initialize cross-validation
        kf = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        
        # Store metrics for each fold
        fold_metrics = []
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(full_resid)):
            print(f"\nTraining fold {fold+1}/{self.config.cv_folds}")
            
            # Split data
            fold_train_resid = full_resid[train_idx]
            fold_train_labels = full_labels[train_idx]
            fold_val_resid = full_resid[val_idx]
            fold_val_labels = full_labels[val_idx]
            
            # Train model for this fold
            _, fold_metrics_dict = self._train_single_model(
                fold_train_resid, 
                fold_train_labels,
                fold_val_resid,
                fold_val_labels,
                verbose=False  # Less verbose during CV
            )
            
            print(f"Fold {fold+1} - Train Accuracy: {fold_metrics_dict['train_accuracy']:.4f}, F1: {fold_metrics_dict['train_f1_score']:.4f}")
            print(f"Fold {fold+1} - Val Accuracy: {fold_metrics_dict['val_accuracy']:.4f}, F1: {fold_metrics_dict['val_f1_score']:.4f}")
            
            fold_metrics.append(fold_metrics_dict)
        
        # Calculate average metrics
        avg_train_acc = np.mean([m['train_accuracy'] for m in fold_metrics])
        avg_val_acc = np.mean([m['val_accuracy'] for m in fold_metrics])
        std_train_acc = np.std([m['train_accuracy'] for m in fold_metrics])
        std_val_acc = np.std([m['val_accuracy'] for m in fold_metrics])
        
        # Calculate average F1 scores
        avg_train_f1 = np.mean([m['train_f1_score'] for m in fold_metrics])
        avg_val_f1 = np.mean([m['val_f1_score'] for m in fold_metrics])
        std_train_f1 = np.std([m['train_f1_score'] for m in fold_metrics])
        std_val_f1 = np.std([m['val_f1_score'] for m in fold_metrics])
        
        print("\nCross-validation results:")
        print(f"Average Train Accuracy: {avg_train_acc:.4f} (±{std_train_acc:.4f})")
        print(f"Average Validation Accuracy: {avg_val_acc:.4f} (±{std_val_acc:.4f})")
        print(f"Average Train F1 Score: {avg_train_f1:.4f} (±{std_train_f1:.4f})")
        print(f"Average Validation F1 Score: {avg_val_f1:.4f} (±{std_val_f1:.4f})")
        
        # Store cross-validation metrics
        self.cv_metrics = {
            'folds': self.config.cv_folds,
            'fold_results': fold_metrics,
            'avg_train_accuracy': float(avg_train_acc),
            'std_train_accuracy': float(std_train_acc),
            'avg_val_accuracy': float(avg_val_acc),
            'std_val_accuracy': float(std_val_acc),
            'avg_train_f1': float(avg_train_f1),
            'std_train_f1': float(std_train_f1),
            'avg_val_f1': float(avg_val_f1),
            'std_val_f1': float(std_val_f1)
        }
        
        # Optionally retrain on all data
        if self.config.retrain_after_cv:
            print("\nRetraining final model on all data...")
            self.probe, final_metrics = self._train_single_model(full_resid, full_labels, verbose=True)
            
            # Store metrics for the final model
            self.train_metrics = {
                'accuracy': final_metrics['train_accuracy'],
                'f1_score': final_metrics['train_f1_score'],
                'precision': final_metrics['train_precision'],
                'recall': final_metrics['train_recall'],
                'size': len(full_resid)
            }
            
            # No validation metrics for final model as we used all data
            self.val_metrics = {}
        
        return self.probe
    
    def analyze_probe(self, data_dict: Dict[str, Union[np.ndarray, torch.Tensor]]):
        """Analyze the trained probe"""
        if self.probe is None:
            raise ValueError("Probe must be trained before analysis")
        
        # Get the weights and bias from the sklearn model
        weights = self.probe.coef_[0]  # First and only class for binary classification
        bias = self.probe.intercept_[0]
        
        # Get torch tensors for directional analysis
        pos_resid = data_dict['torch_pos_resid']
        neg_resid = data_dict['torch_neg_resid']
        
        # For cross-validation, we use the full dataset for final analysis
        if self.config.do_cross_val:
            val_resid = data_dict['full_resid']
            val_labels = data_dict['full_labels']
        else:
            val_resid = data_dict['val_resid']
            val_labels = data_dict['val_labels']
        
        # Analyze predictions
        val_preds = self.probe.predict(val_resid)
        val_probs = self.probe.predict_proba(val_resid)[:, 1]
        
        # Calculate accuracy
        correct_preds = (val_preds == val_labels)
        correct_count = np.sum(correct_preds)
        incorrect_count = len(val_labels) - correct_count
        
        print(f"\nCorrectly predicted: {correct_count} out of {len(val_labels)}")
        print(f"Incorrectly predicted: {incorrect_count} out of {len(val_labels)}")
        print(f"Accuracy: {correct_count / len(val_labels):.4f}")
        
        # Interpretability: Use the weight vector as the direction
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        direction = weight_tensor / torch.norm(weight_tensor)  # Normalize
        
        # Project all examples onto this direction
        pos_proj = torch.matmul(pos_resid, direction)
        neg_proj = torch.matmul(neg_resid, direction)
        
        # Calculate mean and std of projections
        pos_mean, pos_std = pos_proj.mean().item(), pos_proj.std().item()
        neg_mean, neg_std = neg_proj.mean().item(), neg_proj.std().item()
        
        print(f"\nProjection analysis:")
        print(f"Positive examples - Mean: {pos_mean:.4f}, Std: {pos_std:.4f}")
        print(f"Negative examples - Mean: {neg_mean:.4f}, Std: {neg_std:.4f}")
        print(f"Separation (mean difference): {abs(pos_mean - neg_mean):.4f}")
        
        # Store additional metrics
        additional_metrics = {
            'direction_norm': torch.norm(direction).item(),
            'pos_proj_mean': pos_mean,
            'pos_proj_std': pos_std,
            'neg_proj_mean': neg_mean,
            'neg_proj_std': neg_std,
            'separation': abs(pos_mean - neg_mean),
            'bias': bias
        }
        
        # Print information about the model weights
        print(f"Bias: {bias:.4f}")
        print(f"Weight L2 norm: {np.linalg.norm(weights):.4f}")
        
        # Update metrics
        if self.config.do_cross_val and self.config.retrain_after_cv:
            # For cross-validation with final retraining, add to train metrics
            self.train_metrics.update(additional_metrics)
        else:
            # For regular training, add to validation metrics
            self.val_metrics.update(additional_metrics)
        
        return additional_metrics
    
    def save_probe(self):
        """Save the trained probe and its configuration"""
        if self.probe is None:
            raise ValueError("Probe must be trained before saving")
        
        # Create concept-specific directory
        concept_dir = os.path.join(self.config.save_dir, self.config.concept.replace(" ", "_"))
        os.makedirs(concept_dir, exist_ok=True)
        
        # For sklearn models, we use joblib for saving
        try:
            import joblib
            model_path = os.path.join(concept_dir, "probe.joblib")
            joblib.dump(self.probe, model_path)
        except ImportError:
            # Fallback to pickle if joblib is not available
            import pickle
            model_path = os.path.join(concept_dir, "probe.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.probe, f)
        
        # Save config and metrics
        config_dict = {
            'concept': self.config.concept,
            'model_name': self.config.model_name,
            'layer': self.config.layer,
            'hook_name': self.config.hook_name,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'input_dim': self.probe.coef_.shape[1],  # Feature dimension
            'do_cross_val': self.config.do_cross_val,
            'weight_decay': self.config.weight_decay,
            'C': 1.0 / self.config.weight_decay
        }
        
        # Add cross-validation metrics if available
        if self.config.do_cross_val and self.cv_metrics:
            config_dict['cv_metrics'] = self.cv_metrics
        
        config_path = os.path.join(concept_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Saved probe and config to {concept_dir}")
        
        return concept_dir
    
    def visualize_data(self, data_dict: Dict[str, Union[np.ndarray, torch.Tensor]], save_dir: str):
        """Visualize the data using PCA and histograms"""
        try:
            from sklearn.decomposition import PCA
            import plotly.graph_objects as go
            import plotly.io as pio
            
            # Get numpy arrays
            full_resid = data_dict['full_resid']
            full_labels = data_dict['full_labels']
            
            # Apply PCA to reduce dimensions for visualization
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(full_resid)
            
            # Create plotly figure for PCA
            fig = go.Figure()
            
            # Add scatter traces for positive and negative examples
            fig.add_trace(go.Scatter(
                x=pca_result[full_labels == 1, 0],
                y=pca_result[full_labels == 1, 1],
                mode='markers',
                marker=dict(color='blue', opacity=0.5),
                name='Positive Examples'
            ))
            
            fig.add_trace(go.Scatter(
                x=pca_result[full_labels == 0, 0],
                y=pca_result[full_labels == 0, 1],
                mode='markers',
                marker=dict(color='red', opacity=0.5),
                name='Negative Examples'
            ))
            
            # Update layout
            fig.update_layout(
                title=f'PCA of Residual Representations for {self.config.concept}',
                xaxis_title='PC1',
                yaxis_title='PC2',
                legend=dict(x=0.02, y=0.98),
                width=800,
                height=600
            )
            
            # Save the figure
            pca_path = os.path.join(save_dir, "pca_visualization.png")
            pio.write_image(fig, pca_path)
            print(f"PCA visualization saved to '{pca_path}'")
            
            # Extract the direction from the trained probe if available
            if self.probe is not None:
                # Get torch tensors
                pos_resid = data_dict['torch_pos_resid']
                neg_resid = data_dict['torch_neg_resid']
                
                # Convert sklearn weights to torch tensor for projection
                weight_tensor = torch.tensor(self.probe.coef_[0], dtype=torch.float32)
                direction = weight_tensor / torch.norm(weight_tensor)
                
                # Project all examples onto this direction
                pos_proj = torch.matmul(pos_resid, direction)
                neg_proj = torch.matmul(neg_resid, direction)
                
                # Create plotly figure for histogram
                fig_hist = go.Figure()
                
                # Add histogram traces
                fig_hist.add_trace(go.Histogram(
                    x=pos_proj.numpy(),
                    nbinsx=30,
                    opacity=0.5,
                    name='Positive Examples'
                ))
                
                fig_hist.add_trace(go.Histogram(
                    x=neg_proj.numpy(),
                    nbinsx=30,
                    opacity=0.5,
                    name='Negative Examples'
                ))
                
                # Update layout
                fig_hist.update_layout(
                    title=f'Projection onto Probe Direction for {self.config.concept}',
                    xaxis_title='Projection Value',
                    yaxis_title='Frequency',
                    barmode='overlay',
                    width=800,
                    height=600
                )
                
                # Save the figure
                hist_path = os.path.join(save_dir, "projection_histogram.png")
                pio.write_image(fig_hist, hist_path)
                print(f"Projection histogram saved to '{hist_path}'")
                
                return True
        except ImportError:
            print("Skipping visualizations - required libraries not available")
            return False
        except Exception as e:
            print(f"Visualization error: {e}")
            return False
        
    def train_and_analyze(self):
        """Convenience function to run the full training and analysis pipeline"""
        # Extract representations
        data_dict = self.extract_representations()
        
        # Train probe with or without cross-validation
        if self.config.do_cross_val:
            self.cross_validate(data_dict)
        else:
            self.train_probe(data_dict)
        
        # Analyze probe
        self.analyze_probe(data_dict)
        
        # Save probe
        save_dir = self.save_probe()
        
        # Visualize data (optional)
        try:
            self.visualize_data(data_dict, save_dir)
        except Exception as e:
            print(f"Visualization failed: {e}")
        
        return self.probe, save_dir

    def add_general_negative_examples(self, examples: List[str]):
        """Add general negative examples that are unrelated to the concept"""
        self.general_neg_examples = examples
        print(f"Added {len(examples)} general negative examples")
    
    def prepare_data(self):
        """Prepare data for training by extracting representations"""
        if not self.pos_examples or not self.neg_examples:
            raise ValueError("No examples loaded. Call load_examples_from_file() first.")
        
        print(f"Preparing data with {len(self.pos_examples)} positive examples, "
              f"{len(self.neg_examples)} concept-specific negative examples, and "
              f"{len(self.general_neg_examples)} general negative examples")
        
        # Combine all negative examples
        all_neg_examples = self.neg_examples + self.general_neg_examples
        
        # Extract representations for positive examples
        pos_reps = []
        for example in tqdm(self.pos_examples, desc="Processing positive examples"):
            rep = self.get_representation(example)
            pos_reps.append(rep)
        
        # Extract representations for negative examples
        neg_reps = []
        for example in tqdm(all_neg_examples, desc="Processing negative examples"):
            rep = self.get_representation(example)
            neg_reps.append(rep)
        
        # Convert to tensors
        self.X_pos = torch.stack(pos_reps)
        self.X_neg = torch.stack(neg_reps)
        
        # Create labels
        self.y_pos = torch.ones(len(self.X_pos))
        self.y_neg = torch.zeros(len(self.X_neg))
        
        # Combine data
        self.X = torch.cat([self.X_pos, self.X_neg])
        self.y = torch.cat([self.y_pos, self.y_neg])
        
        print(f"Prepared {len(self.X)} examples ({len(self.X_pos)} positive, {len(self.X_neg)} negative)")
        
        # Save the data dimensions in the config
        self.config.data_stats = {
            "num_pos_examples": len(self.X_pos),
            "num_concept_neg_examples": len(self.neg_examples),
            "num_general_neg_examples": len(self.general_neg_examples),
            "total_examples": len(self.X)
        }