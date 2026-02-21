"""
Inference Script for Sleep Stage Classification

This module provides functionality for loading trained models and making
predictions on new data.

Author:  Vojtech Brejtr
Date: February 2026
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Union

from models import DualBranchModel
from data_management.dataset import SleepStageDataset



def predict_single_recording(
    model,
    raw_data: torch.Tensor,
    gcv: Optional[torch.Tensor] = None,
    device: str = 'cuda',
    use_contextual: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make predictions for a single recording.
    
    Works with both old and new model structures.
    
    Args:
        model: Trained model (old or new structure)
        raw_data: Input signal data (num_windows, in_channels, window_size)
        gcv: Optional global context vector (num_windows, feature_dim)
        device: Device to use
        use_contextual: Whether to use contextual predictions (default: True) - only used for new model
    
    Returns:
        predictions: Predicted class labels (num_windows,)
        probabilities: Class probabilities (num_windows, num_classes)
    
    Example:
        >>> predictions, probs = predict_single_recording(model, raw_data)
        >>> print(f"Predicted stages: {predictions}")
    """
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension
        raw_data = raw_data.unsqueeze(0).to(device)
        if gcv is not None:
            gcv = gcv.unsqueeze(0).to(device)
        else:
            # Create dummy GCV if none provided
            num_windows = raw_data.size(1)
            gcv = torch.zeros(1, num_windows, 1).to(device)
        
        # Check if this is old or new model structure
        has_window_classifier = hasattr(model, 'window_classifier')  # New structure
        
        if has_window_classifier:
            # New model structure - supports dual branches
            result = model(raw_data, gcv)
            if len(result) == 3:
                window_logits, contextual_logits, _ = result
            else:
                window_logits, contextual_logits = result
            
            # Select which branch to use for final prediction
            logits = contextual_logits if use_contextual else window_logits
        else:
            # Old model structure - single output
            result = model(raw_data, gcv)
            if isinstance(result, tuple):
                window_logits, _ = result
            else:
                window_logits = result
            logits = window_logits
        
        # Remove batch dimension
        logits = logits.squeeze(0)
        
        # Get predictions and probabilities
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        
    return predictions.cpu(), probabilities.cpu()


def predict_batch(
    model,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    use_contextual: bool = True,
    return_embeddings: bool = False
) -> Dict[str, np.ndarray]:
    """
    Make predictions for a batch of recordings.
    
    Works with both old and new model structures.
    
    Args:
        model: Trained model (old or new structure)
        dataloader: DataLoader with recordings
        device: Device to use
        use_contextual: Whether to use contextual predictions (default: True) - only used for new model
        return_embeddings: Whether to return embeddings (default: False)
    
    Returns:
        results: Dictionary containing:
            - 'predictions': Predicted class labels
            - 'probabilities': Class probabilities
            - 'labels': Ground truth labels (if available)
            - 'embeddings': Feature embeddings (if return_embeddings=True)
    
    Example:
        >>> results = predict_batch(model, val_loader)
        >>> accuracy = (results['predictions'] == results['labels']).mean()
    """
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_embeddings = [] if return_embeddings else None
    
    # Check if this is old or new model structure
    has_window_classifier = hasattr(model, 'window_classifier')  # New structure
    
    with torch.no_grad():
        for batch in dataloader:
            # Handle both 3-tuple and 4-tuple returns from dataset
            if len(batch) == 4:
                raw_data, _, gcv, labels = batch
            else:
                raw_data, gcv, labels = batch
            
            raw_data = raw_data.to(device)
            gcv = gcv.to(device)
            
            # Forward pass
            result = model(raw_data, gcv)
            
            if has_window_classifier:
                # New model structure
                if len(result) == 3:
                    window_logits, contextual_logits, embeddings = result
                else:
                    window_logits, contextual_logits = result
                    embeddings = None
                
                # Select which branch to use
                logits = contextual_logits if use_contextual else window_logits
            else:
                # Old model structure
                if isinstance(result, tuple):
                    window_logits, _ = result
                else:
                    window_logits = result
                logits = window_logits
                embeddings = None
            
            # Compute predictions and probabilities
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            # Collect results
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            if return_embeddings and embeddings is not None:
                all_embeddings.append(embeddings.cpu().numpy())
    
    # Concatenate results
    results = {
        'predictions': np.concatenate(all_predictions, axis=0),
        'probabilities': np.concatenate(all_probabilities, axis=0),
        'labels': np.concatenate(all_labels, axis=0)
    }
    
    if return_embeddings and all_embeddings:
        results['embeddings'] = np.concatenate(all_embeddings, axis=0)
    
    return results


def evaluate_model(
    model,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    use_contextual: bool = True
) -> Dict[str, float]:
    """
    Evaluate model performance on a dataset.
    
    Works with both old and new model structures.
    
    Args:
        model: Trained model (old or new structure)
        dataloader: DataLoader with test data
        device: Device to use
        use_contextual: Whether to use contextual predictions (default: True) - only used for new model
    
    Returns:
        metrics: Dictionary with evaluation metrics (accuracy, per-class F1, etc.)
    
    Example:
        >>> metrics = evaluate_model(model, test_loader)
        >>> print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    # Get predictions
    results = predict_batch(model, dataloader, device, use_contextual)
    
    predictions = results['predictions'].flatten()
    labels = results['labels'].flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    # Per-class metrics
    class_metrics = {}
    for i in range(len(precision)):
        class_metrics[f'class_{i}'] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    # Overall metrics
    macro_precision = float(precision.mean())
    macro_recall = float(recall.mean())
    macro_f1 = float(f1.mean())
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    return {
        'accuracy': float(accuracy),
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'class_metrics': class_metrics,
        'confusion_matrix': cm.tolist()
    }


def save_predictions(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_path: str,
    labels: Optional[np.ndarray] = None
) -> None:
    """
    Save predictions to a file.
    
    Args:
        predictions: Predicted class labels
        probabilities: Class probabilities
        output_path: Path to save predictions
        labels: Optional ground truth labels
    
    Example:
        >>> save_predictions(predictions, probs, 'predictions.npz')
    """
    save_dict = {
        'predictions': predictions,
        'probabilities': probabilities
    }
    
    if labels is not None:
        save_dict['labels'] = labels
    
    np.savez(output_path, **save_dict)
    print(f"Predictions saved to {output_path}")


# Example usage
if __name__ == "__main__":
    import os
    
    # Try converted checkpoint first, fall back to original
    checkpoint_path = "/home/vojtech.brejtr/SleepSegmentationPaperLong/model_weights_renamed.pth"
    save_path = "predictions.npz"

    
    # Example: Load model and make predictions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model - will auto-detect structure based on checkpoint
    model = DualBranchModel(use_gcv=True,gcv_dim=97, embed_dim=512)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.float().to(device)
    
    # Create dataset and dataloader
    from torch.utils.data import DataLoader
    
    test_dataset = SleepStageDataset(
        folder_raw='dataset',
        folder_features='features_cleaned',
        window_size=2000,
        subject_filter=[1, 2, 3, 4],  # Example subjects
        use_augmentation=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, device=device)
    print("\nEvaluation Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall: {metrics['macro_recall']:.4f}")
    
    # Make predictions and save
    results = predict_batch(model, test_loader, device=device)
    save_predictions(
        results['predictions'],
        results['probabilities'],
        save_path,
        results['labels']
    )
