"""
Performance Metrics for Sleep Stage Classification

This module implements various metrics for evaluating model performance:
- Accuracy calculation
- Per-class precision, recall, F1 score
- Comprehensive metrics including ROC-AUC and confusion matrices

Author: Vojtech Brejtr
Date: February 2026
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


def calculate_accuracy(logits, target):
    """
    Calculate classification accuracy.
    
    Args:
        logits: Model predictions (batch_size, num_windows, num_classes)
        target: Ground truth labels (batch_size, num_windows)
    
    Returns:
        accuracy: Accuracy value as float
    """
    pred_classes = torch.argmax(logits, dim=-1)
    correct = (pred_classes == target).sum().item()
    total = pred_classes.numel()
    return correct / total if total > 0 else 0.0


def calculate_per_class_metrics(logits, target, num_classes=4):
    """
    Calculate precision, recall, and F1 score for each class.
    
    Args:
        logits: Model predictions (batch_size, num_windows, num_classes)
        target: Ground truth labels (batch_size, num_windows)
        num_classes: Number of classes (default: 4)
    
    Returns:
        class_metrics: Dictionary with per-class metrics
    """
    pred_classes = torch.argmax(logits, dim=-1).view(-1)
    target_classes = target.view(-1).long()
    
    class_metrics = {}
    for cls in range(num_classes):
        pred_cls = (pred_classes == cls)
        target_cls = (target_classes == cls)
        
        tp = (pred_cls & target_cls).sum().item()
        fp = (pred_cls & ~target_cls).sum().item()
        fn = (~pred_cls & target_cls).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics[f'class_{cls}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return class_metrics


def calculate_comprehensive_metrics(window_logits, contextual_logits, target, num_classes=4):
    """
    Calculate comprehensive evaluation metrics including ROC-AUC and confusion matrices.
    
    Args:
        window_logits: Window branch predictions (batch_size, num_windows, num_classes)
        contextual_logits: Contextual branch predictions (batch_size, num_windows, num_classes)
        target: Ground truth labels (batch_size, num_windows)
        num_classes: Number of classes (default: 4)
    
    Returns:
        metrics: Dictionary with comprehensive metrics
    """
    # Convert to numpy for sklearn metrics
    window_pred = torch.argmax(window_logits, dim=-1).view(-1).cpu().numpy()
    contextual_pred = torch.argmax(contextual_logits, dim=-1).view(-1).cpu().numpy()
    target_np = target.view(-1).cpu().numpy().astype(int)
    
    # Get probabilities for ROC-AUC
    window_probs = torch.softmax(window_logits, dim=-1).view(-1, num_classes).cpu().numpy()
    contextual_probs = torch.softmax(contextual_logits, dim=-1).view(-1, num_classes).cpu().numpy()
    
    # Calculate ROC-AUC (one-vs-rest multiclass)
    try:
        window_roc_auc = roc_auc_score(target_np, window_probs, multi_class='ovr', average='macro')
        contextual_roc_auc = roc_auc_score(target_np, contextual_probs, multi_class='ovr', average='macro')
    except:
        window_roc_auc = 0.0
        contextual_roc_auc = 0.0
    
    # Confusion matrices
    window_cm = confusion_matrix(target_np, window_pred, labels=list(range(num_classes)))
    contextual_cm = confusion_matrix(target_np, contextual_pred, labels=list(range(num_classes)))
    
    # Per-class metrics
    contextual_report = classification_report(
        target_np, contextual_pred,
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0
    )
    
    class_metrics = {}
    for cls in range(num_classes):
        cls_key = f'class_{cls}'
        class_metrics[cls_key] = {
            'precision': contextual_report[str(cls)]['precision'],
            'recall': contextual_report[str(cls)]['recall'],
            'f1': contextual_report[str(cls)]['f1-score'],
            'support': int(contextual_report[str(cls)]['support'])
        }
    
    return {
        'window_accuracy': float((window_pred == target_np).mean()),
        'contextual_accuracy': float((contextual_pred == target_np).mean()),
        'window_roc_auc': float(window_roc_auc),
        'contextual_roc_auc': float(contextual_roc_auc),
        'window_confusion_matrix': window_cm.tolist(),
        'contextual_confusion_matrix': contextual_cm.tolist(),
        'class_metrics': class_metrics,
        'macro_precision': contextual_report['macro avg']['precision'],
        'macro_recall': contextual_report['macro avg']['recall'],
        'macro_f1': contextual_report['macro avg']['f1-score']
    }


def calculate_metrics(window_logits, contextual_logits, target):
    """
    Calculate basic metrics for batch-level tracking during training.
    
    Args:
        window_logits: Window branch predictions
        contextual_logits: Contextual branch predictions
        target: Ground truth labels
    
    Returns:
        metrics: Dictionary with basic metrics
    """
    window_accuracy = calculate_accuracy(window_logits, target)
    contextual_accuracy = calculate_accuracy(contextual_logits, target)
    class_metrics = calculate_per_class_metrics(contextual_logits, target)
    
    return {
        'window_accuracy': window_accuracy,
        'contextual_accuracy': contextual_accuracy,
        'class_metrics': class_metrics
    }
