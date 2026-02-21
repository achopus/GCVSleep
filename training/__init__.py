"""
Training Package

This package contains all components for training the dual-branch sleep stage
classification model, organized into modular files:

- losses: Custom loss functions for classification and contrastive learning
- metrics: Performance metrics calculation (accuracy, F1, ROC-AUC, etc.)
- utils: Utility functions for data splitting and preprocessing
- trainer: Main training and validation loops

Author: Vojtech Brejtr
Date: February 2026
"""

from .losses import DualBranchLoss, SupervisedContrastiveLoss
from .metrics import (
    calculate_accuracy,
    calculate_per_class_metrics,
    calculate_comprehensive_metrics,
    calculate_metrics
)
from .utils import get_subject_split
from .trainer import train_epoch, validate_epoch, train

__all__ = [
    # Losses
    'DualBranchLoss',
    'SupervisedContrastiveLoss',
    # Metrics
    'calculate_accuracy',
    'calculate_per_class_metrics',
    'calculate_comprehensive_metrics',
    'calculate_metrics',
    # Utils
    'get_subject_split',
    # Trainer
    'train_epoch',
    'validate_epoch',
    'train',
]
