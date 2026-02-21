"""
Training Utilities

This module provides utility functions for training:
- Data splitting (cross-validation folds)
- Subject filtering
- Configuration helpers

Author: Vojtech Brejtr
Date: February 2026
"""

import json


def get_subject_split(cv_file='cv_folds.json', fold_idx=0):
    """
    Load train/validation subject split from cross-validation folds.
    
    Args:
        cv_file: Path to JSON file containing fold definitions
        fold_idx: Index of fold to use (0-based)
    
    Returns:
        train_subjects: List of training subject IDs
        val_subjects: List of validation subject IDs
    """
    with open(cv_file, 'r') as f:
        cv_folds = json.load(f)
    
    if fold_idx >= len(cv_folds):
        raise ValueError(f"Fold index {fold_idx} out of range. Available folds: {len(cv_folds)}")
    
    fold_data = cv_folds[fold_idx]
    train_subjects = fold_data['train_ids']
    val_subjects = fold_data['val_ids']
    
    print(f"\nFold {fold_data['fold']}:")
    print(f"  Training subjects: {len(train_subjects)} - {train_subjects}")
    print(f"  Validation subjects: {len(val_subjects)} - {val_subjects}")
    
    return train_subjects, val_subjects
