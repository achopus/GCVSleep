"""
Training Script for Sleep Stage Classification

This script provides a command-line interface for training the dual-branch
sleep stage classification model. It wraps the training functionality from
the training/ submodule.

Usage:
    # Train with default configuration
    python train.py --config full_model --fold 1
    
    # Train with custom parameters
    python train.py --config full_model --fold 1 --batch_size 8 --learning_rate 1e-4
    
    # Train all folds
    python train.py --config full_model --all_folds

Author:  Vojtech Brejtr
Date: February 2026
"""

import argparse
import json
import torch
from pathlib import Path

from training.trainer import train
from training.utils import get_subject_split
import config


def get_config(config_name: str):
    """
    Load a predefined configuration by name.
    
    Args:
        config_name: Name of the configuration (e.g., 'full_model', 'no_contrastive', 'no_gcv')
    
    Returns:
        Configuration dictionary
    """
    config_map = {
        'base': config.BASE_CONFIG,
        'full_model': config.FULL_MODEL_CONFIG,
        'no_contrastive': config.NO_CONTRASTIVE_CONFIG,
        'no_gcv': config.NO_GCV_CONFIG,
    }
    
    if config_name not in config_map:
        raise ValueError(f"Unknown configuration: {config_name}. Available: {list(config_map.keys())}")
    
    return config_map[config_name].copy()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train dual-branch sleep stage classification model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration selection
    parser.add_argument('--config', type=str, default='full_model',
                        choices=['base', 'full_model', 'no_contrastive', 'no_gcv'],
                        help='Configuration preset to use')
    
    # Training setup
    parser.add_argument('--fold', type=int, default=None,
                        help='Cross-validation fold index (0-4 for 5-fold CV)')
    parser.add_argument('--all_folds', action='store_true',
                        help='Train on all folds sequentially')
    
    # Data paths (override config)
    parser.add_argument('--raw_data_folder', type=str, default=None,
                        help='Path to raw data folder')
    parser.add_argument('--features_folder', type=str, default=None,
                        help='Path to GCV features folder')
    parser.add_argument('--cv_file', type=str, default=None,
                        help='Path to cross-validation folds JSON file')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save checkpoints')
    
    # Training parameters (override config)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of data loader workers')
    
    # Model architecture (override config)
    parser.add_argument('--embed_dim', type=int, default=None,
                        help='Embedding dimension')
    parser.add_argument('--transformer_hidden_dim', type=int, default=None,
                        help='Transformer hidden dimension')
    parser.add_argument('--num_transformer_layers', type=int, default=None,
                        help='Number of Transformer layers')
    parser.add_argument('--num_heads', type=int, default=None,
                        help='Number of attention heads')
    parser.add_argument('--global_context_vector_dim', type=int, default=None,
                        help='Learnable global context vector dimension (None to disable)')
    
    # Training options (override config)
    parser.add_argument('--accumulation_steps', type=int, default=None,
                        help='Gradient accumulation steps')
    parser.add_argument('--no_contrastive', action='store_true',
                        help='Disable contrastive learning')
    parser.add_argument('--contrastive_weight', type=float, default=None,
                        help='Weight for contrastive loss')
    parser.add_argument('--disable_tqdm', action='store_true',
                        help='Disable progress bars')
    
    # Device selection
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    
    # Special training modes
    parser.add_argument('--saline_only', action='store_true',
                        help='Train only on saline subjects')
    parser.add_argument('--n_minutes', type=int, default=None,
                        help='Limit training to N minutes per recording')
    parser.add_argument('--minimal_iterations', type=int, default=None,
                        help='Minimum iterations per training epoch')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load base configuration
    cfg = get_config(args.config)
    print(f"Loaded configuration: {args.config}")
    
    # Override with command line arguments
    override_keys = [
        'raw_data_folder', 'features_folder', 'cv_file', 'save_dir',
        'batch_size', 'num_epochs', 'learning_rate', 'num_workers',
        'embed_dim', 'transformer_hidden_dim', 'num_transformer_layers',
        'num_heads', 'global_context_vector_dim', 'accumulation_steps',
        'contrastive_weight'
    ]
    
    for key in override_keys:
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
            print(f"  Override: {key} = {value}")
    
    # Handle boolean flags
    if args.no_contrastive:
        cfg['use_contrastive'] = False
        print("  Override: use_contrastive = False")
    
    if args.disable_tqdm:
        cfg['disable_tqdm'] = True
        print("  Override: disable_tqdm = True")
    
    # Device setup
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Using CPU.")
        device = 'cpu'
    
    print(f"\nDevice: {device}")
    
    # Determine folds to train
    if args.all_folds:
        # Count number of folds from CV file
        with open(cfg['cv_file'], 'r') as f:
            cv_data = json.load(f)
        num_folds = len(cv_data)
        folds = list(range(num_folds))
        print(f"Training all {num_folds} folds\n")
    elif args.fold is not None:
        folds = [args.fold]
        print(f"Training fold {args.fold}\n")
    else:
        raise ValueError("Must specify either --fold or --all_folds")
    
    # Train each fold
    for fold_idx in folds:
        print("=" * 80)
        print(f"TRAINING FOLD {fold_idx + 1}")
        print("=" * 80)
        
        # Create fold-specific save directory
        save_dir = Path(cfg['save_dir']) / f"fold{fold_idx + 1}"
        print(f"Checkpoints will be saved to: {save_dir}\n")
        
        # Train model
        model, history = train(
            raw_data_folder=cfg['raw_data_folder'],
            features_folder=cfg['features_folder'],
            batch_size=cfg['batch_size'],
            num_epochs=cfg['num_epochs'],
            learning_rate=cfg['learning_rate'],
            device=device,
            save_dir=str(save_dir),
            num_workers=cfg['num_workers'],
            window_size=cfg['window_size'],
            cv_file=cfg['cv_file'],
            fold_idx=fold_idx,
            embed_dim=cfg['embed_dim'],
            transformer_hidden_dim=cfg['transformer_hidden_dim'],
            num_transformer_layers=cfg['num_transformer_layers'],
            num_heads=cfg['num_heads'],
            cardinality=cfg['cardinality'],
            base_width=cfg['base_width'],
            accumulation_steps=cfg['accumulation_steps'],
            disable_tqdm=cfg['disable_tqdm'],
            use_contrastive=cfg['use_contrastive'],
            contrastive_weight=cfg['contrastive_weight'],
            contrastive_temperature=cfg['contrastive_temperature'],
            global_context_vector_dim=cfg.get('global_context_vector_dim'),
            saline_only_training=args.saline_only,
            n_minutes_in_training=args.n_minutes,
            minimal_iterations_per_train_epoch=args.minimal_iterations
        )
        
        print(f"\nFold {fold_idx + 1} training completed!")
        print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
        print(f"Best validation F1: {max(history['val_f1']):.4f}\n")
    
    print("=" * 80)
    print("ALL TRAINING COMPLETED!")
    print("=" * 80)


if __name__ == '__main__':
    main()
