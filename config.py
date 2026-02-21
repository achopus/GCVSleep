"""
Configuration File for Sleep Stage Classification

This module provides pre-defined configurations for different experimental setups.
You can use these configurations directly or modify them for your needs.

Author: Vojtech Brejtr
Date: February 2026
"""

from typing import Dict, Any


# ==================== BASE CONFIGURATION ====================

BASE_CONFIG: Dict[str, Any] = {
    # Data paths
    'raw_data_folder': '../dataset',
    'features_folder': '../features_cleaned',
    'cv_file': '../cv_folds.json',
    
    # Training parameters
    'batch_size': 4,
    'num_epochs': 100,
    'learning_rate': 1e-5,
    'num_workers': 4,
    'window_size': 2000,  # 4 seconds at 500Hz
    
    # Model architecture
    'in_channels': 4,  # 3 EEG + 1 EMG
    'num_classes': 4,  # Wake, N1, N2, REM
    'embed_dim': 512,
    'transformer_hidden_dim': 512,
    'num_transformer_layers': 4,
    'num_heads': 8,
    'cardinality': 32,
    'base_width': 4,
    
    # Training options
    'accumulation_steps': 1,
    'disable_tqdm': False,
    'use_contrastive': True,
    'contrastive_weight': 0.25,
    'contrastive_temperature': 0.07,
    
    # Feature configuration
    'use_gcv': False,
    'gcv_dim': None,
    'global_context_vector_dim': 128,
}


# ==================== EXPERIMENT CONFIGURATIONS ====================

# Full model with Global Context Vector (GCV)
FULL_MODEL_CONFIG = {
    **BASE_CONFIG,
    'save_dir': '../checkpoints_full_model',
    'global_context_vector_dim': 128,
    'use_contrastive': True,
    'contrastive_weight': 0.25,
}

# Model without contrastive learning (ablation study)
NO_CONTRASTIVE_CONFIG = {
    **BASE_CONFIG,
    'save_dir': '../checkpoints_no_contrastive',
    'global_context_vector_dim': 128,
    'use_contrastive': False,
}

# Model without Global Context Vector (ablation study)
NO_GCV_CONFIG = {
    **BASE_CONFIG,
    'save_dir': '../checkpoints_no_gcv',
    'global_context_vector_dim': None,
    'use_contrastive': True,
    'contrastive_weight': 0.25,
}

# Small model (faster training)
SMALL_MODEL_CONFIG = {
    **BASE_CONFIG,
    'save_dir': 'checkpoints_small',
    'embed_dim': 256,
    'transformer_hidden_dim': 256,
    'num_transformer_layers': 2,
    'num_heads': 4,
    'global_context_vector_dim': 64,
}

# Large model (maximum capacity)
LARGE_MODEL_CONFIG = {
    **BASE_CONFIG,
    'save_dir': 'checkpoints_large',
    'batch_size': 2,  # Reduce batch size for larger model
    'accumulation_steps': 2,  # Effective batch size: 4
    'embed_dim': 768,
    'transformer_hidden_dim': 768,
    'num_transformer_layers': 6,
    'num_heads': 12,
    'global_context_vector_dim': 256,
}

# Fast training configuration (for debugging)
DEBUG_CONFIG = {
    **BASE_CONFIG,
    'save_dir': 'checkpoints_debug',
    'batch_size': 2,
    'num_epochs': 5,
    'embed_dim': 128,
    'transformer_hidden_dim': 128,
    'num_transformer_layers': 1,
    'num_heads': 4,
    'num_workers': 0,  # Single-threaded for debugging
    'disable_tqdm': False,
}

# Placebo-only training (domain adaptation experiment)
PLACEBO_ONLY_CONFIG = {
    **BASE_CONFIG,
    'save_dir': 'checkpoints_placebo_only',
    'placebo_only_train': True,
    'global_context_vector_dim': 128,
}

# Limited training data (data efficiency experiment)
LIMITED_DATA_CONFIG = {
    **BASE_CONFIG,
    'save_dir': 'checkpoints_limited_data',
    'n_minutes_in_training': 120,  # Use only 2 hours per fold
    'minimal_iterations_per_train_epoch': 100,
    'global_context_vector_dim': 128,
}


# ==================== CONFIGURATION REGISTRY ====================

CONFIGS = {
    'full': FULL_MODEL_CONFIG,
    'no_contrastive': NO_CONTRASTIVE_CONFIG,
    'no_gcv': NO_GCV_CONFIG,
    'small': SMALL_MODEL_CONFIG,
    'large': LARGE_MODEL_CONFIG,
    'debug': DEBUG_CONFIG,
    'placebo_only': PLACEBO_ONLY_CONFIG,
    'limited_data': LIMITED_DATA_CONFIG,
}


def get_config(name: str = 'full') -> Dict[str, Any]:
    """
    Get a pre-defined configuration by name.
    
    Args:
        name: Configuration name (see CONFIGS keys)
    
    Returns:
        config: Configuration dictionary
    
    Available configurations:
        - 'full': Full model with all features
        - 'no_contrastive': Without contrastive learning
        - 'no_gcv': Without Global Context Vector
        - 'small': Smaller model (faster training)
        - 'large': Larger model (maximum capacity)
        - 'debug': Fast configuration for debugging
        - 'placebo_only': Train on placebo recordings only
        - 'limited_data': Limited training data experiment
    
    Example:
        >>> config = get_config('full')
        >>> from train import train
        >>> model, history = train(**config)
    """
    if name not in CONFIGS:
        available = list(CONFIGS.keys())
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    
    return CONFIGS[name].copy()


def update_config(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Update a configuration with custom parameters.
    
    Args:
        config: Base configuration dictionary
        **kwargs: Parameters to update
    
    Returns:
        updated_config: Updated configuration
    
    Example:
        >>> config = get_config('full')
        >>> config = update_config(config, batch_size=8, num_epochs=150)
    """
    updated = config.copy()
    updated.update(kwargs)
    return updated


def print_config(config: Dict[str, Any]) -> None:
    """
    Print configuration in a readable format.
    
    Args:
        config: Configuration dictionary
    
    Example:
        >>> config = get_config('full')
        >>> print_config(config)
    """
    print("Configuration:")
    print("-" * 50)
    for key, value in sorted(config.items()):
        print(f"  {key:30s}: {value}")
    print("-" * 50)


# Example usage
if __name__ == "__main__":
    # List all available configurations
    print("Available configurations:")
    for name in CONFIGS.keys():
        print(f"  - {name}")
    
    print("\n" + "=" * 50 + "\n")
    
    # Print full model configuration
    config = get_config('full')
    print_config(config)
    
    # Example: Customize configuration
    print("\n" + "=" * 50 + "\n")
    print("Customized configuration:")
    custom_config = update_config(
        config,
        batch_size=8,
        num_epochs=150,
        learning_rate=5e-6
    )
    print_config(custom_config)
