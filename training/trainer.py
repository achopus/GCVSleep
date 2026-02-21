"""
Training and Validation Loops

This module implements the main training functionality:
- train_epoch: Training for one epoch with gradient accumulation
- validate_epoch: Validation with comprehensive metrics
- train: Complete training loop with checkpointing and history

Author: Vojtech Brejtr
Date: February 2026
"""

import torch
import torch.optim as optim
import numpy as np
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from .metrics import calculate_metrics, calculate_comprehensive_metrics
from .utils import get_subject_split
from models import DualBranchModel
from data_management.dataset import SleepStageDataset


def train_epoch(
    model, 
    dataloader, 
    criterion, 
    optimizer, 
    device, 
    cosine_scheduler=None, 
    accumulation_steps=1, 
    disable_tqdm=False, 
    contrastive_criterion=None, 
    contrastive_weight=0.0, 
    minimal_iterations_per_train_epoch=None
):
    """
    Train model for one epoch with gradient accumulation and optional contrastive learning.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Classification loss function
        optimizer: Optimizer
        device: Device to use ('cuda' or 'cpu')
        cosine_scheduler: Optional cosine annealing scheduler
        accumulation_steps: Number of batches to accumulate before updating (default: 1)
        disable_tqdm: Whether to disable progress bar (default: False)
        contrastive_criterion: Optional contrastive loss function
        contrastive_weight: Weight for contrastive loss (default: 0.0)
        minimal_iterations_per_train_epoch: If set, ensures at least this many iterations
    
    Returns:
        metrics: Dictionary with training metrics (loss, accuracy, per-class metrics)
    """
    model.train()
    total_loss = 0.0
    total_window_loss = 0.0
    total_contextual_loss = 0.0
    total_contrastive_loss = 0.0
    all_metrics = []
    use_contrastive = contrastive_criterion is not None and contrastive_weight > 0
    
    # Determine total iterations
    total_iterations = minimal_iterations_per_train_epoch or len(dataloader)
    
    # Create an iterator that loops through dataloader if needed
    def infinite_dataloader():
        while True:
            for batch in dataloader:
                yield batch
    
    data_iterator = infinite_dataloader() if minimal_iterations_per_train_epoch is not None else iter(dataloader)
    
    with tqdm(total=total_iterations, desc="Training", disable=disable_tqdm) as pbar:
        for batch_idx in range(total_iterations):
            # Unpack batch based on whether contrastive learning is used
            batch_data = next(data_iterator)
            if contrastive_criterion is not None and contrastive_weight > 0:
                # With contrastive learning: (raw_data, raw_data_augmented, gcv, labels)
                raw_data, raw_data_augmented, gcv, labels = batch_data
                raw_data = raw_data.to(device)
                raw_data_augmented = raw_data_augmented.to(device)
            else:
                # Without contrastive learning: (raw_data, gcv, labels)
                raw_data, gcv, labels = batch_data
                raw_data = raw_data.to(device)
                raw_data_augmented = raw_data  # Use same data (already augmented)
            
            gcv = gcv.to(device)
            labels = labels.to(device)
            
            # Check for NaN in input data
            if torch.isnan(raw_data).any():
                print("Warning: NaN detected in raw_data, skipping batch")
                continue
            if torch.isnan(gcv).any():
                print("Warning: NaN detected in gcv, skipping batch")
                continue
            if torch.isnan(labels).any():
                print("Warning: NaN detected in labels, skipping batch")
                continue
            
            # raw_data shape: (batch_size, num_windows, in_channels, window_size)
            # gcv shape: (batch_size, num_windows, gcv_dim)
            # labels shape: (batch_size, num_windows)
            
            # Forward pass: concatenate original and augmented data for single forward pass
            if contrastive_criterion is not None and contrastive_weight > 0:
                # Need embeddings for contrastive learning - temporarily enable
                original_return_embeddings = model.return_embeddings
                model.return_embeddings = True
                
                # Concatenate along batch dimension: [original; augmented]
                raw_data_combined = torch.cat([raw_data, raw_data_augmented], dim=0)
                gcv_combined = torch.cat([gcv, gcv], dim=0)
                
                # Single forward pass
                window_logits_combined, contextual_logits_combined, embeddings_combined = model(raw_data_combined, gcv_combined)
                
                # Restore original setting
                model.return_embeddings = original_return_embeddings
                
                # Split results
                batch_size = raw_data.size(0)
                window_logits = window_logits_combined[batch_size:]  # Use augmented for classification
                contextual_logits = contextual_logits_combined[batch_size:]  # Use augmented for classification
                # Extract backbone embeddings for contrastive loss
                embeddings_orig = embeddings_combined['backbone_embeddings'][:batch_size]
                embeddings_aug = embeddings_combined['backbone_embeddings'][batch_size:]
            else:
                # Standard forward pass without contrastive learning
                result = model(raw_data_augmented, gcv)
                if len(result) == 3:
                    window_logits, contextual_logits, _ = result
                else:
                    window_logits, contextual_logits = result
                embeddings_orig = None
                embeddings_aug = None
            
            # Check for NaN in outputs
            if torch.isnan(window_logits).any() or torch.isnan(contextual_logits).any():
                print("Warning: NaN detected in model outputs, skipping batch")
                print(f"  window_logits has NaN: {torch.isnan(window_logits).any()}")
                print(f"  contextual_logits has NaN: {torch.isnan(contextual_logits).any()}")
                continue
            
            # Compute classification loss
            loss, window_loss, contextual_loss = criterion(window_logits, contextual_logits, labels)
            
            # Compute contrastive loss if enabled
            contrastive_loss = torch.tensor(0.0, device=device)
            if contrastive_criterion is not None and contrastive_weight > 0:
                contrastive_loss = contrastive_criterion(embeddings_orig, embeddings_aug, labels)
                
                # Add contrastive loss to total loss
                loss = loss + contrastive_weight * contrastive_loss
            
            # Check for NaN loss
            if torch.isnan(loss):
                print("Warning: NaN loss detected, skipping batch")
                continue
            
            # Scale loss by accumulation steps (to maintain same effective learning rate)
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Check for NaN gradients
            has_nan = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN gradient detected in {name}")
                    has_nan = True
                    break
            
            if has_nan:
                print(f"Skipping update due to NaN gradients")
                optimizer.zero_grad()
                continue
            
            # Perform optimizer step only every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                # Step cosine scheduler after each optimizer update
                if cosine_scheduler is not None:
                    cosine_scheduler.step()
            
            # Calculate metrics
            with torch.no_grad():
                metrics = calculate_metrics(window_logits, contextual_logits, labels)
                all_metrics.append(metrics)
            
            # Accumulate losses (multiply back to get original scale for tracking)
            total_loss += (loss.item() * accumulation_steps)
            total_window_loss += window_loss.item()
            total_contextual_loss += contextual_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            
            # Update progress bar
            postfix_dict = {
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'win_acc': f'{metrics["window_accuracy"]:.4f}',
                'ctx_acc': f'{metrics["contextual_accuracy"]:.4f}'
            }
            if use_contrastive:
                postfix_dict['contr'] = f'{contrastive_loss.item():.4f}'
            pbar.set_postfix(postfix_dict)
            pbar.update(1)
        
        # Handle any remaining gradients at the end of epoch
        if (total_iterations % accumulation_steps) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
    
    # Average metrics over actual number of batches processed
    num_batches = len(all_metrics)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_window_loss = total_window_loss / num_batches if num_batches > 0 else 0.0
    avg_contextual_loss = total_contextual_loss / num_batches if num_batches > 0 else 0.0
    avg_contrastive_loss = total_contrastive_loss / num_batches if (use_contrastive and num_batches > 0) else 0.0
    avg_window_accuracy = np.mean([m['window_accuracy'] for m in all_metrics])
    avg_contextual_accuracy = np.mean([m['contextual_accuracy'] for m in all_metrics])
    
    # Average per-class metrics
    num_classes = 4
    avg_class_metrics = {}
    for cls in range(num_classes):
        cls_key = f'class_{cls}'
        avg_class_metrics[cls_key] = {
            'precision': np.mean([m['class_metrics'][cls_key]['precision'] for m in all_metrics]),
            'recall': np.mean([m['class_metrics'][cls_key]['recall'] for m in all_metrics]),
            'f1': np.mean([m['class_metrics'][cls_key]['f1'] for m in all_metrics])
        }
    
    result = {
        'loss': avg_loss,
        'window_loss': avg_window_loss,
        'contextual_loss': avg_contextual_loss,
        'window_accuracy': avg_window_accuracy,
        'contextual_accuracy': avg_contextual_accuracy,
        'class_metrics': avg_class_metrics
    }
    
    if use_contrastive:
        result['contrastive_loss'] = avg_contrastive_loss
    
    return result


def validate_epoch(model, dataloader, criterion, device, compute_comprehensive=False, disable_tqdm=False):
    """
    Validate for one epoch.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to use
        compute_comprehensive: If True, computes comprehensive metrics (ROC-AUC, confusion matrices)
        disable_tqdm: If True, disables tqdm progress bars (useful for log files)
    
    Returns:
        metrics: Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_window_loss = 0.0
    total_contextual_loss = 0.0
    all_metrics = []
    
    # For comprehensive metrics, collect all predictions
    if compute_comprehensive:
        all_window_logits = []
        all_contextual_logits = []
        all_labels = []
    
    with torch.no_grad():
        with tqdm(dataloader, desc="Validation", disable=disable_tqdm) as pbar:
            for raw_data, gcv, labels in pbar:
                raw_data = raw_data.to(device)
                gcv = gcv.to(device)
                labels = labels.to(device)
                
                # Forward pass
                result = model(raw_data, gcv)
                if len(result) == 3:
                    window_logits, contextual_logits, _ = result
                else:
                    window_logits, contextual_logits = result
                loss, window_loss, contextual_loss = criterion(window_logits, contextual_logits, labels)
                
                if compute_comprehensive:
                    all_window_logits.append(window_logits)
                    all_contextual_logits.append(contextual_logits)
                    all_labels.append(labels)
                
                # Calculate metrics
                metrics = calculate_metrics(window_logits, contextual_logits, labels)
                all_metrics.append(metrics)
                
                total_loss += loss.item()
                total_window_loss += window_loss.item()
                total_contextual_loss += contextual_loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'win_acc': f'{metrics["window_accuracy"]:.4f}',
                    'ctx_acc': f'{metrics["contextual_accuracy"]:.4f}'
                })
    
    # Average metrics
    avg_loss = total_loss / len(dataloader)
    avg_window_loss = total_window_loss / len(dataloader)
    avg_contextual_loss = total_contextual_loss / len(dataloader)
    avg_window_accuracy = np.mean([m['window_accuracy'] for m in all_metrics])
    avg_contextual_accuracy = np.mean([m['contextual_accuracy'] for m in all_metrics])
    
    # Average per-class metrics
    num_classes = 4
    avg_class_metrics = {}
    for cls in range(num_classes):
        cls_key = f'class_{cls}'
        avg_class_metrics[cls_key] = {
            'precision': np.mean([m['class_metrics'][cls_key]['precision'] for m in all_metrics]),
            'recall': np.mean([m['class_metrics'][cls_key]['recall'] for m in all_metrics]),
            'f1': np.mean([m['class_metrics'][cls_key]['f1'] for m in all_metrics])
        }
    
    result = {
        'loss': avg_loss,
        'window_loss': avg_window_loss,
        'contextual_loss': avg_contextual_loss,
        'window_accuracy': avg_window_accuracy,
        'contextual_accuracy': avg_contextual_accuracy,
        'class_metrics': avg_class_metrics
    }
    
    # If comprehensive metrics requested, compute them on all collected predictions
    if compute_comprehensive:
        all_window_logits = torch.cat(all_window_logits, dim=0)
        all_contextual_logits = torch.cat(all_contextual_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        comprehensive = calculate_comprehensive_metrics(
            all_window_logits, all_contextual_logits, all_labels, num_classes
        )
        result['comprehensive'] = comprehensive
    
    return result


def train(
    raw_data_folder='dataset',
    features_folder='features_cleaned',
    batch_size=32,
    num_epochs=50,
    learning_rate=1e-5,
    device='cuda',
    save_dir='checkpoints_features_enhanced',
    num_workers=8,
    window_size=2000,
    cv_file='cv_folds.json',
    fold_idx=0,
    embed_dim=512,
    transformer_hidden_dim=512,
    num_transformer_layers=2,
    num_heads=8,
    cardinality=32,
    base_width=4,
    accumulation_steps=1,
    disable_tqdm=False,
    use_contrastive=False,
    contrastive_weight=0.5,
    contrastive_temperature=0.07,
    global_context_vector_dim=None,
    saline_only_training=False,
    n_minutes_in_training=None,
    minimal_iterations_per_train_epoch=None
):
    """
    Main training function for dual-branch sleep stage classification.
    
    Args:
        raw_data_folder: Path to raw dataset folder
        features_folder: Path to GCV features folder (contains manual features from dataset)
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to use ('cuda' or 'cpu')
        save_dir: Directory to save checkpoints
        num_workers: Number of data loader workers
        window_size: Window size in samples (default 2000 = 4 seconds at 500Hz)
        cv_file: Path to cross-validation folds JSON file
        fold_idx: Index of fold to use (0-based, so fold 1 is index 0)
        embed_dim: Embedding dimension for model
        transformer_hidden_dim: Hidden dimension for Transformer
        num_transformer_layers: Number of Transformer layers
        num_heads: Number of attention heads
        cardinality: ResNeXt cardinality
        base_width: ResNeXt base width
        accumulation_steps: Number of batches to accumulate gradients before updating
        disable_tqdm: If True, disables tqdm progress bars
        use_contrastive: If True, enables supervised contrastive learning
        contrastive_weight: Weight for contrastive loss term (default: 0.5)
        contrastive_temperature: Temperature parameter for contrastive loss (default: 0.07)
        global_context_vector_dim: If provided, uses a learnable global context vector
        saline_only_training: If True, trains only on saline subjects
        n_minutes_in_training: If provided, limits training to n minutes per recording
        minimal_iterations_per_train_epoch: If provided, ensures at least this many iterations
    
    Returns:
        model: Trained model
        history: Training history dictionary
    """
    # Import loss functions here to avoid circular imports
    from .losses import DualBranchLoss, SupervisedContrastiveLoss
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load train/val split
    train_subjects, val_subjects = get_subject_split(cv_file=cv_file, fold_idx=fold_idx)
    
    # Determine if we're using GCV or global context vector
    use_gcv = global_context_vector_dim is None
    
    # Create datasets with subject filtering
    train_dataset = SleepStageDataset(
        folder_raw=raw_data_folder,
        folder_features=features_folder if use_gcv else None,
        window_size=window_size,
        subject_filter=train_subjects,
        use_augmentation=True,
        return_all_augmented=use_contrastive,  # Only return both augmented+original when using contrastive
        placebo_only_train=saline_only_training,
        n_minutes_in_training=n_minutes_in_training
    )
    
    val_dataset = SleepStageDataset(
        folder_raw=raw_data_folder,
        folder_features=features_folder if use_gcv else None,
        window_size=window_size,
        subject_filter=val_subjects,
        use_augmentation=False,
    )
    
    # Data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True
    )
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)} recordings")
    print(f"  Validation: {len(val_dataset)} recordings")
    
    # Get GCV dimension from a sample (if using GCV)
    if use_gcv:
        if use_contrastive:
            sample_raw_orig, sample_raw_aug, sample_features, sample_labels = train_dataset[0] # type: ignore
        else:
            sample_raw, sample_features, sample_labels = train_dataset[0]
        gcv_dim = sample_features.shape[1]  # (num_windows, feature_dim)
        print(f"  GCV dimension: {gcv_dim}")
    else:
        gcv_dim = None
        print(f"  Using global context vector with dimension: {global_context_vector_dim}")
    
    # Initialize model
    model = DualBranchModel(
        in_channels=4,
        num_classes=4,
        embed_dim=embed_dim,
        window_size=window_size,
        cardinality=cardinality,
        base_width=base_width,
        num_layers=num_transformer_layers,
        num_heads=num_heads,
        use_gcv=use_gcv,
        gcv_dim=gcv_dim,
        use_cross_attention=True,
        max_seq_len=512,
        return_embeddings=use_contrastive
    )
    model = model.to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = DualBranchLoss(window_weight=0.01, contextual_weight=0.99)
    contrastive_criterion = SupervisedContrastiveLoss(temperature=contrastive_temperature) if use_contrastive else None
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Plateau scheduler to reduce max LR when validation plateaus
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Cosine annealing with warm restarts for cycling within epochs
    steps_per_epoch = len(train_dataloader) // accumulation_steps
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=steps_per_epoch, T_mult=1, eta_min=learning_rate * 0.01
    )
    
    current_max_lr = learning_rate
    
    print(f"\nGradient Accumulation:")
    print(f"  Accumulation steps: {accumulation_steps}")
    print(f"  Effective batch size: {batch_size * accumulation_steps}")
    print(f"  Optimizer updates per epoch: {steps_per_epoch}")
    
    # Training history
    history = {
        'train_loss': [],
        'train_window_loss': [],
        'train_contextual_loss': [],
        'train_window_accuracy': [],
        'train_contextual_accuracy': [],
        'val_loss': [],
        'val_window_loss': [],
        'val_contextual_loss': [],
        'val_window_accuracy': [],
        'val_contextual_accuracy': [],
        'learning_rate': []
    }
    
    if use_contrastive:
        history['train_contrastive_loss'] = []
    
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_train_epoch = -1
    best_val_epoch = -1
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_metrics = train_epoch(
            model, train_dataloader, criterion, optimizer, device, 
            cosine_scheduler, accumulation_steps, disable_tqdm,
            contrastive_criterion, contrastive_weight, minimal_iterations_per_train_epoch
        )
        
        # Validate
        val_metrics = validate_epoch(model, val_dataloader, criterion, device, disable_tqdm=disable_tqdm)
        
        # Update plateau scheduler based on validation loss
        old_lr = optimizer.param_groups[0]['lr']
        plateau_scheduler.step(val_metrics['loss'])
        new_lr = optimizer.param_groups[0]['lr']
        
        # If plateau scheduler reduced LR, update the max LR for cosine scheduler
        if new_lr < current_max_lr:
            current_max_lr = new_lr
            # Reset cosine scheduler with new max LR
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=steps_per_epoch, T_mult=1, eta_min=current_max_lr * 0.01
            )
            print(f"\n  Learning rate reduced by plateau scheduler: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_window_loss'].append(train_metrics['window_loss'])
        history['train_contextual_loss'].append(train_metrics['contextual_loss'])
        history['train_window_accuracy'].append(train_metrics['window_accuracy'])
        history['train_contextual_accuracy'].append(train_metrics['contextual_accuracy'])
        if use_contrastive:
            history['train_contrastive_loss'].append(train_metrics.get('contrastive_loss', 0.0))
        history['val_loss'].append(val_metrics['loss'])
        history['val_window_loss'].append(val_metrics['window_loss'])
        history['val_contextual_loss'].append(val_metrics['contextual_loss'])
        history['val_window_accuracy'].append(val_metrics['window_accuracy'])
        history['val_contextual_accuracy'].append(val_metrics['contextual_accuracy'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  TRAIN - Loss: {train_metrics['loss']:.4f}, "
              f"Win Acc: {train_metrics['window_accuracy']:.4f}, "
              f"Ctx Acc: {train_metrics['contextual_accuracy']:.4f}")
        print(f"  VAL   - Loss: {val_metrics['loss']:.4f}, "
              f"Win Acc: {val_metrics['window_accuracy']:.4f}, "
              f"Ctx Acc: {val_metrics['contextual_accuracy']:.4f}")
        
        # Print per-class metrics
        print("\n  Per-class metrics (Contextual branch):")
        for cls in range(4):
            cls_key = f'class_{cls}'
            cls_metrics = val_metrics['class_metrics'][cls_key]
            print(f"    Class {cls}: P={cls_metrics['precision']:.4f}, "
                  f"R={cls_metrics['recall']:.4f}, F1={cls_metrics['f1']:.4f}")
        
        # Calculate average F1 score from class metrics
        avg_val_f1 = np.mean([val_metrics['class_metrics'][f'class_{cls}']['f1'] for cls in range(4)])
        
        # Save best model with comprehensive metrics (based on F1 score)
        if avg_val_f1 > best_val_f1:
            best_val_loss = val_metrics['loss']
            best_val_f1 = avg_val_f1
            best_val_epoch = epoch
            
            # Compute comprehensive metrics for validation
            print("\n  Computing comprehensive metrics for best validation epoch...")
            comprehensive_val_metrics = validate_epoch(
                model, val_dataloader, criterion, device, 
                compute_comprehensive=True, disable_tqdm=disable_tqdm
            )
            
            # Save model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_metrics['loss'],
                'accuracy': val_metrics['contextual_accuracy'],
                'metrics': val_metrics,
                'model_config': {
                    'in_channels': 4,
                    'num_classes': 4,
                    'embed_dim': embed_dim,
                    'window_size': window_size,
                    'cardinality': cardinality,
                    'base_width': base_width,
                    'num_layers': num_transformer_layers,
                    'num_heads': num_heads,
                    'use_gcv': use_gcv,
                    'gcv_dim': gcv_dim,
                    'use_cross_attention': True,
                    'max_seq_len': 512
                }
            }, save_path / 'best_model.pth')
            
            # Save comprehensive metrics to JSON
            comprehensive = comprehensive_val_metrics.get('comprehensive', {})
            best_val_metrics_json = {
                'epoch': epoch + 1,
                'loss': comprehensive_val_metrics['loss'],
                'window_loss': comprehensive_val_metrics['window_loss'],
                'contextual_loss': comprehensive_val_metrics['contextual_loss'],
                'window_accuracy': comprehensive_val_metrics['window_accuracy'],
                'contextual_accuracy': comprehensive_val_metrics['contextual_accuracy'],
                'window_roc_auc': comprehensive.get('window_roc_auc', 0.0),
                'contextual_roc_auc': comprehensive.get('contextual_roc_auc', 0.0),
                'macro_precision': comprehensive.get('macro_precision', 0.0),
                'macro_recall': comprehensive.get('macro_recall', 0.0),
                'macro_f1': comprehensive.get('macro_f1', 0.0),
                'class_metrics': comprehensive.get('class_metrics', comprehensive_val_metrics.get('class_metrics', {})),
                'window_confusion_matrix': comprehensive.get('window_confusion_matrix', []),
                'contextual_confusion_matrix': comprehensive.get('contextual_confusion_matrix', [])
            }
            
            with open(save_path / 'best_val_metrics.json', 'w') as f:
                json.dump(best_val_metrics_json, f, indent=2)
            
            print(f"\n  *** New best model saved! Val F1: {best_val_f1:.4f}, Val Loss: {best_val_loss:.4f} ***")
            print(f"      Val Acc: {val_metrics['contextual_accuracy']:.4f}")
            print(f"      ROC-AUC: {comprehensive.get('contextual_roc_auc', 0.0):.4f}")
            print(f"      Macro F1: {comprehensive.get('macro_f1', 0.0):.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_metrics['loss'],
                'metrics': train_metrics
            }, save_path / f'checkpoint_epoch_{epoch + 1}.pth')
    
    # Save final model
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_metrics['loss'],
        'metrics': train_metrics
    }, save_path / 'final_model.pth')
    
    # Save training history
    with open(save_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Find best train epoch (highest contextual accuracy)
    best_train_epoch = int(np.argmax(history['train_contextual_accuracy']))
    print(f"\nComputing comprehensive metrics for best train epoch ({best_train_epoch + 1})...")
    
    # We need to do a pass through training data to get comprehensive metrics
    # Use a separate dataloader without augmentation for fair evaluation
    eval_train_dataset = SleepStageDataset(
        folder_raw=raw_data_folder,
        folder_features=features_folder if use_gcv else None,
        window_size=window_size,
        subject_filter=train_subjects,
        use_augmentation=False  # No augmentation for evaluation
    )
    eval_train_dataloader = DataLoader(
        eval_train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True
    )
    
    comprehensive_train_metrics = validate_epoch(
        model, eval_train_dataloader, criterion, device, 
        compute_comprehensive=True, disable_tqdm=disable_tqdm
    )
    
    comprehensive_train = comprehensive_train_metrics.get('comprehensive', {})
    best_train_metrics_json = {
        'epoch': best_train_epoch + 1,
        'loss': history['train_loss'][best_train_epoch],
        'window_loss': history['train_window_loss'][best_train_epoch],
        'contextual_loss': history['train_contextual_loss'][best_train_epoch],
        'window_accuracy': comprehensive_train_metrics['window_accuracy'],
        'contextual_accuracy': comprehensive_train_metrics['contextual_accuracy'],
        'window_roc_auc': comprehensive_train.get('window_roc_auc', 0.0),
        'contextual_roc_auc': comprehensive_train.get('contextual_roc_auc', 0.0),
        'macro_precision': comprehensive_train.get('macro_precision', 0.0),
        'macro_recall': comprehensive_train.get('macro_recall', 0.0),
        'macro_f1': comprehensive_train.get('macro_f1', 0.0),
        'class_metrics': comprehensive_train.get('class_metrics', comprehensive_train_metrics.get('class_metrics', {})),
        'window_confusion_matrix': comprehensive_train.get('window_confusion_matrix', []),
        'contextual_confusion_matrix': comprehensive_train.get('contextual_confusion_matrix', [])
    }
    
    with open(save_path / 'best_train_metrics.json', 'w') as f:
        json.dump(best_train_metrics_json, f, indent=2)
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best train epoch: {best_train_epoch + 1}")
    print(f"Best validation epoch: {best_val_epoch + 1}")
    print(f"Models saved to: {save_path}")
    
    return model, history
