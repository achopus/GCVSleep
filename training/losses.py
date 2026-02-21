"""
Loss Functions for Sleep Stage Classification

This module implements custom loss functions:
- DualBranchLoss: Weighted combination of window and contextual branch losses
- SupervisedContrastiveLoss: Supervised contrastive learning loss (SupCon)

Author: Vojtech Brejtr
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualBranchLoss(nn.Module):
    """
    Combined loss for dual-branch model.
    
    Combines losses from both branches:
    - Branch 1 (window): Local window classification loss
    - Branch 2 (contextual): Context-aware classification loss
    
    Args:
        window_weight: Weight for window branch loss (default: 0.01)
        contextual_weight: Weight for contextual branch loss (default: 0.99)
    """
    
    def __init__(self, window_weight=0.01, contextual_weight=0.99):
        super(DualBranchLoss, self).__init__()
        self.window_weight = window_weight
        self.contextual_weight = contextual_weight
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.025)
    
    def forward(self, window_logits, contextual_logits, target):
        """
        Compute combined loss from both branches.
        
        Args:
            window_logits: Window predictions (batch_size, num_windows, num_classes)
            contextual_logits: Contextual predictions (batch_size, num_windows, num_classes)
            target: Ground truth labels (batch_size, num_windows)
        
        Returns:
            total_loss: Weighted combination of both losses
            window_loss: Window branch loss
            contextual_loss: Contextual branch loss
        """
        batch_size, num_windows, num_classes = window_logits.shape
        
        # Reshape for cross entropy loss
        window_logits_flat = window_logits.view(batch_size * num_windows, num_classes)
        contextual_logits_flat = contextual_logits.view(batch_size * num_windows, num_classes)
        target_flat = target.view(batch_size * num_windows).long()
        
        # Compute individual losses
        window_loss = self.ce_loss(window_logits_flat, target_flat)
        contextual_loss = self.ce_loss(contextual_logits_flat, target_flat)
        
        # Weighted combination
        total_loss = (
            self.window_weight * window_loss + 
            self.contextual_weight * contextual_loss
        )
        
        return total_loss, window_loss, contextual_loss


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss (SupCon).
    
    Based on "Supervised Contrastive Learning" (Khosla et al., 2020)
    https://arxiv.org/abs/2004.11362
    
    For each anchor:
    - Positives: augmented version + same-class samples
    - Negatives: different-class samples
    
    Args:
        temperature: Temperature scaling parameter (default: 0.07)
        contrast_mode: Use 'all' anchors or 'one' (default: 'all')
    """
    
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
    
    def forward(self, embeddings_orig, embeddings_aug, labels):
        """
        Compute supervised contrastive loss.
        
        Args:
            embeddings_orig: Original embeddings (batch_size, num_windows, embed_dim)
            embeddings_aug: Augmented embeddings (batch_size, num_windows, embed_dim)
            labels: Class labels (batch_size, num_windows)
        
        Returns:
            loss: Scalar contrastive loss value
        """
        device = embeddings_orig.device
        batch_size, num_windows, embed_dim = embeddings_orig.shape
        
        # Flatten to treat each window independently
        embeddings_orig_flat = embeddings_orig.view(batch_size * num_windows, embed_dim)
        embeddings_aug_flat = embeddings_aug.view(batch_size * num_windows, embed_dim)
        labels_flat = labels.view(batch_size * num_windows).long()
        
        # Normalize embeddings to unit sphere
        embeddings_orig_flat = F.normalize(embeddings_orig_flat, p=2, dim=1)
        embeddings_aug_flat = F.normalize(embeddings_aug_flat, p=2, dim=1)
        
        # Concatenate [original; augmented] as [anchor; positive]
        embeddings_all = torch.cat([embeddings_orig_flat, embeddings_aug_flat], dim=0)
        labels_all = torch.cat([labels_flat, labels_flat], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings_all, embeddings_all.T) / self.temperature
        
        # Create mask for positive pairs (same class, excluding self)
        labels_equal = labels_all.unsqueeze(0) == labels_all.unsqueeze(1)
        mask_self = torch.eye(2 * batch_size * num_windows, dtype=torch.bool, device=device)
        labels_equal = labels_equal & ~mask_self
        
        # Numerical stability: subtract max
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log probabilities
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True) - exp_logits * mask_self.float() + 1e-12
        )
        
        # Check for valid positive pairs
        mask_pos_pairs = labels_equal.sum(dim=1) > 0
        if mask_pos_pairs.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute mean log-likelihood over positive pairs
        mean_log_prob_pos = (labels_equal * log_prob).sum(dim=1) / (labels_equal.sum(dim=1) + 1e-12)
        
        # Negative log-likelihood loss
        loss = -mean_log_prob_pos[mask_pos_pairs].mean()
        
        return loss
