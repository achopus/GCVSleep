"""
Dual-Branch Sleep Stage Classification Model

This module implements the complete dual-branch architecture that combines:
- Branch 1: Window-level predictions without temporal context
- Branch 2: Contextual predictions with Transformer self-attention
- Cross-attention: Bidirectional fusion of CNN and GCV features

Architecture Overview:
    Signal → ResNeXt Backbone → Embeddings
                                    ↓
    Branch 1: Window Classifier (no context)
    Branch 2: Contextual Classifier (with Transformer)
    
    Optional: Bidirectional cross-attention for GCV fusion

Author: Vojtech Brejtr
Date: February 2026
"""

import torch
import torch.nn as nn
from .embeddings import WindowEmbedding
from .classifiers import WindowClassifier, ContextualClassifier


class DualBranchModel(nn.Module):
    """
    Complete dual-branch model for sleep stage classification.
    
    This model processes 4-second signal windows and produces predictions through
    two parallel branches: one without temporal context (faster inference) and 
    one with global context via Transformer (better accuracy).
    
    Args:
        in_channels: Number of input signal channels (default: 4)
        num_classes: Number of sleep stages to classify (default: 4)
        embed_dim: Embedding dimension for ResNeXt backbone (default: 512)
        window_size: Samples per 4-second window (default: 2000 = 500Hz × 4s)
        cardinality: ResNeXt cardinality parameter (default: 32)
        base_width: ResNeXt base width parameter (default: 4)
        num_layers: Number of Transformer encoder layers (default: 2)
        num_heads: Number of attention heads in Transformer (default: 8)
        dropout: Dropout probability (default: 0.3)
        use_gcv: Whether to use global context vector (default: True)
        gcv_dim: Dimension of global context vector (default: 128)
        use_cross_attention: Whether to use bidirectional cross-attention (default: True)
        max_seq_len: Maximum sequence length for positional encoding (default: 512)
        return_embeddings: Whether to return intermediate embeddings (default: False)
    """
    
    def __init__(
        self,
        in_channels=4,
        num_classes=4,
        embed_dim=512,
        window_size=2000,
        cardinality=32,
        base_width=4,
        num_layers=2,
        num_heads=8,
        dropout=0.3,
        use_gcv=True,
        gcv_dim=128,
        use_cross_attention=True,
        max_seq_len=512,
        return_embeddings=False
    ):
        super(DualBranchModel, self).__init__()
        
        self.use_gcv = use_gcv
        self.use_cross_attention = use_cross_attention
        self.return_embeddings = return_embeddings
        
        # Feature extraction: 4-second windows → embeddings
        # Note: Cross-attention is applied inside ResNeXt1D after each layer
        self.embedding = WindowEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            window_size=window_size,
            cardinality=cardinality,
            base_width=base_width,
            use_gcv=use_gcv,
            gcv_dim=gcv_dim
        )
        
        self.use_gcv = use_gcv
        self.use_cross_attention = use_cross_attention and use_gcv and gcv_dim is not None
        embedding_gcv_dim = gcv_dim if self.use_cross_attention else None
        
        # Branch 1: Window-level classifier (no temporal context)
        self.window_classifier = WindowClassifier(
            embed_dim=embed_dim,
            num_classes=num_classes,
            hidden_dim=128,
            embedding_gcv_dim=embedding_gcv_dim,
            return_embeddings=return_embeddings
        )
        
        # Branch 2: Contextual classifier (with Transformer self-attention)
        self.contextual_classifier = ContextualClassifier(
            embed_dim=embed_dim,
            num_classes=num_classes,
            hidden_dim=512,
            num_layers=4,
            num_heads=num_heads,
            dropout=dropout,
            embedding_gcv_dim=embedding_gcv_dim,
            max_seq_len=max_seq_len,
            return_embeddings=return_embeddings
        )
    
    def forward(self, x, gcv=None):
        """
        Forward pass through dual-branch architecture.
        
        Args:
            x: Input signals (batch_size, num_windows, in_channels, window_size)
            gcv: Optional global context vector 
                           (batch_size, num_windows, gcv_dim)
        
        Returns:
            window_logits: Branch 1 predictions (batch_size, num_windows, num_classes)
            contextual_logits: Branch 2 predictions (batch_size, num_windows, num_classes)
            embeddings_dict: Optional intermediate embeddings if return_embeddings=True
        """
        batch_size, num_windows = x.shape[:2]
        
        # Step 1: Extract embeddings from signal windows
        # Note: Cross-attention is already applied inside ResNeXt if use_gcv=True
        embeddings, gcv_embedding = self.embedding(x, gcv)
        
        # Step 3: Dual-branch classification
        if self.return_embeddings:
            # Return intermediate embeddings for analysis
            window_logits, window_emb = self.window_classifier(embeddings, gcv_embedding)
            contextual_logits, contextual_emb = self.contextual_classifier(embeddings, gcv_embedding)
            
            embeddings_dict = {
                'backbone_embeddings': embeddings,
                'gcv_embeddings': gcv_embedding,
                'window_embeddings': window_emb,
                'contextual_embeddings': contextual_emb
            }
            return window_logits, contextual_logits, embeddings_dict
        else:
            # Standard forward pass
            window_logits = self.window_classifier(embeddings, gcv_embedding)
            contextual_logits = self.contextual_classifier(embeddings, gcv_embedding)
            return window_logits, contextual_logits
    
    def get_num_params(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        """Return detailed model configuration information."""
        info = {
            "model_name": "DualBranchModel",
            "total_parameters": self.get_num_params(),
            "backbone": "ResNeXt-29",
            "branch_1": "WindowClassifier (no context)",
            "branch_2": "ContextualClassifier (Transformer)",
            "use_gcv": self.use_gcv,
            "use_cross_attention": self.use_cross_attention
        }
        return info
