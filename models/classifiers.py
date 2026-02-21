"""
Classification Heads

This module implements the two classification branches:
- Window-level classifier (Branch 1): No temporal context
- Contextual classifier (Branch 2): With Transformer self-attention

Author:  Vojtech Brejtr
Date: February 2026
"""

import torch
import torch.nn as nn


class WindowClassifier(nn.Module):
    """
    Classify individual windows independently without temporal context (Branch 1).
    
    Args:
        embed_dim: Input embedding dimension
        num_classes: Number of output classes (default: 4 sleep stages)
        hidden_dim: Hidden layer dimension
        embedding_gcv_dim: Optional GCV dimension to concatenate
        return_embeddings: Whether to return intermediate embeddings
    """
    
    def __init__(
        self, 
        embed_dim=256, 
        num_classes=4, 
        hidden_dim=128, 
        embedding_gcv_dim=None, 
        return_embeddings=False
    ):
        super(WindowClassifier, self).__init__()
        
        input_dim = embed_dim + (embedding_gcv_dim if embedding_gcv_dim else 0)
        self.return_embeddings = return_embeddings
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with truncated normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, embeddings, gcv_embedding=None):
        """
        Forward pass for window-level classification.
        
        Args:
            embeddings: Window embeddings (batch_size, num_windows, embed_dim)
            gcv_embedding: Optional GCV to concatenate
        
        Returns:
            logits: Class logits (batch_size, num_windows, num_classes)
            embeddings: If return_embeddings=True, intermediate embeddings
        """
        batch_size, num_windows, _ = embeddings.shape
        
        # Flatten and optionally concatenate GCV
        x = embeddings.view(batch_size * num_windows, -1)
        if gcv_embedding is not None:
            x_gcv = gcv_embedding.view(batch_size * num_windows, -1)
            x = torch.cat([x, x_gcv], dim=-1)
        
        # Save for optional return
        x_emb = x.clone() if self.return_embeddings else None
        
        # Classification
        x = self.mlp(x)
        logits = x.view(batch_size, num_windows, -1)
        
        if self.return_embeddings:
            return logits, x_emb.view(batch_size, num_windows, -1)
        return logits


class ContextualClassifier(nn.Module):
    """
    Classify windows with global temporal context using Transformer (Branch 2).
    
    This branch uses self-attention to model temporal dependencies between
    windows, enabling context-aware predictions.
    
    Args:
        embed_dim: Input embedding dimension
        num_classes: Number of output classes (default: 4 sleep stages)
        hidden_dim: Feed-forward hidden dimension
        num_layers: Number of Transformer encoder layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        embedding_gcv_dim: Optional GCV dimension
        max_seq_len: Maximum sequence length for positional encoding
        return_embeddings: Whether to return intermediate embeddings
    """
    
    def __init__(
        self, 
        embed_dim=256, 
        num_classes=4, 
        hidden_dim=256, 
        num_layers=2, 
        num_heads=8, 
        dropout=0.3, 
        embedding_gcv_dim=None, 
        max_seq_len=512, 
        return_embeddings=False
    ):
        super(ContextualClassifier, self).__init__()
        
        self.original_embed_dim = embed_dim
        self.use_gcv = embedding_gcv_dim is not None
        self.return_embeddings = return_embeddings
        
        # Project concatenated features to embed_dim (if using GCV)
        if embedding_gcv_dim is not None:
            input_dim = embed_dim + embedding_gcv_dim
            self.input_proj = nn.Linear(input_dim, embed_dim)
        else:
            self.input_proj = None
        
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # Transformer encoder with self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LayerNorm for better stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        
        # Output layers
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with truncated normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, embeddings, gcv_embedding=None):
        """
        Forward pass with temporal self-attention.
        
        Args:
            embeddings: Window embeddings (batch_size, num_windows, embed_dim)
            gcv_embedding: Optional GCV to concatenate
        
        Returns:
            logits: Context-aware class logits (batch_size, num_windows, num_classes)
            embeddings: If return_embeddings=True, context-aware embeddings
        """
        batch_size, num_windows, _ = embeddings.shape
        x = embeddings
        
        # Concatenate and project GCV if provided
        if gcv_embedding is not None:
            x = torch.cat([x, gcv_embedding], dim=-1)
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        # Add temporal positional embeddings
        x = x + self.pos_embedding[:, :num_windows, :]
        
        # Transformer self-attention for temporal context
        x = self.transformer(x)
        
        # Final normalization
        emb = self.norm(x)
        
        # Per-window classification with global context
        logits = self.classifier(emb)
        
        if self.return_embeddings:
            return logits, emb
        return logits
