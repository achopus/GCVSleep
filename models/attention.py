"""
Attention Mechanisms

This module implements cross-attention mechanisms for fusing CNN features
with manually extracted features.

Author:  Vojtech Brejtr
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention module for fusing CNN and GCV features.
    
    This module enables bidirectional information flow:
    - CNN features attend to GCV
    - GCV attends to CNN features
    - Self-attention captures inter-window temporal relationships
    
    Args:
        cnn_dim: Dimensionality of CNN features
        gcv_dim: Dimensionality of global context vector
        num_heads: Number of attention heads
        dropout: Dropout probability
        max_seq_len: Maximum sequence length for positional embeddings
    """
    
    def __init__(self, cnn_dim, gcv_dim, num_heads=8, dropout=0.1, max_seq_len=512):
        super(BidirectionalCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.cnn_dim = cnn_dim
        self.gcv_dim = gcv_dim
        self.head_dim = cnn_dim // num_heads
        assert cnn_dim % num_heads == 0, "cnn_dim must be divisible by num_heads"
        
        # Learnable positional embeddings for temporal awareness
        self.cnn_pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, cnn_dim))
        self.gcv_pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, gcv_dim))
        nn.init.trunc_normal_(self.cnn_pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.gcv_pos_embedding, std=0.02)
        
        # Self-attention layers for inter-window temporal modeling
        self.cnn_self_attn = nn.MultiheadAttention(
            embed_dim=cnn_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        # Adapt num_heads for GCV to ensure divisibility
        gcv_num_heads = min(num_heads, gcv_dim)
        while gcv_dim % gcv_num_heads != 0:
            gcv_num_heads -= 1
        self.gcv_self_attn = nn.MultiheadAttention(
            embed_dim=gcv_dim, num_heads=gcv_num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer norms for self-attention
        self.cnn_self_norm = nn.LayerNorm(cnn_dim)
        self.gcv_self_norm = nn.LayerNorm(gcv_dim)
        
        # Cross-attention: CNN -> GCV (CNN attends to GCV)
        self.cnn_to_gcv_q = nn.Linear(cnn_dim, cnn_dim)
        self.cnn_to_gcv_k = nn.Linear(gcv_dim, cnn_dim)
        self.cnn_to_gcv_v = nn.Linear(gcv_dim, cnn_dim)
        self.cnn_to_gcv_out = nn.Linear(cnn_dim, cnn_dim)
        
        # Cross-attention: GCV -> CNN (GCV attends to CNN)
        self.gcv_to_cnn_q = nn.Linear(gcv_dim, cnn_dim)
        self.gcv_to_cnn_k = nn.Linear(cnn_dim, cnn_dim)
        self.gcv_to_cnn_v = nn.Linear(cnn_dim, cnn_dim)
        self.gcv_to_cnn_out = nn.Linear(cnn_dim, gcv_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Layer norms
        self.cnn_norm1 = nn.LayerNorm(cnn_dim)
        self.cnn_norm2 = nn.LayerNorm(cnn_dim)
        self.gcv_norm1 = nn.LayerNorm(gcv_dim)
        self.gcv_norm2 = nn.LayerNorm(gcv_dim)
        
        # Feed-forward networks
        self.cnn_ffn = nn.Sequential(
            nn.Linear(cnn_dim, cnn_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cnn_dim * 4, cnn_dim),
            nn.Dropout(dropout)
        )
        
        self.gcv_ffn = nn.Sequential(
            nn.Linear(gcv_dim, gcv_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gcv_dim * 4, gcv_dim),
            nn.Dropout(dropout)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using SOTA methods"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _cross_attention(self, queries, keys, values, out_proj):
        """
        Perform scaled dot-product cross-attention.
        
        Args:
            queries: Query tensor (batch_size, num_windows, dim_q)
            keys: Key tensor (batch_size, num_windows, dim_k)
            values: Value tensor (batch_size, num_windows, dim_v)
            out_proj: Output projection layer
        
        Returns:
            Attention output (batch_size, num_windows, dim_out)
        """
        batch_size, num_windows, _ = queries.shape
        
        # Reshape for multi-head attention: (batch_size, num_heads, num_windows, head_dim)
        Q = queries.view(batch_size, num_windows, self.num_heads, self.head_dim).transpose(1, 2)
        K = keys.view(batch_size, num_windows, self.num_heads, self.head_dim).transpose(1, 2)
        V = values.view(batch_size, num_windows, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_windows, -1)
        attn_output = out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        return attn_output
    
    def forward(self, cnn_features, gcv, batch_size, num_windows):
        """
        Forward pass implementing bidirectional cross-attention with self-attention.
        
        Args:
            cnn_features: CNN features (batch_size * num_windows, cnn_dim, temporal_dim)
            gcv: Global context vector (batch_size, num_windows, gcv_dim)
            batch_size: Number of recordings in batch
            num_windows: Number of windows per recording
        
        Returns:
            updated_cnn: Enhanced CNN features (batch_size * num_windows, cnn_dim, temporal_dim)
            updated_gcv: Enhanced global context vector (batch_size, num_windows, gcv_dim)
        """
        batch_windows, cnn_dim, temporal_dim = cnn_features.shape
        gcv_dim = gcv.shape[2]
        
        # Verify dimension consistency
        assert batch_windows == batch_size * num_windows, \
            f"Shape mismatch: {batch_windows} != {batch_size} * {num_windows}"
        
        # Convert CNN features to sequence format via global pooling
        cnn_pooled = F.adaptive_avg_pool1d(cnn_features, 1).squeeze(-1)
        cnn_seq = cnn_pooled.view(batch_size, num_windows, cnn_dim)
        
        # Add temporal positional embeddings
        cnn_seq = cnn_seq + self.cnn_pos_embedding[:, :num_windows, :]
        gcv_pos = gcv + self.gcv_pos_embedding[:, :num_windows, :]
        
        # Self-attention: CNN windows attend to each other
        cnn_self_attn_out, _ = self.cnn_self_attn(cnn_seq, cnn_seq, cnn_seq)
        cnn_seq = self.cnn_self_norm(cnn_seq + cnn_self_attn_out)
        
        # Self-attention: GCV windows attend to each other
        gcv_self_attn_out, _ = self.gcv_self_attn(gcv_pos, gcv_pos, gcv_pos)
        gcv_pos = self.gcv_self_norm(gcv_pos + gcv_self_attn_out)
        
        # Cross-attention: CNN attends to GCV
        Q_cnn = self.cnn_to_gcv_q(cnn_seq)
        K_gcv = self.cnn_to_gcv_k(gcv_pos)
        V_gcv = self.cnn_to_gcv_v(gcv_pos)
        cnn_attended = self._cross_attention(Q_cnn, K_gcv, V_gcv, self.cnn_to_gcv_out)
        
        # Update CNN with cross-attention and feed-forward
        cnn_updated = self.cnn_norm1(cnn_seq + cnn_attended)
        cnn_ffn_out = self.cnn_ffn(cnn_updated)
        cnn_updated = self.cnn_norm2(cnn_updated + cnn_ffn_out)
        
        # Cross-attention: GCV attends to CNN
        Q_gcv = self.gcv_to_cnn_q(gcv_pos)
        K_cnn = self.gcv_to_cnn_k(cnn_seq)
        V_cnn = self.gcv_to_cnn_v(cnn_seq)
        gcv_attended = self._cross_attention(Q_gcv, K_cnn, V_cnn, self.gcv_to_cnn_out)
        
        # Update GCV with cross-attention and feed-forward
        gcv_updated = self.gcv_norm1(gcv + gcv_attended)
        gcv_ffn_out = self.gcv_ffn(gcv_updated)
        gcv_updated = self.gcv_norm2(gcv_updated + gcv_ffn_out)
        
        # Broadcast updated CNN features back to convolutional format
        cnn_updated_conv = cnn_updated.view(batch_windows, cnn_dim, 1).expand(-1, -1, temporal_dim)
        cnn_features_out = cnn_features + cnn_updated_conv  # Residual connection
        
        return cnn_features_out, gcv_updated
