"""
Window Embedding Module

This module extracts embeddings from signal windows using the ResNeXt backbone.

Author: Vojtech Brejtr
Date: February 2026
"""

import torch.nn as nn
from .backbone import ResNeXt1D


class WindowEmbedding(nn.Module):
    """
    Extract embeddings from 4-second signal windows using ResNeXt-29.
    
    Args:
        in_channels: Number of input channels (default: 4 for 3 EEG + 1 EMG)
        embed_dim: Output embedding dimension
        window_size: Samples per window (default: 2000 = 4s at 500Hz)
        cardinality: ResNeXt cardinality parameter
        base_width: ResNeXt base width parameter
        use_gcv: Whether to integrate global context vector
        gcv_dim: Dimension of global context vector (if used)
    """
    
    def __init__(
        self, 
        in_channels=4, 
        embed_dim=256, 
        window_size=2000, 
        cardinality=32, 
        base_width=4,
        use_gcv=False, 
        gcv_dim=None
    ):
        super(WindowEmbedding, self).__init__()
        self.window_size = window_size
        self.use_gcv = use_gcv
        
        # ResNeXt-29 backbone
        self.resnext = ResNeXt1D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            cardinality=cardinality,
            base_width=base_width,
            use_gcv=use_gcv,
            gcv_dim=gcv_dim
        )
    
    def forward(self, x, gcv=None):
        """
        Extract embeddings from windowed signals.
        
        Args:
            x: Input signals (batch_size, num_windows, in_channels, window_size)
            gcv: Optional global context vector (batch_size, num_windows, gcv_dim)
        
        Returns:
            embeddings: Feature embeddings (batch_size, num_windows, embed_dim)
            updated_gcv: Updated global context vector or None
        """
        batch_size, num_windows, in_channels, window_size = x.shape
        
        # Reshape: maintain batch structure for attention mechanisms
        x = x.view(batch_size * num_windows, in_channels, window_size)
        
        # Extract features using ResNeXt
        if self.use_gcv and gcv is not None:
            x, updated_gcv = self.resnext(x, gcv, batch_size, num_windows)
        else:
            x, updated_gcv = self.resnext(x)
        
        # Reshape back to separate batch and window dimensions
        embeddings = x.view(batch_size, num_windows, -1)
        
        return embeddings, updated_gcv
