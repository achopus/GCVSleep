"""
ResNeXt Backbone Architecture

This module implements the 1D ResNeXt-29 backbone for feature extraction
from raw physiological signals.

Author:  Vojtech Brejtr
Date: February 2026
"""

import torch
import torch.nn as nn


class ResNeXtBlock1D(nn.Module):
    """
    1D ResNeXt block with cardinality (grouped convolutions).
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (should be divisible by cardinality)
        cardinality: Number of groups for grouped convolution
        base_width: Base width for each group
        stride: Stride for the convolution
        downsample: Optional downsampling layer for residual connection
    """
    def __init__(self, in_channels, out_channels, cardinality=32, base_width=4, stride=1, downsample=None):
        super(ResNeXtBlock1D, self).__init__()
        
        # Calculate width per group
        width = int(out_channels * (base_width / 64.0)) * cardinality
        
        # 1x1 conv to reduce dimensionality
        self.conv1 = nn.Conv1d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=width)
        
        # 3x3 grouped conv (core ResNeXt operation)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, stride=stride, 
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.GroupNorm(num_groups=8, num_channels=width)
        
        # 1x1 conv to restore dimensionality
        self.conv3 = nn.Conv1d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        
        self.gelu = nn.GELU()
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gelu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.gelu(out)
        
        return out


class ResNeXt1D(nn.Module):
    """
    1D ResNeXt-29 architecture for EEG+EMG signals with progressive GCV integration.
    
    Architecture follows ResNeXt-29 (3, 3, 3) structure:
    - Initial conv layer
    - 3 residual blocks with 64 channels + cross-attention fusion
    - 3 residual blocks with 128 channels + cross-attention fusion
    - 3 residual blocks with 256 channels + cross-attention fusion
    - Global average pooling
    
    Args:
        in_channels: Number of input channels (4 for 3 EEG + 1 EMG)
        embed_dim: Dimension of output embedding
        cardinality: Number of groups for ResNeXt (default: 32)
        base_width: Base width for each group (default: 4)
        use_gcv: Whether to integrate global context vector (default: False)
        gcv_dim: Dimension of global context vector (required if use_gcv=True)
    """
    def __init__(self, in_channels=4, embed_dim=256, cardinality=32, base_width=4,
                 use_gcv=False, gcv_dim=None):
        super(ResNeXt1D, self).__init__()
        
        self.cardinality = cardinality
        self.base_width = base_width
        self.use_gcv = use_gcv
        
        # Initial convolution - kernel size suitable for 500 Hz EEG
        # 50ms window = 25 samples, use kernel=51 for ~100ms receptive field
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=51, stride=2, padding=25, bias=False)
        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.gelu = nn.GELU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNeXt blocks (3, 3, 3 for ResNeXt-29)
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=3, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=3, stride=2)
        
        # Cross-attention modules after each layer
        if use_gcv:
            from .attention import BidirectionalCrossAttention
            assert gcv_dim is not None, "gcv_dim must be specified"
            self.cross_attn1 = BidirectionalCrossAttention(64, gcv_dim, num_heads=8, dropout=0.1)
            self.cross_attn2 = BidirectionalCrossAttention(128, gcv_dim, num_heads=8, dropout=0.1)
            self.cross_attn3 = BidirectionalCrossAttention(256, gcv_dim, num_heads=8, dropout=0.1)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection to embedding dimension
        self.fc = nn.Linear(256, embed_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using SOTA methods"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Kaiming initialization for conv layers (optimal for GELU/ReLU)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                # Kaiming initialization for linear layers
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                # Standard normalization layer initialization
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Create a layer with multiple ResNeXt blocks"""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_channels)
            )
        
        layers = []
        # First block may have stride > 1 and/or different in/out channels
        layers.append(ResNeXtBlock1D(in_channels, out_channels, self.cardinality, 
                                     self.base_width, stride, downsample))
        
        # Remaining blocks have stride 1 and same in/out channels
        for _ in range(1, blocks):
            layers.append(ResNeXtBlock1D(out_channels, out_channels, 
                                        self.cardinality, self.base_width))
        
        return nn.Sequential(*layers)
    
    def forward(self, x, gcv=None, batch_size=None, num_windows=None):
        """
        Forward pass through ResNeXt backbone.
        
        Args:
            x: Input tensor of shape (batch_size * num_windows, in_channels, time_steps)
            gcv: Optional global context vector (batch_size, num_windows, gcv_dim)
            batch_size: Batch size (required if gcv is provided)
            num_windows: Number of windows per recording (required if gcv is provided)
        
        Returns:
            embeddings: Feature embeddings (batch_size * num_windows, embed_dim)
            updated_gcv: Updated global context vector or None
        """
        if self.use_gcv and gcv is not None:
            assert batch_size is not None and num_windows is not None, \
                "batch_size and num_windows required for GCV integration"
        
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.maxpool(x)
        
        # Layer 1 with optional cross-attention fusion
        x = self.layer1(x)
        if self.use_gcv and gcv is not None:
            x, gcv = self.cross_attn1(x, gcv, batch_size, num_windows)
        
        # Layer 2 with optional cross-attention fusion
        x = self.layer2(x)
        if self.use_gcv and gcv is not None:
            x, gcv = self.cross_attn2(x, gcv, batch_size, num_windows)
        
        # Layer 3 with optional cross-attention fusion
        x = self.layer3(x)
        if self.use_gcv and gcv is not None:
            x, gcv = self.cross_attn3(x, gcv, batch_size, num_windows)
        
        # Global pooling and projection
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        # Maintain consistent return format
        if self.use_gcv and gcv is not None:
            return x, gcv
        return x, None
