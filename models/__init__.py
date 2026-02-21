"""
Neural Network Models Package

This package contains all neural network components for sleep stage classification.

Modules:
    - backbone: ResNeXt-29 backbone architecture
    - attention: Cross-attention mechanisms
    - embeddings: Window embedding modules
    - classifiers: Classification heads
    - dual_branch: Main dual-branch model

Author:  Vojtech Brejtr
Date: February 2026
"""

from .backbone import ResNeXtBlock1D, ResNeXt1D
from .attention import BidirectionalCrossAttention
from .embeddings import WindowEmbedding
from .classifiers import WindowClassifier, ContextualClassifier
from .dual_branch import DualBranchModel

__all__ = [
    'ResNeXtBlock1D',
    'ResNeXt1D',
    'BidirectionalCrossAttention',
    'WindowEmbedding',
    'WindowClassifier',
    'ContextualClassifier',
    'DualBranchModel',
]
