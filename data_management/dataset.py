"""
Sleep Stage Classification Dataset

This module provides a PyTorch Dataset class for loading and preprocessing
sleep stage data with support for data augmentation.

Author:  Vojtech Brejtr
Date: February 2026
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torch.types import Tensor
from typing import List, Tuple, Optional


class SleepStageDataset(Dataset):
    """
    PyTorch Dataset for sleep stage classification.
    
    This dataset loads raw physiological signals (EEG/EMG) and corresponding
    manually extracted features, with support for data augmentation and
    flexible filtering options.
    
    Args:
        folder_raw: Path to folder containing raw signal data (.pth files)
        folder_features: Path to folder containing manual features (.csv files).
                        If None, uses default 'features_cleaned' directory
        window_size: Number of samples per window (default: 2000 = 4s at 500Hz)
        subject_filter: List of subject IDs to include. If None, includes all
        use_augmentation: Whether to apply data augmentation
        return_all_augmented: If True, returns both original and augmented data
        placebo_only_train: If True, only includes placebo/saline recordings
        n_minutes_in_training: Maximum number of recordings to use (for efficiency)
    """
    
    def __init__(
        self,
        folder_raw: str,
        folder_features: Optional[str] = None,
        window_size: int = 2000,
        subject_filter: Optional[List[str]] = None,
        use_augmentation: bool = False,
        return_all_augmented: bool = False,
        placebo_only_train: bool = False,
        n_minutes_in_training: Optional[int] = None
    ) -> None:
        super().__init__()
        self.folder_raw = Path(folder_raw) if folder_raw else None
        self.folder_features = Path(folder_features) if folder_features else None
        self.window_size = window_size
        self.subject_filter = subject_filter
        self.use_augmentation = use_augmentation
        self.return_all_augmented = return_all_augmented
        self.placebo_only_train = placebo_only_train
        self.n_minutes_in_training = n_minutes_in_training
        
        # Augmentation parameters optimized for physiological signals
        self.noise_std = 0.02  # Gaussian noise standard deviation
        self.scale_range = (0.95, 1.05)  # Amplitude scaling range
        self.time_shift_max = 20  # Maximum time shift in samples
        self.channel_dropout_prob = 0.05  # Probability of dropping a channel
        
        # Find and filter file pairs
        self.file_pairs = self._find_file_pairs()
        if placebo_only_train:
            self.filter_placebo_only()
        if n_minutes_in_training is not None:
            self.filter_by_training_time()
    
    def filter_placebo_only(self) -> None:
        """
        Filter dataset to only include placebo/saline recordings.
        
        Useful for training models on placebo-only data for domain adaptation
        or baseline experiments.
        """
        filtered_file_pairs = []
        for raw_file, feature_file in self.file_pairs:
            filename = os.path.basename(raw_file)
            if 'saline' in filename.lower():
                filtered_file_pairs.append((raw_file, feature_file))
        self.file_pairs = filtered_file_pairs
        
    def filter_by_training_time(self) -> None:
        """
        Limit the number of recordings to control training time.
        
        Randomly selects a subset of recordings. Each recording typically
        contains 15 windows (1 minute at 4-second windows).
        """
        assert (
            self.n_minutes_in_training < len(self.file_pairs) 
            and self.n_minutes_in_training > 0
        ), "n_minutes_in_training must be between 1 and total recordings"
        
        shuffled_pairs = self.file_pairs.copy()
        np.random.shuffle(shuffled_pairs)
        self.file_pairs = shuffled_pairs[:self.n_minutes_in_training]
        
    
    def augment_data(self, data: Tensor) -> Tensor:
        """
        Apply data augmentation to physiological signals.
        
        Augmentation techniques include:
        - Gaussian noise addition
        - Amplitude scaling
        - Time shifting
        - Channel dropout
        - Sign flipping (for EEG polarity invariance)
        
        Args:
            data: Input tensor of shape (num_windows, in_channels, window_size)
        
        Returns:
            Augmented tensor of the same shape
        """
        augmented = data.clone()
        
        # Gaussian noise addition
        if torch.rand(1).item() < 0.2 or self.return_all_augmented:
            signal_std = augmented.std()
            noise = torch.randn_like(augmented) * (self.noise_std * signal_std)
            augmented = augmented + noise
        
        # Amplitude scaling
        if torch.rand(1).item() < (0.1 if not self.return_all_augmented else 0.5):
            scale = torch.FloatTensor(1).uniform_(*self.scale_range).item()
            augmented = augmented * scale
        
        # Time shifting
        if torch.rand(1).item() < (0.1 if not self.return_all_augmented else 0.5):
            shift = torch.randint(-self.time_shift_max, self.time_shift_max + 1, (1,)).item()
            if shift != 0:
                augmented = torch.roll(augmented, shifts=shift, dims=2)
        
        # Channel dropout
        for c in range(augmented.shape[1]):
            if torch.rand(1).item() < self.channel_dropout_prob:
                augmented[:, c, :] = 0
        
        # Sign flipping (useful for EEG polarity invariance)
        if torch.rand(1).item() < (0.05 if not self.return_all_augmented else 0.5):
            flip_mask = torch.randint(0, 2, (1, augmented.shape[1], 1), dtype=torch.float32) * 2 - 1
            augmented = augmented * flip_mask
        
        return augmented
    
    def __len__(self) -> int:
        return len(self.file_pairs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get a single recording with its features and labels.
        
        Args:
            idx: Index of the recording
        
        Returns:
            If use_augmentation and return_all_augmented:
                (raw_data_original, raw_data_augmented, features, labels)
            Otherwise:
                (raw_data, features, labels)
            
            Shapes:
                raw_data: (num_windows, in_channels, window_size)
                features: (num_windows, feature_dim)
                labels: (num_windows,)
        """
        raw_file, feature_file = self.file_pairs[idx]
        
        # Load raw signal data
        loaded_data = torch.load(raw_file, weights_only=False)
        
        # Handle both new format (just 'data' key) and original format (both 'data' and 'annotations' keys)
        if isinstance(loaded_data, dict) and 'data' in loaded_data:
            raw_data = loaded_data['data']
            has_annotations = 'annotations' in loaded_data
        else:
            # Fallback for other formats
            raw_data = loaded_data
            has_annotations = False
        
        in_channels, seq_len = raw_data.shape
        
        # Calculate number of complete windows
        num_windows = seq_len // self.window_size
        seq_len_truncated = num_windows * self.window_size
        raw_data_truncated = raw_data[:, :seq_len_truncated]

        # Reshape into windows: (in_channels, num_windows, window_size) -> (num_windows, in_channels, window_size)
        raw_data_windowed = raw_data_truncated.view(in_channels, num_windows, self.window_size)
        raw_data_windowed = raw_data_windowed.permute(1, 0, 2)
        
        # Apply augmentation if enabled
        if self.use_augmentation:
            if self.return_all_augmented:
                raw_data_windowed_original = raw_data_windowed.clone()
            raw_data_windowed = self.augment_data(raw_data_windowed)
        
        # Load labels and features
        if feature_file is not None:
            # Load labels from CSV file (no header row in data files)
            feature_data = pd.read_csv(feature_file, header=None).values[:, 1:].astype(np.float32)
            labels = feature_data[:, -1]
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
            
            # Load or create feature tensor
            if self.folder_features is not None:
                features = feature_data[:, :-1]
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                feature_tensor = torch.tensor(features, dtype=torch.float32)
                # Note: Set to zeros for ablation study without manual features
                feature_tensor = torch.zeros_like(feature_tensor)
            else:
                # Create dummy feature tensor (will be replaced by GCV in model)
                feature_tensor = torch.zeros((num_windows, 1), dtype=torch.float32)
        elif has_annotations:
            # Load labels from annotations in the same file (original annotated_sample format)
            annotations = loaded_data['annotations']
            # annotations shape: (num_classes, seq_len) - one-hot encoded
            num_classes, _ = annotations.shape
            
            # Truncate and reshape into windows
            annotations_truncated = annotations[:, :seq_len_truncated]
            annotations_windowed = annotations_truncated.view(num_classes, num_windows, self.window_size)
            annotations_windowed = annotations_windowed.permute(1, 0, 2)  # (num_windows, num_classes, window_size)
            
            # For each window, get the label (majority vote within window)
            labels = torch.zeros(num_windows, dtype=torch.long)
            for w in range(num_windows):
                # Get the most frequent class in this window
                window_labels = annotations_windowed[w]  # (num_classes, window_size)
                # Convert one-hot to class indices
                class_indices = torch.argmax(window_labels, dim=0)  # (window_size,)
                # Get mode (most frequent class)
                mode_class = torch.mode(class_indices)[0].item() # Unnecessary in our dataset since we have clean windows, implemented for generalization to noisier datasets
                labels[w] = mode_class
            
            labels_tensor = labels.float()
            # Create dummy feature tensor
            feature_tensor = torch.zeros((num_windows, 1), dtype=torch.float32)
        else:
            # No features file and no annotations - load labels from separate file
            # Support both .pth and legacy .pt naming
            if raw_file.endswith('_raw.pth'):
                labels_file = raw_file.replace('_raw.pth', '_labels.pth')
            elif raw_file.endswith('_raw.pt'):
                labels_file = raw_file.replace('_raw.pt', '_labels.pt')
            else:
                # Fallback: replace extension only
                raw_path = Path(raw_file)
                labels_file = str(raw_path.with_name(raw_path.stem + '_labels' + raw_path.suffix))
            labels_tensor = torch.load(labels_file)
            # Create dummy feature tensor
            feature_tensor = torch.zeros((num_windows, 1), dtype=torch.float32)
        
        # Return appropriate format based on augmentation settings
        if self.use_augmentation and self.return_all_augmented:
            return raw_data_windowed_original, raw_data_windowed, feature_tensor, labels_tensor
        else:
            return raw_data_windowed, feature_tensor, labels_tensor
    
    def get_recording_id_for_idx(self, idx: int) -> str:
        """
        Get the recording ID for a given index.
        
        Args:
            idx: Dataset index
        
        Returns:
            Recording ID in format 'subject_treatment_mode'
        """
        raw_file, _ = self.file_pairs[idx]
        filename = os.path.basename(raw_file)
        stem = Path(filename).stem  # handles both .pt and .pth
        recording_id = "_".join(stem.split('_')[:-1])
        return recording_id
    
    def _find_file_pairs(self) -> List[Tuple[str, str]]:
        """
        Find matching pairs of raw data and feature files.
        
        Note: Each file represents a chunk (e.g., _annotated_sample0, _annotated_sample1),
        not a complete recording. We treat each chunk as a separate dataset item.
        
        Returns:
            List of (raw_file_path, feature_file_path) tuples
        """
        raw_files = []
        
        # Support both new format (_raw.pth) and original format (_annotated_sample*.pth)
        # (and keep legacy .pt support)
        for f in os.listdir(self.folder_raw):
            if not f.endswith(('.pth', '.pt')):
                continue

            if f.endswith('_raw.pth') or f.endswith('_raw.pt'):
                raw_files.append(f)
            elif '_annotated_sample' in f:
                raw_files.append(f)
        
        # Convert to full paths
        raw_file_paths = [os.path.join(self.folder_raw, f) for f in raw_files]
        
        # Filter by subject IDs if specified
        if self.subject_filter is not None:
            filtered_paths = []
            for path in raw_file_paths:
                filename = os.path.basename(path)
                subject_str = filename.split('_')[0]  # Extract subject ID (first element)
                # Handle both numeric IDs (e.g., "11") and "subjectX" format
                try:
                    subject_id = int(subject_str) if subject_str.isdigit() else int(subject_str.replace('subject', ''))
                except ValueError:
                    continue  # Skip if subject ID can't be parsed
                if subject_id in self.subject_filter:
                    filtered_paths.append(path)
            raw_file_paths = filtered_paths
        
        # If no features folder specified, return raw files only
        if self.folder_features is None:
            return [(path, None) for path in raw_file_paths]
        
        # Use default features path if not specified but needed
        features_path = self.folder_features or '../features_cleaned'
        
        # TODO: Feature matching for annotated_sample files is not yet implemented in this version
        # The current approach would need to handle the sample numbering in the filename
        # For now, we just skip feature matching and return raw files only
        # This is fine for the annotated_sample files which contain both data and annotations
        print("Warning: Feature file matching not implemented for annotated_sample format. "
              "Returning raw files only.")
        return [(path, None) for path in raw_file_paths]
        