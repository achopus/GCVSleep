# GCVSleep — Dual-branch sleep stage classification

Research code for automatic sleep stage classification from raw physiological signals (EEG/EMG). The core model is a dual-branch network:

- Branch 1: CNN based feature extraction
- Branch 2: GCV based memory bank

The training loop supports k-fold cross-validation, checkpointing, and optional supervised contrastive learning.

## Repository layout

```
.
├── config.py
├── train.py
├── inference.py
├── requirements.txt
├── data_management/
│   └── dataset.py
├── models/
│   ├── __init__.py
│   ├── attention.py
│   ├── backbone.py
│   ├── classifiers.py
│   ├── dual_branch.py
│   └── embeddings.py
└── training/
    ├── __init__.py
    ├── losses.py
    ├── metrics.py
    ├── trainer.py
    └── utils.py
```

## Installation

Python 3.9+ is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Note: Installing PyTorch with CUDA varies by system. If `pip install -r requirements.txt` installs a CPU-only build and you want GPU, install PyTorch per the official instructions for your CUDA version, then install the remaining requirements.

## Data expectations

The dataset loader is implemented in `data_management/dataset.py` (`SleepStageDataset`). Each dataset item corresponds to a single `.pth` file in `folder_raw`.

### Supported raw file formats

The loader scans `folder_raw` for `.pth` files ending in `_raw.pth` or containing `_annotated_sample`.

1) **Tensor format**

- A `.pth` file containing a `torch.Tensor` shaped `(in_channels, seq_len)`.

2) **Dict format (recommended when labels are embedded)**

- A `.pth` file containing a dict with:
  - `data`: `torch.Tensor` shaped `(in_channels, seq_len)`
  - optionally `annotations`: `torch.Tensor` shaped `(num_classes, seq_len)` (one-hot)

Windows are created by truncating `seq_len` to a multiple of `window_size` and reshaping into `(num_windows, in_channels, window_size)`.

### Labels

Labels can be provided in one of these ways:

- If the raw `.pth` dict contains `annotations`, labels are derived per-window (majority vote).
- Otherwise, for a file named `..._raw.pth`, the loader will try to load `..._labels.pth` from the same directory.

### Subject IDs (for CV splits)

Subject filtering is based on the *first underscore-delimited token* in the filename (e.g., `11_..._raw.pth` or `subject11_..._raw.pth`). Cross-validation train/val splits should therefore use integer subject IDs matching that token.

## Cross-validation folds format

Training expects a JSON file (default: `cv_folds.json`) with a list of folds:

```json
[
  {"fold": 1, "train_ids": [1, 2, 3], "val_ids": [4, 5]},
  {"fold": 2, "train_ids": [4, 5], "val_ids": [1, 2, 3]}
]
```

## Training

The CLI entrypoint is `train.py`. It wraps `training/trainer.py::train`.

Important: `--fold` is **0-based** (so `--fold 0` corresponds to the first fold in the JSON file).

### Train a single fold

```bash
python train.py --config full_model --fold 0 \
  --raw_data_folder path\to\dataset \
  --cv_file path\to\cv_folds.json \
  --save_dir path\to\checkpoints
```

### Train all folds

```bash
python train.py --config full_model --all_folds
```

### Available CLI config presets

`train.py` currently exposes these presets:

- `base`
- `full_model`
- `no_contrastive`
- `no_gcv`

Other configs in `config.py` (e.g., `SMALL_MODEL_CONFIG`, `DEBUG_CONFIG`, …) are intended for programmatic use.

### Programmatic training

```python
from training.trainer import train
import config

cfg = config.FULL_MODEL_CONFIG.copy()
cfg.update({
    "raw_data_folder": "path/to/dataset",
    "cv_file": "path/to/cv_folds.json",
    "save_dir": "checkpoints/run1",
    "device": "cuda",
})

model, history = train(**cfg)
```

## Training outputs

For each fold, `train.py` writes to `<save_dir>/foldN/`:

- `best_model.pth` (checkpoint dict with `model_state_dict` + `model_config`)
- `best_val_metrics.json`
- `best_train_metrics.json`
- `training_history.json`
- `checkpoint_epoch_10.pth`, `checkpoint_epoch_20.pth`, ...
- `final_model.pth`

## Inference / evaluation

There is no separate inference CLI; use `inference.py` from Python.

### Load a trained checkpoint and evaluate

```python
import torch
from torch.utils.data import DataLoader

from models import DualBranchModel
from data_management.dataset import SleepStageDataset
from inference import evaluate_model

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load("checkpoints/run1/fold1/best_model.pth", map_location=device)
model_cfg = ckpt.get("model_config", {})

model = DualBranchModel(**model_cfg).to(device)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()

ds = SleepStageDataset(folder_raw="path/to/dataset", folder_features=None, use_augmentation=False)
dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

metrics = evaluate_model(model, dl, device=device, use_contextual=True)
print(metrics["accuracy"], metrics["macro_f1"])
```

## Notes / current limitations

- The dataset supports loading labels from embedded `annotations` or from a separate `*_labels.pth` file.
- CSV feature pairing is present in the API, but feature-file matching for the `_annotated_sample` format is not implemented in `_find_file_pairs()`.
- `config.py` uses relative default paths like `../dataset`. If your data lives elsewhere, override via CLI flags (recommended) or by editing the config dict.

## License

No license file is included in this repository at the moment.
