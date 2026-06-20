# Leveraging Machine Learning to Estimate C. Elegans Lifespan from Movement Data

## Project Overview

This project analyzes movement trajectories of *Caenorhabditis elegans* (*C. elegans*) to answer two complementary questions:

1. **Lifespan prediction** — Estimate how many recording segments remain in a worm's life from its movement time series alone.
2. **Treatment classification** — Distinguish worms treated with **Terbinafine** (Terbinafine+, lifespan-extending) from untreated controls (Terbinafine-) based on behavioral data, using both trajectory-image CNNs and classical tabular features.

Terbinafine extends *C. elegans* lifespan. By learning directly from preprocessed trajectory data — coordinates, speed, turning rate, and lifetime index — we investigate whether movement patterns carry enough signal to predict remaining lifespan and treatment group without heavy manual feature engineering.

---

## Setup

### Installation

1. Clone the repository.
2. Create and activate a Python virtual environment (Python 3.10+ recommended).
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure feature columns in a `.env` file at the project root (used by `LPBSDataset`):

```env
features_cols_pytorch = ["X", "Y", "ComputedSpeed_frames", "Lifetime"]
features_cols_rock = ["X", "Y", "ComputedSpeed_frames"]
features_cols_sklearn = ["Age_hours", "Mean_Speed", "Median_Speed", "Net_Displacement", "Tortuosity"]
```

### Data layout

Place raw tracking CSVs under `data/`. The folder names for treated and control groups are defined in `data/config.json`:

```json
{
    "control_folder": "TERBINAFINE- (control)",
    "treated_folder": "TERBINAFINE+"
}
```

Expected structure:

```
data/
├── config.json
├── lifespan_summary.csv          # Per-worm metadata (lifespan, treatment label, …)
├── TERBINAFINE+/                 # Treated worm tracking CSVs
└── TERBINAFINE- (control)/       # Control worm tracking CSVs
```

See [`data/DATA.md`](data/DATA.md) for column definitions and recording schedule (1 frame every 2 s, 900-frame sessions every 6 h).

### Usage

All commands below are run from the project root.

#### 0. Preprocessing (required first step)

Clean trajectories, add segment/lifetime columns, and write model-ready CSVs:

```bash
python scripts/preprocess.py data/ --output-dir preprocessed_data/
```

Useful flags:

| Flag | Description |
|------|-------------|
| `--death-crop` | Trim frames after the worm stops moving (end-of-life detection) |
| `--normalize` | Min-max normalize coordinates and speed columns per worm |
| `--distance-threshold` | Max per-frame displacement before a jump is removed (default: `16`) |
| `--file <name.csv>` | Process a single file only |
| `--generate-images` | Also build multichannel trajectory images for the CNN classifier |
| `--cnn-output-dir` | Output directory for CNN images (default: `cnn_dataset/`) |
| `--only-cnn` | Skip CSV preprocessing and only regenerate CNN images from existing CSVs |

**Classifier feature extraction** (segment-level tabular features for sklearn models):

```bash
python scripts/extract_features.py
```

Reads from `preprocessed_data/` and writes per-worm segment summaries to `preprocessed_data_for_classifier/`.

#### 1. Lifespan prediction — training

Train regression models with **Group K-Fold** cross-validation (all segments of one worm stay in the same fold):

```bash
python scripts/training_pipeline.py --pytorch_dir preprocessed_data/ --scaler standard
```

| Flag | Description |
|------|-------------|
| `--plot` | Plot average results across folds |
| `--augment_data` / `-a [N]` | Apply trajectory augmentations (default `5` if flag given without value) |
| `--prod` | Save the best checkpoint per model to `ckpts/` |
| `--scaler` / `-s` | `none`, `minmax`, or `standard` (writes `scaler_config.json` when not `none`) |
| `--output_json` / `-o` | Basename for the results JSON (default: `avg_results`) |

Edit the `models_config` dict inside the script to choose architectures (TCN, Gaussian/Weibull heads) and hyperparameters.

#### 2. Lifespan prediction — benchmarking

Evaluate saved checkpoints on a held-out set with survival-oriented metrics (MAE, tier MAE, CRPS, coverage, earlyness, …):

```bash
python scripts/benchmark_pipeline.py \
  --pytorch_dir preprocessed_data/ \
  --scaler_config_path preprocessed_data/scaler_config.json \
  --output_dir benchmark_results/
```

Checkpoints are expected under `ckpts/best_<model_name>_*.pth`.

#### 3. Lifespan prediction — interactive visualization

Compare model predictions on individual worm trajectories:

```bash
python scripts/visualization_pipeline.py \
  --pytorch_dir preprocessed_data/ \
  --scaler_config_path preprocessed_data/scaler_config.json
```

#### 4. Treatment classification — CNN pipeline

Train image-based classifiers (ResNet, DenseNet) on multichannel trajectory images:

```bash
python scripts/cnn_pipeline.py --data_dir cnn_dataset/
```

Modify the `models_config` dict in the script to change architectures, batch size, or learning rate.

#### 5. Plotting results

```bash
python scripts/plot_results.py --results_file avg_results.json
```

---

## Pipelines & Architecture

The project is organized around two modeling tracks that share the same preprocessing stage.

### Lifespan prediction (regression)

| Script | Role |
|--------|------|
| [`scripts/training_pipeline.py`](scripts/training_pipeline.py) | Train TCN / BiLSTM + CNN-attention regressors with Group K-Fold CV |
| [`scripts/benchmark_pipeline.py`](scripts/benchmark_pipeline.py) | Benchmark saved models with probabilistic and survival metrics |
| [`scripts/visualization_pipeline.py`](scripts/visualization_pipeline.py) | Interactive per-worm prediction plots |
| [`scripts/plot_regression_interpretation.py`](scripts/plot_regression_interpretation.py) | Attention / interpretability plots for regression models |

**Models** live in [`models/cnn_attention_models/`](models/cnn_attention_models/) and compose reusable blocks from [`models/building_blocs/`](models/building_blocs/) (CNN feature extractor, TCN, BiLSTM, gated attention, time embedding). Wrappers in [`models/cnn_attention_models/regression_wrappers.py`](models/cnn_attention_models/regression_wrappers.py) handle training, benchmarking, and visualization.

**Loss functions** include Huber, Gaussian NLL, and Weibull survival variants (standard, shifted, beta-penalized) for uncertainty-aware remaining-lifespan estimates.

**Data** is loaded through [`LPBSDataset`](utils/train_utils/dataset.py), which reads preprocessed CSVs into padded segment tensors and supports optional standard/min-max scaling.

### Treatment classification

| Script | Role |
|--------|------|
| [`scripts/cnn_pipeline.py`](scripts/cnn_pipeline.py) | Train CNN classifiers on multichannel trajectory images with Stratified Group K-Fold CV |
| [`scripts/extract_features.py`](scripts/extract_features.py) | Build segment-level tabular features (speed, tortuosity, displacement) for classical ML |

CNN classifiers use [`CElegansCNNDataset`](utils/train_utils/dataset.py) and legacy model factories in [`models/deprecated/`](models/deprecated/). Older tabular classifiers (Logistic Regression, Random Forest, ROCKET, Tail-MIL, XGBoost) are kept in `models/deprecated/` for reference.

### Data augmentation

`LPBSDataset.augment_data()` applies random rotations, translations, and scaling on trajectory tensors during regression training to improve generalization.

---

## Project Structure

```
.
├── data/                              # Raw tracking CSVs, lifespan summary, config
│   ├── config.json
│   ├── lifespan_summary.csv
│   └── DATA.md
├── preprocessed_data/                 # Cleaned trajectory CSVs (gitignored)
├── preprocessed_data_for_classifier/  # Segment-level tabular features (gitignored)
├── cnn_dataset/                       # Multichannel trajectory images (gitignored)
├── ckpts/                             # Saved model checkpoints (gitignored)
├── models/
│   ├── building_blocs/                # TCN, BiLSTM, CNN extractor, attention, HMM, …
│   ├── cnn_attention_models/          # Regression model + training/benchmark wrappers
│   ├── deprecated/                    # Legacy classification models (LR, RF, ROCKET, CNN, …)
│   ├── model_dummies.py
│   └── wrappers.py                    # Base training / benchmark / visualization wrappers
├── scripts/
│   ├── preprocess.py                  # Trajectory cleaning, segmentation, CNN image generation
│   ├── extract_features.py            # Tabular feature extraction for classifiers
│   ├── training_pipeline.py           # Lifespan regression training
│   ├── benchmark_pipeline.py          # Regression model benchmarking
│   ├── visualization_pipeline.py      # Interactive regression visualization
│   ├── cnn_pipeline.py                # CNN classification training
│   ├── plot_results.py                # Results plotting utility
│   └── plot_regression_interpretation.py
├── utils/
│   ├── train_utils/
│   │   └── dataset.py                 # LPBSDataset, CElegansCNNDataset
│   └── plot_utils/                    # Result presentation and plotting helpers
├── notebook/                          # Exploratory visualization notebooks
├── data_analysis/                     # Statistical analysis scripts and notebooks
├── .env                               # Feature column configuration
└── requirements.txt
```

---

## Preprocessing Methodology

Raw data consists of per-worm movement trajectories recorded over the worms' lifespans (see [`data/DATA.md`](data/DATA.md)). Because tracking can be noisy and sessions are separated by multi-hour pauses, a dedicated pipeline in [`scripts/preprocess.py`](scripts/preprocess.py) standardizes every file before modeling.

Steps are applied in the following order:

### 1. Timestamp & initialization cleaning

**Objective:** Remove rows with invalid or inconsistent timestamps at the start of a recording.

- The first 10 rows are scanned for missing timestamps or gaps larger than 1 000 seconds between consecutive frames.
- Invalid rows are dropped and `GlobalFrame` is re-indexed to start at 0.

### 2. Segmentation & lifetime indexing

**Objective:** Align trajectories with the recording schedule and provide a continuous life-stage index.

- **`Segment`**: `GlobalFrame // 900` — each segment corresponds to one 30-minute recording session (900 frames at 0.5 fps).
- **`Lifetime`**: `GlobalFrame + Segment × 9900` — a frame counter that accounts for the ~5 h 30 min pause between sessions, so models can reason about absolute life stage across gaps.

### 3. Trajectory reconstruction (per segment)

**Objective:** Remove tracking jumps and restore a continuous path within each session.

1. **Displacement thresholding** — Frames where displacement from the previous point exceeds the threshold (default 16 px) are treated as tracking errors.
2. **Coordinate stitching** — Valid displacements are cumulatively summed from the segment start to reconstruct `(X, Y)` without the spurious jump.

Reconstruction is applied **per segment** so inter-session gaps are never bridged.

### 4. Derived kinematic features

**Objective:** Recompute movement descriptors on the repaired trajectories.

1. **`ComputedSpeed_frames`** and **`ComputedSpeed_timestamp`** — Instantaneous speed from reconstructed coordinates, using frame index and real timestamps respectively.
2. **`Turning_rate`** — Change in heading (`atan2` of displacement), wrapped to \([-π, π]\). Set to 0 when speed ≤ 0.05 px/frame to suppress angular noise while the worm is stationary.

### 5. Optional end-of-life cropping (`--death-crop`)

**Objective:** Remove post-mortem frames where the worm is immobile.

- Cumulative displacement between smoothed reference points is monitored; when it stays below a threshold, the trajectory is cropped at that index.

### 6. Optional normalization (`--normalize`)

Min-max scaling of `X`, `Y`, `Speed`, and both computed speed columns to \([0, 1]\) per worm.

### 7. Dataset statistics

After processing all files, global mean/std/min/max for `X`, `Y`, and `ComputedSpeed_frames` are saved to `dataset_stats.json` in the output directory.

---

## CNN-Specific Preprocessing

For the classification track, trajectories are rendered as **multichannel images** so CNNs can learn spatial-temporal patterns directly.

### Windowing strategy

Preprocessed segments (900 frames) are further sliced into **300-frame clips** with a stride of 150 inside each segment. Clips containing `NaN` values are skipped so partial gaps do not discard an entire segment.

### Multichannel encoding (128 × 128 px)

Each clip is centered and scaled to a global spatial span, then drawn with OpenCV:

| Channel | Content |
|---------|---------|
| **Red** | Binary path occupancy — where the worm has been |
| **Green** | Local time gradient (0 → 255) — when the worm visited each location |
| **Blue** | Speed intensity — faster movement is brighter |

**Example of generated input:**

![Trajectory Example](/results/seg_6_frame_150_to_450.png)

Images are stored under `cnn_dataset/<treatment>/<worm_id>/photos_trajectories/`.

Generate them with:

```bash
python scripts/preprocess.py data/ --output-dir preprocessed_data/ --generate-images --cnn-output-dir cnn_dataset/
```

---

## Data Leakage Prevention

Because each worm contributes many segments (or image clips), a random train/test split would place segments from the **same worm** in both sets. The model would then memorize individual worms rather than generalizing to new animals.

**Solution:** All cross-validation uses **group-aware splitting by worm ID**:

- **Regression** ([`scripts/training_pipeline.py`](scripts/training_pipeline.py)) — `GroupKFold`: every segment from one worm stays in the same fold.
- **CNN classification** ([`scripts/cnn_pipeline.py`](scripts/cnn_pipeline.py)) — `StratifiedGroupKFold`: groups by worm while keeping the treated/control ratio balanced across folds.

This ensures reported metrics reflect generalization to **unseen worms**.

---

## Key Files

| File | Purpose |
|------|---------|
| [`scripts/preprocess.py`](scripts/preprocess.py) | Trajectory cleaning, segmentation, feature computation, CNN image generation |
| [`scripts/training_pipeline.py`](scripts/training_pipeline.py) | Lifespan regression training orchestrator |
| [`scripts/benchmark_pipeline.py`](scripts/benchmark_pipeline.py) | Regression evaluation with survival metrics |
| [`scripts/cnn_pipeline.py`](scripts/cnn_pipeline.py) | CNN classification orchestrator |
| [`utils/train_utils/dataset.py`](utils/train_utils/dataset.py) | `LPBSDataset` and `CElegansCNNDataset` data loaders |
| [`models/cnn_attention_models/`](models/cnn_attention_models/) | Regression model architecture and wrappers |
