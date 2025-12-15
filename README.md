# Classification of C. elegans Longevity under Terbinafine Treatment

## Project Overview
This project focuses on the automated classification of *Caenorhabditis elegans* (*C. elegans*) nematodes based on their longevity phenotypes. Specifically, we aim to distinguish between worms treated with **Terbinafine** (Terbinafine+) and untreated control worms (Terbinafine-).

Terbinafine helps extending the lifespan of *C. elegans*. By analyzing movement trajectories and derived features, we investigate whether machine learning models can accurately predict the treatment group (and thus the longevity potential) from behavioral data.

A key objective of this project was to adopt a **featureless approach** as much as possible. Instead of relying heavily on handcrafted biological metrics, we prioritized methods (such as ROCKET, Tail-MIL, and CNNs) that learn directly from raw data representations, minimizing the bias introduced by manual feature selection.

---

## Preprocessing Methodology

The raw data consists of movement trajectories tracked over the worms' lifespans. Since the raw tracking data can be noisy and inconsistent, a rigorous preprocessing pipeline was applied to ensure data quality and relevance. The steps were applied in the following order:

### 1. Data Cleaning & Artifact Removal
**Objective**: Remove non-biological noise and tracking errors.
1.  **Drop First Row**: The first row of many files contained inconsistent timestamps or initialization artifacts. It was systematically removed.
2.  **Death Clipping**: Frames recorded after the annotated frame of death were removed to focus analysis on the living phase.

### 2. Trajectory Reconstruction & Smoothing
**Objective**: Fix tracking "jumps" where the camera lost the worm or swapped identity, causing unrealistic displacements.
1.  **Speed Cap & Outlier Removal**: We identified instances of impossible acceleration (e.g., speed > 4). These are caused by tracking errors.
2.  **Displacement Thresholding**: We removed frames where the sudden displacement exceeded a biological threshold (e.g., > 16 pixels/frame).
3.  **Coordinate Reconstruction**: When gaps or jumps were removed, the worm's trajectory was stitched back together (cumulative summation of valid displacements) to recreate a continuous, biologically plausible path.

### 3. Segmentation
**Objective**: Handle the variable lifespan of worms.
- Tracks were divided into fixed-length **segments** (e.g., 900 frames). This standardizes the input for models that process sequential data and allows us to analyze behavior at different life stages.

### 4. Feature Extraction (Tabular Models)
For models like Logistic Regression, Random Forest, and SVM, we extracted scalar features per segment to capture behavioral dynamics:
- **Mean & Median Speed**: Proxies for general activity levels.
- **Net Displacement**: Distance between start and end points of a segment.
- **Tortuosity**: Ratio of total path length to net displacement. High tortuosity indicates "searching" behavior (frequent turning), while low tortuosity indicates "roaming" behavior.

---

## CNN Specific Preprocessing

For the Convolutional Neural Network (CNN) approach, we treated the trajectory classification as a computer vision problem. Instead of scalar features, we generated **visual representations** of the worm's movement.

### Multichannel Trajectory Imaging (@Windowing Strategy)
**Windowing Strategy**: although the tabular models use full 900-frame segments, for the CNN we further slice these segments into smaller **clips of 150 frames** (with a stride of 75).
- This allows us to filter out clips containing `NaNs` (gaps) without discarding the entire segment.
- It focuses the network on shorter, more detailed movement patterns.

We then converted these time-series coordinates $(x, t)$ into 3-channel images ($128 \times 128$ pixels) to encode spatial and temporal information simultaneously:

- **Red Channel (Path)**: Binary occupancy map. Indicates *where* the worm has been.
- **Green Channel (Time)**: Gradient from 0 to 255. Encodes *when* the worm was at a specific position (fading from dark to bright). This preserves temporal order in a static image.
- **Blue Channel (Speed)**: Intensity mapped to instantaneous speed. Brighter pixels indicate faster movement at that location.

**Example of Generated Input:**
![Trajectory Example](/results/seg_6_frame_150_to_450.png)

This encoding allows the CNN (e.g., ResNet) to learn complex patterns like "slowing down while turning" or "looping behavior" that scalar features might miss.

---

---

## Data Leakage Prevention

One of the most critical aspects of our methodology was ensuring zero data leakage between training and validation sets. Since we segmented the lifespan of each worm into multiple data points:
- A simple random split would put segments of the **same worm** in both the training and validation sets.
- The model would then learn to recognize the specific worm's movement style rather than the treatment effect, leading to massively inflated performance metrics.

**Solution**: We implemented **Stratified Group K-Fold Cross-Validation**, encapsulated in our custom module [`utils/train_utils/fold_utils.py`](utils/train_utils/fold_utils.py).
- **Group**: We grouped data by `WormID`. All segments from a single worm are forced to be in the same fold (either all in train or all in validation).
- **Stratified**: We ensured that the ratio of Treated vs. Control worms remains balanced across folds.

This rigorous validation strategy ensures that our reported metrics reflect the model's ability to generalize to **new, unseen worms**.

---

## Pipelines & Architecture

We implemented two distinct pipelines to robustly evaluate different modeling approaches.

### 1. Main Pipeline (`main_pipeline.py`)
This pipeline handles traditional Machine Learning models and Time Series classifiers.
- **Models Supported**: Logistic Regression, Random Forest, SVM, XGBoost, and Time Series models like ROCKET and Tail-MIL.
- **Data Augmentation**: To improve model robustness, we implemented a `UnifiedCElegansAugmentedDataset` that expands the training data by a factor of 6 using:
    - **Rotations**: 3 random rotations per sample.
    - **Translation**: Random X/Y offsets.
    - **Scaling**: Random scaling (0.8x to 1.2x).
- **Workflow**:
    1.  Loads preprocessed tabular data (or raw time-series for ROCKET).
    2.  Performs **Stratified Group K-Fold Cross-Validation** (ensuring all segments of one worm stay in the same fold prevent leakage).
    3.  Training, Validation, and Metric reporting (Accuracy, F1, Precision, Recall).

### 2. CNN Pipeline (`cnn_pipeline.py`)
A dedicated pipeline for Deep Learning models processing the image datasets.
- **Models Supported**: ResNet18, ResNet50, DenseNet121.
- **Workflow**:
    1.  **Custom Dataset Class (`CElegansCNNDataset`)**: Loads images and extracts labels/worm IDs.
    2.  **Augmentation**: Applies random rotations, flips, and normalization to improve generalization.
    3.  **Training Loop**: Runs a PyTorch training loop with Stratified Group K-Fold.
    4.  **Comparison**: Automatically plots and compares the performance of different architectures.

---


## Project Structure
```
.
├── cnn_dataset/               # Generated dataset for CNNs
├── data/                      # Raw data and summary files
├── models/                    # Model definitions
│   ├── model_cnn.py           # CNN factory (ResNet, DenseNet)
│   ├── model_lr.py            # Logistic Regression wrapper
│   └── ...                    # Other model wrappers
├── scripts/                   # Execution scripts
│   ├── cnn_pipeline.py        # CNN training pipeline
│   ├── main_pipeline.py       # Main pipeline for tabular/time-series models
│   ├── preprocess.py          # Data cleaning and reconstruction
│   └── extract_features.py    # Feature extraction for tabular models
├── utils/                     # Utility functions
│   ├── train_utils/           # Datasets and Stratified Group K-Fold logic
│   └── plot_utils/            # Plotting functions
└── requirements.txt           # Python dependencies
```

## Installation

1.  Clone the repository.
2.  Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

## Usage

### 0. Preprocessing (Required First Step)
Before running any analysis, the raw data must be cleaned, reconstructed, and prepared.

**Standard Cleaning & Reconstruction:**
```bash
python scripts/preprocess.py
```
This script reads from `data/`, cleans the trajectories, and outputs validation-ready CSVs to `preprocessed_data/`.

**Generating CNN Images:**
To generate the dataset for the CNN pipeline (images):
```bash
python scripts/preprocess.py --generate-images --cnn-output-dir "cnn_dataset/"
```

**Feature Extraction (for tabular models):**
After standard preprocessing, extract scalar features (speed, tortuosity, etc.):
```bash
python scripts/extract_features.py
```

### 1. Running the Main Pipeline
Train and evaluate tabular models (Logistic Regression, Random Forest, etc.) or Time Series models (ROCKET).

```bash
python scripts/main_pipeline.py --pytorch_dir "preprocessed_data/"
```

**Options:**
- `--plot`: Generate plots of the results.
- `--augmented_data` / `-a`: Use the augmented dataset (6x larger).
- `--prod`: Run in production mode (saves the best model).
- `-o`: Specify output JSON filename for results.

### 2. Running the CNN Pipeline
Train and compare Deep Learning models (ResNet18, ResNet50, DenseNet).

```bash
python scripts/cnn_pipeline.py --data_dir "cnn_dataset"
```

**Configuration:**
- You can modify the `models_config` dictionary inside `scripts/cnn_pipeline.py` to change architectures, batch sizes, or learning rates.

---

## Key Files
- `scripts/preprocess.py`: Implementation of cleaning and trajectory reconstruction logic.
- `scripts/main_pipeline.py`: Orchestrator for tabular and time-series models.
- `scripts/cnn_pipeline.py`: Orchestrator for CNN models.
- `models/`: Directory containing model definitions (ResNet factory, LogisticRegression wrapper, etc.).
- `utils/train_utils/dataset.py`: Unified data loading logic.
