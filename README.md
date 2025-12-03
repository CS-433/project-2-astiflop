# Worm Classification ML Project

## Project Structure

```
project-2-astiflop/
│
├── dataset.py
├── extract_features.py
├── fold_utils.py
├── main_pipeline.py
├── preprocess.py
├── presents_results.py
├── README.md
├── data/
│   ├── lifespan_summary.csv
│   ├── TERBINAFINE- (control)/
│   └── TERBINAFINE+/
├── data_analysis/
│   ├── exploration.ipynb
│   └── generate_comparison_plot.py
├── models/
│   ├── base.py
│   ├── model_lr.py
│   ├── model_rf.py
│   ├── model_rocket.py
│   ├── model_svm.py
│   ├── model_tail_mil.py
│   └── model_xgboost.py
├── preprocessed_data/
├── preprocessed_data_for_classifier/
└── tail_mil/
```

## File Descriptions

### Data Preparation

- **preprocess.py**
  Cleans and preprocesses raw worm tracking CSV files.
  - Renames files for consistency.
  - Drops invalid rows.
  - Segments time series.
  - Caps speed, normalizes coordinates, and interpolates gaps.
  - Produces cleaned files in `preprocessed_data/`.

- **extract_features.py**
  Converts preprocessed worm data into segment-level features for classical machine learning models.
  - Computes features (age, speed, displacement, tortuosity) per segment.
  - Aggregates and saves segment data for each worm in `preprocessed_data_for_classifier/`.

- **dataset.py**
  Defines `UnifiedCElegansDataset`, a PyTorch Dataset class that handles data loading for all models.
  - Loads raw time series for ROCKET and Deep Learning models.
  - Loads feature-based data for Scikit-Learn models.
  - Handles data splitting and formatting.

### Modeling

- **main_pipeline.py**
  The central script for training and evaluating models.
  - Orchestrates the training process using Stratified K-Fold Cross-Validation.
  - Supports multiple models: Logistic Regression, Random Forest, XGBoost, SVM, ROCKET, and Tail-MIL.
  - Aggregates results across folds.

- **models/**
  Contains modular implementations of the different models:
  - `model_rocket.py`: ROCKET (MiniRocketMultivariate) implementation for time series classification.
  - `model_tail_mil.py`: Implementation of **TAIL-MIL** (Time-Aware Instance Learning - Multiple Instance Learning). This is a custom Deep Learning model that uses CNNs for feature extraction, followed by hierarchical attention mechanisms (Variable-level and Time-level) with positional encodings to classify worms based on their entire lifespan of movement segments.
  - `model_lr.py`, `model_rf.py`, `model_svm.py`, `model_xgboost.py`: Wrappers for classical ML models (Logistic Regression, Random Forest, SVM, XGBoost).
  - `base.py`: Base classes or common utilities for models.

- **tail_mil/**
  Contains specific implementation details, experiments, or trained weights for the Tail-MIL model.

### Utilities

- **fold_utils.py**
  Provides `get_stratified_worm_splits` to generate stratified K-Fold splits based on worm IDs, ensuring no data leakage between train and validation sets.

- **presents_results.py**
  Utilities for processing and visualizing results.
  - Calculates average metrics (Accuracy, Precision, Recall, F1) across folds.
  - Saves results to JSON.
  - Generates performance comparison plots.

### Data Analysis

- **data_analysis/**
  Contains notebooks and scripts for data exploration and generating specific plots.

## Typical Workflow

1. **Preprocessing:**
   Run `preprocess.py` to clean and segment raw data.

2. **Feature Extraction:**
   Run `extract_features.py` to generate segment-level features for classical models.

3. **Model Training & Evaluation:**
   Run `main_pipeline.py` to train and evaluate selected models.
   ```bash
   python main_pipeline.py --plot
   ```
   This script will:
   - Load data using `dataset.py`.
   - Perform Stratified K-Fold CV.
   - Train models defined in `models/`.
   - Save results to `avg_results.json`.
   - Plot results if `--plot` is specified.

## Data Leakage Prevention

The project uses `fold_utils.py` to ensure that all segments or time series belonging to the same worm are kept together in either the training or validation set. This prevents the model from learning to recognize specific worms rather than general patterns associated with the treatment.

