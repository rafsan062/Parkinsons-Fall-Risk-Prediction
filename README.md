# Multi-Class Fall Risk Stratification in Parkinson's Disease

This repository contains the modeling pipeline and evaluation code for predicting multi-class fall risk (no fall, rare falls, recurrent falls) in Parkinson's Disease (PD) patients. The methodology emphasizes strict temporal separation (a "landmark" design) to prevent data leakage and robust cross-validation.

## 📂 Repository Structure

*   **`data/`**: Expected location for raw patient data (not included due to privacy).
*   **`docs/`**: Documentation of the methodology.
    *   `results_report.md` - Summary of model performance and findings.
    *   `evaluation_protocol.md` - Detailed modeling setup, grids, and metrics.
    *   `data_cleaning.md` & `imputation_rules.md` - Details on range checks and missing data handling.
    *   `aggregation_and_delta_rules.md` - Rules for deriving longitudinal features.
*   **`notebooks/`**: Jupyter notebooks for sequential data processing.
    *   `01_data_extraction.ipynb` - Feature extraction and temporal truncation.
    *   `02_imputation.ipynb` - Missing data handling and imputation.
    *   `03_eval_no_deltas.ipynb` / `03b_eval_with_deltas.ipynb` - Model evaluation wrappers.
*   **`src/`**: Core source code.
    *   `eval_core.py` - Core execution script for training, tuning, and evaluation.
*   **`results/`**: Model evaluation outputs (JSON summaries, predictions).

## 🚀 How to Run the Pipeline

Due to data privacy regulations, the raw data files are not included in this repository. To run this project locally, you must supply the necessary data files inside the `data/` directory.

### 1. Data Setup
Place your raw clinical data CSV files inside the `data/` folder. Ensure the feature names match the extraction scripts.

### 2. Feature Engineering & Preprocessing
Run the Jupyter notebooks in sequential order:
1.  **`notebooks/01_data_extraction.ipynb`**: Generates the temporally correct base modeling datasets.
2.  **`notebooks/02_imputation.ipynb`**: Handles missing values and produces the final `.csv` files for tree-based models and the imputation strategy for sklearn.

### 3. Model Evaluation
The final model evaluation (tuning and multi-seed testing) is designed to be executed via the Python script: `src/eval_core.py`.

To evaluate the primary clinical model across random seeds (0-4), run:
```bash
python src/eval_core.py --which no_deltas --seeds 0-4
```

To run the sensitivity analysis including longitudinal Delta features:
```bash
python src/eval_core.py --which with_deltas --seeds 0-4
```

The script performs GridSearch tuning based on Inner-CV Macro F1, trains both Two-Stage and Direct 3-class models, and saves the summary metrics to the `results/` folder.
