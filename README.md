# Parkinson's Disease Fall-Risk Prediction

This repository contains the final 26-feature landmark analysis for predicting three fall-risk classes in Parkinson's disease:

- no fall;
- rare fall; and
- recurrent fall.

The predictor landmark is 12 months before the outcome visit. Only information available on or before that patient-specific date is eligible. The outcome is the fall category reported for the following 12-month window.

## Final results

Across 20 paired stratified 70/30 splits:

| System | Macro F1 | Rare-fall recall | No-fall recall | Recurrent-fall recall |
| --- | ---: | ---: | ---: | ---: |
| Direct three-class | 0.513 ± 0.019 | 0.384 ± 0.056 | 0.710 ± 0.045 | 0.557 ± 0.102 |
| Two-stage | 0.510 ± 0.019 | 0.351 ± 0.068 | 0.743 ± 0.038 | 0.522 ± 0.159 |

The paired macro-F1 comparison did not establish an architecture-level winner. Final full-cohort artifacts are therefore retained for the direct model and both two-stage components:

- direct: balanced CatBoost with corrected-FDR selection of 20 clinical groups;
- Stage 1: balanced Extra Trees with corrected-FDR selection of the same 20 groups;
- Stage 2: random forest with median-threshold Extra-Trees selection of 13 groups.

See [`docs/final_run/final_results_report.md`](docs/final_run/final_results_report.md) for the full configuration and results. Full-cohort fitted models and SHAP values are interpretation and future-use artifacts, not independent performance estimates.

## Repository layout

- `notebooks/`: final executable workflow, numbered 01–12.
- `data/`: dated raw CSV inputs used by Notebooks 01–02.
- `results/final_pipeline/`: saved checkpoints, predictions, summaries, fitted pipelines, validation files, and interpretation artifacts.
- `docs/`: final cleaning, aggregation, imputation, evaluation, methodology, results, and limitation documents.
- `llm_layer_handoff/`: direct, Stage 1, and Stage 2 SHAP packages; raw patient alignment; feature dictionaries; plots; validation; and implementation guidance.
- `.agents/`: reusable leakage-safe feature-selection instructions for coding-agent handoff.

## Execution order

Run the notebooks from the repository root in filename order. Every notebook locates the root through `AGENTS.md` and saves into its matching `results/final_pipeline/` directory. Long modeling notebooks resume from validated checkpoints.

The saved final run is already complete. Re-execution is needed only to reproduce or deliberately refresh artifacts.

## Environment

The saved models were produced with Python 3.13.12, scikit-learn 1.8.0, CatBoost 1.2.10, XGBoost 3.2.0, LightGBM 4.7.0, and cloudpickle 3.1.2. SHAP artifacts use SHAP 0.52.0. Install the declared environment with:

```bash
python -m pip install -r requirements.txt
```

## LLM explanation layer

Start with [`llm_layer_handoff/llm_layer_recommendations.md`](llm_layer_handoff/llm_layer_recommendations.md). The package includes:

- class-specific direct, Stage 1, and Stage 2 SHAP arrays;
- encoded and clinically grouped contributions;
- raw non-imputed patient inputs aligned by `row_index` and `PATNO`;
- a 26-feature clinical dictionary and raw-input schema;
- model-output alignment, global summaries, plots, and validation.

Never present an internal imputation value as an observed clinical measurement. Stage 1 and Stage 2 explanations must remain separate because hard routing is not additive.

## Scientific boundaries

All data-dependent preprocessing, feature selection, engineering, and tuning are fitted within training data. Outer test results do not select the model. The final estimates are internal validation because the cohort also supported prior exploratory development. See [`docs/final_run/methodological_limitations.md`](docs/final_run/methodological_limitations.md) for the maintained interpretation boundaries.
