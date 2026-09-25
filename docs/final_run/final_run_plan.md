# Locked revision-aligned final run

**Status:** 26-feature run executed through Notebook 12 on 25 September 2026; all saved validation checks passed
**Locked:** 25 September 2026

This is the final manuscript-oriented analysis. It keeps the recognizable
revision workflow while applying the leakage safeguards and useful findings
from the completed updated-data experiments. The completed Notebooks 21–29c
remain supporting exploratory work; they are not overwritten.

## 1. Primary analysis input

| Item | Locked choice |
| --- | --- |
| Analysis unit | One row per patient |
| Primary cohort | Historical revision eligibility: 1,040 patients (712 no fall, 227 rare fall, 101 recurrent fall) |
| Temporal design | Landmark design: predictors at or before the patient-specific index month, which is 12 months before the outcome visit |
| Outcome | `FLNFR12M` at the outcome visit: 0 = no fall, 1 = rare fall, and 2–4 = recurrent fall |
| Candidate universe | 25 eligible paper-aligned variables plus maximum eligible `NP1CNST` constipation severity (26). Neuro-QoL mobility enters only as the revision's 8-item Gaussian score; its seven individual items are not separate candidates (amended by D28 after the initial 33-feature run) |
| Main Part III representation | Combined-state maximum valid pre-landmark value |
| Main 101 handling | Codebook-defined unable-to-rate values become missing; assessment-status flags are omitted from the primary input |
| Longitudinal deltas | Omitted; the no-delta design remains primary |
| Stored input | One unimputed 26-feature landmark table plus identifiers, outcome, and split metadata |

The final modeling workflow uses only this 26-feature universe; it does not
build separate 21- or 32-feature candidate tables. The older sets remain
historical context in prior results. Separate ON/OFF values, 101 flags, the
1,043-patient cohort, revision-compatible preprocessing, and latest rather
than maximum constipation remain sensitivity inputs.

## 2. Paired evaluation design

Use the revision-style repeated holdout framework so the final results can be
integrated into the existing manuscript with limited structural change:

1. Use seeds 0–19 to create 20 stratified 70/30 train/test splits.
2. Reuse the same patient split for direct and two-stage classification.
3. Freeze each test set before any imputation, engineering, feature selection,
   or tuning.
4. Use five stratified inner folds within that seed's training set for every
   data-dependent decision.
5. Train Stage 2 only on true fallers from the applicable training fold.
6. Evaluate the selected direct and hard-routed two-stage pipelines once on
   that seed's untouched test set.

Because this cohort has already supported exploratory development, these are
internal validation estimates. They are not independent external validation.

## 3. Training-only pipeline

Each inner-training fold fits the complete pipeline from raw aggregated
predictors:

1. Apply the approved missing-data rules and learn any imputation values.
2. Encode categorical variables and scale when required by the model. Use the
   enhanced dense branch for models requiring complete matrices and the
   approved native-missing branch for compatible boosting models.
3. Construct the four prespecified interactions whose source variables are
   contained in the final 26-feature universe when that branch is tested.
4. Fit one within-family PCA alternative using only final-run source
   variables: the three included Part I patient items. Test the frozen 0.80,
   0.90, and 0.95 variance thresholds inside the training fold. The earlier
   Neuro-QoL PCA branch was removed with the individual items (D28).
5. Compare four feature-selection choices:
   - retain all candidates;
   - corrected univariate FDR screening;
   - L1-regularized selection;
   - Extra-Trees importance selection.
6. Fit and score the model.

Every learned value, selected column, interaction decision, and PCA loading is
derived from the inner-training fold only. The outcome is never imputed.

The earlier systolic-pressure interaction and the larger Part I rater, Part I
patient, Part IV, Neuro-QoL, and orthostatic PCA definitions require source
variables outside the final 26-feature universe. They are not silently added
to this manuscript run. Notebook 05 records the exclusions and the one
26-compatible PCA redefinition explicitly.

## 4. Feature and model search

Run the search separately for direct classification, Stage 1, and Stage 2
within every outer-training set.

### Stage A — feature-pipeline screening

Screen the raw 26-feature representation, the four selectors, and the locked
interaction/PCA alternatives with the same fixed balanced logistic-regression
and shallow Extra-Trees references used in the completed screening design.
Retain the best two feature-pipeline configurations for that target and seed.

### Stage B — nine-family screen

Compare one frozen starting configuration from each family on the retained
feature pipelines:

1. logistic regression;
2. Linear SVC;
3. RBF SVC;
4. random forest;
5. Extra Trees;
6. histogram gradient boosting;
7. XGBoost;
8. LightGBM;
9. CatBoost.

Retain the best two feature-pipeline/model combinations for that target and
seed. This broad screen is necessary because CatBoost, Extra Trees, and RBF
SVC frequently won training-only partitions in the completed expanded search.
Any probability calibration or score conversion required for a reported
metric must also be fitted using training data only.

### Stage C — focused tuning

Tune only the two retained combinations using the parameter grids frozen
before Notebook 29b. Do not expand a grid after viewing final-test results.

### Selection rule

- Primary selection metric: mean inner-fold macro F1.
- Within 0.01 macro F1: use rare-fall recall for direct classification and
  Stage 2, and any-fall recall for Stage 1.
- If still tied: prefer fewer selected features and then the prespecified
  simpler model.
- Accuracy, weighted F1, AUC, and outer-test results never select a pipeline.

## 5. Execution notebooks and saved state

The shareable GitHub workflow uses a fresh global numbering sequence. The
completed internal notebooks retain their original 22–29c names as the audit
trail; the final export uses the following clean execution order:

| Order | Final GitHub notebook | Source | Purpose |
| --- | --- | --- | --- |
| 01 | `01_data_cleaning.ipynb` | Based on internal Notebook 22 | Rebuild and validate landmark-eligible cleaned visit tables |
| 02 | `02_patient_level_aggregation.ipynb` | Based on internal Notebook 23 | Build the unimputed 1,040-patient, 26-feature landmark table and sensitivity inputs |
| 03 | `03_final_input_and_split_setup.ipynb` | Final-run setup | Validate the final input and freeze the paired split/configuration manifests |
| 04 | `04_statistical_feature_analysis.ipynb` | Final-run statistical analysis | Run corrected univariate tests within training partitions and summarize effects, FDR results, and stability |
| 05 | `05_feature_pipeline_screening.ipynb` | Final-run screening | Run resumable target-specific selector and engineering screening |
| 06 | `06_model_family_screening.ipynb` | Final-run screening | Screen all nine families and retain two combinations per target/seed |
| 07 | `07_tuning_and_outer_evaluation.ipynb` | Final-run evaluation | Tune retained combinations and evaluate each frozen test split once |
| 08 | `08_sensitivity_analyses.ipynb` | Final-run sensitivity | Evaluate the five one-change sensitivity branches using the locked procedure |
| 09 | `09_full_cohort_fit.ipynb` | Final-run full-cohort fit | Rerun internal selection on all 1,040 patients and fit the seed-42 direct and two-stage artifacts |
| 10 | `10_model_interpretation.ipynb` | Final-run interpretation | Produce final model explanations and feature summaries without redefining the model |
| 11 | `11_final_performance_summary.ipynb` | Complete; 7/7 checks passed | Summarize performance, sensitivities, statistical analysis, selected features, and model frequencies |
| 12 | `12_shap_and_llm_handoff.ipynb` | Complete; 8/8 checks passed | Explain the frozen direct and two-stage components with SHAP and package the governed LLM handoff artifacts |

All final publication notebooks live together in
`notebooks/04_final_pipeline/06_final_run/`. Their filenames match the final
GitHub numbers shown above so export does not require renaming. Notebook 28b is
supporting preflight and remains with Notebooks 28–29c in
`05_feature_selection_and_model_tuning/`. Result folders are unchanged.
Notebooks 01 and 02 are clean publication implementations of the already
verified cleaning and aggregation work rather than new methodological
experiments.

Notebook 04 provides the statistical analysis requested by the professors.
For model selection, every test and FDR correction is recalculated inside the
applicable training fold; a pooled significance list cannot select features
for all splits. It reports outcome-group summaries, the test used, effect size,
raw and adjusted significance, and stability across training partitions.
Notebook 11 may add a clearly labeled full-cohort descriptive appendix after
all model choices are frozen. That appendix is for communication and cannot
change the fitted pipeline.

Long-running notebooks must save fold-level checkpoints, manifests, selected
configurations, predictions, and metrics under matching numbered folders in
`results/final_pipeline/06_final_fit_and_performance_summary/`. A rerun resumes
from validated checkpoints rather than restarting completed units.

The full-cohort fits are deployable/explanation artifacts. Their training-set
fit is not another performance estimate.

## 6. Sensitivity analyses

After the primary procedure is frozen, evaluate one change at a time in
Notebook 08 (`notebooks/04_final_pipeline/06_final_run/08_sensitivity_analyses.ipynb`):

1. add the three Part III unable-to-rate flags;
2. replace combined-state Part III with separate ON/OFF features;
3. use the broader 1,043-patient cohort;
4. use revision-compatible preprocessing;
5. use latest eligible rather than maximum eligible constipation.

Sensitivities reuse the locked primary procedure and paired splits where the
cohort permits. They cannot redefine the primary pipeline from their test
results.

## 7. Final reporting and repository export

Maintain concise, final-state versions of these documents in the active
project:

- `docs/final_run/final_data_cleaning.md`;
- `docs/final_run/final_aggregation_rules.md`;
- `docs/final_run/final_imputation_rules.md`;
- `docs/final_run/final_evaluation_protocol.md`;
- `docs/final_run/final_results_report.md` after results exist.

After the final run and sensitivity/reporting work are complete, update
`github_repo_export` as the shareable repository. Replace its revision-era
method documents with the final versions, add the final notebooks and required
reproducibility artifacts, update its README and results report, and remove or
clearly label superseded outputs. Review the Git diff before committing or
publishing. The active project remains authoritative until that synchronization
is complete. The export already contains uncommitted SHAP/handoff artifacts;
preserve and reconcile those files during synchronization rather than
overwriting or deleting them implicitly.

## 8. Completion criteria

The final run is complete when:

- every seed has validated, paired direct and two-stage predictions;
- no test patient influenced preprocessing, feature selection, tuning, or
  model-family choice;
- macro F1, per-class recall, balanced accuracy, accuracy, and macro AUC are
  summarized with paired uncertainty;
- the selected features and model families are reported across seeds;
- full-cohort direct and two-stage artifacts are saved;
- planned sensitivities and limitations are reported; and
- the five final method/results documents and `github_repo_export` agree with
  the executed analysis.
