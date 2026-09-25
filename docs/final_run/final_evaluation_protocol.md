# Final evaluation protocol

## Purpose

This protocol defines how the final fall-risk pipelines are selected and evaluated. It applies to the direct three-class model and the two-stage model. The completed split assignments are authoritative and must be reused by every downstream notebook.

The results are internal validation estimates. The same PPMI cohort supported earlier exploratory work, so these holdouts are not an independent external validation cohort.

## Primary input

- One row per patient.
- 1,040 patients: 712 no fall, 227 rare fall, and 101 recurrent fall.
- One unimputed 26-feature candidate universe (amended from 33 by D28: Neuro-QoL enters only as its Gaussian score).
- Outcome and timing columns remain separate from predictors.
- Predictors occur at or before the patient-specific landmark, 12 months before the outcome visit.

## Outer evaluation splits

Seeds 0–19 define 20 stratified random holdouts. Each seed contains:

- 728 outer-training patients, used for all development;
- 312 outer-test patients, used once after the seed-specific pipeline is selected.

Stratification preserves all three outcome classes. Direct and two-stage classification use the same patient assignments for every seed, making their performance comparisons paired.

The 20 test sets are distinct. A patient may appear in test sets for more than one seed, so the 20 scores are correlated repeated-holdout estimates rather than independent experiments.

## Inner validation folds

Each 728-patient outer-training set is divided into five stratified inner folds using the same integer as the outer split seed.

For one inner iteration:

- four folds fit every learned step;
- one fold evaluates candidate pipelines;
- the process rotates until each outer-training patient has served once in inner validation.

Inner validation folds contain 145 or 146 patients. Every inner training and validation role contains no-fall, rare-fall, and recurrent-fall patients.

Stage 2 uses only the true rare and recurrent fallers within the applicable saved inner or outer training partition. Outer-test patients never enter an inner fold.

## Training-only pipeline rule

Within each inner-training partition, fit the entire candidate pipeline in this order:

1. derive approved missing-state indicators and apply fixed clinical missingness rules;
2. learn any required imputation values;
3. fit categorical encoding and scaling when required;
4. construct the prespecified interaction or PCA representation when tested;
5. fit the candidate feature selector;
6. fit the candidate classifier.

Apply the fitted pipeline unchanged to the corresponding inner validation patients. No statistic, category vocabulary, scaling value, PCA loading, selected feature, hyperparameter, model family, or decision threshold may use outer-test data.

## Targets

| Target | Training patients | Labels | Recall used for a near tie |
| --- | --- | --- | --- |
| Direct | All applicable training patients | 0 = no fall, 1 = rare fall, 2 = recurrent fall | Rare-fall recall |
| Stage 1 | All applicable training patients | 0 = no fall, 1 = any fall | Any-fall recall |
| Stage 2 | True fallers only | 0 = rare fall, 1 = recurrent fall | Rare-fall recall |

The final two-stage prediction uses hard routing: Stage 1 predicts no fall versus any fall; patients predicted as fallers are classified by Stage 2 as rare or recurrent fallers.

## Selection rule

Choose candidates separately for direct classification, Stage 1, and Stage 2 within each outer-training set.

- Primary criterion: mean inner-fold macro F1.
- If candidates are within 0.01 macro F1, apply the target-specific recall in the table above.
- If still tied, prefer fewer selected features and then the prespecified simpler model.
- Accuracy, weighted F1, AUC, and outer-test results do not select a pipeline.

The candidate pipeline search starts from the single 26-feature universe. Feature selection occurs inside the training pipeline; no full-cohort selected subset is applied to every split.

Feature engineering also remains inside that source universe. The final
screen includes four interactions that can be formed from the 26 variables
and PCA alternatives for the three included Part I patient items, at 0.80,
0.90, and 0.95 explained variance. Earlier engineering definitions that require additional raw
variables are excluded from this manuscript run. Every PCA loading is fitted
inside the applicable training fold.

## One-time outer evaluation

After inner validation selects a seed-specific procedure:

1. refit the complete procedure on all 728 outer-training patients;
2. transform the 312 outer-test patients using only the fitted training values;
3. generate direct or end-to-end two-stage predictions once;
4. save patient-level predictions, probabilities, selected features, fitted configuration, and metrics.

Outer-test results are reporting evidence. They cannot change feature handling, the selected model, or a sensitivity definition.

## Reported performance

Report at minimum:

- macro F1;
- recall for no fall, rare fall, and recurrent fall;
- balanced accuracy;
- accuracy;
- macro one-versus-rest AUC when valid probabilities are available;
- confusion matrices.

Summaries must preserve the paired seed structure. Report seed-level values and paired direct-versus-two-stage differences. Any corrected repeated-holdout uncertainty interval must state that overlapping patient samples and prior cohort exploration limit its interpretation.

## Sensitivity analyses

Notebook 08 (`notebooks/04_final_pipeline/06_final_run/08_sensitivity_analyses.ipynb`) refits the 60 frozen Notebook 07 winners on each branch input using the same outer splits and refit seeds. It does not retune. It first reruns the primary refits and proceeds only if Notebook 07's predictions are reproduced exactly.

- **`101` flags, latest constipation:** the changed input columns only.
- **Separate ON/OFF:** eight state-specific scores replace the four combined-state scores, and `ON_STATE_AVAILABLE`/`OFF_STATE_AVAILABLE` are grouped with their scores. Winners whose interaction uses combined-state scores omit that term and keep their model and dense input.
- **1,043-patient cohort:** every primary train/test assignment is unchanged. The three added patients join outer training in all seeds with `NO_ELIGIBLE_LONGITUDINAL_HISTORY`, so the test patients remain paired with the primary run.
- **Revision-compatible preprocessing:** blanket Part IV zero, nominal training mode, dopamine training mode plus `DOPTHERST_MISSING`, and no form indicators. Dense models use training medians; native boosting models keep numeric `NaN`.

Artifacts are saved under `results/final_pipeline/07_sensitivity_analyses/08_sensitivity_analyses/`. Sensitivity test results describe robustness and cannot redefine the primary pipeline.

## Saved authoritative files

Notebook 03 saves the split and configuration files under:

`results/final_pipeline/06_final_fit_and_performance_summary/03_setup_and_splits/`

The key files are:

- `outer_split_assignments.csv`;
- `inner_fold_assignments.csv`;
- `analysis_configuration.csv`;
- `primary_feature_manifest.csv`;
- `target_manifest.csv`;
- `split_validation.csv`.

Notebook 03 passed all 15 validation checks. It created 20 distinct test sets, found zero outer-test overlap with corresponding inner folds, and confirmed all required classes in every training and validation role.
