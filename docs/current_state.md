# Current state and handoff

**Updated:** 25 September 2026, after completing the SHAP/LLM handoff

## Active analysis

The manuscript-oriented analysis uses the historical revision cohort of 1,040
patients and one unimputed table with 26 candidate predictors: 25 eligible
paper-aligned variables plus maximum eligible `NP1CNST`. Neuro-QoL mobility
enters only through the revision's 8-item Gaussian composite
`NQ_GAUSSIAN_REVISION`; its seven component items are used to construct that
score but are not separate model inputs (D28).

The broader 1,043-patient cohort, Part III unable-to-rate flags, separate Part
III ON/OFF values, revision-compatible preprocessing, and latest rather than
maximum constipation are one-change sensitivities. Combined-state Part III,
no 101 flags, maximum constipation, and the current train-only preprocessing
remain primary. The no-delta landmark design remains unchanged.

The final notebooks are in
`notebooks/04_final_pipeline/06_final_run/`. Notebooks 01–11 have been run on
the 26-feature design. Saved validations passed:

| Notebook | Saved validation |
| --- | ---: |
| 01 data cleaning | 68/68 |
| 02 patient-level aggregation | 17/17 |
| 03 input and split setup | 15/15 |
| 04 statistical feature analysis | 13/13 |
| 05 feature-pipeline screening | 15/15 |
| 06 model-family screening | 11/11 |
| 07 focused tuning | 9/9 |
| 07 outer evaluation | 13/13 |
| 08 sensitivity analyses | 14/14 |
| 09 full-cohort fit | 13/13 |
| 10 model interpretation | 6/6 |
| 11 final performance summary | 7/7 |
| 12 SHAP and LLM handoff | 8/8 |

Notebook 11 consolidated the saved performance, sensitivity, statistical,
selection-frequency, full-fit, and interpretation results without refitting
or selecting models. Its results agree with the current final report.

Notebook 12 explains the frozen direct, Stage 1, and Stage 2 models
separately, verifies SHAP additivity, and packages grouped clinical
contributions plus restricted patient-level handoff artifacts. All 8 saved
checks passed. The methodological limitations are maintained separately in
`docs/final_run/methodological_limitations.md`.

## Current evidence

The paired 20-seed outer evaluation is internal validation, not an independent
external test. It produced:

| System | Macro F1 | Rare-fall recall | No-fall recall | Recurrent-fall recall |
| --- | ---: | ---: | ---: | ---: |
| Direct | 0.5132 | 0.3838 | 0.7098 | 0.5567 |
| Two-stage | 0.5098 | 0.3507 | 0.7425 | 0.5217 |

The direct-minus-two-stage macro-F1 difference was 0.0034, with an approximate
corrected 95% interval of -0.0338 to 0.0406. The rare-fall recall difference
was 0.0331, with interval -0.0529 to 0.1191. These results do not establish an
architecture advantage; keep both architectures in reporting.

Notebook 08 exactly reproduced all 12,480 primary prediction rows before
running sensitivities. True and predicted classes and selected columns were
identical, and the maximum score difference was 5.55e-16. Every
sensitivity-minus-primary macro-F1 corrected interval included zero. The
revision-compatible preprocessing branch had the highest observed mean macro
F1 (direct 0.5216; two-stage 0.5173), but this descriptive sensitivity result
does not justify replacing the locked primary branch.

Notebook 09 fit explanation/deployment artifacts on all 1,040 primary
patients. These full-cohort fits are not performance estimates:

| Target | Final artifact |
| --- | --- |
| Direct | Balanced CatBoost; corrected-FDR selection |
| Stage 1 | Balanced Extra Trees; corrected-FDR selection |
| Stage 2 | Unweighted random forest; median-threshold Extra-Trees selection |

The model files and verified hashes are recorded in
`results/final_pipeline/06_final_fit_and_performance_summary/09_full_cohort_fit/models/model_manifest.json`.

Notebook 10's grouped out-of-fold permutation analysis found the largest
direct-system macro-F1 drops for constipation (0.0209), freezing (0.0206), the
Neuro-QoL composite (0.0189), age (0.0134), and Part III motor total (0.0087).
The two-stage system's largest drops were for freezing (0.0542), Part IV motor
complications (0.0397), and the Neuro-QoL composite (0.0321). These results are
descriptive and non-causal and cannot redefine the frozen model.

## Audit conclusions

- The active primary and sensitivity tables contain none of the seven removed
  raw Neuro-QoL items. The composite remains present.
- Candidate, selection, tuning, sensitivity, and full-fit artifacts use the
  26-feature D28 configuration. Current run manifests and model hashes match
  their inputs.
- Notebook 01 and Notebook 02 appropriately retain raw Neuro-QoL item fields
  only to clean one coherent form and calculate the 8-item composite.
- Notebook 04 may mention the raw items in the documented historical-test
  lookup, but its active feature list contains only the 26 final candidates.
- Notebook 05 lists the removed Neuro-QoL PCA branch only as an excluded legacy
  branch. The active engineering manifest has four interactions and three Part
  I patient-item PCA thresholds.
- The superseded 33-feature outputs are isolated under
  `results/final_pipeline/superseded_33_feature_run_2026-09-25/`.

## Next task

1. Synchronize the final notebooks, documentation, and approved shareable
   results into `github_repo_export/`. Keep governed patient-level PPMI
   artifacts out of any public export unless permission is confirmed.

The completed Notebooks 21–29c remain supporting exploratory evidence. They
are not the manuscript analysis. Independent PPMI data or a future cohort is
still required for publication-strength confirmation.
