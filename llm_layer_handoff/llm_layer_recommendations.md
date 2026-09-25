# Final LLM layer implementation guidance

## Purpose

This handoff supports patient-facing explanations for the frozen 26-feature fall-risk models. The direct model predicts no fall, rare fall, or recurrent fall. The two-stage system first predicts no fall versus any fall, then applies Stage 2 only to patients routed to any fall.

## Required interpretation rules

1. **Use raw patient values for narrative text.** Read `patient_raw_inputs.csv` and `raw_input_schema.csv` when describing a patient. Never present a training median, mode, structural placeholder, or encoded category column as an observed clinical measurement.
2. **Keep missingness explicit.** Say “not recorded” or “unavailable” when the raw input is missing. The model may use a missingness indicator or an internal fill value, but the explanation must not convert that fill into a clinical fact.
3. **Describe the time meaning accurately.** “Latest eligible” means the closest recorded value on or before the patient-specific landmark. “Maximum eligible” means the greatest recorded pre-landmark burden and may come from an older visit. It is not necessarily the patient's condition on the prediction date.
4. **State the prediction horizon.** Inputs are available by the landmark, which is 12 months before the outcome visit. The prediction concerns falls during the following 12-month window.
5. **Use the final 101 rule.** Codebook-defined `101 = unable to rate` is converted to missing for applicable clinical fields. The primary model does not use separate 101-status flags. Do not describe a patient as severely impaired merely because a 101 code occurred.
6. **Use the Neuro-QoL score carefully.** `NQ_GAUSSIAN_REVISION` is a custom Gaussian-weighted score from one coherent 8-item form. It is not a standard Neuro-QoL total or T-score. The seven component items are not separate model predictors.
7. **Interpret SHAP by model and class.** A positive SHAP value pushes the model output toward the named class; a negative value pushes away from it. Direct-model SHAP values are on CatBoost's raw class-score scale. Stage 1 and Stage 2 SHAP values reconstruct class probabilities.
8. **Do not add Stage 1 and Stage 2 SHAP values.** The two-stage system uses hard routing, so its component explanations must remain separate. Explain Stage 2 only when Stage 1 routes the patient to any fall.
9. **Prefer grouped clinical contributions.** Use `grouped_shap_values` and `group_names` in each NPZ artifact. They combine one-hot and form-missing columns back into one clinical concept while preserving additivity.
10. **Avoid causal or prescriptive claims.** SHAP describes how a fitted model used recorded inputs. It does not show that a feature caused falls, that changing it will change risk, or that treatment should be altered.
11. **Do not report full-cohort fit as performance.** Use the repeated outer-evaluation results in the final report for performance. The SHAP artifacts explain models fitted on the full cohort.
12. **Communicate uncertainty.** Rare-fall recall remains limited, and paired evaluation did not establish one architecture as superior. Present the output as estimated risk support, not a diagnosis or certainty.

## Artifact map

- `direct_shap_artifact.npz`: direct three-class encoded and grouped SHAP arrays.
- `stage_1_shap_artifact.npz`: no-fall versus any-fall component.
- `stage_2_shap_artifact.npz`: rare-fall versus recurrent-fall component.
- `patient_explanation_index.csv`: row alignment, observed class, predictions, and probabilities.
- `patient_raw_inputs.csv`: the corresponding non-imputed 26-feature patient values.
- `clinical_feature_dictionary.csv`: clinical meaning, coding, aggregation, missingness, and final-model membership.
- `raw_input_schema.csv`: machine-readable raw-input contract.
- `feature_map.csv`: mapping from transformed model columns to clinical source groups.
- `shap_global_summary.csv` and PNG files: one complete direct all-class
  magnitude overview and five full-feature beeswarms. Each beeswarm includes
  every retained clinical group and names the fitted model and output class.
- `shap_handoff_manifest.json` and `shap_validation.csv`: provenance and validation.

## Prompt construction order

For a patient, first state the prediction horizon and model output. Then identify the largest class-specific grouped SHAP contributions, recover their raw values, and describe each using the feature dictionary's aggregation meaning. Mention unavailable measurements explicitly. End with the non-causal and uncertainty boundaries above.
