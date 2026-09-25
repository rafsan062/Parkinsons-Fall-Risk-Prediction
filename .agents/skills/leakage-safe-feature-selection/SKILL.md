---
name: leakage-safe-feature-selection
description: Design, implement, audit, or rerun feature selection for the PD fall-risk models while preventing test leakage and preserving paired direct-versus-two-stage evaluation.
---

# Leakage-Safe Feature Selection

Read `docs/research_protocol.md` before acting. Treat its locked decisions as authoritative and its open feature-selection questions as unresolved; do not silently decide them from test performance.

## Required invariants

- Split by patient before any target-aware screening or learned preprocessing.
- Never use an outer test fold to select features, choose a threshold, tune a model, impute values, or resolve ties.
- In repeated outer CV, do not pool outcome-guided evidence from all outer-training partitions to select a common feature or representation for evaluation. A patient held out in one partition contributes outcomes to other partitions. Apply any advancement rule using only the corresponding partition's inner folds; pooled selection frequencies are descriptive.
- Put supervised feature selection inside the inner cross-validation workflow fitted on the outer training fold.
- Fit imputation and scaling within the same training-only pipeline when the selected method requires them.
- Keep the target, raw fall measures, identifiers, dates, outcome-window variables, and post-index information out of the candidate matrix.
- Recode `101 = unable to rate` before selection. A corresponding binary flag is a separate candidate, not a replacement numeric score.
- Use macro F1 for selection and report balanced accuracy plus class-specific recalls. Do not choose a subset from test accuracy or SHAP values.
- Direct, Stage 1, and Stage 2 targets differ. Do not reuse target-derived rankings between them unless the protocol explicitly requires a common prespecified subset.

## Before implementation

Confirm that `docs/research_protocol.md` locks the candidate universe, clinical mandatory features, selection method, stage-specific versus common subset policy, tuning grid, stability rule, outer seeds, and final-fit convention. If a material item remains open, analyze options without running outcome-guided selection and ask the user to lock the decision.

Create a candidate manifest that records source variable, timing/aggregation rule, data type, missingness handling, clinical rationale, and exclusion reason. Remove only ineligible predictors—leakage, identifiers, target proxies, impossible deployment variables, or exact duplicates—before supervised selection.

## Evaluation workflow

For every outer split:

1. Reserve the outer test patients untouched.
2. On outer training patients only, run the complete imputation/selection/model pipeline using stratified inner cross-validation.
3. Choose selector and model settings by inner-CV macro F1.
4. Refit the chosen pipeline on the full outer training fold.
5. Evaluate once on the untouched outer test fold.
6. Save selected names, selection scores, model parameters, patient IDs, predictions, probabilities, and metrics.

Use the same outer patient splits for direct and two-stage comparisons. Stage 2 feature selection uses only true fallers from the outer training fold. End-to-end two-stage evaluation still uses Stage 1 routing on the full outer test fold.

## Required reporting

Write analysis outputs to `results/feature_selection/` and include:

- per-split and per-stage selected-feature lists;
- selection frequency and stability summaries;
- performance with selection versus all eligible features;
- paired architecture metrics on identical outer splits;
- class-specific confusion matrices and rare/recurrent recall;
- the final seed-42 subset and reproducibility manifest.

If selected subsets vary, report that instability rather than constructing a consensus subset using outer test results. SHAP and LLM artifacts must be regenerated only after the final feature-selection convention and final model are locked.

## Archive boundary

Treat `archive/` as read-only evidence. Copy reusable logic into `src/`, add tests, and record provenance. Do not run or modify archived notebooks as the active analysis.
