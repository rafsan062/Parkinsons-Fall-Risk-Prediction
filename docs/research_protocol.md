# Research Protocol: Parkinson's Disease Fall-Risk Prediction

**Status:** 26-feature manuscript run executed through Notebook 12; all saved validation checks passed
**Updated:** 25 September 2026

## 1. Objective

Develop and compare clinically interpretable models that classify Parkinson's disease patients into:

- 0: no fall;
- 1: rare fall;
- 2: recurrent fall.

The two architectures are a direct three-class classifier and a two-stage classifier: Stage 1 predicts no fall versus any fall, and Stage 2 separates rare from recurrent fall among fallers.

### Performance ambition and scope

The publication objective is the strongest defensibly validated predictive model achievable within the study's scientific constraints. The existing direct and two-stage implementations remain reference comparisons, but the current model families and feature list do not restrict the search. Other ML models and additional PPMI variables remain possible future research directions. The first expanded, partition-local feature/model search and internal outer evaluation are complete; no publication model has been selected.

Low rare-fall recall in the current results is a central motivation for this work. Macro F1 remains the primary model-selection metric, with rare-fall recall and the other class recalls explicitly assessed. Any recall target, alternative selection objective, or acceptable class tradeoff must be agreed and documented before the corresponding search; none is assumed here.

Performance improvements must be established through training-side validation and an unbiased final evaluation, preserving the landmark design and all leakage safeguards. Historical test results may motivate a new study, but repeated model development against a held-out test set compromises its role as unbiased evidence. Review evaluation independence before broadening the search, including whether a fresh holdout or an appropriately nested evaluation is needed.

When additional PPMI data are considered, verify field meanings, collection timing, cohort coverage, and applicable access, use, and publication requirements using official documentation. Explain the implications in plain language and record sources and unresolved questions. Additional variables must satisfy the predictor cutoff and cannot be outcomes or follow-up information.

## 2. Chronological project context

1. The legacy pipeline assembled predictors from PPMI sources, performed exploratory univariate tests, manually fixed a 21-feature list, filled missing values broadly with zero, and evaluated a two-stage model on one split.
2. The revision rebuilt the cohort with a landmark design so predictors occur at or before the index date, 12 months before the outcome visit. The outcome is never imputed.
3. Missing-data handling was revised to occur after splitting. PPMI sentinel `101 = unable to rate` was removed from Gait, Postural Stability, and Hoehn and Yahr score values and represented with binary assessment-status flags.
4. No-delta and delta feature sets were evaluated; the no-delta set was retained as the manuscript-primary analysis.
5. Two-stage and direct approaches were compared over 20 paired seeds. Direct classification had significantly better macro F1 after multiplicity correction and substantially better rare-fall recall. Balanced accuracy was subsequently added.
6. Seed-42 error analysis showed that most missed rare fallers were predicted as no fall rather than recurrent fall.
7. SHAP outputs were generated for both two-stage components and the direct three-class model.
8. The professor then required feature selection to be repeated because the source data changed. The audit of legacy work found no reproducible selection algorithm: the final list combined univariate tests, manual decisions, clinical judgment, and exploratory feature importances, with screening performed before the train/test split.
9. The active rebuild completed a partition-local feature-selection and model-search phase followed by internal repeated outer evaluation. Earlier revision model results remain historical comparators, not validation of the rebuilt procedure.

## 3. Locked data principles

- One analysis row per patient unless a later protocol amendment explicitly changes the unit of analysis.
- The fall outcome must have been observed; it is never filled or inferred from missingness.
- Predictors must respect the landmark cutoff and cannot use information from the outcome or follow-up window.
- When a field-specific PPMI codebook defines `101` as unable to rate, it is not a clinical score and must become missing before aggregation or modeling. This applies to the audited Part III gait, postural-stability, and Hoehn–Yahr fields and to six candidate Part I items. A measured vital-sign value of 101 remains an ordinary measurement. Retain an assessment-status flag only as explicitly specified.
- All learned preprocessing values come from training data only.
- Patient identifiers, target columns, raw fall counts, dates, and follow-up/history bookkeeping remain metadata.

### Current baseline decisions (15 September 2026)

- Reproduce revision cohort eligibility: `COHORT == 1` and status in `Enrolled`, `Complete`, or `Withdraw Deceased`. Select the latest dated, observed `FLNFR12M` per patient; stop for unresolved latest-month ties. This field means falls not related to freezing in the past 12 months and is an ordinal category, not an exact count. Map 0 to no fall, 1 to rare fall, and 2–4 to recurrent fall. Set the index month 12 calendar months before the outcome month. Final modeling eligibility still requires pre-landmark predictor assessment.
- The user approved revision-style maximum valid pre-landmark values per Part III feature across treatment states as a provisional baseline. Final ON/OFF representation remains open (D01 in `decision_register.md`). This does not authorize reuse of the BMI leakage path. Notebook 08 has implemented the provisional baseline and separate ON/OFF candidate representations.
- The candidate review must cover every non-outcome variable that Devi's preserved notebooks explicitly extracted, screened, engineered, or used, including variables omitted from the final 21-feature model. The repository does not document all discussions between Devi and Professor Lin, so omission from the final model is not treated as evidence that a variable was clinically rejected. Each recovered variable and each materially different representation will be classified in the candidate manifest. Outcomes, identifiers, post-index information, fall-consequence variables, and other target proxies remain ineligible as predictors even if they appeared in exploratory legacy code.
- Use two coordinated analysis tracks. The paper-aligned primary track uses the union of eligible predictors that Devi statistically screened and predictors used in the final 21-feature paper/revision model, followed by formal leakage-safe feature selection on the updated data, including the requested statistical tests. In parallel, an enhancement track may test a small, prespecified set of promising eligible additions and alternative representations from the manifest. Prioritize that set without using outcomes, based on clinical plausibility, pre-landmark coverage, deployability, and redundancy with existing inputs. All performance-based choices must use training-side validation only, using the locked primary metric and class-specific recalls; the outer test set cannot choose an addition or representation. Consistently better additions or replacements will be reported to Professors Li and Lin as proposed amendments rather than adopted silently. The paper-aligned results remain available even if the enhancement track is unsuccessful or not approved.
- Apply `INFODT <= index_date` to demographic, family-history, and diagnosis-history source forms. Later-only values remain missing; this rule alone does not remove a patient. Any required predictor imputation is fitted on training data only. Do not use complete-case deletion merely because a static candidate is unavailable.
- Construct family history from eligible records as `Yes` if any eligible `Yes` is present; otherwise use an eligible `No`; otherwise leave it missing. This resolves the 11 audited conflicts using their supporting relative-level history fields.
- For BMI, apply the landmark filter before any longitudinal consistency check. Use only the patient's eligible height/weight history to correct clear unit or entry errors, and set unresolved implausible records missing. Do not replace them with a cohort mean or delete the patient because of an invalid BMI record.
- Preserve `FEATPOSHYP=2` as the codebook-defined nominal category `Uncertain`. Do not convert it to missing, combine it with `No`, or treat the numeric code as an ordered severity above `Yes`. If modeled, encode it categorically using training-only preprocessing.
- Calculate a visit-level GDS-15 total only when all 15 items are recorded. Reverse-score the five positively worded items according to the instrument and sum the complete form; otherwise leave that visit's total missing. Do not treat a partial sum as a full-scale score.
- For Part III gait, postural stability, and Hoehn–Yahr, create a record-level binary indicator before converting a codebook-defined `101` score to missing. Do not exclude patients merely because an eligible record contains `101`: that rule would remove 325/1,280 outcome-eligible patients, and all 325 also have a valid eligible value for the affected field. Later compare two otherwise matched branches, one carrying the binary indicators into modeling and one treating 101 only as missing. Keep this comparison separate from the paper-aligned versus enhanced-feature comparison, and make performance-based choices using training-side validation only. Define patient-level flag aggregation with the final Part III representation.
- Aggregate the revision-compatible predictors as follows: compute age and disease duration at the landmark; use the most recent valid MoCA, complete GDS total, BMI, and Neuro-QoL composite; use historical maximum severity for freezing, SCOPA symptoms, Part I, Part III, and Part IV scores; use eligible-history ever-started for dopaminergic therapy; and use eligible-history ever-Yes for family history. Preserve the approved combined-state Part III maximum as the provisional baseline.
- Construct experimental representations without adopting them as final model inputs: use the most recent coherent Neuro-QoL assessment for its items and composites; historical maxima for individual Part I and Part IV items; state-specific historical maxima for Part III ON and OFF examinations; and one most recent coherent vital-sign visit for raw orthostatic measurements and within-visit standing-minus-supine changes. Aggregate the Part III unable-to-rate indicators as ever observed within the corresponding combined or state-specific history.
- For categorical clinical-history fields `FEATPOSHYP` and `FEATSUGRBD`, use eligible-history `Yes` if ever recorded; otherwise retain the most recent eligible `Uncertain` or `No`. Preserve `Uncertain` as a nominal category.
- The primary modeling cohort requires at least one recorded, landmark-eligible predictor from the paper-aligned candidate union. This retains 1,043 of 1,280 outcome-eligible patients. Keep the historical revision criterion—at least one longitudinal predictor from its final 21-feature set, retaining 1,040 patients—as a replication sensitivity analysis. Apply neither criterion using outcome class or model performance.
- Do not create or save one globally imputed analysis table. Fit every imputer, encoder, scaler, interaction transformation, and PCA transformation using the relevant training fold only, then apply it unchanged to validation or test data.
- Carry three prespecified preprocessing candidates into training-side comparison: a revision-compatible dense branch, an enhanced dense branch with explicit missing-state handling, and a native-missing boosting branch. The internal preprocessing-design Notebook 11 manifests define their exact defaults and exceptions; none was the selected final branch at that stage.
- For Part IV, distinguish a wholly absent six-item form from an isolated invalid or missing item. In the current 1,043-patient cohort, 228 patients have the whole form absent and one additional patient has an otherwise present form with an unable-to-rate item. The enhanced dense branch may encode structural zero only for the wholly absent form and must also retain a form-missing indicator; isolated gaps use a training-fold estimate. The native-missing branch compares native missing values with that same structural-zero representation.
- Prespecify the five interactions and five within-family PCA blocks recorded by internal preprocessing-design Notebook 11 as enhancement candidates. Generate interactions after fold-specific imputation/scaling. Fit PCA loadings and select variance thresholds (0.80, 0.90, or 0.95) within training data only. Preserve non-PCA paper-aligned results.

## 4. Feature-selection phase

Feature selection must be completed before rerunning the model comparison. The implementation must follow `.agents/skills/leakage-safe-feature-selection/SKILL.md`.

### Locked search design for Notebooks 12–17 (updated 16 September 2026)

- Compare the historical final 21 predictors, all 32 paper-aligned candidates, corrected fold-specific FDR screening of those 32 candidates, and controlled enhancement representations. The enhancement search covers Neuro-QoL items, Part I rater and patient items, Part IV items, measured orthostatic changes, separate Part III ON/OFF values, sex, and the recorded REM-sleep-behavior symptom. Race is reserved for subgroup/transportability analysis rather than used as a predictive enhancement in this initial search. Competing totals and items or combined and ON/OFF values must be represented as named alternatives; do not silently treat all 81 columns as independent, simultaneous measurements.
- No feature is forced into every selected subset. Clinical interpretability is preserved by reporting the full final-21 and paper-aligned reference models, the selected raw clinical features, and a non-PCA analysis.
- Compare four selection conventions: no supervised removal, corrected FDR at 0.05, L1-regularized model selection, and extremely randomized-tree importance selection. Calculate supervised selection anew for the direct target, Stage 1 target, and Stage 2 target using only the applicable training patients. Corrected statistical screening is one requested branch, not the sole admissible selection procedure. Preserve the all-candidate model as a sensitivity analysis.
- Search regularized logistic regression, random forest, extremely randomized trees, histogram gradient boosting, and XGBoost. Use small prespecified grids: logistic `C` in 0.1, 1, or 10; forest models with 300 trees, maximum depth unrestricted or 6, and minimum leaf size 1 or 5; histogram boosting with 15 or 31 leaves, learning rate 0.05 or 0.1, and L2 regularization 0 or 1; XGBoost with 300 trees, depth 2 or 4, learning rate 0.03 or 0.08, subsampling and column sampling 0.8, and L2 regularization 1 or 5. Compare unweighted and balanced class/sample weighting for every family. Notebook 12 may remove invalid or exactly redundant combinations before execution but may not expand the grid based on outer-test results.
- Use enhanced dense preprocessing for conventional dense models and native-missing preprocessing for compatible boosting models. Retain revision-compatible dense preprocessing as a historical sensitivity branch. The five prespecified interactions compete only in the enhancement track. PCA is a secondary enhancement branch, with variance thresholds 0.80, 0.90, or 0.95, and does not replace the non-PCA paper-aligned analysis.
- Choose configurations separately for direct classification, Stage 1, and Stage 2. Stage 2 selection uses only true fallers from the applicable training partition. Choose by mean inner-CV macro F1. If configurations are within 0.01 macro F1, use rare-fall recall for direct classification and Stage 2, and any-fall recall for Stage 1; then prefer the smaller feature set and simpler model in the prespecified order. This target-specific rule replaces the earlier generic binary-minority tie-break because recurrent fall is the Stage 2 minority class but rare-fall recall is the project’s stated clinical concern.
- Compare the Part III unable-to-rate indicators in a matched with-flags versus without-flags sensitivity after the base representation is defined. Keep this factorial comparison separate from candidate-feature enhancement.
- Report selected features and their frequencies across the 20 outer evaluations. Do not manufacture a consensus subset from outer-test performance. After the nested evaluation selects a procedure, refit that locked procedure on all 1,043 patients using five-fold internal selection; use random seed 42 for the final explanation/deployment artifact. Its apparent full-data fit is not an additional unbiased performance estimate.
- Use the staged execution plan saved by Notebook 12 within each outer-training set: Notebook 13 screens compatible representation/selector pairs with fixed balanced logistic and shallow Extra Trees references and retains the best two per target; Notebook 14 compares default versions of all five model families for those two plus the revision-21 all-feature, paper-32 all-feature, and paper-32 corrected-FDR references, then tunes the best two combinations over the locked family grid; Notebook 15 performs rare-faller diagnostics; Notebook 16 screens interactions and PCA; Notebook 17 will refit the locally chosen procedures on each outer-training fold and evaluate its outer test once. All pruning uses the saved inner folds and the locked tie rule.
- Pooled summaries across the 20 repeated outer-training partitions are descriptive only. A patient's outcome is in the training data of other outer partitions, so a pooled diagnostic must never define one candidate list for all outer evaluations. The focused `NP1CNST` addition and interaction/PCA candidates advance separately for each outer-training partition using only its own five inner folds: mean paired macro-F1 gain must be positive, at least three folds must improve, and mean rare-fall recall must not drop by more than 0.01. At most two interaction/PCA branches advance per target and partition. Notebook 14 then performs its full model-family comparison and tuning only for locally advanced branches. This correction precedes any outer predictions.
- The pooled rare-faller and interaction/PCA diagnostics remain useful for scientific interpretation but cannot establish unbiased performance. Use paired inner-validation addition/ablation comparisons and class-specific permutation importance as descriptive evidence. Model-specific impurity importance and SHAP, when available, are supplementary explanations rather than selectors because correlated predictors can exchange attribution.

### Completed expanded-search freeze (21 September 2026; historical context)

- The completed expanded search carried three main candidate universes: revision 21, paper-aligned 32, and paper 32 plus `NP1CNST`. These were candidate universes, with supervised selection refitted within the applicable training fold. D28 later replaced them with the single 26-feature manuscript universe described below; do not treat this historical freeze as the active final-run specification.
- Use combined-state Part III historical maxima as the primary representation for the current final run. Keep separate ON/OFF values as a named experimental alternative. Keep ever-101 indicators out of primary inputs and evaluate them only as a separate sensitivity.
- The five-family preflight scope was reopened before final model-family screening because it had not been explicitly reviewed with the user and omitted two historical revision families. The final family screen covers logistic regression, Linear SVC, RBF SVC, random forest, Extra Trees, histogram gradient boosting, XGBoost, LightGBM, and CatBoost. A dummy classifier is reported only as a benchmark and cannot win. This amendment was made before Notebook 29a and before any final-family results were generated; do not add or remove families after inspecting those results.
- Notebook 29a uses one fixed, documented starting configuration per family to screen the nine families on the partition-specific Notebook 28 candidate queue. Notebook 29b tunes only the two locally retained candidate/model combinations using prespecified grids. The existing grids remain fixed for logistic regression, random forest, Extra Trees, histogram gradient boosting, and XGBoost. Before Notebook 29b was run, the added-family grids were frozen as follows: Linear SVC uses `C` 0.1, 1, or 10 with unweighted or balanced classes; RBF SVC uses those `C` values, gamma `scale`, 0.01, or 0.1, and both weighting choices; LightGBM uses 300 trees, 15 or 31 leaves, learning rate 0.03 or 0.08, L2 regularization 1 or 5, and both weighting choices while holding the documented sampling settings fixed; CatBoost uses 300 or 600 iterations, depth 4 or 6, learning rate 0.03 or 0.08, L2 leaf regularization 5, and unweighted or automatically balanced classes. Families with no locally retained Notebook 29a combination remain documented but generate no Notebook 29b fits.
- Use dense fold-fitted preprocessing for logistic regression, both SVMs, random forest, and Extra Trees. Use the native-numeric-missing branch for compatible boosting families when the candidate representation does not intrinsically require dense standardized values. Interaction and PCA branches remain dense by construction. Fit the full preprocessing, engineering, selection, and model pipeline inside each inner-training fold.
- Notebook 28b freezes five one-change-at-a-time sensitivity branches before the final-family results: three combined-state Part III ever-101 flags; separate ON/OFF Part III values with state-availability indicators; the alternative cohort; revision-compatible dense preprocessing; and latest rather than maximum eligible constipation. Under D23 the 1,040-patient cohort is primary, so the cohort branch adds the three patients without longitudinal history (D25). Evaluate them only after the primary procedure is locked, reuse that procedure, and do not combine branches or use their results to redefine the primary model. Final Notebook 08 implements them by refitting the frozen Notebook 07 winners without retuning (D25).
- Fit final full-cohort artifacts for both direct and two-stage procedures because the internal evaluation did not establish a clear architecture advantage. Use seed 42 and five-fold internal selection, then refit the selected pipelines on all 1,040 primary patients (D23, D26). These full-cohort fits are not new performance estimates.
- The exact final sequence and machine-readable decisions are recorded by Notebook 21 under `results/final_pipeline/01_preflight/`.
- Notebook 24 audited final-feature missingness before imputation was implemented. Missingness percentage measures the reach of a rule but does not determine the fill value; instrument meaning distinguishes structural absence, unknown categories, unable-to-rate values, and ordinary measurement gaps. No globally imputed predictor table may be created.
- The Notebook 24 imputation rules are approved and recorded in `docs/final_run/final_imputation_rules.md`. Use explicit missing categories for unknown family and dopamine history; conditional Part IV structural zero only when the whole form is absent and therapy is recorded `No`; training-fold placeholders plus form indicators for missing freezing, Neuro-QoL, and nonstructural Part IV values; native numeric missingness where supported; and state-availability indicators for the separate ON/OFF sensitivity. Keep the declared source-plus-indicator families grouped during feature selection.
- Notebooks 25–27 implement the approved rules across the saved outer partitions and verify train-only imputation, encoding/scaling, interactions, and PCA without writing globally transformed patient data. These implementation checks do not select features, tune a model, or provide performance evidence; those tasks begin in Notebook 28 using the saved inner folds.

### Revision-aligned manuscript analysis freeze (25 September 2026)

The expanded Notebooks 21–29c analysis is complete and remains supporting
internal evidence. For the final manuscript-oriented run, use the historical
revision eligibility cohort of 1,040 patients as primary and the 1,043-patient
cohort as a sensitivity. Use one unimputed candidate table containing 25
paper-aligned variables plus maximum eligible `NP1CNST` (26). Neuro-QoL
mobility enters only as the revision's 8-item Gaussian score; its seven
individual items were removed as redundant duplicates after the initial
33-feature run (D28). The final workflow
does not rebuild the older 21- or 32-feature sets as separate modeling inputs;
they remain context in the completed exploratory results. Keep the no-delta design,
combined-state Part III maxima, and omission of 101 status flags as primary.

Evaluate direct and two-stage architectures on the same 20 stratified 70/30
holdouts generated with seeds 0–19. Within each outer-training set, use five
stratified inner folds for all imputation, encoding, scaling, interaction/PCA
fitting, supervised feature selection, model-family screening, and tuning.
Screen all nine frozen model families and tune only the two locally retained
feature-pipeline/model combinations per target and seed. Select by inner macro
F1 with the established 0.01 target-specific recall tie rule. Score each outer
test set once. The exact workflow and deliverables are locked in
`docs/final_run/final_run_plan.md`.

For this 26-source final run, feature engineering cannot reintroduce variables
from the broader exploratory manifest. Screen the four previously specified
interactions that are fully constructible from the 26 sources. Screen one
within-set PCA group—the three included Part I patient items (sleepiness,
urinary problems, and constipation)—at 0.80, 0.90, and 0.95 explained
variance. The Neuro-QoL PCA branches were removed with the seven raw mobility
items (D28). The earlier blood-pressure interaction and larger PCA groups remain
historical exploratory definitions because their required raw variables are
outside the locked candidate universe.

Retain the professor-requested statistical feature analysis as an explicit
part of the final workflow. Report outcome-group summaries, appropriate
univariate tests, effect sizes, raw p-values, within-partition FDR-adjusted
values, and stability across training partitions. When statistical screening
is used for prediction, recompute it inside the applicable training fold and
allow it to compete as the corrected-FDR selector branch; never use one pooled
or full-cohort significance list to filter every outer split. A conventional
full-cohort table may be produced only as a descriptive appendix after model
choices are frozen and cannot change the pipeline.

This final run was designed after extensive exploration on the same cohort.
Its repeated holdout estimates are internal validation and require independent
confirmation before a claim of external generalization.

## 5. Completed expanded-search validation framework

The framework below describes the completed repeated-fold analyses through
Notebook 29c. It remains part of the scientific record but does not replace
the revision-aligned final-run amendment above.

- Use five stratified patient-level outer folds repeated across four fixed shuffles (`random_state` 0, 1, 2, and 3), yielding 20 outer evaluations. Within every outer-training partition, use five stratified inner folds with the deterministic seeds saved by Notebook 09.
- Reuse the saved patient assignments for every feature set, direct-versus-two-stage architecture, preprocessing branch, and 101-indicator branch. Each patient is in one outer test fold per repeat and never appears in the corresponding inner folds.
- Fit supervised selection, imputation, scaling, and hyperparameter tuning inside the outer training data.
- Use stratified inner cross-validation and macro F1 for selection unless an amendment is documented before execution.
- Evaluate direct and two-stage approaches on the same paired outer splits.
- Primary model-selection metric: macro F1.
- Report at minimum: macro F1, weighted F1, accuracy, balanced accuracy, macro one-vs-rest AUC, and recall for each class.
- Report paired direct-versus-two-stage differences with approximate corrected resampled *t* intervals using the repeated-fold test/train ratio, alongside descriptive fold and repeat summaries. Treat these intervals as exploratory: patient overlap and prior human-guided development on this cohort limit their interpretation. Do not use them to change the locked candidate procedure after outer evaluation.
- Do not treat the 20 fold results as independent observations when estimating uncertainty or testing paired differences; the folds and repeats reuse patients.

## 6. Outputs required from feature selection

- Candidate-feature manifest with clinical meaning and eligibility rationale.
- Selected-feature list for every outer split and, when applicable, every stage.
- Selection frequency/stability table across splits.
- Inner-CV score and selector/model parameters.
- Held-out predictions and all locked metrics.
- Comparison with the all-eligible-features model as a sensitivity analysis.
- Final seed-42 feature list and artifacts for the explanation layer.

Write generated outputs under `results/feature_selection/`. New analysis work belongs in Jupyter notebooks under `notebooks/`, with readable code, explanatory Markdown, and visible intermediate results. Keep cells concise and logically ordered, and ensure the notebooks run top to bottom from a fresh kernel. Reusable functions may live in notebook cells. Propose `.py` modules under `src/` only when they offer a concrete advantage, and obtain the user's approval before creating them or moving logic into them. This notebook preference supersedes local skill guidance that otherwise directs reusable logic into `src/`.

## 7. Historical material

- `archive/legacy_project/` contains the former working project, including `revision_work/` and the detailed `legacy_feature_selection_review.md`.
- `github_repo_export/` is the existing Git repository and LLM-layer handoff. It retains its history and current working changes.
- Existing statistical, rare-fall, and SHAP reports describe the pre-feature-selection pipeline and must be labeled historical if cited.
