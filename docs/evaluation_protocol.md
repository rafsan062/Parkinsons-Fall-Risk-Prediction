# Evaluation and model-selection protocol

**Status:** locked for Phase 3 (selection metric + legacy alignment). Do not change the selection metric after seeing numbers.

**Implements:** reviewer Comments 5 (inconsistent selection) and 6 (one split, no CI, no test of two-stage vs direct).

---

## The rule (edit here if you disagree)

**Selection metric at every stage: `f1_macro`.**

| Step | What wins | Metric |
|------|-----------|--------|
| Tune hyperparameters (`GridSearchCV`) | best params | inner-CV **`f1_macro`** |
| Choose among **primary** candidates | best algorithm | inner-CV **`f1_macro`** (`gs.best_score_`) |
| Claim two-stage beats direct 3-class | the architecture | **test** **`f1_macro`** on the 3-class label (paired across **5** seeds; 20 later if needed) |

The same string, `f1_macro`, is used for Stage 1 (none vs any fall), Stage 2 (rare vs recurrent), and the direct 3-class model.

**Never used to choose a model:** test-set accuracy, weighted F1, AUC, or “best on any column of the comparison table.” Those are **reported**, not selection criteria.

---

## 1. What we are trying to do clinically

The tool has two jobs, in order:

1. **Stage 1** — Is this patient likely to fall at all in the next 12 months? (none vs any)
2. **Stage 2** — If they fall, is it rare or **recurrent**? Recurrent is the group that most needs closer follow-up.

The submitted paper is a **3-class** problem (no fall / rare / recurrent). Classes are unbalanced (landmark: 712 / 227 / 101). Missing a recurrent faller is worse clinically than missing a non-faller, but a model that calls everyone “no fall” is useless and would look good on accuracy.

So the selection metric must:

- treat the minority class as **real**, not noise (Comment 5: accuracy / weighted F1 favor the majority);
- be **the same rule** at Stage 1 and Stage 2 (Comment 5: “optimized for macro F1” then picked on accuracy);
- be a **standard** sklearn / ML reporting choice, not a custom score invented after the table was printed.

---

## 2. Why macro F1 (and not the other usual options)

**Macro F1** = unweighted mean of per-class F1. For a binary stage that is `(F1_class0 + F1_class1) / 2`. For the 3-class headline it is the mean of the three class F1s.

| Metric | What it rewards | Keep as selection? |
|--------|-----------------|--------------------|
| **Accuracy** | Overall % correct. With 68% no-fall, a “always 0” model scores ~0.68. | **No.** This is what the pipeline actually used (`Accuracy.idxmax()`). Favors the majority. |
| **Weighted F1** | F1 weighted by class size. Almost accuracy. | **No.** Same bias. |
| **AUC** | Ranking, not the decision threshold we deploy. Two models can have similar AUC and very different F1 at the default 0.5 cutoff. | Report only. |
| **Recall of the high-risk class only** | Stage 1: catch fallers. Stage 2: catch recurrent. Matches “don’t miss risk.” | Tempting, but a model can recall 1.0 by labeling everyone high-risk. F1 penalizes that. Also Stage 1 and Stage 2 would be “different positive classes” unless we write extra prose. |
| **Balanced accuracy** (macro recall) | Mean of per-class recall. Standard, imbalance-aware. Ignores precision (false alarms). | Close cousin. We are **not** using it, so we do not shop after the fact. |
| **Macro F1** | Each class’s precision–recall tradeoff, then unweighted average. Standard for imbalanced classification. Already what the paper *said* and what `GridSearchCV` used. | **Yes.** |

For binary Stage 1 / Stage 2, macro F1 still cares about the minority: if Stage 2 ignores recurrent fallers, F1_recurrent collapses and so does the average. It is not as “catch every recurrent faller” as positive-class F1, but it is the metric the manuscript already claimed, it is one sklearn string at every stage, and it does not favor the majority the way accuracy does.

**If we instead wanted positive-class F1** (Stage 1: F1 of “any fall”; Stage 2: F1 of “recurrent”): that is also standard (`sklearn` `f1` with an explicit `pos_label`). It is *more* aligned with “detect the clinical event.” It is a **change from what the paper wrote**. Only switch to it in this file *before* running Phase 3, and say so in the response. Do not mix: do not tune on macro F1 and pick on positive F1.

---

## 3. Same rule at both stages

Both stages are **binary**. The direct comparator is **3-class**. Macro F1 is defined for both (mean of 2 class F1s vs mean of 3). That is the consistency Comment 5 asked for.

| Model | Label | `f1_macro` means |
|-------|--------|------------------|
| Stage 1 | 0 = no fall, 1 = any fall | mean(F1_none, F1_any) |
| Stage 2 | 0 = rare, 1 = recurrent (mapped from `falls_class` 1 vs 2) | mean(F1_rare, F1_recurrent) |
| Direct | 0 / 1 / 2 | mean of the three class F1s |
| Two-stage **end-to-end** | 0 / 1 / 2 on the full test set | same 3-class macro F1 as direct (so they are comparable) |

Stage 2 is trained only on **true** fallers in the training set. On test, patients are routed by Stage 1; end-to-end 3-class F1 includes Stage 1 errors (fallers sent to “no fall,” or non-fallers sent to Stage 2). That is the number that is compared to direct classification.

---

## 4. Where the number is allowed to come from

This is the other half of “post hoc.”

1. Split train / test (**stratified**, 70/30). Test is frozen.
2. On **train only**, inner stratified CV (`GridSearchCV`, same k as the paper — 5-fold unless a class is too thin, then 3).
3. Each candidate family gets a `best_score_` = CV macro F1.
4. The winner is `argmax(best_score_)`. **Not** the model with the best *test* row.
5. Refit that winner on full train; score **test** once, for reporting.
6. Repeat for **5** seeds (`random_state = 0 … 4`). The algorithm may differ by seed (RF Stage 1 on seed 3, LightGBM on seed 7). That is honest. Do **not** then take the algorithm that won the most *test* macro F1 and call it “the” model.

**Optional extension (not the first pass):** seeds `5 … 19` (20 total) if the professor wants a tighter CI / more powered Wilcoxon. Same code, `resume=True`. Do not start that extension after peeking at the 5-seed table and disliking it — decide from compute/reviewer needs, not from the point estimate.

The submitted bug was step 4 using **test accuracy** after step 2 had already used macro F1.

---

## 5. Robust comparison (Comment 6)

- **Manuscript-primary file:** `modelling_landmark_trees.csv` (no deltas). Same clinical predictors as the paper, plus the imputation flags the revision already requires (`dopamine_missing`, three `*_missing_due_to_101`). This is the model the Methods section can still describe as the original feature family.
- **Parallel file:** `modelling_landmark_with_deltas_trees.csv`. Same patients, same **5** seeds, same selection rule. Professor-requested velocity features. Reported as a paired sensitivity, not as a silent swap of the paper model. See §8.
- **Fills:** sklearn models → `apply_dense_imputation(train, test)` after the split. XGBoost / LightGBM leave NaNs.
- **Seeds:** **headline `0 … 4` (5 paired splits).** Same indices for two-stage and direct **and** for the no-delta vs with-delta comparison. Optional later: continue `5 … 19` (20 total) if the professor wants more; Wilcoxon on n=5 is underpowered.
- **Candidates (primary):** LogReg, LinearSVC, RandomForest, XGBoost, LightGBM (if import works). Same list as the paper. This is the only set that can win Stage 1 / Stage 2 / direct.
- **Candidates (exploratory):** see §5.1. Screened; cannot replace a primary winner after looking at test metrics.
- **Report for every system:** mean ± SD, median, 95% CI of test macro F1, accuracy, weighted F1, macro AUC, **per-class recall** (especially recurrent).
- **Two-stage vs direct:** mean difference with 95% *t* CI on the **5** paired **test macro F1** scores, plus Wilcoxon (underpowered at n=5). If the CI includes 0, do not claim an improvement.
- **External validation:** none on this extract. Do not relabel a random holdout as external. Optional later: time-based split (earlier vs later `index_date`) as sensitivity, stated as such.

---

## 5.1 Extra models (exploratory screen — pre-specified)

Worth running. **Not** worth hiding.

Comment 5 is exactly “we looked at several numbers and kept the convenient one.” If we try CatBoost / ExtraTrees / etc. and only mention them when they look good, we repeat that. The allowed version:

- The **primary** horse race stays the paper’s five families (plus Dummy as a baseline).
- Extra models are named **here**, before Phase 3.
- They are scored with the **same** inner-CV `f1_macro` rule.
- We **always** record the result (notebook + a small supplement table or one sentence). “No material gain” is a finding. Silence is not.
- An extra model **cannot** become the reported Stage 1/2 learner because it won on the **test** set. Promotion only if it beats the primary winner on **inner-CV macro F1** by ≥ 0.02 on the **same seed**, and that rule is not changed later. Even then it is labeled as a sensitivity, not a silent swap.

**Baseline (always):** `DummyClassifier(strategy="most_frequent")` — shows why accuracy is a bad selection metric.

**Extra trees (if they import):** ExtraTrees, `HistGradientBoosting` (sklearn; NaN-capable), CatBoost.

**Extra non-trees:** GaussianNB. That is enough. We already have LogReg and LinearSVC. Skip MLP, RBF-SVM, k-NN — n is small and they are not standard fall-risk comparators.

**Compute:** full **5** seeds (`0…4`) for **primary + Dummy**. Exploratory families: same 5 seeds. If none is within 0.02 CV-macro-F1 of that seed’s primary winner, stop and write that. Optional 20-seed extension (`0…19`) is only for the primary families, if requested.

---

## 6. Checklist before a Phase 3 commit

- [ ] Selection metric is `f1_macro` in GridSearch **and** in `argmax` across models.
- [ ] No `results_df["Accuracy"].idxmax()` (or weighted F1 / AUC) anywhere in the winner line.
- [ ] Stage 1, Stage 2, and direct 3-class use that same line of code.
- [ ] Test set is not used to choose hyperparameters or the algorithm.
- [ ] Two-stage vs direct is paired on the same **5** splits; headline is 3-class macro F1. 20 seeds remain an optional extension.
- [ ] Exploratory extras (if run) are the pre-specified list; supplement records all of them, including nulls.
- [ ] Grids, class weights, scaler pipelines, and two-stage routing match `Revised_Modelling/Revised_Model_Results.ipynb` except the rows in §8 marked **change**.
- [ ] No-delta and with-delta runs share `eval_core.py` (do not fork grids).
- [ ] SHAP is computed only after the final pipeline is chosen, via `explanation.export.export_explanation_artifacts`.

---

## 7. Manuscript one-liner (after you sign this off)

> Model hyperparameters and the choice among candidate learners were selected by stratified inner-loop cross-validated macro-F1, the same criterion at both stages and for the direct three-class comparator. Test-set accuracy and weighted F1 were reported but not used for selection. Two-stage versus direct classification was compared on 5 paired stratified splits using test-set macro-F1 (mean, 95% CI). A 20-split extension is available if a tighter interval is requested.

---

## 8. Follow the legacy modelling code (keep vs change)

Source of truth for *how* we train: `Revised_Modelling/Revised_Model_Results.ipynb` (same families, grids, and two-stage routing as `Model_Development.ipynb` cells 47–57). Do not invent a new architecture. Change only what Comments 1 / 3 / 5 / 6 and the landmark extract already force.

### Keep (copy, do not “improve”)

| Piece | Legacy | Why |
|-------|--------|-----|
| Outer split | 70/30, `stratify=Y` | Paper design. **5** seeds (`0…4`) instead of one `random_state=42`; 20 optional. |
| Inner CV | `StratifiedKFold(n_splits=5, shuffle=True)` | Same k. Inner `random_state` = the outer seed (deterministic per split). |
| Five families | LogReg, LinearSVC, RF, XGBoost, LightGBM | Comment 5 is about *selection*, not a new horse race. |
| LogReg / LinearSVC | `StandardScaler` pipeline; LinearSVC inside `CalibratedClassifierCV(sigmoid, cv=5)` | Needed for `predict_proba` and SHAP-ready scores. |
| Stage 1 imbalance | LogReg/SVC `class_weight="balanced"`; RF `{0:1, 1:2}`; XGB `scale_pos_weight=2`; LGBM `{0:1, 1:2}` | Copy, do not retune. |
| Stage 2 imbalance | LogReg/SVC `balanced`; RF `class_weight="balanced"`; XGB **no** `scale_pos_weight` (legacy omitted it); LGBM `balanced` | Copy, including the XGB asymmetry. |
| Hyperparameter grids | Exact lists in that notebook (Stage 1 RF has `max_features`; Stage 2 RF does not; Stage 2 XGB `max_depth` is `[2,3,4]` not `[3,5,7]`) | Do not enlarge the grid after seeing numbers. |
| Stage labels | S1: `(Y != 0)`; S2: train only on **true** `{1,2}` in **this split’s train**, map `1→0`, `2→1` | Same clinical two-stage. |
| End-to-end routing | S1 predicts; routed (`pred==1`) go to S2; final `0` / (`s2==0→1`) / (`s2==1→2`) | Same mapping the LLM layer already assumes. |
| Reported metrics | accuracy, macro F1, weighted F1, macro AUC, classification report, confusion matrices | Still **report** them. Stop using accuracy to **choose**. |
| SHAP producer | `explanation.export.export_explanation_artifacts` | Do not rewrite TreeSHAP / artifact filenames. Nadine’s consumer is the contract. |

### Change (required; say so in the response letter)

| Piece | Legacy | Revision | Why |
|-------|--------|----------|-----|
| Data | `Modelling.csv`, latest/max across all visits | Landmark tree CSVs | Comment 1. |
| FOG column | `Freezing of gait in past 36 months` | `Freezing of gait (peak severity)` | Same construct, peak-in-window under the landmark. |
| Extra columns | none | `dopamine_missing`, `NP3GAIT/NP3PSTBL/NHY_missing_due_to_101` | Comment 3 + 101 cleanup. SEX / RACE / dates / `history_months` stay **meta** (see §11). |
| Imputation | cohort `fillna(0)` before split | split first; sklearn `apply_dense_imputation`; trees keep NaN | Comment 3. |
| Winner line | `results_df["Accuracy"].idxmax()` | `argmax(gs.best_score_)` = inner-CV `f1_macro` | Comment 5. |
| Stage 2 comparison split | a **second** 70/30 of all fallers, independent of Stage 1 | fallers from the **same** outer train fold | The independent S2 split is not the pipeline’s train set; copying it would keep a second selection leak. |
| Robustness | one seed (42) | **5** paired seeds (`0…4`); 20 if requested | Comment 6. |
| Direct 3-class | not compared | same families, same seeds, same `f1_macro` selection | Comment 6. |
| Dummy | none | `most_frequent` | Shows why accuracy is a bad selector. |

Do **not** copy: expanding grids, promoting meta columns (SEX / RACE / dates / `history_months`) into `X`, MLP/RBF-SVM/k-NN, or picking the “lucky” seed for the paper table.

---

## 9. Two feature sets, two notebooks, one shared module

Separate notebooks are the right idea **if they cannot drift**. Duplicate GridSearch cells will fork (Comment 5 again). Structure:

| File | Role |
|------|------|
| `revision_work/eval_core.py` | Candidates, grids, class weights, split + (sklearn) impute, tune-on-CV-macro-F1, two-stage predict, metrics. Copied from the legacy notebook, not rewritten. |
| `revision_work/03_eval_no_deltas.ipynb` | **Manuscript-primary.** Loads `modelling_landmark_trees.csv`. 20-seed two-stage vs direct. |
| `revision_work/03b_eval_with_deltas.ipynb` | **Parallel sensitivity.** Loads `modelling_landmark_with_deltas_trees.csv`. Same seeds, same `eval_core`. |
| `revision_work/04_shap_export.ipynb` | After a winner exists. Fits that pipeline on the **canonical explanation split** and calls `export_explanation_artifacts`. |

Both eval notebooks are thin: set `CSV`, `OUT_DIR`, `USE_DELTAS`, then call the same functions. Results go to `revision_work/results/no_deltas/` and `revision_work/results/with_deltas/`.

**How deltas can become the paper model (pre-specified, not after peeking):** only if mean **test** 3-class macro F1 (two-stage, **5** seeds) is ≥ 0.02 higher than no-deltas **and** the paired Wilcoxon two-sided p < 0.05. Wilcoxon on n=5 is weak — if the mean gain is ≥ 0.02 but p is not < 0.05, do not promote; say so and offer the 20-seed extension. Otherwise the paper leads with no-deltas and deltas stay a one-paragraph / supplement result. Do not promote because Stage 1 CV looked nicer, or because seed 42 liked deltas.

---

## 10. SHAP after the final model (LLM layer)

The explanation package does **not** train models. It reads artifacts produced by `explanation.export.export_explanation_artifacts` (`shap_values_stage1.npy`, `shap_values_stage2.npy`, `X_test.csv`, `y_pred_proba.npy`, `patient_ids_test.csv`, `routing_mask.npy`, `y_pred_s2.npy`, `y_pred_final.npy`). That cell already exists at the end of `Revised_Model_Results.ipynb`. Reuse it.

**When:** after §5 and §9, not during the 5-seed loop. SHAP on every seed is unnecessary for the LLM and would not match one patient-facing report.

**Which split (canonical, not the lucky one):** `test_size=0.3`, `random_state=42`, `stratify=falls_class` — the paper’s split settings — on the **chosen** feature file. Report 20-seed CIs in the manuscript; explanations use seed 42 so they are reproducible and not cherry-picked. Inner CV / model `random_state` also 42 on that fit.

**Which models:** the Stage 1 and Stage 2 families that won by inner-CV `f1_macro` **on this canonical split** (same rule as every other seed). Refit on that train set. Do not take “the algorithm that won the most of the 20 test scores.”

**Background:** `X_train` for Stage 1, `X_train_12` for Stage 2 (legacy export). Sklearn models must be explained on the **imputed** matrices they were trained on; trees on the NaN-keeping matrices. Column order of background and `X_test` must match (`compute_shap` already checks).

**`feature_map.csv` must list exactly the modelled columns** (`export._validate` requires set equality). Before export, add rows that the paper map is missing:

- `dopamine_missing`
- `NP3GAIT_missing_due_to_101`, `NP3PSTBL_missing_due_to_101`, `NHY_missing_due_to_101`

The ten `Delta *` rows are already in the map. If the no-delta model wins, drop those rows from the copy used for export (or keep a `feature_map_no_deltas.csv` / `feature_map_with_deltas.csv`). Do not leave extra map rows; validation will fail.

**Write to** `revision_work/explanation_artifacts/` first. Do not overwrite the project-root `explanation_artifacts/` until Nadine is ready to point the LLM at the new files.

**Do not:** Kernel-SHAP a 20-seed ensemble; explain a model trained on the full 1040 (no held-out `X_test`); or hand-write SHAP in the modelling notebook. Imputed values in the LLM prose are Nadine’s later pass (flag or hide); export still writes the numeric matrix the model saw.

---

## 11. Last locks before the first fit

These have no legacy cell (the paper had no direct 3-class horse race). Write them here so we do not invent them after a smoke-test score.

**What goes into `X` (predictors) vs what stays in the CSV as meta**

The tree CSVs include `SEX`, `RACE`, `index_date`, `outcome_date`, `first_visit_date`, `history_months`, `n_falls_visits`, `PATNO`, `falls_raw`. That is **not** the paper’s feature list. The published model’s `cols` (same list in `Revised_Model_Results.ipynb` and `Model_Development.ipynb`) were the 21 clinical scores below. SEX and RACE were extracted into the merged table, then **never passed to `train_test_split` as `X`**. Dates and `history_months` did not exist in the old `Modelling.csv`; we added them for the landmark design and for later history-length tables (Devi / professor), not as predictors.

So this is not “legacy dropped demographics.” It is “legacy never trained on them; do not start now.” Putting SEX / RACE into `X` would be a new analysis (and PPMI race is ~90% White, so it would mostly be a sparse dummy). `history_months` is collinear with how much PPMI follow-up exists and is a **stratum**, not a risk factor we would deploy on a new patient.

- **Predictors, no-deltas (03):** the 21 clinical columns, FOG renamed to `Freezing of gait (peak severity)`: `No_of_years`, `Age`, `Postural instability present at dx?`, `Rigidity present at diagnosis?`, `Dopaminergic therapy started for participant`, `MoCA Total Score`, `Freezing of gait (peak severity)`, `lightheaded after standing`, `fainted`, `Total Depression Score`, `able_weighted_score`, `MDS-UPDRS Part I Score`, `BMI`, `Daytime_Sleepiness`, `Urinary_Problems`, `Gait`, `Postural_Stability`, `Hoehn_And_Yahr_Stage`, `Postural_hypotension`, `MDS-UPDRS Part III Score`, `MDS-UPDRS PartIV score`
- **Plus revision flags (always in `X`):** `dopamine_missing`, `NP3GAIT_missing_due_to_101`, `NP3PSTBL_missing_due_to_101`, `NHY_missing_due_to_101`
- **Plus deltas (03b `X` only):** the ten `Delta *` columns
- **Meta only (split / tables / IDs, never `X`):** `PATNO`, dates, `history_months`, `n_falls_visits`, `SEX`, `RACE`, `falls_raw`. Target = `falls_class`.

**Which matrix each family sees:** sklearn LogReg / LinearSVC / RF (and Dummy) → `apply_dense_imputation` after the split. XGBoost / LightGBM → tree CSV NaNs as-is. Sklearn `RandomForestClassifier` cannot take NaN; it is not “a tree” for this purpose.

**Direct 3-class (no paper grid to copy):**

- Same five families. Hyperparameter grids = **Stage 1** grids (RF `max_features`, XGB `max_depth` `[3,5,7]`).
- LogReg / LinearSVC / RF / LightGBM: `class_weight="balanced"`.
- Logistic Regression grid: same `C` values as Stage 1, but **`lbfgs` only**. Current sklearn refuses `liblinear` when `n_classes >= 3`; Stage 1 / Stage 2 stay binary and keep the paper `{lbfgs, liblinear}` grid.
- XGBoost: **no** `scale_pos_weight` (that flag is binary). Do not invent per-class `sample_weight` after seeing numbers.
- Dummy stays `most_frequent`.

**Reporting maths (not selection):**

- 95% CI on a 5-seed mean: Student’s *t* interval (`scipy.stats.t.interval`, df = 4) on the 5 test scores. If the 20-seed extension is run, df = 19.
- 3-class AUC: `roc_auc_score(..., multi_class="ovr", average="macro")` on `predict_proba`. Binary stages: `roc_auc_score(y, proba[:, 1])`.

**Code, not a second copy of the rules:** `apply_dense_imputation` lives in `02_imputation.ipynb` today. Copy it into `eval_core.py` (same function, same constants). Do not `%run` the notebook.

**First pass vs extras:** run primary five + Dummy for **5** seeds on both CSVs first. Optional: `--seeds 0-19` later if the professor wants a tighter CI. Exploratory ExtraTrees / HistGB / CatBoost / GaussianNB are a later cell on the same 5 seeds, not a reason to delay the headline comparison.

**Pre-flight:** LightGBM must actually import. If it does not, install it **before** seed 0. Do not drop it after a partial table exists. CatBoost / SHAP are not required for notebooks 03 / 03b.
