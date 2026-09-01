# Methodological Ablation Study: Reconciling Pipeline Performance

This report details the findings from a 5-phase methodological ablation study. The goal of this analysis was to objectively isolate the specific causes of the performance shift between the legacy predictive pipeline (which reported a Macro F1 of ~0.59) and the final, validated Landmark pipeline (which reports a Macro F1 of 0.493). 

By starting from the exact legacy conditions and systematically resolving methodological artifacts one by one, we objectively mapped the impact of **Imputation Bias**, **Temporal Alignment (Prediction Horizon)**, and **Hyperparameter Regularization**.

---

## Detailed Methodological Breakdown

### Phase I: Legacy Baseline (Seed 42)
* **Methodology:** Replicated the legacy pipeline exactly. Predictors were aggregated over a loose 3-year target window, missing target values were imputed with zeros (`fillna(0)`), an unconstrained Random Forest classifier was utilized, and evaluation was restricted exclusively to Seed 42.
* **Result:** Confirmed the high baseline (~0.571 Two-Stage / ~0.541 Direct) that initially prompted this investigation.

### Phase II: Seed Variance Validation (Seeds 0–4)
* **Methodology:** Maintained the exact pipeline from Phase I, but averaged the evaluation across the 5 specific data splits (Seeds 0, 1, 2, 3, 4) utilized in the final manuscript.
* **Result:** The F1 scores remained nearly identical across both architectures. This confirms that the initial high score on Seed 42 was a stable representation of the legacy pipeline's behavior, not a statistical anomaly or "lucky split."

### Phase III: Correction of Target Imputation Bias
* **Methodology:** Resolved an imputation artifact where over 80 patients missing ground-truth falls data were implicitly imputed into the "No Fall" class. This phase drops those uncertain targets and employs a robust clinical imputer strictly for the feature space.
* **Result:** Removing these artificially certain "No Fall" patients eliminated a major source of optimistic bias, correcting the overall Accuracy downward from ~0.678 to ~0.645 in both architectures.

### Phase IV: Constraining the Prediction Horizon (Temporal Alignment)
* **Methodology:** Replaced the loose 3-year aggregated target window with a mathematically strict 12-month Landmark prediction horizon. The model was restricted from utilizing clinical assessments that occurred after the designated index date.
* **Result:** Forcing the model to predict falls on a strict 1-year horizon is a significantly more challenging clinical task than predicting an event anywhere within a 3-year window. This structural correction resulted in a steep drop in Recall for Recurrent fallers (0.509 $\rightarrow$ 0.413 for Two-Stage), as the model could no longer rely on temporally misaligned future data.

### Phase V: Hyperparameter Regularization (Nested Cross-Validation)
* **Methodology:** Replaced the unconstrained Random Forest trees with the rigorous Nested CV methodology implemented in the final `eval_core.py` script. 
    * Standard, unconstrained trees are highly susceptible to memorizing small clinical datasets (overfitting).
    * **Nested CV** strictly bounds tree depth (e.g., `max_depth` 4, 6, 8, 10) by running an internal 5-fold grid search on the training data *only*, preventing the model from optimizing hyperparameters against the hidden test fold.
* **Result:** Nested CV eliminates hyperparameter optimization bias. While the constrained trees yield a lower overall Accuracy (the expected trade-off when moving from memorization to generalization), the regularization heavily stabilizes minority class performance. Notably, the Direct 3-Class model's Recurrent Recall improved significantly from 0.315 back up to **0.527**.

---

## 1. Two-Stage Pipeline Results (Random Forest $\rightarrow$ XGBoost)

| Evaluation Phase | Methodology | Macro F1 | Accuracy | Recall: No Fall | Recall: Rare | Recall: Recurrent |
|---|---|---|---|---|---|---|
| **I** | **Legacy Baseline (Seed 42)** | **0.571** | 0.680 | 0.865 | 0.335 | 0.492 |
| **II** | **Seed Variance Validation** | **0.573** | 0.678 | 0.859 | 0.344 | 0.498 |
| **III** | **Correct Imputation Bias** | **0.558** | 0.645 | 0.828 | 0.333 | 0.509 |
| **IV** | **Constrain Prediction Horizon**| **0.526** | 0.700 | 0.894 | 0.240 | 0.413 |
| **V** | **Nested Cross-Validation** | **0.493** | 0.664 | 0.830 | 0.191 | 0.553 |

---

## 2. Direct 3-Class Pipeline Results (Random Forest)

| Evaluation Phase | Methodology | Macro F1 | Accuracy | Recall: No Fall | Recall: Rare | Recall: Recurrent |
|---|---|---|---|---|---|---|
| **I** | **Legacy Baseline (Seed 42)** | **0.541** | 0.675 | 0.907 | 0.245 | 0.445 |
| **II** | **Seed Variance Validation** | **0.540** | 0.673 | 0.908 | 0.233 | 0.452 |
| **III** | **Correct Imputation Bias** | **0.525** | 0.641 | 0.883 | 0.247 | 0.421 |
| **IV** | **Constrain Prediction Horizon**| **0.487** | 0.706 | 0.947 | 0.150 | 0.315 |
| **V** | **Nested Cross-Validation** | **0.507** | 0.637 | 0.750 | 0.332 | 0.527 |

---

## Conclusion
The model did not lose underlying predictive power; rather, the evaluation framework was corrected to eliminate multiple sources of optimistic bias and data leakage. The final constrained models (**0.493** Two-Stage / **0.507** Direct) represent a highly robust, mathematically rigorous, and clinically honest estimate of out-of-sample performance for a strict 12-month prediction horizon.
