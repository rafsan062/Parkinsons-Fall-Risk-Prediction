# Modeling Results Report: PD Fall-Risk Paper Revision

## 1. Executive Summary

This report summarizes the findings from the revised modeling pipeline (Phase 3) for the multi-class fall risk stratification project. The pipeline was completely overhauled to address methodological flaws (data leakage, improper imputation, and flawed model selection) present in the original submitted manuscript. 

**Key Takeaways:**
*   **Performance:** The strict, temporally correct design yields a Two-Stage test **Macro F1 of 0.493** (95% CI: 0.476–0.509) across 5 random seeds. While lower than the ~0.60+ reported in the original paper, this reflects a robust and honest evaluation of the model's predictive power without future data leakage. The baseline dummy model (most frequent) achieves 0.271, demonstrating that the task is still learnable.
*   **Two-Stage vs. Direct:** The Two-Stage architecture does **not** significantly outperform a Direct 3-class classifier on this extract. 
*   **Delta Features:** Adding change-over-time (delta) features provided only a marginal boost (+0.010 Macro F1) and failed to meet our pre-specified threshold for promotion to the primary model.
*   **Winning Algorithm:** Random Forest consistently dominated model selection across all stages and seeds.

---

## 2. Model Performance (Primary Results)

Evaluation was conducted over 5 paired seeds (0–4) using **CV Macro F1** for model selection. The primary model relies strictly on clinical features without deltas (`modelling_landmark_trees.csv`), maintaining a cohort of 1040 patients.

Below is the comprehensive evaluation of the primary No-Deltas dataset across all evaluated metrics:

### Detailed Metrics Comparison (No-Deltas)
| Metric | Two-Stage (Mean ± SD) | Direct 3-Class (Mean ± SD) |
| :--- | :--- | :--- |
| **Macro F1** | **0.493 ± 0.013** | **0.507 ± 0.013** |
| **Weighted F1** | 0.648 ± 0.013 | 0.649 ± 0.020 |
| **Accuracy** | 0.664 ± 0.018 | 0.637 ± 0.023 |
| **Recall: No Fall (Class 0)** | 0.830 ± 0.047 | 0.750 ± 0.036 |
| **Recall: Rare (Class 1)** | 0.191 ± 0.037 | 0.332 ± 0.029 |
| **Recall: Recurrent (Class 2)** | 0.553 ± 0.128 | 0.527 ± 0.060 |
| **AUC OVR (Macro)** | 0.739 ± 0.017 | 0.750 ± 0.012 |
*(Dummy Baseline Macro F1: 0.271)*

### Two-Stage vs. Direct Classifier
The mean difference in Macro F1 between the Two-Stage and Direct architectures is **-0.015** (Wilcoxon $p=0.3125$). Because the confidence interval of the difference includes zero, we **cannot claim the Two-Stage approach is superior** to the Direct 3-class approach on this specific dataset. While the Two-Stage approach improves accuracy (0.664 vs 0.637) largely by better identifying the "No Fall" class, it struggles more with the "Rare" class compared to the Direct model.

### Model Family Winners
Model selection heavily favored **Random Forest**:
*   **Stage 1 (No vs. Any Fall):** 4 Random Forest, 1 XGBoost
*   **Stage 2 (Rare vs. Recurrent):** 5 Random Forest
*   **Direct 3-class:** 5 Random Forest

---

## 3. Sensitivity Analysis: Delta Features

We tested whether including temporal velocity (Delta features, e.g., $\Delta$ MoCA, $\Delta$ UPDRS) improved performance over the primary model. 

### Detailed Metrics Comparison (With Deltas)
| Metric | Two-Stage (Mean ± SD) | Direct 3-Class (Mean ± SD) |
| :--- | :--- | :--- |
| **Macro F1** | **0.503 ± 0.011** | **0.504 ± 0.013** |
| **Weighted F1** | 0.658 ± 0.014 | 0.646 ± 0.016 |
| **Accuracy** | 0.663 ± 0.019 | 0.633 ± 0.016 |
| **Recall: No Fall (Class 0)** | 0.814 ± 0.041 | 0.740 ± 0.024 |
| **Recall: Rare (Class 1)** | 0.247 ± 0.043 | 0.332 ± 0.059 |
| **Recall: Recurrent (Class 2)** | 0.533 ± 0.091 | 0.553 ± 0.122 |
| **AUC OVR (Macro)** | 0.738 ± 0.017 | 0.746 ± 0.018 |

*   **Performance Gain:** The Two-Stage model gained ~+0.010 in Macro F1 vs. the no-deltas dataset.
*   **Decision:** The pre-specified rule for promoting delta features to the primary model required a gain of $\geq +0.02$ and a Wilcoxon $p < 0.05$. The results failed this threshold. Therefore, **Deltas will not be promoted** to the main paper.

---

## 4. Differences from Legacy/Submitted Paper

The drop in performance metrics compared to the original manuscript is entirely expected and is the result of fixing fundamental design flaws. When communicating these numbers, we should emphasize that the decrease in score is due to **fixing the evaluation design**, not a failure of the model.

| Metric / Methodology | Legacy / Submitted Paper | Current Revision | Impact / Why it Changed |
| :--- | :--- | :--- | :--- |
| **Temporal Design** | Latest/max values over the entire patient record. FOG measured at outcome. | **Landmark approach:** Predictors strictly restricted to $\le$ Index Date (12 mos prior to outcome). | **Major.** Removed severe data leakage where the model "saw the future." |
| **Imputation** | Blanket `fillna(0)` across the entire cohort before splitting. Target variable imputed. | **Split-then-impute:** Zero-filling avoided for dopamine. NaNs kept for trees. | **Major.** Removed data leakage between train/test and corrected biased clinical assumptions. |
| **Model Selection** | Grid search tuned on Macro F1, but best model picked via Test Accuracy. | Best model picked via **Inner-CV Macro F1**. | **Moderate.** Honest evaluation. Test data is no longer used for tuning. |
| **Robustness** | 1 Seed (Seed 42) | **5 Paired Seeds** (with 95% CIs) | Accounts for variance and "lucky" seeds. |
| **Dataset Size** | ~1168 patients (included invalid labels) | **1040** clean landmark patients | Cleaned BMI voids and mislabeled diagnoses. |

---

## 5. Other Important Findings & Caveats

1.  **Recurrent Class Recall is Noisy:** The mean recall for the recurrent faller class in the Two-Stage model is **0.553**, but the confidence interval is very wide (0.394 – 0.713). This indicates high variance in predicting the rarest class across different data splits. We must report this honestly and avoid swapping metrics post-hoc to hide it.
2.  **Next Steps for SHAP (LLM Explanations):** 
    *   The paper pipeline for SHAP is confirmed as the **No-Deltas Two-Stage** model. 
    *   We will refit the chosen Stage 1 and Stage 2 architectures (Random Forest for both) on the canonical 70/30 Seed 42 split to generate the final `explanation_artifacts/`. 
    *   The `feature_map.csv` will be updated to drop unused delta columns and include the new `dopamine_missing` and `101` flags.
3.  **Composite Neuro-QoL Score:** The Gaussian composite score (`able_weighted_score`) was retained in the primary feature set, as it effectively captures the "Fall Risk Paradox" (high risk in impaired patients still attempting mobility) better than the 8 raw items.

---

### Appendix A: Performance Stability Across Repeated Runs
Shows the stability of the primary **Two-Stage** model across the five random seeds.
| Metric | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± SD |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Macro F1** | 0.502 | 0.498 | 0.502 | 0.471 | 0.490 | 0.493 ± 0.013 |
| **Weighted F1** | 0.646 | 0.662 | 0.655 | 0.650 | 0.629 | 0.648 ± 0.013 |
| **Accuracy** | 0.641 | 0.679 | 0.679 | 0.673 | 0.647 | 0.664 ± 0.018 |
| **Recall: No Fall (Class 0)** | 0.762 | 0.855 | 0.864 | 0.869 | 0.799 | 0.830 ± 0.047 |
| **Recall: Rare (Class 1)** | 0.250 | 0.191 | 0.191 | 0.176 | 0.147 | 0.191 ± 0.037 |
| **Recall: Recurrent (Class 2)** | 0.667 | 0.533 | 0.467 | 0.400 | 0.700 | 0.553 ± 0.128 |
| **AUC OVR (Macro)** | 0.737 | 0.746 | 0.712 | 0.743 | 0.758 | 0.739 ± 0.017 |

---

### Appendix B: Statistical Significance Analysis
Compares the proposed **Two-Stage** model against the **Direct 3-Class** baseline across the 5 paired random seeds.
| Metric | Two-Stage (Mean) | Baseline (Mean) | Difference (95% CI) | p-Value |
| :--- | :--- | :--- | :--- | :--- |
| **Macro F1** | 0.493 | 0.507 | [-0.0439, 0.0143] | 0.3125 |
| **Weighted F1** | 0.648 | 0.649 | [-0.0227, 0.0217] | 0.8125 |
| **Accuracy** | 0.664 | 0.637 | [0.0142, 0.0397] | 0.0625 |
| **Recall: No Fall (Class 0)** | 0.830 | 0.750 | [0.0527, 0.1080] | 0.0625 |
| **Recall: Rare (Class 1)** | 0.191 | 0.332 | [-0.2039, -0.0785] | 0.0625 |
| **Recall: Recurrent (Class 2)** | 0.553 | 0.527 | [-0.1425, 0.1958] | 0.8750 |
| **AUC OVR (Macro)** | 0.739 | 0.750 | [-0.0411, 0.0187] | 0.6250 |
