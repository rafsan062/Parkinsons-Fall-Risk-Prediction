# Post-Submission Revision Updates & Findings
**Date:** Week of Aug 24, 2026

This report summarizes the major updates, methodological fixes, and modeling results completed to address reviewer comments (specifically Comments 1, 2, 3, 5, and 6) on the multi-class fall risk stratification manuscript. 

The entire pipeline has been rebuilt into a clean, reproducible GitHub repository format. Detailed justifications for our rules and full statistical outputs can be found in the `docs/` folder of the provided repository.

---

## 1. Dataset Refresh & Feature Updates
*   **PPMI Data Refresh:** The dataset was updated to the latest Aug 24, 2026 export. The final clean "Landmark" cohort for modeling consists of **1,040 patients** (No fall: 712, Rare: 227, Recurrent: 101).
*   **Gait / Wearable Data Excluded:** We evaluated the requested Opals and Gait Substudy Mobility Assessments. However, coverage was extremely low (4.2% and 3.2% of patients, respectively), making them unviable for model features.
*   **Neuro-QoL Update:** The 8th Neuro-QoL item ("Able to push open a heavy door") was confirmed and added to the pipeline. However, we retained the Gaussian composite score (`able_weighted_score`) as our primary feature over the 8 raw items because it best captures the geriatric "Fall Risk Paradox."

## 2. Major Methodological Fixes
We overhauled the data pipeline to address fundamental flaws in the original manuscript's design:
*   **Fixed Temporal Data Leakage (Comment 1):** The legacy code used "latest" or "max" values across a patient's entire record, allowing the model to "see the future" at the time of the fall outcome. We fixed this by adopting a strict **Landmark Design**, where all predictors are strictly truncated to ≤ 12 months before the outcome.
*   **Refined Feature Aggregation & Delta Rules:** Instead of arbitrary aggregation, we defined a strict "Mixed Aggregation" approach for the truncated visits: `Last` was used for dynamic scores (MoCA, BMI), while `Max/Ever` was used for progressive symptoms (FOG, UPDRS). We also explicitly formulated the Δ velocity features. *(See `docs/aggregation_and_delta_rules.md` in repo).*
*   **Pre-Imputation Data Audits:** Before running models, we audited the raw data for plausibility, correctly voiding impossible BMI values (e.g., extreme height/weight mis-entries) and fixing the mislabeled postural instability diagnosis column (`DXPOSINS`). *(See `docs/data_cleaning.md` in repo).*
*   **Imputation Overhaul (Comment 3):** The legacy code used a blanket `fillna(0)` across the entire cohort (including target variables) before splitting. We corrected this to a strict **split-then-impute** strategy. We also applied targeted clinical rules (e.g., retaining missing value indicators for tree models and avoiding zero-filling for missing dopaminergic therapy). *(See `docs/imputation_rules.md` in repo).*
*   **Robust Evaluation Protocol (Comments 5 & 6):** We replaced the single-seed evaluation (which was incorrectly selected via Test Accuracy) with a much more rigorous setup. Model selection is now strictly driven by **Inner-CV Macro F1**, and final evaluation is aggregated over **5 paired random seeds** to ensure statistical stability. *(See `docs/evaluation_protocol.md` in repo).*

## 3. Modeling Results & Key Findings

With the data leakage fixed and honest cross-validation in place, the resulting predictive metrics are notably lower than the ~0.60+ Macro F1 reported in the original paper. **This drop is entirely expected and reflects the true predictive power of the model.** In the legacy pipeline, the model had access to symptoms (like FOG) recorded *at the time of the outcome*, and targets were inappropriately zero-filled. Now, the model is strictly forecasting based on historical data (≤ 12 months prior), which is a much harder, but clinically valid, task.

*   **Primary Architecture Performance:** The primary Two-Stage model achieves a mean test **Macro F1 of 0.493** and an **Accuracy of 0.664**. The baseline Dummy classifier scores 0.271, proving that despite the drop from the legacy paper, the task remains learnable. Random Forest overwhelmingly dominated as the best algorithm across all stages.
*   **Two-Stage vs. Direct 3-Class (Important Context):** A core premise of the paper was that a Two-Stage architecture is superior to a standard Direct 3-Class approach. However, our rigorous testing showed that the Direct 3-Class model actually achieved a **slightly higher Macro F1 (0.507 vs 0.493)**. This difference is not statistically significant (p=0.3125), meaning we cannot definitively claim one is better than the other. 
    *   *Why the difference?* The Two-Stage model achieves higher overall accuracy (0.664 vs 0.637) because it is much better at identifying the majority "No Fall" class (Recall: 83% vs 75%). However, it struggles more than the Direct model when identifying "Rare" fallers, which drags down its Macro F1.
    *   *Manuscript Framing Strategy:* While it doesn't statistically beat the baseline, retaining the Two-Stage model remains highly defensible. It aligns perfectly with sequential clinical decision-making (screening out non-fallers first, where it objectively excels), and it provides critical interpretability for the downstream LLM layer to explain the mechanism of falling separately from the mechanism of recurrence.
*   **Delta Features Did Not Work Out:** We hypothesized that adding longitudinal velocity features (e.g., Δ MoCA, Δ UPDRS) would significantly improve predictions. Unfortunately, this was not the case. The Deltas provided only a marginal, non-significant boost (+0.010 F1). Because it failed our pre-specified success threshold (≥ +0.02 F1 and p < 0.05), we are **not** promoting Delta features to the primary paper model. The static clinical baseline features already capture the vast majority of the available predictive signal.

*(For a comprehensive breakdown of all metrics, confidence intervals, stability across runs, and exact p-values, please point to `docs/results_report.md` in the GitHub repo).*

## 4. Next Steps
*   **SHAP & LLM:** We have locked in the final model (No-Deltas, Two-Stage). We will now fit this model on a canonical 70/30 split (Seed 42) to generate the clean SHAP explanation artifacts required by the LLM layer.
