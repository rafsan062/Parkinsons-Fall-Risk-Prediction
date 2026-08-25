# LLM Layer Implementation Notes & Recommendations

This document captures key observations and rules discovered during the revision modeling phase. It is intended for the team building the downstream Large Language Model (LLM) explanation layer to ensure that the patient prompts are medically accurate, semantically rich, and avoid hallucination.

---

## 1. Handling Missing & Imputed Values
The Random Forest model requires dense data, meaning we used **median imputation** for missing values. 
*   **Recommendation:** **Do not show the median-imputed values to the LLM.** 
*   If a value was missing in the raw data, the LLM prompt should explicitly state "Not Recorded" or "Unknown." Feeding the LLM an imputed median value (e.g., a median MoCA score of 26) as if it were ground truth will cause the LLM to generate confident, but completely fabricated, clinical narratives about the patient.
*   *Note:* Use the custom indicator flags we created (e.g., `dopamine_missing == 1`) to trigger the "Unknown" logic in the prompt builder.

## 2. Neuro-QoL Mobility Score (Raw vs. Composite)
The predictive model relies on a single engineered feature: the Gaussian-weighted composite score (`able_weighted_score`). 
*   **Recommendation:** While the SHAP value will be tied to this single composite score, the LLM will generate much better, personalized narratives if it can see the **8 individual raw item scores** (e.g., "Able to get in and out of a car", "Able to push open a heavy door"). 
*   Providing the raw items in the text prompt allows the LLM to explicitly mention the patient's specific daily struggles, rather than vaguely referencing a "Composite Mobility Score of 2.1."

## 3. Explaining Temporal Aggregation
To fix data leakage, we anchored all predictions to a $\le 12$-month historical cutoff, using "Mixed Aggregation" rules. The LLM must be explicitly told *how* a feature was recorded so it doesn't hallucinate the timeline.
*   **Recommendation:** Use the precise terminology from our aggregation rules when constructing the prompt text.
    *   For dynamic metrics (MoCA, BMI), tell the LLM: *"The patient's **Most Recent** MoCA score prior to the index date was..."*
    *   For episodic/progressive metrics (FOG, Fainting, UPDRS III), tell the LLM: *"The patient's **Peak Historical Severity** of Freezing of Gait on record is..."*
    *   This prevents the LLM from falsely assuming a peak historical FOG score is what the patient was experiencing exactly on the day of the index visit.

## 4. The "101" Medical Flags
During data cleaning, we discovered that the MDS-UPDRS dataset uses the code `101` to indicate a clinician was "Unable to rate due to medical reasons."
*   **Recommendation:** We extracted these into specific binary flags (`NP3GAIT_missing_due_to_101`, `NP3PSTBL_missing_due_to_101`, `NHY_missing_due_to_101`). If any of these are triggered, the LLM prompt should explicitly state: *"The clinician was physically unable to evaluate this metric due to the patient's medical condition."* This provides critical context to the LLM that the patient is likely severely impaired, rather than just having a "missing" data row.

## 5. SHAP Value Interpretation (Directionality)
When the LLM reads the raw SHAP values from the arrays, it needs strict guidance on how to interpret positive vs. negative values so it doesn't state a patient's risk is dropping when it's actually rising.
*   **Recommendation for Stage 1 (Any Fall):** A **positive** SHAP value increases the probability that the patient is a "Faller". A negative value pushes them toward "No Fall".
*   **Recommendation for Stage 2 (Recurrent):** A **positive** SHAP value increases the probability that the patient will have **Recurrent Falls**. A negative value pushes them toward "Rare Falls".

---
*Included in this directory:*
*   `04_shap_export.ipynb`: Final notebook used to extract SHAP features.
*   `feature_map.csv`: The finalized feature-to-name mapping.
*   `plot_shap.py`: Python script used to render the SHAP summary plots.
*   `shap_summary_stage1.png` & `shap_summary_stage2.png`: Rendered global SHAP plots showing feature importance and effect directionality.
