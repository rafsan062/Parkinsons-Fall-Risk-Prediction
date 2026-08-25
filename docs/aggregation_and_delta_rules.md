# Final Aggregation & Delta Rule Decisions & Manuscript Explanations

**Purpose:** Document the final decisions for each predictor's aggregation rule. The terminology uses standard academic terms (e.g., "Peak Historical Severity") for the manuscript, while explanations remain logical and easy to communicate.

**Terminology:**
- **Most Recent Status** = The patient's most recent valid score before the prediction date.
- **Peak Historical Severity** = The patient's worst (highest) valid score recorded at any point before the prediction date.
- **Lifetime History** = A binary yes/no check of whether the patient *ever* experienced this before the prediction date.

---

## 1. Dopaminergic Therapy (`DOPTHERST`)

- **Methodology Term:** Lifetime History (Ever Initiated)
- **Explanation:** We track whether the patient had *ever* initiated dopaminergic therapy prior to the index date. Since starting this medication is a permanent milestone in treating Parkinson's, any past record confirms their status.

## 2. MoCA Total Score (`MCATOT`)

- **Methodology Term:** Most Recent Status 
- **Explanation:** We use the patient's most recent score prior to the index date to represent their *current* cognitive state. Using their historical best score would artificially mask any recent cognitive decline.

## 3. Freezing of Gait (`FRZGT12M`)

- **Methodology Term:** Peak Historical Severity
- **Explanation:** We use the patient's highest recorded score to capture the *worst* period of gait freezing they have experienced. A history of severe freezing remains a strong predictor for future falls, even if it was not actively observed at their most recent visit.

## 4. Lightheaded after standing (`SCAU14`)

- **Methodology Term:** Peak Historical Severity
- **Explanation:** We use the patient's highest recorded score to capture their *worst* historical episode of orthostatic lightheadedness. Because sudden dizziness upon standing is a primary mechanism for falls, any history of severe episodes is a critical risk factor.

## 5. Fainted (`SCAU16`)

- **Methodology Term:** Peak Historical Severity
- **Explanation:** This asks how often the patient fainted. Fainting is episodic, so capturing the highest frequency reported *before* the index date captures the patient's peak autonomic severity risk.

## 6. Neuro-QoL Mobility Composite (`able_weighted_score`)

- **Methodology Term:** Gaussian-Weighted Recent Status
- **Clinical Justification for Reviewers:** The Neuro-QoL items were aggregated into a composite using a Gaussian function (μ=2.5, σ=1.2). While reviewers noted this is non-monotonic, **this is an intentional design to capture the geriatric "Fall Risk Paradox".** Patients scoring 5 (No difficulty) are healthy and rarely fall. Patients scoring 1 (Unable to do) are often wheelchair-bound or require full physical assists, meaning their risk of *independent* incidental falls actually decreases. The highest risk of injurious falls occurs in the "transition zone"—patients scoring 2 or 3 (Much/Some difficulty)—who are severely impaired but still attempting independent mobility. The Gaussian weighting mathematically isolates this specific high-risk mobility window, which is why it provides superior predictive accuracy.
- **LLM Adaptation Plan:** Because this composite score represents "High-Risk Transitional Mobility" rather than simple linear weakness, the LLM explanation layer will be programmed to interpret high scores not as "total inability to move", but specifically as: *"The patient is in a high-risk mobility transition phase, where they are still mobile but face significant difficulty, putting them at peak risk for incidental falls."*

## 7. GDS Total Depression Score (`GDS_TOTAL`)

- **Methodology Term:** Most Recent Status
- **Explanation:** We use the most recent score to capture the patient's *current* depressive symptoms. Because this survey specifically asks about symptoms over the "past week", historical high scores from years ago do not reflect the patient's active psychiatric state.

## 8. MDS-UPDRS Part I Score (`NP1RTOT`)

- **Methodology Term:** Peak Historical Severity
- **Explanation:** We use the patient's highest recorded score to capture the *worst* burden of non-motor symptoms (like apathy or hallucinations) they have experienced. These symptoms can fluctuate, but a history of severe episodes indicates higher vulnerability.

## 9. Daytime Sleepiness (`NP1SLPD`) & Urinary Problems (`NP1URIN`)

- **Methodology Term:** Peak Historical Severity
- **Explanation:** We use the highest recorded severity for these symptoms. For example, a history of severe nighttime urinary urgency significantly increases the risk of nighttime falls, even if the symptom is currently well-managed.

## 10. MDS-UPDRS Part III Score (`NP3TOT`)

- **Methodology Term:** Peak Historical Severity
- **Explanation:** We use the highest recorded score to represent the patient's *worst* motor (movement) impairment. Motor tests can appear artificially improved if the patient recently took medication, so tracking the historical peak provides a more reliable measure of their underlying disease severity.

## 11. MDS-UPDRS Part III Sentinels (Gait, Postural Stability, H&Y)

- **Methodology Term:** Peak Historical Severity (Data-Cleaned)
- **Explanation:** After excluding invalid administrative codes (such as "101" which meant "unable to rate"), we use the patient's highest valid score to capture the *worst* walking and balance impairment they have historically demonstrated.

## 12. MDS-UPDRS Part IV (`NP4TOT`)

- **Methodology Term:** Peak Historical Severity
- **Explanation:** We use the highest recorded score to capture the *worst* motor complications (like dyskinesias or twitching) the patient has ever experienced. Even if these side-effects were later resolved by lowering medication doses, a history of them indicates a more complex disease state.

## 13. Postural Hypotension (`FEATPOSHYP`)

- **Methodology Term:** Lifetime History (Ever Present)
- **Explanation:** We track whether the patient has *ever* had a documented episode of postural hypotension (blood pressure dropping when standing). Because this directly causes dizziness and fainting, any historical occurrence is a primary fall risk factor.

---

# Delta Features (Change-Over-Time)

**Purpose:** Document the rationale for selecting specific features to be calculated as "deltas" (the difference between the most recent and second-most recent scores). 

**Methodology Term:** Longitudinal Trajectory (Velocity of Change)
**Explanation for Manuscript:** To capture the velocity of disease progression prior to the index date, longitudinal trajectories (deltas) were calculated for core clinical scales. Delta features were derived by subtracting the second-most recent valid score from the most recent valid score prior to the index date. Missing deltas for patients with insufficient longitudinal history (≤1 visit) were left as missing values (NaN) to be handled natively by tree-based models, rather than assuming artificial clinical stability. 

**Justification for Selected Features:**

- **Delta MoCA, Delta MDS-UPDRS (Parts I, III, IV), Delta GDS (Depression):** These represent the gold-standard, continuously scaled metrics for assessing the velocity of cognitive, motor, non-motor, and psychiatric decline in Parkinson's disease. Rapid worsening across these specific domains is an established prognostic indicator of advancing disease and imminent fall risk.
- **Delta Gait, Delta Postural Stability, Delta H&Y:** Rather than aggregating every individual item of the UPDRS, we specifically isolated the trajectory of these three items because mechanical deterioration in posture, balance, and gait are the most direct biomechanical precursors to falling.
- **Delta BMI:** A rapid reduction in BMI over time serves as a quantitative surrogate for advancing frailty and muscle wasting, a critical independent risk factor for injurious falls in the geriatric population.
- **Delta Neuro-QoL (Weighted Composite):** Tracking the longitudinal change in the Gaussian-weighted mobility composite isolates the patient's velocity into (or out of) the high-risk "transitional mobility" zone. A rapid decline in independent transfer ability is highly predictive of imminent falls.

**Justification for Excluded Features:**
- Binary history markers (e.g., *Dopaminergic Therapy Initiation*, *Fainting*, *Postural Hypotension*) were excluded from delta calculations because they represent permanent clinical milestones or sporadic events rather than continuous clinical deterioration.
