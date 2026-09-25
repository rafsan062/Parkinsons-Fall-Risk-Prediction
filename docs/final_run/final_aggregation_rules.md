# Final predictor aggregation rules

## Purpose

This document explains how repeated, pre-landmark PPMI records are converted into one predictor value per patient. The locked manuscript run uses 25 eligible paper-aligned variables plus maximum eligible constipation (`NP1CNST`), for 26 candidate predictors. Neuro-QoL mobility enters only through the 8-item Gaussian composite; its seven component items are retained solely to calculate that score. The revision-21, paper-32, and superseded 33-feature tables remain historical references; the named alternative representations have been evaluated as one-change sensitivities.

Aggregation happens only after each patient's landmark date is fixed and only records at least 12 months before the outcome visit are eligible. The rule for a variable is chosen according to the clinical meaning we want the resulting value to have.

## How the rules were chosen

Four questions determine the rule:

1. **Is the value defined by the prediction date?** Age and disease duration must be calculated at the landmark so that they describe the patient at the moment the prediction is made.
2. **Does the feature represent current status?** For cognition, mood, body size, and current function, the most recent eligible assessment is the closest available description of the patient at the landmark.
3. **Does the feature represent prior burden?** For episodic symptoms and motor or treatment-complication scores, the maximum eligible value summarizes the greatest recorded pre-landmark burden. This preserves the interpretation used in the revision work, but it is a historical-severity representation rather than a current-status measurement.
4. **Is the feature a durable occurrence or history?** Treatment initiation, family history, and documented symptomatic orthostatic hypotension are represented as an eligible history of occurrence because a later negative or missing entry should not erase a prior documented occurrence.

These choices are made without examining held-out outcomes. They do not prove that one representation is clinically superior. Where two interpretations are both plausible, the alternatives are kept separate and compared inside the training-only modeling process.

## The four aggregation groups

| Group | Question answered by the patient-level value | General rule |
|---|---|---|
| Landmark-derived | What was true at the prediction date? | Calculate from dates relative to the landmark. |
| Most recent status | What was the latest eligible recorded state? | Use the latest eligible measurement or coherent assessment. |
| Peak historical severity | What was the greatest eligible recorded burden? | Use the maximum eligible score. |
| Historical occurrence | Had the condition or event been documented by the landmark? | Preserve an eligible positive occurrence; keep missing distinct from negative. |

## Group 1: landmark-derived predictors

These variables would be wrong if calculated at data-extraction time because that would add time occurring after the prediction date.

| Feature | Meaning | Rule | Why this rule fits | Limitation or rejected alternative |
|---|---|---|---|---|
| `AGE_ASSESS` | Age at prediction | Landmark date minus birth date. | Fall risk is being estimated at the landmark, so age must describe the patient on that date. | Current age at export would include future follow-up time; age at the outcome would also use information from after prediction. |
| `PD_DURATION` | Years since Parkinson's diagnosis at prediction | Landmark date minus diagnosis date. | The clinically relevant exposure is disease duration already accumulated when the prediction is made. | Duration at the outcome or at data extraction would add the prediction horizon or later follow-up. Diagnosis dates have month-level precision, so the result is approximate. |

## Group 2: most recent eligible status

The latest eligible measurement is used when the variable is intended to describe status near the landmark. A mean would blend remote and recent states, while a maximum or minimum would deliberately select an extreme rather than the closest state.

For multi-item instruments, the selected values come from one coherent assessment whenever the source permits it. This avoids constructing a synthetic assessment from items recorded at different visits.

The seven Neuro-QoL item rows below describe the **components of the Gaussian score**. They were separate candidates in the initial 33-feature run but are not model inputs in the amended 26-feature primary analysis (D28).

| Feature | Meaning | Rule | Why this rule fits | Limitation or rejected alternative |
|---|---|---|---|---|
| `DXTREMOR` | Tremor at diagnosis | Latest eligible coherent diagnosis record. | This is a diagnosis-characteristic field, so the latest record is used only to retain the most up-to-date eligible documentation or correction. | A maximum is inappropriate for a categorical field. There were no eligible patient-level conflicts in the current cohort, so first versus latest currently makes no difference. |
| `DXBRADY` | Bradykinesia at diagnosis | Latest eligible coherent diagnosis record. | This preserves the most recent eligible diagnosis documentation and keeps it aligned with the other diagnosis fields. | A maximum would impose an ordering on codes. No eligible conflicts were found in the current cohort. |
| `DOMSIDE` | Dominant side at diagnosis | Latest eligible coherent diagnosis record. | It is a categorical diagnosis characteristic and should come from a single diagnosis record. | Averaging or maximizing category codes has no clinical meaning. No eligible conflicts were found in the current cohort. |
| `MCATOT` | Montreal Cognitive Assessment total | Latest eligible complete total. | The target concept is cognitive status closest to prediction. | The maximum would preferentially select the patient's best historical performance, the minimum the worst, and the mean would obscure when the status occurred. |
| `GDS_TOTAL` | Geriatric Depression Scale total | Latest eligible complete total. | The target concept is depressive-symptom status closest to prediction. | A historical maximum would instead represent the worst prior episode. A partial questionnaire is not substituted for a complete total. |
| `BMI` | Body mass index | Latest eligible cleaned BMI. | BMI can change, so the closest reliable eligible measurement best represents body size at prediction. | A maximum would emphasize the highest historical weight state; a mean would mix measurements from different times. Physiologically implausible source records are handled in cleaning before aggregation. |
| `NQMOB_TOTAL` (`NQ_GAUSSIAN_REVISION`) | Neuro-QoL Gaussian mobility score: each of 8 items (the seven below plus `NQMOB25`) weighted by exp(−(response − 2.5)² / (2 × 1.2²)), averaged, and multiplied by 8. **This is the only Neuro-QoL model input (D28).** | Latest eligible coherent form. | The score is intended to represent recent self-reported mobility and must be calculated from items completed together. | Selecting each item's latest value independently could combine different visits. Selecting the maximum would favor the best historical mobility because higher item scores indicate less difficulty. |
| `NQMOB37` | Able to get on and off the toilet | Same latest coherent Neuro-QoL form. | It is a current functional-ability item, and keeping all mobility items from one form preserves a real assessment. | Per-item latest values can create a form that was never completed; an extreme value would answer a best/worst-history question instead. |
| `NQMOB30` | Able to step up and down curbs | Same latest coherent Neuro-QoL form. | It describes functional ability near the landmark and belongs to the coherent mobility assessment. | Independent item selection would break within-form coherence. |
| `NQMOB26` | Able to get in and out of a car | Same latest coherent Neuro-QoL form. | It describes recent transfer ability and is therefore represented by the latest complete form. | A maximum would favor best historical function; a minimum would favor worst historical function. |
| `NQMOB32` | Able to get out of bed into a chair | Same latest coherent Neuro-QoL form. | It describes recent transfer and mobility ability and should remain tied to the assessment occasion. | Combining item values from separate visits could misstate the patient's actual form response pattern. |
| `NQMOB33` | Able to run errands and shop | Same latest coherent Neuro-QoL form. | It describes recent community mobility and is part of the same functional snapshot. | An extreme or average across history would no longer mean status closest to prediction. |
| `NQMOB31` | Able to get off the floor without help | Same latest coherent Neuro-QoL form. | It describes recent floor-recovery ability, which is directly relevant to falls. | Per-item aggregation would sacrifice the coherent-form interpretation. |
| `NQMOB28` | Able to go for a walk of at least 15 minutes | Same latest coherent Neuro-QoL form. | It describes recent walking endurance and is retained from the latest complete assessment. | The worst historical response is a different construct and is not silently substituted. |

## Group 3: peak eligible historical severity

These features retain the revision work's maximum-over-eligible-history rule. The resulting value means **greatest recorded pre-landmark burden**, not status at the landmark. This is defensible when symptoms fluctuate and a severe prior episode may remain informative even if the latest score improves.

The tradeoff is that an old episode can dominate the value, and item-wise maxima can combine observations from different visits. We retain this definition for the main comparison because it is explicit and compatible with the established revision model; plausible alternatives must be tested as separate representations rather than silently changing the meaning.

| Feature | Meaning | Rule | Why this rule fits | Limitation or rejected alternative |
|---|---|---|---|---|
| `FRZGT12M` | Freezing of gait during the preceding 12 months | Maximum eligible value. | Each assessment summarizes a preceding-year symptom window; the maximum records whether substantial freezing burden had occurred anywhere in the eligible history. | The latest value would describe only the most recent window. A remote severe episode may dominate the maximum. |
| `SCAU14` | Lightheadedness on standing up (SCOPA-AUT) | Maximum eligible value. | Orthostatic symptoms may be intermittent, so the maximum preserves the greatest documented burden rather than treating a later symptom-free visit as erasing it. | It does not indicate that the symptom was still present at the landmark. |
| `SCAU16` | Fainting in the preceding six months (SCOPA-AUT) | Maximum eligible value. | Fainting is episodic, so the maximum preserves the greatest documented pre-landmark burden. | It is an orthostatic/autonomic item, not a falls measure. The latest value would measure recent rather than historical burden. |
| `NP1RTOT` | Rater-completed MDS-UPDRS Part I total | Maximum eligible value after field-specific 101 handling. | The main representation treats the strongest recorded non-motor burden as potentially prognostic even if later scores improve. | It can reflect a remote visit and is not a current-status total. A latest-total representation is a plausible future alternative but is not mixed into this one. |
| `NP1SLPD` | Daytime sleepiness | Maximum eligible value. | Sleepiness can fluctuate, so the maximum records the most severe eligible report and matches the established historical-burden definition. | It may overstate current symptoms if the problem resolved; latest would answer a different question. |
| `NP1URIN` | Urinary problems | Maximum eligible value. | The rule preserves the strongest documented urinary-symptom burden, which may carry information about disease burden even after fluctuation. | It is not proof of ongoing symptoms at prediction, and the latest value may be preferable if current status is the scientific target. |
| `NP3TOT_ONOFF` | Combined-state MDS-UPDRS Part III motor total | Maximum eligible value across eligible medication states. | The primary representation summarizes the greatest recorded motor impairment and maintains comparability with the revision model. | Medication ON and OFF examinations measure different states. Combining them sacrifices that distinction, so separate-state values are retained as a sensitivity analysis. |
| `NP3GAIT_ONOFF` | Combined-state gait item | Maximum eligible value across eligible medication states. | It captures the greatest recorded examiner-rated gait impairment in the eligible history. | The value can come from either medication state and may be remote; state-specific versions test that assumption. |
| `NP3PSTBL_ONOFF` | Combined-state postural-stability item | Maximum eligible value across eligible medication states. | It captures the greatest recorded examiner-rated postural-instability burden, a clinically relevant historical signal for fall risk. | It may not represent current function and can mix medication-state interpretation. Values coded 101 are treated as missing, not as extreme severity. |
| `NHY_ONOFF` | Combined-state Hoehn-Yahr stage | Maximum eligible value across eligible medication states. | It summarizes the highest documented pre-landmark disease stage and preserves the established main-model definition. | A later improvement or medication-state difference is not represented. Code 101 is missing rather than a stage. |
| `NP4TOT` | MDS-UPDRS Part IV total | Maximum eligible value. | Motor complications can fluctuate; the maximum retains the strongest documented complication burden. | It can emphasize a remote episode. The latest total would instead represent recent treatment-complication status. |
| `NP4DYSKI` | Functional impact of dyskinesia | Maximum eligible value. | It records the greatest documented functional impact of dyskinesia rather than losing a prior severe episode after later improvement. | It does not establish that the impact persists at the landmark. |

## Group 4: eligible history of occurrence

These variables answer whether an event or condition had been documented by the landmark. The implementation uses a positive-dominant rule: any eligible `Yes` produces `Yes`; otherwise an eligible `No` produces `No`; otherwise the value remains missing. This avoids treating absence of a record as evidence of absence.

| Feature | Meaning | Rule | Why this rule fits | Limitation or rejected alternative |
|---|---|---|---|---|
| `DOPAMINE` | Dopaminergic therapy initiated | Any eligible documented `Yes`; otherwise eligible `No`; otherwise missing. | Starting therapy is a treatment-history milestone. A later missing or differently recorded entry should not erase known prior exposure. | This does not encode dose, duration, adherence, or medication status exactly at the landmark. |
| `ANYFAMPD` | Any family history of Parkinson's disease | Any eligible relative-specific `Yes`; otherwise eligible `No`; otherwise missing. | Family history is a durable background characteristic; once an eligible relative-specific positive is recorded, a later negative should not erase it. | Eleven patients have conflicting eligible source records, so the positive-dominant resolution is a documented cleaning assumption rather than a proven truth. |
| `FEATPOSHYP` | Symptomatic orthostatic hypotension documented | Any eligible `Yes`; otherwise eligible `No`; preserve `Uncertain` when no eligible `Yes` exists; otherwise missing. | The representation asks whether symptomatic orthostatic hypotension had been documented before prediction. Preserving `Uncertain` avoids converting ambiguity into a negative. | It represents documented history, not necessarily current symptoms. `Uncertain` is a real response category and must not be recoded as missing without a separate decision. |

## Constipation: two scientifically different candidates

Constipation is the one new candidate that advanced from the broader feature exploration. Maximum eligible severity is locked for the primary analysis; latest eligible status remains a one-change sensitivity because the two clinically interpretable definitions disagree often enough to matter.

| Candidate | Rule | Meaning | Why it may be useful | Main limitation |
|---|---|---|---|---|
| `NP1CNST_LATEST` | Latest eligible value. | Constipation status closest to prediction. | It aligns with the usual interpretation of a mutable symptom measured near the landmark. | It can miss substantial earlier burden. In the frozen sensitivity, it did not show a clear macro-F1 advantage over the primary maximum representation. |
| `NP1CNST_MAX` | Maximum eligible value. | Greatest recorded constipation burden before prediction. | It matches the historical-burden definition used in the exploratory work that identified constipation as promising. | It can overstate current status when a remote episode resolved. In the audit, it exceeded the latest value for 381 of 1,040 measured patients, so it is not interchangeable with `LATEST`. |

The two values must never enter the same feature set as duplicate versions of constipation. `MAX` enters the primary 26-feature universe, while `LATEST` replaces it only in the named sensitivity branch using the otherwise locked procedure.

## Experimental representations

The following variables are alternatives or extensions, not automatic additions to the final model. They use the same semantic test: recent state uses a coherent latest assessment; prior burden uses a maximum; changes use two measurements from the same visit context.

### Additional recent-status candidates

| Feature(s) | Meaning | Rule and rationale |
|---|---|---|
| `NQMOB25` | Able to push open a heavy door | Use the latest coherent Neuro-QoL form because it describes recent mobility function. Do not select it independently from another visit when it is evaluated with the other Neuro-QoL items. |
| `NQMOB_COMPOSITE` | Expanded Neuro-QoL mobility composite | Calculate from one latest coherent form so the composite corresponds to a real questionnaire occasion. Per-item maxima would preferentially select best historical function and could combine visits. |
| `NP1COG`, `NP1HALL`, `NP1DPRS`, `NP1ANXS`, `NP1APAT`, `NP1DDS` | Six rater-completed Part I items | A latest-item representation would test current non-motor status. Code 101 is missing only because the audited codebook defines it as unable to rate for these fields. These are alternatives to, or components of, the Part I total and must not be assumed to add independent information. |
| `NP1SLPN`, `NP1PAIN`, `NP1LTHD`, `NP1FATG` and alternative versions of `NP1SLPD`, `NP1URIN`, `NP1CNST` | Patient-reported Part I symptoms | Use the latest value when the intended construct is symptom status near prediction. This must be labeled separately from maximum-based historical burden. |
| `NP4WDYSK`, `NP4OFF`, `NP4FLCTI`, `NP4FLCTX`, `NP4DYSTN` and alternative `NP4DYSKI` | Individual Part IV motor-complication items | A latest-item representation tests current complication status. The items are alternatives to, or components of, the Part IV total and can be redundant if entered together without selection. |

### Additional historical-burden candidates

| Feature(s) | Meaning | Rule and rationale |
|---|---|---|
| Maximum versions of the Part I patient-reported items | Greatest recorded non-motor symptom severity | Use the maximum only when the candidate is explicitly defined as historical burden. This matches the main revision convention but can preserve remote symptoms after improvement. |
| Maximum versions of the six rater-completed Part I items | Greatest recorded rater-assessed non-motor severity | Use the maximum after converting field-specific 101 codes to missing. This is an item-level alternative to the total, not an additional fact that should automatically accompany the total. |
| Maximum versions of the Part IV items | Greatest recorded motor-complication severity | Use the maximum to retain prior complication burden. These can overlap strongly with `NP4TOT`, so inclusion must be decided inside training folds. |

### Visit-matched change candidates

| Feature(s) | Meaning | Rule and rationale |
|---|---|---|
| Orthostatic systolic and diastolic blood-pressure change | Change from supine to standing | Calculate from measurements collected in the same eligible visit because a difference across two different visits would mix temporal change with posture change. A measured value of 101 is retained as an ordinary vital-sign measurement. |
| Heart-rate change | Change from supine to standing | Calculate from the same eligible visit for the same reason: the feature is a within-occasion physiological response. |

## Sensitivity representations

Sensitivity branches test whether a modeling conclusion depends on a defensible but uncertain representation choice. They are evaluated separately from the main branch.

| Representation | Rule | Why it is separate |
|---|---|---|
| Part III ON and OFF values | Aggregate each medication state separately using the same eligible-history severity rule. | ON and OFF exams describe different motor states. Keeping them separate tests whether the combined-state maximum hides useful state-specific information. |
| `NP3GAIT_101FLAG` | `1` if an eligible gait item was ever coded 101; otherwise `0`. | The flag records assessment inability, not motor severity. It is tested only to learn whether that missingness pattern adds predictive information. |
| `NP3PSTBL_101FLAG` | `1` if an eligible postural-stability item was ever coded 101; otherwise `0`. | Code 101 means unable to rate. The flag preserves that event without treating 101 as the largest clinical score. |
| `NHY_101FLAG` | `1` if an eligible Hoehn-Yahr field was ever coded 101; otherwise `0`. | The flag represents rating availability, not disease stage, and therefore must remain distinct from the cleaned stage value. |

Patients are not excluded merely because an eligible Part III assessment contains 101. The primary branch treats the value as missing and continues through training-only imputation; the sensitivity branch adds the corresponding flag.

## Deterministic same-month ties

Some source dates have month-level rather than day-level precision. If multiple eligible records share the latest normalized month, selection must use a documented deterministic source ordering. This makes reruns reproducible; it does not imply that one tied record is clinically later within that month. Conflicting tied records should be surfaced rather than resolved with an undocumented row-order choice.

## Boundary with later feature engineering

Interactions, polynomial terms, scaling, PCA components, and learned feature subsets are not aggregation rules. They are created after the patient-level table is split and must be fitted using training data only. This separation keeps the raw clinical representation understandable and prevents information from validation or test patients from influencing learned transformations.
