# Final data-cleaning protocol

## Purpose and boundary

Final GitHub Notebook 01 converts the raw 24 August 2026 PPMI exports into
cleaned visit-level predictor tables. Cleaning occurs after each patient's
landmark is established and before patient-level aggregation.

The notebook does not aggregate patients, fill missing values, select
features, engineer predictors, or fit models. The fall outcome is used only to
establish the outcome month and patient-specific landmark; it is never written
to a predictor table.

## Temporal eligibility

1. Retain Parkinson's disease participants with status `Enrolled`, `Complete`,
   or `Withdraw Deceased`.
2. Select each patient's latest dated, observed `FLNFR12M` record. Stop for an
   unresolved tie in the latest month.
3. Define the index month as 12 calendar months before the outcome month.
4. Link each predictor source to the patient-specific index month.
5. Exclude undated records and records after the index month before applying
   any history-based cleaning rule.

This order prevents later measurements from influencing an earlier prediction
profile.

## Approved cleaning rules

| Area | Final rule | Reason |
| --- | --- | --- |
| Categorical codes | Translate codes using the annotated project code list and fail on unexpected nonblank codes. | Numeric category codes must not be treated as ordered clinical quantities. |
| Diagnosis history | Use `DXPOSINS` for postural instability at diagnosis. | The earlier legacy label incorrectly pointed to `DXOTHSX`, which represents another symptom. |
| `FEATPOSHYP` | Preserve `No`, `Yes`, and `Uncertain` as distinct nominal categories. | Code 2 is a recorded uncertain response, not missing and not greater severity. |
| `FEATSUGRBD` | Preserve `No`, `Yes`, and `Uncertain` as distinct nominal categories. | Missing, negative, and uncertain histories have different meanings. |
| Part I `101` | Convert `101` to missing for the six audited rater-completed items. | Their field-specific codebooks define 101 as unable to rate. |
| Part III `101` | Create a record-level unable-to-rate flag, then convert the gait, postural-stability, or Hoehn–Yahr score to missing. | The code is assessment status, not clinical severity; flags remain available only for sensitivity analysis. |
| Part IV `101` | Convert `101` to missing for the six audited items. | Their field-specific codebooks define 101 as unable to rate. |
| Measured vital value 101 | Retain it as an ordinary measurement when it falls in a measured vital-sign field. | The sentinel interpretation is field specific and must not be applied globally. |
| GDS-15 | Reverse-score the five positively worded items and calculate a total only when all 15 items are observed. | A partial sum is not a complete GDS-15 total. |
| BMI timing | Apply the landmark filter before examining a patient's height or weight history. | Later visits cannot correct or invalidate an earlier predictor. |
| BMI height | Use a supported eligible-history consensus to correct a clear adult-height inconsistency; otherwise leave unresolved values missing and retain an audit flag. | Adult height should not jump substantially, but an unsupported replacement would invent data. |
| BMI weight | Invalidate an extreme eligible-history weight disagreement rather than deleting the patient. | Clear entry errors should not become clinical BMI values or patient exclusions. |
| BMI range | Retain calculated BMI only within 10–60. Do not replace invalid BMI with a cohort mean. | An invented mean would appear to be a measured patient value and could distort explanations. |

## Output tables

Notebook 01 writes 15 cleaned visit-level tables containing:

- record and patient identifiers;
- source month and patient-specific index month;
- cleaned predictor fields;
- Part III assessment-status flags;
- GDS completeness and total fields; and
- BMI correction and unresolved-history flags.

The tables contain no fall outcome, outcome class, outcome month, imputed
predictor, or model-derived value.

## Verification

The final publication notebook passed all 68 scientific and structural
cleaning checks. Its 15 regenerated cleaned tables matched the verified
Notebook 22 tables exactly by SHA-256.

Notebook 02 must use these final Notebook 01 outputs as its only cleaned input.
