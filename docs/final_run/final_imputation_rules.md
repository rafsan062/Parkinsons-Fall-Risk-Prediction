# Final imputation rules

## Purpose

This document defines how missing predictor values are handled after final landmark-safe aggregation. The locked manuscript run applies these rules to one 26-feature universe: 25 paper-aligned variables plus maximum eligible constipation. Neuro-QoL mobility enters only as the revision's 8-item Gaussian score (D28). Older feature-set outputs remain historical context and are not rebuilt as competing final inputs. The declared alternative representations remain sensitivities.

The aggregated CSV files remain the unimputed source of truth. No globally imputed patient-level dataset is created. Any median, mode, encoding vocabulary, scaling parameter, or multivariate transformation must be fitted using only the applicable training patients and then applied unchanged to validation or evaluation patients.

## How the strategy was chosen

Missingness percentage measures how many patients a rule affects, but it does not determine what the missing value means. The final rules also use:

- whether an eligible source form exists;
- whether an entire instrument or only one item is absent;
- whether a codebook-defined `101` was removed;
- whether another valid eligible visit supplies the aggregate;
- whether missingness indicates an unrecorded clinical value or that an assessment was not applicable;
- whether the model can preserve numeric missing values natively.

The supporting outcome-blind audit is in Notebook 24. It covers all 82 broader representations and all four main dataset variants.

## Core safeguards

1. Never impute the fall outcome.
2. Never restore `101` as a clinical score.
3. Create missing-state indicators before filling their source values.
4. Fit statistical fills on training patients only.
5. Keep categorical `Missing` distinct from a recorded clinical category such as `No`, `Yes`, or `Uncertain`.
6. Do not delete a patient merely because a predictor is missing.
7. Do not present an imputed value as a measured patient fact in explanations or reports.

## Model backends

| Backend | Numeric missing values | Nominal missing values |
|---|---|---|
| Dense input required | Apply the feature-specific rule below, then use a training-fold median or mode where specified. | Add an explicit `Missing` category and fit encoding on the training fold. |
| Native missingness supported | Retain `NaN` unless a fixed clinical structural rule applies. | Retain an explicit `Missing` category because nominal values still require encoding. |

The native-missing branch applies only to model implementations that support missing numeric values. Other tree models use the dense branch.

## Feature-specific rules

### Complete features

Observed values remain unchanged. A preprocessing pipeline may fit a declared scaler or encoder, but it must not replace observed values merely for consistency with another column.

### Ordinary isolated numeric or ordinal gaps

For isolated gaps without a special clinical meaning:

- dense models use the training-fold median;
- compatible native-missing models retain `NaN`;
- no generic missingness flag is added.

This covers the small residual gaps in MoCA, GDS, BMI, SCOPA-AUT, Part I totals or items, combined-state Part III scores, constipation, and eligible experimental vital-sign measures unless a more specific rule below applies.

### Patients without eligible longitudinal measures

Three patients in the broader 1,043-patient sensitivity cohort have static predictors but lack the historical revision longitudinal eligibility. They are excluded by definition from the primary 1,040-patient cohort.

- In the 1,043-patient sensitivity only, numeric values follow their feature-specific dense or native-missing rule.
- Nominal values use the explicit `Missing` category.
- Create `NO_ELIGIBLE_LONGITUDINAL_HISTORY` before filling.
- Treat this indicator as availability metadata during selection and report whether it is retained.
- Do not add this constant-zero indicator to the primary 1,040-patient analysis.

The indicator prevents a fully imputed longitudinal profile from being mistaken for a set of measured typical values.

### Freezing of gait: `FRZGT12M`

All 66 missing patient-level values occur because no eligible freezing form exists. These patients have greater observed disease and motor burden than patients with a recorded zero, so missing must not silently become confirmed absence of freezing.

- Dense models use the training-fold mode as a placeholder. It was zero in every saved outer-training partition, but it must still be calculated inside the training fold.
- Native-missing models retain `NaN`.
- Create `FOG_FORM_MISSING` before filling.
- Treat `FRZGT12M` and `FOG_FORM_MISSING` as one representation group during feature selection.

### Neuro-QoL mobility block

All 67 missing cases lack an eligible Neuro-QoL form. The revision Gaussian score is the only Neuro-QoL model input; it is missing exactly when no eligible form exists.

- Dense models fit the median of every included Neuro-QoL representation on the training fold and use the resulting set of medians as one placeholder profile.
- Native-missing models retain the numeric `NaN` values.
- Create one `NQ_FORM_MISSING` indicator, not one indicator per item.
- Group the included Neuro-QoL inputs and `NQ_FORM_MISSING` during feature selection.
- Record that the filled profile is synthetic and must not be narrated as a completed questionnaire.

The composite remains a separately defined representation and receives its own training-fold median. The missing-form indicator preserves the distinction between a measured form and the placeholder profile.

### Family history: `ANYFAMPD`

Sixty-one patients lack a usable value: 22 have no eligible form and 39 have an eligible but blank form.

- Use an explicit nominal `Missing` category.
- Never convert missing family history to `No`.
- Do not add a duplicate binary flag; the encoded `Missing` category already represents the state.

### Dopaminergic therapy history: `DOPTHERST`

Missing therapy history is not equivalent to recorded untreated status. The 205 missing patients have longer disease duration than the recorded-No group, and many have observed Part IV complications.

- The primary preprocessing branch uses an explicit nominal `Missing` category.
- Do not add a second binary flag in this branch; it would duplicate the missing category.
- The revision-compatible sensitivity fills with the training-fold mode and adds `DOPTHERST_MISSING`, reproducing the earlier strategy without changing the primary definition.

### MDS-UPDRS Part IV

Part IV measures complications of dopaminergic treatment. Its handling depends on both form availability and recorded therapy history.

#### Whole form absent and therapy recorded `No`

This applies to 145 patients.

- Fill the Part IV total or included items with fixed structural zero.
- Create `PART_IV_FORM_MISSING` before filling.
- This zero is a clinical structural rule, not a statistic estimated from other patients.

#### Whole form absent and therapy recorded `Yes` or missing

This applies to 84 patients: 18 with recorded therapy and 66 with missing therapy history.

- Dense models use a training-fold median placeholder.
- Native-missing models retain `NaN`.
- Set `PART_IV_FORM_MISSING = 1`.
- Do not interpret absence as zero treatment complications.

#### Form present with an isolated gap

One patient has a final `NP4OFF` gap after a codebook-defined `101` was removed.

- Dense models use the training-fold median.
- Native-missing models retain `NaN`.
- Do not apply structural zero because the form exists.

Group the included Part IV values and `PART_IV_FORM_MISSING` during feature selection.

### Codebook-defined `101` values

`101` means unable to rate only in the audited fields. It was converted to missing before aggregation.

- All patients with Part III `101` records have another valid eligible value for the affected combined-state aggregate. Therefore, `101` creates no additional missing primary Part III aggregate.
- Removed Part I `101` values create no final item-level gaps.
- One experimental Part IV `NP4OFF` value remains missing and follows the isolated-gap rule above.
- Never fill a score with `101` or treat it as high severity.
- The Part III ever-101 indicators remain an independent sensitivity branch. They may be selected or ignored within that branch, and their selection frequency must be reported.

### Separate Part III ON/OFF sensitivity

State-specific values have approximately 24–30% missingness because the corresponding medication state was not assessed.

- Create `ON_STATE_AVAILABLE` and `OFF_STATE_AVAILABLE` before filling.
- Dense models use training-fold median placeholders for missing state-specific scores.
- Native-missing models retain `NaN`.
- Keep state-availability indicators distinct from unable-to-rate indicators.
- Group each state-specific score block with its matching availability indicator during feature selection.

These rules apply only to the separate-state sensitivity. Combined-state historical maxima remain primary.

## Representation groups during feature selection

The following source values and indicators are evaluated as representation groups:

| Group | Members |
|---|---|
| Freezing form | `FRZGT12M`, `FOG_FORM_MISSING` |
| Neuro-QoL form | Neuro-QoL Gaussian score, `NQ_FORM_MISSING` |
| Part IV form | Included Part IV total/items, `PART_IV_FORM_MISSING` |
| Part III ON state | Included ON-state scores, `ON_STATE_AVAILABLE` |
| Part III OFF state | Included OFF-state scores, `OFF_STATE_AVAILABLE` |

A selector must not retain a placeholder-filled source representation while silently discarding an indicator required to distinguish unmeasured from measured values. Implement group-aware keep/drop behavior for these families. The direct no-removal branch retains the entire applicable group.

The explicit nominal `Missing` levels for `DOPTHERST` and `ANYFAMPD` are part of their categorical encodings rather than separate flags.

## Training-fold sequence

For every inner or outer training partition:

1. Start from the unimputed aggregated rows for that partition.
2. Derive fixed missing-state and availability indicators from the original missingness pattern.
3. Apply the conditional fixed Part IV structural-zero rule.
4. Fit all required modes and medians using only the training partition.
5. Transform validation or evaluation rows using the already fitted rules and statistics.
6. Fit categorical encoding and scaling using only the transformed training partition.
7. Apply feature engineering and feature selection inside the same training boundary.
8. Save the fitted fill values, derived-column manifest, and selected representation groups for audit.

## Prohibited shortcuts

- No full-cohort median or mode before splitting.
- No blanket `fillna(0)`.
- No conversion of missing dopamine or family history to `No`.
- No blanket structural zero for Part IV when therapy is recorded `Yes` or unknown.
- No separate Neuro-QoL missing flag for every item.
- No use of outcome values to determine an imputation rule.
- No removal of patients solely because a predictor is missing.
- No patient-level dense imputed CSV presented as source data.

## Sensitivity branches

The final workflow retains these named comparisons:

- primary explicit-missing dopamine representation versus revision mode-plus-flag handling;
- conditional Part IV structural handling versus native missingness where supported;
- primary combined-state Part III versus separate ON/OFF with state-availability indicators;
- primary omission of Part III ever-101 flags versus the matched with-flags branch;
- primary 1,040-patient historical-revision cohort versus the broader 1,043-patient cohort.

Sensitivity results cannot redefine the primary procedure using held-out performance.
