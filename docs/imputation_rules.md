# Phase 2 — Imputation rules (plan, not executed)

**Dataset this is about:** `data/modelling_landmark_with_deltas.csv` (1040 patients, Aug 2026 extract). Landmark, mixed last/max **at or before index**, 101 already recoded to NaN then peak taken, Uncertain postural hypotension already dropped. **Do not use** the old `03_imputation` / `imputation_justification.csv` numbers (those were last-value, 1025 patients, 44% dopamine).

Range / plausibility (BMI visit typos, `DXOTHSX` vs `DXPOSINS`) was **Phase 1.5** and is applied in the current extract. PI-at-dx Yes is 92 (`DXPOSINS`). BMI max 46.2.

**Outcome is never imputed.** No `FLNFR12M` → not in the cohort.

**How to fill (when a fill is required):** statistics from the **training split only**, then applied to test. There is no pre-filled dense CSV — that would leak. Tree CSVs keep NaN; sklearn models call `apply_dense_imputation(train, test)` inside each seed. See **Two backends**.

Overall recurrent-fall rate: **9.7%** (101 / 1040).

---

## What missingness looks like now

After peak/last over the pre-index window, most clinic scores are **complete**. Imputation is no longer a 30-column `fillna(0)` problem.

| Tier | % missing | Features |
|------|----------:|----------|
| High | 19–33% | Δ Part IV (32.8%), Part IV score (21.7%), Δ Neuro-QoL (20.3%), **dopamine (19.4%)** |
| Moderate | 5–14% | Other deltas (5–14%), Neuro-QoL composite (6.2%), FOG (6.1%) |
| Tiny / none | ≤0.5% | Postural hypotension (5 people), one diagnosis item; Age, duration, MoCA, BMI, GDS, UPDRS I/III, gait, PS, H&Y, sleepiness, urinary, SCOPA, 101 flags all **complete** |

101 flags are 0/1 and fully filled. Gait / PS / H&Y **scores** are complete because extraction already took the peak *valid* value. The flag means “last visit was unable to rate,” not “this score is missing.”

---

## Feature-by-feature

### 1. Dopaminergic therapy (`DOPTHERST`) — **do not fill with 0**

**What it is:** PPMI form *Initiation of Dopaminergic Therapy*. Codebook: `0 = No` (has not started), `1 = Yes` (has started). Our feature is **ever started before the index date**.

#### Why “blank = not on therapy, so fill 0” sounds right — and why it is not

The protocol story is: staff skip the form if the patient has not started levodopa / a dopamine agonist, so a blank should mean No.

That would be true **if the form had no No box**. It does. `0` is a real answer someone typed. When PPMI means “not started,” they write `0`. A blank is a third state: **this form was never filled before our prediction date.**

A simple check: PATNO 3010 has `DOPTHERST = 0` on later visits, then `1`. Untreated people are not left empty; they are scored No until they start.

If blank really meant “too early for the form / not started,” the missing patients should look like the explicit-No patients: short disease, almost no Part IV (no drug side-effects yet), few falls.

They look like the opposite:

| Group | What we know | n | Recurrent falls | Median years with PD |
|-------|----------------|--:|----------------:|---------------------:|
| Form says **No (0)** | Not started (recorded) | 153 | **3.3%** | **2.4** |
| Form says **Yes (1)** | Started (recorded) | 685 | 9.8% | 4.1 |
| Form **blank** | Unknown | 202 | **14.4%** | **6.5** |

Read the table as three different people:

1. **Recorded untreated** — early PD, safest group. This is what `0` *actually* means.
2. **Recorded treated** — typical PPMI PD.
3. **Blank** — longest disease, *more* falls than people already on treatment. These are not “therapy hasn’t started.” They are “the initiation form never landed in our window” (started before enrollment, started at another site, form not used, visit after index, etc.).

Same picture on follow-up: blanks have **more** PPMI history (median 63 months) than explicit No (16 months), and Part IV is missing only 31% of the time (vs 95% in explicit No). If they were “not started,” Part IV should be almost entirely missing, like the real untreated group.

Filling those blanks with `0` does this: take the **highest-risk** 202 people and write the **lowest-risk** label on them. The model then learns “0 = safe” from the real untreated patients and applies it to the people who fall the most.

That is the whole argument. Part IV *can* be filled with 0 because a missing Part IV form means “no drug complications to rate,” and those patients really are the early/untreated group (1.3% recurrent, 2.0 y duration). Dopamine blanks are not that group. Same missingness rate, opposite clinical story.

**Rule:** add a **`dopamine_missing` indicator** (0/1) so “unknown” is a feature, not a fake No. Trees: leave `DOPTHERST` as NaN. Dense models that cannot take NaN: training-set **mode**, which will be **1** (the common, treated state) — still with the indicator. Never structural 0.

---

### 2. MDS-UPDRS Part IV total — **structural zero is justified**

**What it is:** motor *complications of dopaminergic therapy* (dyskinesia, OFF time, fluctuations) over the past week. MDS-UPDRS Part IV is not a general motor exam. PPMI de novo work often treats missing Part IV as 0 when building a total score, because untreated patients have no therapy complications to rate.

| Group | n | Recurrent | Median duration | Dopamine = Yes |
|-------|--:|----------:|----------------:|---------------:|
| Missing | 226 | **1.3%** | 2.0 y | 8% |
| Observed 0 | 273 | 4.0% | 2.9 y | 90% |
| Observed >0 | 541 | 16.1% | 9.6 y | 78% |

Missing is healthier than explicit 0, short duration, almost never on recorded therapy. That is “form not applicable,” not “we forgot the score.”

**Rule:** fill **0**. Optional: missingness indicator if you want “never opened Part IV” separate from “opened and scored 0.” Not required.

---

### 3. Freezing of gait (peak severity)

**What it is:** `FRZGT12M`, 12-month recall, 0–4 (none → frequent falls from freezing). We keep the **peak** before index.

**Why missing (6.1%, n=63):** 58/63 are single-falls-visit patients (no prior falls form by construction). Recurrent rate among missing ≈ overall (9.5% vs 9.7%). Observed 0 is 3.9% recurrent; any FOG is 23.7%.

**Rule:** trees — **leave NaN** (or mode 0; clinically the same as “not documented”). Dense — training-set **mode (0)**. Do not invent a FOG score. A flag is almost the same as `n_falls_visits == 1`; skip unless you want it explicit. **0 is on the scale** (none).

---

### 4. Neuro-QoL Gaussian composite (`able_weighted_score`)

**What it is:** 8 lower-extremity items (1 = unable … 5 = no difficulty), Gaussian-weighted (μ=2.5, σ=1.2) **per visit**, then the last valid visit before index. Range in data ~0.91–7.3. **0 is not a legal value.**

**“Use the most recent available” is already what Phase 1 did.** Extraction uses `_last_valid` on this column: last *non-missing* Neuro-QoL visit at or before index, not the last clinic row (which might not have had the form).

The 6.2% (n=64) who are still missing have **no Neuro-QoL form at all** before the index date (58 of them are single-falls-visit patients). There is no earlier score to carry forward. Options that look like “most recent available” but are not allowed:

| Idea | Why not |
|------|---------|
| Last Neuro-QoL *after* the index (or at the outcome visit) | Future information. The model would peek into the 12-month window we are trying to predict. |
| Last *row* even if the form is blank | That is how the old extract created extra missingness. We already switched to last-valid. |
| Borrow gait / UPDRS III as a stand-in | Those features are already in the model. A second model just to invent Neuro-QoL is extra complexity for 64 people. |

So “impute with most recent available” cannot fill these 64. What is left is a *between-patient* fill: pretend they scored like a typical measured patient.

**Rule:** trees — **leave NaN** (no invented mobility score). Dense models — **training-set median**. Never 0.

---

### 5. Deltas (all 10)

**What they are:** last valid minus second-last valid, after truncation. NaN means **fewer than two** measurements before index, not “no change.”

Most delta-missing groups have **0% recurrent** and short history (the low-risk, little-follow-up patients). Filling 0 (“stable”) would code “new to PPMI” as “not progressing,” which is confounded with the outcome.

Δ Part IV is the exception: 32.8% missing, 1.8% recurrent — mostly people who also lack Part IV itself.

**Rule:** trees — **leave NaN**. Dense models — **0 plus a `_missing` flag** (the 0 is a placeholder; the flag is the real information). Do **not** bare-0 as “stable.” Experiment plan already chose NaN for trees; ignore the older “impute deltas as 0” sentence in `aggregation_and_delta_rules.md`.

---

### 6. Scores that are already complete — **no imputation**

MoCA (0–30, higher = better), BMI (10–60 kept), GDS-15 (0–15), UPDRS I, UPDRS III, gait 0–4, postural stability 0–4, H&Y 0–5, daytime sleepiness 0–4, urinary 0–4, lightheaded 0–3, fainted 0–3, Age, `No_of_years`.

Filling any of these with 0 would still be clinically wrong (MoCA 0 = profound impairment; gait 0 = normal). They just no longer have holes.

**101 flags:** already 0/1, no missing. Keep as features. Do not recode the peak gait/PS/H&Y again.

---

### 7. Tiny holes

| Feature | n missing | Meaning | Rule |
|---------|----------:|---------|------|
| Postural hypotension | 5 | Ever Yes vs No after dropping Uncertain (2). The 5 are Uncertain-only. | Train **mode** (0 = never documented). Leave 0/1 as 0/1. |
| Postural instability at dx | 1 | 0/1/2=Unknown | Train **mode**. Leave **2 as 2**. |
| Rigidity at dx | 0 | same 0/1/2 | None. Leave 2 as 2. |

---

## Summary table (what to implement)

| Feature | Missing | Trees (NaN OK) | Dense (no NaN) | Do not |
|---------|--------:|----------------|----------------|--------|
| Dopamine (ever started) | 19.4% | **NaN** + `dopamine_missing` | Train **mode (1)** + same indicator | Structural 0 |
| UPDRS Part IV | 21.7% | **0** (clinical, both files) | **0** | Median |
| FOG peak | 6.1% | **NaN** (or mode 0; same story) | Train **mode (0)** | Invent a FOG score |
| able_weighted_score | 6.2% | **NaN** | Train **median** | 0; “last available” (already used) |
| All deltas | 5–33% | **NaN** | **0 + `_missing` flag** | Bare 0 as “stable” |
| Postural hypotension | 0.5% | Train **mode** (5 people; fill both) | Train **mode** | |
| Dx postural instability | 0.1% | Train **mode**; leave **2 as 2** | same | Recode 2 |
| Everything else in §6 | 0% | None | None | Blanket `fillna(0)` |
| `falls_class` | 0% | **Never** | **Never** | |

Train-only median/mode. Apply the same constants to test. Part IV = 0 is a *clinical* fill, not a sklearn workaround — it is already in the **tree** CSVs. Dense statistical fills are **not** saved to disk.

---

## Two backends (tree vs dense)

Phase 1 CSVs stay the NaN source of truth. Do not overwrite them.

| Backend | Examples | What to load / do |
|---------|----------|-------------------|
| **Trees** | XGBoost, LightGBM | `modelling_landmark_trees.csv` or `_with_deltas_trees.csv`. NaNs kept. |
| **Dense** | sklearn `RandomForestClassifier`, logistic regression | Same tree file → **split** → `apply_dense_imputation(train, test)` in `02_imputation.ipynb`. No `*_dense.csv`. |

Rule-based fills that are already in the tree files (not estimated from other patients):

- Part IV missing → 0
- `dopamine_missing` indicator
- hypotension / PI-at-dx tiny holes → mode (5 and 1 people; mode is 0 on any split)

Statistical fills that **must not** be a CSV (would leak the test fold):

- dopamine mode, FOG mode, Neuro-QoL median
- delta → 0 + `{col}_missing` (the 0 is a placeholder; the flag is the information)

`modelling_no_landmark.csv` stays a sensitivity extract.

**Why not one filled file for everyone:** a dense fill on deltas (0 = “stable”) is a different scientific claim from “this patient had only one visit.” Trees can keep that distinction. A full-cohort median/mode file looks official and will get used by accident — that is why those files were deleted.

---

## LLM layer (Nadine)

Imputed dopamine (dense fill), Part IV 0s that were missing, FOG 0s that were missing, and a median composite should not be narrated as measured facts. Prefer hide or “(not recorded; treated as …).” Deltas left as NaN, and `dopamine_missing`, should not be read out as clinical findings.

101 flags: already in the data; the explanation builder already drops “unable to rate” from display. Keep that. Peak gait can still drive SHAP while the flag is on.

---

## Why this differs from the meeting brief

The brief’s 44.4% / 34.5% / 6.9% were **last-value** on the old 1025-patient file. Peak-over-window + data refresh filled most clinic scales. Dopamine missing **fell** (44% → 19%) but the **shape** of missing did not flip: it still does not look like untreated. Rizvan’s “impute dopamine with 0” and the project-mate protocol story are noted; the table above is why we are not following them without an indicator.

Part IV structural zero is the one high-missing fill that the data and the instrument both support (and that published PPMI progression papers have used for totals).
