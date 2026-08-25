# Phase 1.5 — Range / plausibility audit (plan)

**When:** after Phase 1 extract, **before** imputation fills. Do not overwrite the Phase 1 CSVs until the rules below are signed off.

**Why:** Comment 3 is about *missing* values. It does not catch a filled-in wrong number. The submitted paper already had one of those (an impossible BMI). A short codebook-and-range pass is worth doing once; a free-form “explore all features” EDA is not.

**Where to put it:** first section of the rebuilt `02_imputation.ipynb` (audit → missingness → fills). Not a separate fishing notebook.

Checked on `modelling_landmark_with_deltas.csv` (n=1040) against `CodeBook/` and the Aug 2026 vitals. **Applied in `01_data_extraction.ipynb` (re-run 25 Aug 2026):** `DXPOSINS` swap + BMI visit cleanup. Landmark n still 1040; PI-at-dx Yes is 92; BMI max 46.2; `|ΔBMI|>10` is 0. 35 weight voids and 128 height fixes on the raw vitals table (not all in the landmark cohort).

---

## What this audit should (and should not) do

| Do | Do not |
|----|--------|
| Check every modelling column against the codebook **codes** (0–4, 101, 2=Unknown, …) | Drop a patient because they are obese, old, or a long-duration PD case |
| Check **which ITM** we actually pulled (name vs meaning) | Treat “values look rare” as an error |
| Void **visits** that are physically impossible, then re-take last-valid / peak | Overwrite a bad BMI with the cohort mean (legacy `25.05736`) |
| Hand-review a short list of huge within-patient jumps | IQR / z-score trims on clinical scales |

Two layers, because a legal code can still be the wrong variable:

1. **Legality** — is this value in the codebook?
2. **Identity / plausibility** — is this the right field, and could a human have this body?

---

## 1. The paper’s extreme BMI

Legacy (`Model_Development.ipynb`):

```text
BMI > 123  → replace with 25.05736   # cohort mean, one patient
BMI > 50   → drop the patient
BMI ≤ 0    → drop (this also dropped missing BMI after fillna(0))
```

**The one extreme patient is PATNO 42171.** Latest height/weight in the old extract was Dec 2019: `HTCM=82`, `WGTKG=83` → BMI **123.4**. Height 82 cm is a toddler; their other visits are 182 cm / BMI ~26. That is a data-entry error (likely 182 typed as 82), not a person. They are **not in the current landmark 1040** (cohort construction, not BMI). Legacy kept them by inventing BMI = 25.06.

**Do not repeat the mean overwrite.** An invented 25 looks like a real measurement in SHAP / the LLM summary.

**Do not repeat BMI > 50 as a patient drop.** Class III obesity (BMI ≥ 40) is real. The current extract’s maximum is PATNO **157041 at 54.6**, which *looks* extreme but is inside 10–60 so the visit filter kept it. Their vitals:

| Visit | Height | Weight | BMI | Notes |
|-------|-------:|-------:|----:|-------|
| 01/2024 | 155 cm | 77.0 kg | 32.0 | plausible |
| **02/2025** | 180 cm | **177.0 kg** | **54.6** | **this is the last pre-index value** |
| 04/2026 | 180 cm | 79.4 kg | 24.5 | after outcome; 177 → 79 kg is not real weight loss |

177 kg next to 77 / 79 kg is an extra digit, not morbid obesity. The 10–60 window cannot see that. **Void that visit, do not drop the patient.** Last-valid BMI would then be ~32.

Phase 1 already treats BMI outside **10–60** as missing at the visit, then last-valid. That correctly kills BMI 123, 300, 56200 (`HTCM=5`). It is **necessary and not sufficient**.

---

## 2. BMI: 10–60 misses unit / typo visits

Six landmark patients have `|Δ BMI| > 10` between the last two *kept* visits. Spot-checking vitals, they are almost all height-unit mix-ups or lb-as-kg, not diet:

| PATNO | What last-valid used | What the series actually is |
|------:|----------------------|-----------------------------|
| 157041 | BMI 54.6 (177 kg) | Extra digit; true weight ~77–79 kg |
| 201630 | BMI 21.5 after a 48.3 | First weight **119** then **53 kg** at the same height — 119 **lb** ≈ 54 kg |
| 219153 | BMI 44.5 at **155 cm** | Heights flip 180 / 155 / 180; weight stable ~100 kg. 155 cm visit is the wrong height |
| 137482 | BMI 37.2 at 125 cm | One visit `HTCM=5` already voided; remaining 152 vs 125 cm still looks like a height error |
| 167222 / 143835 | | Same 152–155 vs 180–184 cm flip |

**Proposed visit rule (extraction, not a patient filter):**

1. Keep BMI outside 10–60 as NaN (already done).
2. If a patient’s **non-missing heights** differ by more than ~10 cm, treat the minority height’s visits as NaN (or recompute BMI from that patient’s median height). Adult height does not jump 25 cm.
3. If consecutive valid BMIs differ by more than ~10 points, void the visit that disagrees with the patient’s other weights (hand list is six people; do not automate a harsh cut until we see the notebook).
4. Recompute last-valid BMI and Δ BMI after those voids.
5. **Never** fill the hole with 25.06. If every visit is voided, BMI is missing → Phase 2 median (dense) / NaN (trees).

---

## 3. Codebook legality — already clean on the extract

Every discrete modelling column is inside the codebook. No leftover **101**. No duplicate `PATNO`. No infinities.

| Feature | Codebook | Observed |
|---------|----------|----------|
| FOG `FRZGT12M` | 0–4 | 0–4 |
| Gait, PS | 0–4 (101 = unable) | 0–4; 101 already NaN then peak |
| H&Y | 0–5 (101 = unable) | 0–5 |
| Sleepiness, urinary | 0–4 | 0–4 |
| SCOPA lightheaded / fainted | 0–3 | 0–3 |
| Dopamine | 0/1 | 0/1 / NaN |
| Hypotension | 0/1/2=Uncertain | 0/1 / 5 NaN (Uncertain-only; Phase 1 already dropped 2) |
| Dx rigidity / “PI” | 0/1/2=Unknown | 0/1/2 |
| MoCA | 0–30 | 4–30 (no illegal 0) |
| GDS-15 | 0–15 | 0–15 |
| Part I | 0–52 | 0–17 |
| Part III | 0–132 | 2–96 |
| Part IV | 0–24 | 0–19 |
| Neuro-QoL composite | Gaussian of 1–5 × 8 | 0.91–7.33 (matches all-5 and all-2/3 bounds) |
| Age | — | 35–92 |
| Duration | — | 0.3–25.3 y (none negative) |
| `falls_class` | 0/1/2 | 0/1/2 |

A range check **alone would have passed the mislabeled diagnosis field below.** That is why layer 2 exists.

---

## 4. Wrong variable: “postural instability at dx”

Codebook (`PDDXHIST`):

| ITM | Meaning |
|-----|---------|
| `DXPOSINS` | **Postural instability present at dx?** |
| `DXOTHSX` | Other symptom present at diagnosis? |
| `DXRIGID` | Rigidity present at diagnosis? |

Legacy and `01_data_extraction.ipynb` both pull **`DXOTHSX`** and rename it to `'Postural instability present at dx?'`. The values are legal 0/1/2, so nothing looks “out of range.” They are the **wrong question**.

On the landmark 1040, `DXOTHSX` vs `DXPOSINS` agree only **73%**. True PI-at-dx Yes is **92** people; the column we model as PI-at-dx Yes is **187** (other symptoms).

**Fix in Phase 1:** load `DXPOSINS`, keep the display name. Leave `2 = Unknown` as 2. Then re-save the CSVs. This is a one-line extract change, not an imputation rule.

---

## 5. Huge deltas that are probably not typos

`Δ` UPDRS III of +62 / −41 is large but MDS-UPDRS III moves with ON vs OFF, rater, and real progression. Phase 1 already collapses same-day rows with `max`. **Do not trim these.** Trees can ignore wild deltas; dense fills should not invent a “typical change.”

`Δ` MoCA −13, GDS ±12, gait ±3 are on-scale (one step of the instrument). Leave them.

---

## 6. Summary — what to change vs leave

| Issue | Action |
|-------|--------|
| PATNO 42171 BMI 123 | Already out of the 1040; visit rule 10–60 would have skipped it anyway. Do not mean-impute. |
| PATNO 157041 BMI 54.6 | **Do not drop.** Void the 177 kg visit; last-valid ~32. |
| Other `|ΔBMI|>10` | Void the bad height/weight visit; recompute. |
| BMI 10–60 | Keep. |
| BMI > 50 patient drop (paper) | **Do not bring back.** |
| 101 in gait / PS / H&Y | Already handled. |
| `DXOTHSX` labeled as PI at dx | **Switch to `DXPOSINS`.** |
| UPDRS III deltas | Leave. |
| `falls_class` | Never touch. |

After the BMI visit voids and the `DXPOSINS` swap, **re-run `01_data_extraction.ipynb`**, then do Phase 2 fills on the new CSVs. Missingness % will move slightly (a few BMI NaNs if every visit is voided; PI-at-dx counts will change).
