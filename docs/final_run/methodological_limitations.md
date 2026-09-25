# Methodological limitations and interpretation boundaries

**Applies to:** Final 26-feature PD fall-risk analysis
**Documented:** 25 September 2026

These limitations are maintained separately from the concise final results
report. They apply when interpreting, presenting, or publishing the results.

1. **Internal validation:** design decisions followed exploratory work on this
   cohort. Independent PPMI data or a future cohort are needed for publication-
   strength confirmation.
2. **Repeated patient use:** the 20 train/test splits overlap. Corrected
   resampled intervals partly address this dependence but remain approximate.
3. **Rare and recurrent classes:** only 227 rare and 101 recurrent fallers are
   available, so class-specific estimates vary across splits.
4. **Historical aggregation:** maximum values summarize greatest recorded
   pre-landmark burden and may reflect remote rather than current symptoms.
5. **Part III states:** combined-state maxima mix ON and OFF examinations. The
   matched sensitivity found no clear advantage for separating them, but the
   preferred clinical interpretation remains open.
6. **Neuro-QoL score:** the Gaussian composite is inherited from the revision
   and is not a standard Neuro-QoL score. Its interpretation must be stated
   explicitly.
7. **Model interpretation:** permutation importance measures dependence of a
   fitted model on a variable group. It does not measure causality, clinical
   benefit, or the value of collecting that variable in isolation.

8. **SHAP explanations:** the final SHAP artifacts explain the fitted
   full-cohort direct, Stage 1, and Stage 2 models. They are not independent
   performance estimates or causal explanations. Stage 1 and Stage 2 SHAP
   values must remain separate because hard routing is not additive.
