# PD Fall-Risk Prediction Project

## Project goal

Predict three Parkinson's disease fall-risk classes:

- no fall
- rare fall
- recurrent fall

This is a research project. Scientific validity, leakage prevention,
reproducibility, and transparent evaluation take priority over maximizing
a single performance metric.

## Performance ambition and search scope

- Aim to develop the strongest defensibly validated model for publication. The existing model families and feature list are starting points, not restrictions.
- Consider other ML approaches and additional PPMI variables when there is a justified prospect of improving prediction. Do not assume that either existing architecture must be the final published model.
- Low rare-fall recall is a central concern. Evaluate improvements using macro F1 and per-class recall, with explicit attention to rare fallers and tradeoffs for the other classes.
- Broader exploration must preserve the landmark design, training-only development, and unbiased evaluation. Never choose the final model, features, or decision thresholds by repeatedly inspecting held-out test performance.
- Verify clinical definitions, collection schedules, temporal availability, and applicable access/use/publication requirements against official PPMI documentation when needed; do not assume that every PPMI variable is available or eligible.
- Explain clinical and PPMI-specific choices in plain language. Discuss additional data acquisition and unresolved scientific choices with the user before implementing them.
- Follow the phase-by-phase roadmap. This expanded ambition does not authorize skipping ahead to modeling or feature selection.

## Scientific invariants

Do not violate these rules unless the user explicitly changes the study design.

- Use the landmark design.
- Predictor information must occur at least 12 months before the outcome visit.
- Outcome variables must never be used as predictors.
- Outcomes must never be imputed.
- Split data before fitting any data-dependent preprocessing.
- Imputation statistics must be learned from training data only.
- Scaling, encoding, feature selection, and other learned transformations must
  be fitted using training data only.
- Treat 101 as missing only when the field-specific PPMI codebook defines it as
  "unable to rate." This includes gait, postural stability, Hoehn-Yahr, and the
  six audited rater-completed MDS-UPDRS Part I items. A measured vital-sign
  value of 101 remains an ordinary measurement.
- Do not exclude patients because an eligible Part III assessment contains 101.
  Compare otherwise matched branches with and without 101 assessment-status
  flags, separately from the candidate-feature enhancement track.
- The no-delta dataset is currently the primary dataset.
- The manuscript-oriented final run uses 26 candidate predictors: 25 eligible
  paper-aligned variables plus maximum eligible `NP1CNST`. Neuro-QoL mobility
  enters modeling only through the revision's 8-item Gaussian composite
  (`NQ_GAUSSIAN_REVISION`). Its seven component items may be used to construct
  that composite but must not also enter the model as separate predictors
  unless the research protocol is explicitly amended.

## Evaluation

Do not use test-set performance to select features, hyperparameters,
preprocessing decisions, or model architecture.

With repeated outer folds, do not pool outcome-guided diagnostics across outer-training partitions to choose one candidate list for every fold. A patient held out in one fold belongs to other outer-training partitions. Advance candidates using only the corresponding partition's inner folds; pooled frequencies are descriptive.

Primary evaluation should emphasize:

- macro F1
- per-class recall, especially rare-fall recall

Accuracy/balanced accuracy would be reported but must not be treated as the primary measure.

When comparing modeling approaches across the existing experimental framework,
preserve paired seeds so that models are evaluated on corresponding splits.

Do not draw general conclusions from seed 42 or any other single seed.
Single-seed analyses may be used for diagnostic/error analysis only.

## Current modeling context

Both approaches exist:

1. Two-stage classification:
   - Stage 1: no fall vs any fall
   - Stage 2: rare fall vs recurrent fall

2. Direct three-class classification.

The current evidence favors direct classification for macro F1 and rare-fall
recall, while the two-stage system has shown stronger no-fall recall.

Do not assume the two-stage architecture is the preferred architecture merely
because it existed in the legacy pipeline.

## Research workflow

Before changing the study design, preprocessing, feature selection,
model evaluation, or outcome definition, read:

`docs/research_protocol.md`

Do not silently change scientific methodology.

If implementation and documented methodology disagree, flag the discrepancy
rather than guessing which one is correct.

## Coding behavior

- Use Jupyter notebooks (`.ipynb`) in `notebooks/` as the default for new analysis work, with code and results visible cell by cell.
- Before creating a `.py` file or moving notebook logic into one, explain the concrete advantage and obtain the user's approval. This applies even when local skill guidance recommends `src/`; reusable functions can initially live in clearly organized notebook cells.
- Keep code readable, concise, and easy to understand. Use descriptive names, straightforward logic, and avoid unnecessary abstractions.
- Keep notebook code cells short and focused on one clear task. Longer cells are acceptable when a complete function, pipeline, or similarly cohesive unit would become harder to understand if split apart.
- Precede every code cell with a short Markdown cell stating what the code will do and why. Use section-level Markdown cells to explain the broader purpose of each group of steps.
- Show a concise result after a code cell when the output helps verify or interpret the step. Avoid noisy output and never dump raw patient records.
- Keep notebooks runnable from a fresh kernel in top-to-bottom order, with explicit configuration, paths, and random seeds; avoid hidden execution state.
- Document non-obvious decisions and reusable functions, and update active documentation when the workflow changes.
- Use `archive/legacy_project/revision_work/` as the primary reference for the updated work. Consult earlier legacy work only when relevant to a specific question, and do not mix older raw exports into the updated data without discussing it with the user.
- Inspect the existing implementation before modifying it.
- Prefer small, traceable changes over unnecessary rewrites.
- Preserve reproducibility through explicit random seeds.
- Avoid hard-coding results that can be calculated or are stored safely.
- Keep data processing, modeling, evaluation, and plotting logically separated.
- Organize notebooks under numbered phase folders that show workflow chronology.
- Do not overwrite existing experiment outputs unless explicitly requested.
- When adding an experiment, make its configuration and outputs reproducible.
- After changing analysis code, run the relevant checks or tests when available.

## Documentation and agent handoff

- Keep important decisions portable between coding agents without documenting every trivial code change or interrupting implementation for unnecessary edits.
- Update `AGENTS.md` only when a permanent project rule, coding convention, or methodological guardrail changes.
- Update `docs/research_protocol.md` for important scientific or methodological decisions.
- Update a `SKILL.md` only when new learning changes the reusable procedure it describes.
- Maintain `docs/current_state.md` as a concise handoff covering meaningful progress, major decisions, completed experiments, unresolved issues, and the next task. Replace obsolete state instead of keeping a chronological diary.
- Treat Git history and current code as the authoritative implementation record; do not duplicate implementation history in documentation.
- At meaningful milestones or the end of substantial work sessions, check for important conversation-only information that should be preserved in the repository.
