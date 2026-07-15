# <Experiment title>

## Status

<Planned | Running | Complete | Inconclusive | Abandoned>

## Goal and decision

<State the specific decision this experiment can support.>

## Hypothesis

<State a falsifiable hypothesis and its expected measurable effect.>

## Theory and assumptions

<Describe the relevant WPT/Transformer theory, coordinate conventions, and
assumptions. Link to track documentation instead of duplicating general theory.>

## Configuration

- Model and immutable revision: `<model-id>@<revision>`
- Tokenizer revision: `<revision>`
- Source format and precision: `<format>` / `<precision>`
- Data and revision: `<source>@<revision>`
- Split and selection rule: `<rule>`
- Seed(s): `<seed>`
- Hardware and software: `<details>`
- Command: `<exact command>`
- Full declarative config: [configs/experiment.yaml](configs/experiment.yaml)

## Baseline and controls

<List the unmodified baseline and every control, including random controls where
the claim concerns WPT structure.>

## Method

<Describe exactly what is transformed, measured, masked, or trained. Include
tensor shapes and transform/block definitions when relevant.>

## Evaluation and acceptance criteria

<Declare metrics, aggregation, numerical tolerances, and pass/stop criteria
before reporting final results.>

## Planned ablations

- <Ablation and why it matters>
- <Ablation and why it matters>

## Results

<Add dated result entries. Link raw files under `results/`; distinguish measured
values from interpretation. State “No results yet” initially.>

## Conclusion and next decision

<State what the current evidence supports, does not support, and what should
happen next.>

## Reproducibility notes

<Record deviations, unavailable artifacts, nondeterminism, failed runs, and
retrieval/checksum information.>
