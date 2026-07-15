# WPT-LLM project guidance

## Purpose

This project develops, tests, evaluates, and documents Wavelet Packet Transform
(WPT)-based approaches to large language models. It is both a research project
and a learning exercise: tools, methods, and conclusions may change as evidence
accumulates.

The work should remain scientifically useful even when an idea does not work.
Every experiment must therefore be reproducible, documented, and planned with
cost efficiency in mind.

## Core principles

- **Reproducible:** record the model and tokenizer revisions, data source and
  split, code revision, configuration, seed, hardware, precision, software
  environment, commands, and evaluation protocol.
- **Documented:** keep the relevant living experiment documents current before
  reporting an experiment as complete, inconclusive, or abandoned.
- **Cost-aware:** begin with the smallest experiment that can falsify or support
  a hypothesis. Use smoke tests, calibration subsets, and microbenchmarks before
  expensive training, broad sweeps, or custom kernel work.
- **Evidence-driven:** separate exact algebraic claims, empirical observations,
  approximations, and speculation. Always compare to an appropriate baseline.
- **Learning-oriented:** record dead ends, unexpected outcomes, changed methods,
  and terminology decisions. Do not rewrite history to make an experiment look
  more linear than it was.

## Experiment layout

All experiment work belongs under `experiments/`. Do not place run artifacts,
configs, scripts, or ad-hoc notebooks at the repository root.

Use one directory per experiment track and one directory per individual
experiment:

```text
experiments/
  <track-name>/
    README.md
    experiment.md
    <experiment-name>/
      README.md
      experiment.md
      configs/
      scripts/
      results/
      artifacts/          # optional; do not commit large model/data files
```

Examples of tracks include feature-space WPT checkpoint compilation,
block-sparse kernel feasibility, local continual adaptation, and sequence-axis
time-scale/block-causal generation. A track may be split or renamed when the
research question changes; document the reason in its `experiment.md`.

## Required documentation

### `experiments/<track>/README.md`

This is the high-level entry point for a track. Write plainly and naturally, not
as a rigid template. It should explain:

- what the track is trying to learn;
- why it matters to WPT-based LLMs;
- its current status and nearest next step;
- how the experiments within it relate to one another;
- links to the track `experiment.md` and individual experiments.

Keep it short enough to orient a new reader quickly. It is an overview, not a
run log.

### `experiments/<track>/experiment.md`

This is the track-level living research document. Keep it synchronized with the
actual state of the track. It must cover:

- goals and scope;
- hypotheses and the theory motivating them;
- baseline and comparison strategy;
- evaluation criteria and success/failure gates;
- planned ablations and their rationale;
- current results, including negative or inconclusive results;
- current conclusions, limitations, open questions, and next steps;
- links to detailed individual experiments.

The track document gives the coherent current story. Detailed commands, run
tables, raw observations, and implementation specifics belong in the individual
experiment directories.

### `experiments/<track>/<experiment>/README.md`

Provide a concise, plain-language overview of one concrete experiment: what was
run, what it was meant to decide, current status, and links to its detailed
document, configuration, scripts, and results.

### `experiments/<track>/<experiment>/experiment.md`

This is the highest-detail living record. It must describe:

- the precise goal and hypothesis;
- relevant theory and assumptions;
- model/checkpoint, tokenizer, data, and environment;
- exact configuration, seed, commands, and implementation choices;
- baselines, controls, evaluation metrics, and acceptance criteria;
- ablations, including those not yet run;
- results with enough detail to reproduce the comparison;
- conclusions, failure modes, limitations, and the next decision.

Update this file as the experiment changes. Do not wait until the end to write
it, and do not present planned work as completed work.

## Reproducibility requirements

Before a non-trivial run, create a committed configuration file under that
experiment's `configs/` directory. Prefer declarative configuration over
parameters embedded only in shell history or notebooks.

For every reported result, retain or record:

- an immutable model identifier and revision/commit when available;
- dataset identity, revision, split, preprocessing, and sample selection;
- random seed(s) and deterministic settings;
- hardware and software versions;
- precision, quantization, context length, batch size, and generation settings;
- exact command or script entry point;
- metric definitions and aggregation method;
- enough raw outputs or summaries to audit the reported metric.

Do not claim an implementation is function-preserving without a numerical
equivalence test on fixed inputs. For WPT checkpoint compilation, record maximum
logit difference, loss/perplexity difference, and the numerical precision used.

## Baselines, ablations, and evaluations

Every experimental claim needs a comparison that could disprove it.

For WPT basis-selection work, include relevant controls such as original
coordinates, random permutations, and random orthogonal bases. For pruning or
structured kernels, distinguish nominal sparsity from measured end-to-end
latency. For continual-learning claims, report both new-domain performance and
retention on a fixed prior evaluation set.

Keep these categories separate in documentation and plots:

1. exact coordinate compilation;
2. measured structural properties;
3. approximate masking, pruning, quantization, or local updates;
4. custom-kernel performance;
5. sequence-axis/block-causal architectural experiments.

Feature-space checkpoint compilation and sequence-axis WPT generation are
different tracks. Do not combine their results into one claim without a direct
experiment demonstrating the interaction.

## Cost discipline

Use a staged decision process:

```text
theory and shape checks
  → exact smoke test
  → small calibration measurement
  → targeted ablation
  → small training/adaptation run
  → larger confirmation run
  → kernel engineering or broad sweep
```

Advance only when the preceding stage produces evidence worth pursuing. Record
stop conditions in `experiment.md`, including when an experiment should be
abandoned or redesigned.

Before downloading large checkpoints, launching long training, or building
specialized kernels, confirm that a smaller model or microbenchmark cannot
answer the current question. Prefer reusable measurement harnesses and cached
calibration data where that does not compromise reproducibility.

Never commit model weights, large datasets, or large generated artifacts unless
the repository explicitly adopts a storage solution for them. Store artifact
locations, hashes, and retrieval instructions in the experiment record instead.

## Implementation and reporting

- Keep scripts scoped to their experiment directory unless they are genuinely
  reusable; put shared utilities under a clearly named project-level location.
- Make scripts fail clearly on missing prerequisites and print the resolved
  configuration and output location.
- Keep raw result files separate from interpreted summaries.
- Include unsuccessful runs that materially affect a conclusion; label them
  clearly rather than deleting them.
- Update the track and experiment living documents whenever a result changes
  the current hypothesis, next step, or decision gate.
- Do not describe benchmarks, speedups, quality preservation, or conclusions as
  established without linking them to recorded evaluation evidence.

## Changelog

Maintain the repository-root `CHANGELOG.md`. It documents all notable project
changes, including research plans, experiment tracks, implementation changes,
evaluation results that materially change conclusions, and removals or
deprecations.

Follow the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) categories:
`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, and `Security`. Use the
categories that apply; do not add empty headings solely to fill a template.

Use an `Unreleased` section for work not yet released. When creating releases,
date and version them according to [Semantic Versioning](https://semver.org/).
Update the changelog in the same change set as the work it describes. Keep it a
concise record of externally meaningful changes, not a replacement for detailed
experiment documentation.
