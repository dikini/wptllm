# Feature-space compilation playbook

## Purpose

This playbook turns the feature-space WPT track into small, reproducible,
cost-aware experiments. It is a guide for preparing an individual experiment,
choosing a model and software stack, deciding when to advance, and keeping the
living documentation current.

The working principle is simple:

```text
prove exactness → measure structure → test approximation → test value
```

Do not skip a stage because a transformed weight visualization looks promising.

## Model-selection policy

Choose models for the uncertainty they remove, not only their parameter count.
The first portfolio should include one clean reference model, one GQA/shape
stress case, and one modern confirmation model.

| Role | Recommended model | Why use it | Main limitation |
|---|---|---|---|
| Reference | [`HuggingFaceTB/SmolLM2-1.7B`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) | Conventional Llama decoder, $d=2048$, full MHA, tied embeddings, simple RoPE/RMSNorm. | Does not exercise GQA. |
| Cheap stress case | `Qwen/Qwen2.5-0.5B` or its base equivalent | GQA and $d=896$ test non-dyadic channel grouping and asymmetric Q/KV shapes. | Less convenient WPT geometry; do not treat it as the only basis-selection model. |
| Dense confirmation | [`HuggingFaceTB/SmolLM3-3B-Base`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base) | $d=2048$, GQA, larger depth, and a modern positional configuration. | More layers and layer-specific positional behavior. |
| Later extension | Qwen3.5 text or multimodal checkpoint | Tests hybrid linear/full attention and broader architecture transfer. | Separate research problem; do not use for the first compilation experiment. |

### Selection checklist

Before selecting a checkpoint, record:

- official model identifier and immutable revision;
- base versus instruction checkpoint, with base preferred for NLL/PPL work;
- source precision: prefer official BF16/FP16 `safetensors`;
- residual width, head dimension, query/KV head counts, layer count, and MLP
  width;
- norm, activation, embedding tying, RoPE/positioning, and attention layout;
- tokenizer/vocabulary size and license;
- availability of a supported reference implementation;
- expected memory, download, and calibration cost.

Use GGUF only as a later deployment target or an explicitly quantized-source
experiment. A quantized GGUF source cannot establish exact equivalence to the
original full-precision checkpoint.

### Width and packet geometry

A dyadic residual width such as $2048=2^{11}$ permits a full packet tree. For a
non-dyadic width such as $896$, predeclare a grouping rule, for example a
blockwise transform over 128-wide channel groups. Do not pad dimensions silently
or choose a grouping only after seeing favorable results.

## Software stack policy

Start with an inspectable PyTorch reference implementation. Add deployment or
kernel tools only after the exact and structural gates pass.

### Bootstrap stack

Pin versions in the experiment configuration and environment lock file.

| Need | Initial choice | Purpose |
|---|---|---|
| Model execution | PyTorch + Hugging Face Transformers | Load official checkpoints and expose hidden states. |
| Weights | Safetensors + Hugging Face Hub | Immutable checkpoint download and tensor access. |
| WPT analysis | PyWavelets for offline candidate analysis; a tested Torch implementation for model transforms | Basis discovery versus reproducible tensor conversion. |
| Data | Datasets or versioned local text fixtures | Calibration and held-out evaluation. |
| Configuration | YAML or JSON in the experiment `configs/` directory | Declarative, reviewable runs. |
| Tests | Pytest | Exactness, shape, and serialization checks. |
| Metrics | Torch plus a small project-owned evaluation harness | NLL, PPL, logit divergence, timing, structural metrics. |

### Later additions

- `accelerate` only when a run genuinely needs multi-device or memory management;
- GGUF/llama.cpp only after FP16/BF16 correctness is established;
- Triton, CUDA, or backend-specific tooling only after a block pattern has a
  positive quality/structure Pareto point;
- LoRA/PEFT tooling only for the local-adaptation stage.

Do not let specialized inference packages, quantization, or a custom kernel
become an untracked second source of model differences during F0–F2.

## Bootstrap an individual experiment

Create one directory under this track for one decision. Use descriptive ordered
names, for example:

```text
experiments/feature-space-compilation/
  01-exact-compilation-smoke/
  02-basis-structure-comparison/
  03-block-mask-pareto/
```

Copy the project-wide templates from
[`templates/individual-experiment/`](../../templates/individual-experiment/):

```text
<experiment>/
  README.md
  experiment.md
  configs/
    experiment.yaml
  scripts/
  results/
```

Before executing a non-trivial run:

1. Fill in `configs/experiment.yaml`, including model revision, dataset revision,
   seed, precision, device, transform, block layout, and output path.
2. Write the hypothesis, baseline, acceptance criterion, and stop condition in
   `experiment.md` before viewing final results.
3. Add or identify the test for the implementation property being relied upon.
4. Record the exact command that will generate results.
5. Run the smallest smoke test first and record its outcome, even if it fails.

The individual experiment is not complete until its `README.md`,
`experiment.md`, configuration, and results links agree about its status.

## Decision points for the track

### D0 — select a source and define the reference

Choose one source checkpoint and a fixed calibration/evaluation bundle. Record
the baseline before modifying weights. If the source cannot be loaded in a
reproducible full-precision reference path, choose another source before writing
conversion code.

### D1 — establish exact compilation

Run the transform harness in this order:

```text
identity → permutation → signed permutation → Haar → selected packet basis
```

For every step, compare source and compiled logits on fixed token sequences.
Declare the tolerance in advance and make it precision-aware. Record maximum and
mean absolute logit difference, NLL/PPL difference, and exact configuration.

Do not continue to basis selection, pruning, or kernel work until this gate
passes.

### D2 — establish structural evidence

Compare the selected WPT basis with original order, multiple random
permutations, and random orthogonal bases where practical. Predeclare the
metrics, such as:

$$
E_{\rm offblock}(W')=
\frac{\|\operatorname{OffBlock}(W')\|_F^2}{\|W'\|_F^2},
$$

activation energy concentration, cross-block covariance, and thresholded
coefficient sparsity.

Advance only if the basis improves the declared metric over the distribution of
random controls and the result survives an independent calibration split.

### D3 — test approximate structure

Introduce a declared block mask only after D1 and D2:

$$
\widehat W=M\odot W^{(W)}.
$$

Sweep a small predeclared set of layouts and retained-energy budgets. Report
quality, divergence, retained energy, nominal sparsity, memory, and timing.
The output is a quality/structure Pareto curve, not a single best number.

### D4 — test local learning

Only if D3 yields a useful operating point, compare unrestricted adaptation,
WPT-block LoRA or adapters, local block-only updates, and optional periodic
global consolidation. Evaluate both new-domain learning and retention on a fixed
prior set.

### D5 — test kernel value

Only if D3 yields a stable block pattern, benchmark the actual tensor shapes for
batch-1 decode and prefill. Compare against the best dense implementation on the
same hardware. A sparse kernel is successful only if it improves realized
latency, memory, or cost at an acceptable quality point.

## First three individual experiments

### 01-exact-compilation-smoke

Use SmolLM2-1.7B or a smaller compatible reference. Implement only the source
baseline, identity/permutation/Haar compilation path, and fixed-input numerical
equivalence checks. No basis optimization or pruning.

### 02-basis-structure-comparison

Use the exact harness from experiment 01. Compare fixed WPT candidates and
controls on a small versioned calibration set. This experiment produces
structural measurements, not speed claims.

### 03-block-mask-pareto

Use the best basis selected by a declared rule from experiment 02. Apply a small
mask sweep and produce the first quality/structure curves. Do not start kernel
work unless this result has a credible operating point.

## Documentation and result handling

Update the individual `experiment.md` after each material run. Update this
track-level `experiment.md` when a result changes the current conclusion, the
selected basis, a gate, or the next step. Update `CHANGELOG.md` in the same
change set for notable track, implementation, or conclusion changes.

Raw model weights, datasets, and large activation dumps must not be committed.
Record their source, revision, checksum where available, retrieval instructions,
and local artifact location instead.

## Bootstrap exit condition

The bootstrap is complete when `01-exact-compilation-smoke` exists with a filled
configuration, precise source revision, fixed test inputs, declared tolerance,
reproducible command, and living documents. It is not complete merely because a
checkpoint has been downloaded.
