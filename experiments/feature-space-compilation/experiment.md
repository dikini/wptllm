# Feature-space WPT checkpoint compilation

## Status

Planned. No checkpoint has yet been downloaded, compiled, or evaluated.

Use [playbook.md](playbook.md) to select a model, bootstrap an individual
experiment, and apply the track decision gates.

## Goal

Determine whether a decoder-only checkpoint can be exactly compiled into a
feature-space WPT basis and whether the resulting coordinate system exposes
block structure with practical quality, adaptation, or performance value.

For an orthonormal feature basis $S$, define $z=Sh$. The source checkpoint frame
is denoted $(O)$ and the compiled WPT residual frame $(W)$.

## Scope and non-goals

This track covers feature-axis WPT compilation of existing models. It does not
cover sequence-axis WPTs, block-causal generation, learned lifting, or generic
cross-model merging. Those are separate research tracks.

The initial implementation should retain conventional Q/K/V head coordinates
and RoPE. It should not apply a WPT explicitly at inference.

## Exact compilation theory

Compile residual-stream boundaries once into stored tensors:

$$
W_e^{(W)}=S W_e^{(O)},
$$

$$
W_q^{(W)}=W_q^{(O)}S^\top,\quad
W_k^{(W)}=W_k^{(O)}S^\top,\quad
W_v^{(W)}=W_v^{(O)}S^\top,\quad
W_o^{(W)}=S W_o^{(O)},
$$

$$
W_{\rm up}^{(W)}=W_{\rm up}^{(O)}S^\top,\quad
W_{\rm down}^{(W)}=S W_{\rm down}^{(O)},\quad
W_h^{(W)}=W_h^{(O)}S^\top.
$$

The norm implementation requires model-specific care. For RMSNorm, an ordinary
per-channel scale is generally not diagonal after a general basis change; where
valid, fold its source scale into the following projections and use a unit scale
in the compiled residual frame. Record the exact treatment in each individual
experiment.

Exact compilation alone preserves parameter count and does not imply speedup.
The empirical question is whether selected bases produce useful approximate
block diagonality or activation concentration.

## Hypotheses

### H1 — exact re-expression

For fixed inputs, the compiled model matches source logits and NLL within the
expected numerical tolerance of the selected precision.

### H2 — structural advantage

A WPT basis chosen from a declared basis family improves one or more structural
metrics over original coordinates and random controls.

### H3 — useful approximation

At matched retained block energy, a WPT-space block mask gives a better
quality-versus-structure trade-off than controls.

### H4 — local adaptation or kernel value

For a successful H3 operating point, local block adaptation improves a defined
retention/cost metric, or a structured kernel improves actual target-shape
latency over the best dense baseline.

## Initial models and protocol

Start with a small open-weight decoder model, then confirm a positive result on
a second architecture. A practical smoke-test candidate is
[`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct);
use a base checkpoint when available for perplexity-oriented evaluation.

Each individual experiment must record model/tokenizer revision, precision,
hardware, software, seeds, context lengths, evaluation data, commands, and
metric definitions. Evaluate fixed held-out NLL/perplexity, continuation prompts,
and separate prefill and batch-1 decode timing.

$$
\operatorname{NLL},\quad \operatorname{PPL},\quad
D_{\rm KL}(p_{\rm base}\|p_{\rm variant}),\quad
\text{prefill tokens/s},\quad \text{decode tokens/s},\quad
\text{peak memory}.
$$

## Experimental sequence

### F0 — source baseline

Save source-model metrics, fixed-input logits, selected activations, and timing
under a versioned configuration.

### F1 — exact transformation harness

Validate identity, permutation, signed-permutation, Haar, then WPT transforms.
The model must match source behavior before proceeding.

### F2 — basis and structure comparison

Compare original order, random permutations, random orthogonal bases, Haar, and
declared WPT packet trees. Measure off-block energy:

$$
E_{\rm offblock}(W')=
\frac{\|\operatorname{OffBlock}(W')\|_F^2}{\|W'\|_F^2},
$$

plus activation energy concentration, sparsity, covariance, and basis stability.

### F3 — block-mask quality sweep

Apply masks only after F1:

$$
\widehat W=M\odot W^{(W)}.
$$

Sweep declared block layouts and retained-energy budgets. Report quality,
divergence, retained energy, nominal sparsity, memory, and measured latency.

### F4 — local adaptation and retention

Compare unrestricted adaptation, WPT-block LoRA, local block-only updates, and
local updates plus scheduled global consolidation. Evaluate new-domain learning
and retention on a fixed prior set.

### F5 — kernel feasibility

After a positive F3 result, benchmark the actual block sizes and Q/K/V or MLP
shapes in prefill and decode. Favor fusions such as RMSNorm plus structured QKV
or MLP projection. The kernel consumes precompiled tensors and adds no runtime
WPT transform.

## Controls and decision gates

- F1 must meet numerical-equivalence tolerance; otherwise stop.
- F2 must outperform random controls on predeclared metrics across more than one
  calibration split; otherwise do not pursue masks or kernels.
- F3 must show a useful quality/structure Pareto point before F4 or F5.
- F4 and F5 are independent value tests; success in one does not imply success
  in the other.
- Distinguish exact conversion, approximation, quantization, and kernel effects
  in every report.

## Cost discipline

Run the smallest fixed-input and calibration tests first. Use small models and
limited versioned calibration sets before broad sweeps. Do not download larger
models, begin fine-tuning, or build custom kernels until the preceding gate has
positive evidence.

## Current results

No results yet.

## Next step

Create the first individual experiment directory for F0/F1, including a
declarative configuration and a precise numerical-equivalence acceptance rule.
