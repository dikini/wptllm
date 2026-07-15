> **Historical reference.** This root-level plan has been superseded by the
> living experiment track in
> [`experiments/feature-space-compilation/`](experiments/feature-space-compilation/).
> It is retained for provenance and should not be used as the current experiment
> record.

# Feature-Space WPT Compilation and Sparse-Kernel Experiments

## Purpose

Test whether a WPT on the Transformer residual-feature axis reveals stable
block structure that preserves model quality while enabling local adaptation or
faster structured kernels. This track uses existing decoder-only checkpoints.
It is separate from the sequence-axis architecture in
[wpt-time-scale-experiments.md](wpt-time-scale-experiments.md).

## Research questions

1. Can a checkpoint be exactly re-expressed in a WPT feature basis?
2. Does a selected WPT basis beat original and random bases on block structure?
3. What is the quality-versus-structure trade-off of block pruning?
4. Do local WPT-block updates reduce adaptation cost or forgetting?
5. Do the discovered blocks yield real speedup at target matrix shapes?

## Suggested checkpoints

| Role | Checkpoint | Reason |
|---|---|---|
| Smoke test | [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Small GQA model; exercises Q/KV asymmetry. |
| Confirmation | [`HuggingFaceTB/SmolLM2-1.7B`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) | Compact larger model for cross-architecture confirmation. |

Prefer base checkpoints for perplexity and continuation evaluation. Instruction
checkpoints are useful for qualitative prompts.

## Fixed protocol

Record model and tokenizer revision, precision, GPU, driver, software versions,
seeds, context lengths, and generation parameters. Use a fixed held-out text
set, continuation prompts, and small reasoning/code/retrieval prompt sets.
Measure prefill and batch-1 decode separately.

$$
\operatorname{NLL},\quad \operatorname{PPL},\quad
D_{\rm KL}(p_{\rm base}\|p_{\rm variant}),\quad
\text{prefill tokens/s},\quad \text{decode tokens/s},\quad
\text{peak memory}.
$$

## Phase A — baseline and exact compilation

### A1. Baseline

Run the fixed protocol with the original checkpoint. Save logits on a small
calibration set, selected layer activations, and timing results.

### A2. Exact conversion harness

Validate identity, permutation, signed permutation, Haar, and then selected WPT
bases. For an original checkpoint frame $(O)$ and WPT residual frame $(W)$:

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

Keep Q/K/V and RoPE in ordinary head coordinates. All WPT factors are fused
into stored weights: no explicit WPT multiplication may remain at inference.

### A3. Gate

The exact compiled model must agree with source logits and NLL within the
expected numerical tolerance. Stop if it does not.

## Phase B — basis and structure measurement

Compare original coordinates, random permutations, random orthogonal bases,
Haar/WPT packet trees, and optionally PCA/whitening. Use identical calibration
data for every candidate. If width is not dyadic, use fixed power-of-two channel
groups rather than untracked padding.

For every layer and matrix, measure:

$$
E_{\rm offblock}(W')=
\frac{\|\operatorname{OffBlock}(W')\|_F^2}{\|W'\|_F^2}.
$$

Also record retained block energy, threshold sparsity, activation entropy and
kurtosis, activation/gradient cross-block covariance, and basis stability across
prompts, data, and layers.

Advance only when the WPT consistently improves relevant metrics over random
orthogonal and permutation controls.

## Phase C — pruning Pareto sweep

Apply masks only to an exact compiled model:

$$
\widehat W=M\odot W^{(W)}.
$$

Sweep diagonal-only blocks, diagonal-plus-neighbor blocks, per-layer versus
shared masks, and retained-energy budgets such as 90%, 75%, 50%, and 25%.
Plot quality loss against retained energy, latency, and memory. Do not confuse
nominal sparsity with speed.

## Phase D — local adaptation and retention

Compare unrestricted LoRA, WPT-block LoRA, block-only updates, and block-only
updates with periodic global consolidation on a small new-domain dataset:

$$
\Delta W_{\ell,kk}'=-\eta\nabla_{W_{\ell,kk}'}\mathcal L,
\qquad
\Delta W_{\ell,ij}'=0\quad(i\ne j).
$$

Evaluate new-domain learning and retention on the fixed baseline evaluation
set. This directly tests the forgetting hypothesis.

## Phase E — kernel feasibility

Only after a good Phase C point, microbenchmark its actual block sizes and
matrix shapes for batch-1 decode and batched prefill. Include GQA Q/K/V and MLP
gate/up/down projections. Likely first targets are:

```text
RMSNorm -> block-sparse Q/K/V projection
RMSNorm -> block-sparse MLP gate/up projection -> activation
```

Compare with the best dense implementation on the target hardware. The kernel
consumes precompiled weights and must not perform a separate WPT transform.

## Milestone schedule

| Milestone | Deliverable | Advance condition |
|---|---|---|
| M0 | Baseline report | Reproducible logits, quality, and timing. |
| M1 | Exact compilation report | Numerical equivalence within tolerance. |
| M2 | Basis report | WPT structure beats random-basis controls. |
| M3 | Pruning Pareto curves | At least one useful quality/structure point. |
| M4 | Retention report | Local adaptation has a measurable advantage. |
| M5 | Kernel microbenchmark | Structured kernel wins on actual shapes. |
| M6 | End-to-end prototype | Gain survives complete prefill/decode timing. |

The low-cost path is M0 through M3 on the 0.5B model. Start M4 or M5 only if
M3 yields a clear operating point.
