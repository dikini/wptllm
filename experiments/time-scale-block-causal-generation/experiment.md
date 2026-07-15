# Time-scale WPT and block-causal generation

## Status

Planned. No architecture prototype or training run has been completed.

## Goal

Test whether sequence-axis WPT states can support block-causal generation with
useful local parallelism. The model must preserve causality between completed
blocks while allowing an active block to coordinate masked, noisy, or latent
states internally.

## Theory and factorization

For hidden states $H\in\mathbb{R}^{B\times T\times d}$, a temporal WPT acts
on sequence position:

$$
Z=\Psi_{\rm time}H.
$$

Partition the token sequence into blocks $\mathcal B_1,\ldots,\mathcal B_R$
and use a block-causal factorization:

$$
p(x_{1:T})=
\prod_{r=1}^{R}p_\theta(x_{\mathcal B_r}\mid x_{\mathcal B_{<r}}).
$$

This is not ordinary token-by-token causal decoding. The current block needs an
internal generation mechanism, such as a coarse latent, masked refinement,
denoising, or semi-autoregressive groups.

## Non-negotiable causality rule

The model may not consume observed future/current-block target tokens merely
because they are available under teacher forcing. Any future-like signal must be
derived from completed context:

$$
\widehat z_{\rm future}=f(z_{\leq t}).
$$

For temporal tree nodes $n_i,n_j$, a conceptual mask is:

$$
M_{ij}=
\begin{cases}
0, & n_j\in\operatorname{Past}(n_i),\\
0, & n_i=n_j\text{ and its states are masked, noisy, or latent},\\
-\infty, & \text{otherwise}.
\end{cases}
$$

Every individual experiment must include an adversarial leakage check.

## Scope

The first prototype should use short fixed blocks of 4, 8, or 16 tokens, a tiny
or small model, a fixed Haar transform, and a matched ordinary causal baseline.
Do not combine dynamic best-basis search, learned lifting, adaptive routing,
colored-noise learning, and kernel work in the initial experiment.

## Candidate mechanisms

### T-A — coarse block latent plus causal renderer

Predict a block latent from completed context,

$$
z_{\rm coarse,r}=f(x_{\mathcal B_{<r}}),
$$

refine through a time-scale hierarchy, then render tokens causally.

### T-B — masked iterative refinement

Use masked current-block token slots and update them in parallel over several
iterations, conditioned only on completed blocks and permitted latent state.

### T-C — WPT-coefficient denoising

Train a conditional denoiser over block coefficients:

$$
\widetilde z_{k,r}=z_{k,r}+\sigma_{k,r}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

Start with a fixed scale-dependent schedule.

### T-D — semi-autoregressive groups

Generate groups causally within a block while allowing parallel computation
inside a group or packet-tree level.

## Evaluation

Compare against a parameter- and data-matched causal baseline. Record:

- NLL/perplexity where applicable;
- block-generation and continuation quality;
- wall-clock latency and number of sequential generation steps;
- sensitivity to block size, WPT level, mask, and noise schedule;
- coherence across block boundaries;
- train/inference leakage tests.

For masked or denoising objectives, report their objective separately from
autoregressive NLL.

## Experimental sequence and gates

| Step | Experiment | Advance condition |
|---|---|---|
| T0 | Matched ordinary causal baseline | Stable train/evaluate harness. |
| T1 | Block-causal mask without WPT | Leakage-free and useful block behavior. |
| T2 | Fixed Haar time-scale states | Comparable quality at one block size. |
| T3 | Compare T-A through T-D | Select by quality, steps, and stability. |
| T4 | Fixed colored-noise sweep | Beats no-noise and uniform-noise controls. |
| T5 | Adaptive masks/routing | Improves fixed hierarchy without leakage. |
| T6 | Best-basis or drift adaptation | Stable selection improves a declared metric. |

The low-cost initial study ends at T2. Stop or redesign if T1 fails leakage or
quality gates; do not add WPT complexity to an invalid block-generation setup.

## Cost discipline

Start with short blocks, small models, and a versioned limited dataset. Measure
sequential steps as well as runtime. Test several block sizes rather than
selecting one favorable window. Scale only after a matched baseline and leakage
tests establish a credible positive signal.

## Current results

No results yet.

## Next step

Create an individual T0 baseline experiment directory with a fixed data/model
budget, evaluation protocol, and leakage-test design.
