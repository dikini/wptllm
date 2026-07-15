> **Historical reference.** This root-level plan has been superseded by the
> living experiment track in
> [`experiments/time-scale-block-causal-generation/`](experiments/time-scale-block-causal-generation/).
> It is retained for provenance and should not be used as the current experiment
> record.

# Time-Scale WPT and Block-Causal Generation Experiments

## Purpose

Explore a new architecture in which a WPT acts on the sequence axis, creating
localized time/scale states. Causality is preserved between completed blocks,
while inference inside an active block can be parallel, bidirectional, masked,
or denoising-based.

This is not function-preserving checkpoint compilation. Keep results separate
from the feature-space experiments in
[wpt-feature-compilation-experiments.md](wpt-feature-compilation-experiments.md).

## Central hypothesis

Replace total token order with a partial order over blocks:

$$
p(x_{1:T})=
\prod_{r=1}^{R}p_\theta(x_{\mathcal B_r}\mid x_{\mathcal B_{<r}}).
$$

Within the active block $\mathcal B_r$, a WPT hierarchy may support parallel
coordination and local reconstruction without reading ground-truth future
tokens.

## Low-cost initial scope

- Fixed blocks of 4, 8, or 16 tokens.
- A small model or a limited-data tiny-model training run.
- Strict causality between blocks.
- Masked/noisy token slots or block latents inside the active block.
- Haar WPT before any best-basis search.
- A parameter- and data-matched ordinary causal baseline.

Do not combine adaptive basis search, experts, colored-noise learning, and
custom kernels in the first prototype.

## Candidate variants

### A. Block latent plus causal renderer

Predict a coarse current-block latent from completed blocks,

$$
z_{\rm coarse,r}=f(x_{\mathcal B_{<r}}),
$$

refine it through a WPT tree, then render tokens causally. This is lowest risk:
token likelihood can remain autoregressive.

### B. Masked iterative block refinement

Represent current-block positions by masked slots, condition on earlier blocks,
and iteratively update all current positions in parallel. Ground-truth current
block token embeddings must never be visible as attention keys or values.

### C. Denoising over WPT block coefficients

Transform active-block states into time-scale coefficients and train a denoiser
conditioned on completed blocks:

$$
\widetilde z_{k,r}=z_{k,r}+\sigma_{k,r}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

Start with a fixed scale-dependent noise schedule. Learn the schedule or route
noise only after fixed-schedule controls have been tested.

### D. Semi-autoregressive groups

Generate groups causally inside a block, allowing parallel computation within
each group or WPT level. This is the midpoint between ordinary AR decoding and
fully joint block generation.

## Hierarchical attention mask

Let $n_i$ and $n_j$ be temporal WPT-tree nodes associated with query and key
states. One conceptual mask is:

$$
M_{ij}=
\begin{cases}
0, & n_j\in\operatorname{Past}(n_i),\\
0, & n_i=n_j\text{ and its states are masked, noisy, or latent},\\
-\infty, & \text{otherwise}.
\end{cases}
$$

This allows completed context and local parallel interaction, but prevents
observed-future leakage. A valid future-like signal is a forecast computed only
from available history:

$$
\widehat z_{\rm future}=f(z_{\leq t}).
$$

## Evaluation

Compare against a parameter- and data-matched causal baseline. Measure:

- NLL/perplexity when the training objective permits it;
- block generation and continuation quality;
- prefill and block-generation latency;
- number of sequential generation steps;
- sensitivity to block size, WPT level, masks, and noise schedule;
- long-context coherence at block boundaries;
- adversarial tests for train/inference leakage.

For masked or denoising variants, report their training objective separately
from autoregressive NLL.

## Milestone schedule

| Milestone | Experiment | Advance condition |
|---|---|---|
| T0 | Matched causal baseline | Stable train/evaluate harness. |
| T1 | Block-causal mask without WPT | No leakage and useful block behavior. |
| T2 | Fixed Haar time-scale states | Comparable quality at one block size. |
| T3 | Compare variants A–D | Select by quality, sequential steps, stability. |
| T4 | Colored-noise sweep | Beats no-noise and uniform-noise controls. |
| T5 | Adaptive masks/routing | Improves over fixed hierarchy without leakage. |
| T6 | Best-basis/drift adaptation | Stable selection improves the chosen metric. |

The first low-cost study ends at T2. It establishes whether local parallel
time-scale processing is viable before diffusion, adaptive routing, or dynamic
WPT re-basing are introduced.

## Non-negotiable controls

1. Train with only the information available at inference.
2. Compare to a matched causal baseline, not an unoptimized reference.
3. Report sequential generation steps as well as wall-clock latency.
4. Keep feature-space and sequence-axis WPT ablations separate.
5. Test several block sizes rather than selecting one favorable window.
