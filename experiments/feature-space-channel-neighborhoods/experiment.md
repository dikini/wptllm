# Channel-neighborhood discovery for feature-space WPTs

## Status

Planned follow-up to exact feature-space WPT checkpoint compilation. No results
have been collected yet.

## Goal

Discover and evaluate data-derived residual-channel orderings or groupings that
make a fixed, orthonormal WPT basis more likely to expose useful approximate
block structure in a compiled Transformer.

Let $P$ be a channel permutation or grouping-induced ordering and
$W_{\rm packet}$ a fixed packet transform. The basis under evaluation is:

$$
S = W_{\rm packet}P.
$$

The goal is not to prove that $P$ is the unique or semantically true order. It
is to determine whether it gives a reproducible advantage over original and
random channel orders for the stated structural and quality metrics.

## Motivation and theory

Feature dimensions in independently trained Transformers have no inherent
one-dimensional spatial order. A WPT over the feature axis therefore has an
arbitrary neighborhood relation unless an ordering is supplied. The hypothesis
is that channels with similar activation, gradient, weight, or task-conditioned
response profiles can form local neighborhoods. Applying a packet transform to
such neighborhoods may concentrate energy and weight interactions into fewer
blocks.

This is deliberately weaker than a semantic-neuron claim. A channel may be
permuted, mixed, or only conditionally useful. The relevant question is whether
the resulting basis improves operational structure and downstream trade-offs.

The track is compatible with exact checkpoint compilation: after choosing $S$,
compile the model using the feature-space equations in
[`wpt-feature-compilation-experiments.md`](../../wpt-feature-compilation-experiments.md).
No runtime WPT operation is required for the exact compiled model.

## Prerequisites and scope

Before running this track, complete the exact-compilation gate in the
feature-space compilation track. The channel-order experiment must start from a
compiled model that matches source logits and loss within numerical tolerance.

Initial scope:

- one small open-weight decoder model;
- one fixed wavelet family, initially Haar;
- residual-feature ordering only;
- a shared global ordering across layers as the primary design;
- small calibration datasets and offline analysis before any fine-tuning;
- no learned lifting, dynamic re-basing, pruning, or kernel implementation in
  the first individual experiment.

Layer-specific orderings are a later ablation. They require basis conversions
between layers and may undermine the simplicity and efficiency sought here.

## Hypotheses

### H1 — data-derived order improves structure

A channel order derived from calibration data yields lower off-block weight
energy and lower cross-block activation dependence than original order and
random-order controls.

### H2 — improvement is stable

The selected ordering and its metrics remain meaningfully similar across
independent calibration splits and representative prompt/data mixtures.

### H3 — structural improvement has practical value

At matched retained block energy, a mask derived from the data-ordered WPT basis
has lower output divergence or quality loss than masks derived from controls.

The first experiments test H1 and H2. H3 is a follow-up gate before any lifting
or custom-kernel work.

## Candidate channel affinity signals

For channels $i$ and $j$, construct an affinity graph $K$ from one or more
signals, aggregated across selected layers:

$$
K_{ij}
=
\sum_\ell \alpha_\ell
\left[
\lambda_a\,|\operatorname{corr}(h_{\ell,i},h_{\ell,j})|
+\lambda_g\,|\operatorname{corr}(g_{\ell,i},g_{\ell,j})|
+\lambda_w\,\operatorname{sim}(w_{\ell,i}^{\rm in/out},w_{\ell,j}^{\rm in/out})
+\lambda_s\,\operatorname{sim}(s_{\ell,i},s_{\ell,j})
\right].
$$

Where:

- $h_{\ell,i}$ is the activation of residual channel $i$ at layer $\ell$;
- $g_{\ell,i}$ is its gradient or update signal, when adaptation data are used;
- $w_{\ell,i}^{\rm in/out}$ is an incoming/outgoing weight fingerprint;
- $s_{\ell,i}$ is a labelled task-conditioned response signature.

A simple semantic/task signature for class or condition $c$ is:

$$
s_{\ell,i}=
\left[
\mathbb E(h_{\ell,i}\mid c_1),\ldots,
\mathbb E(h_{\ell,i}\mid c_m)
\right].
$$

Candidate conditions may include ordinary prose, code, retrieval, controlled
reasoning examples, analogy/metaphor examples, multilingual text, and
uncertainty/conflict prompts. Such labels are evidence about response profiles,
not proof that a dimension represents a named concept.

## From affinity graph to packet tree

Evaluate several ways to derive $P$ and a packet-compatible hierarchy:

1. original channel order;
2. random permutation controls;
3. activation-correlation clustering;
4. weight-fingerprint clustering;
5. task-signature clustering;
6. combined-affinity spectral seriation or hierarchical graph partitioning.

The hierarchy should respect the intended packet block sizes. If necessary,
constrain clustering to yield dyadic groups and use the resulting binary tree as
the packet tree. Record the ordering, tree, affinity definition, calibration
data, and seed as experiment artifacts.

## Evaluation

### Structural metrics

For each relevant transformed weight matrix, measure:

$$
E_{\rm offblock}(W')=
\frac{\|\operatorname{OffBlock}(W')\|_F^2}{\|W'\|_F^2}.
$$

Also measure:

- activation energy concentration by packet block;
- activation cross-block covariance and correlation;
- thresholded coefficient sparsity;
- stability of ordering/tree assignments across calibration splits;
- similarity of affinity graphs and resulting cluster assignments.

### Quality and approximation metrics

Exact compilation must first preserve source behavior. For subsequent masking
experiments, report:

$$
D_{\rm KL}(p_{\rm base}\|p_{\rm masked}),\quad
\Delta\operatorname{NLL},\quad
\Delta\operatorname{PPL},
$$

along with the retained block energy, nominal sparsity, and actual latency when
a suitable kernel exists.

### Controls

Every comparison must include the original coordinate order and multiple random
permutation controls. Where feasible, include random orthogonal and
weight-only/activation-only ordering controls. Report distribution over random
seeds, not only the best random draw.

## Ablations

Planned ablations include:

- calibration data size and composition;
- activation-only, gradient-only, weight-only, semantic-only, and combined
  affinities;
- signed versus absolute correlation;
- one global ordering versus selected layer-family orderings;
- packet block size and tree depth;
- Haar versus other fixed wavelet families;
- fixed labels versus unlabeled calibration data;
- stability after a small local adaptation run.

## Experimental sequence and decision gates

| Step | Experiment | Advance condition |
|---|---|---|
| N0 | Exact WPT compilation prerequisite | Source-equivalent logits/loss within tolerance. |
| N1 | Activation-only ordering on a small calibration set | Reproducible analysis pipeline and random controls. |
| N2 | Structural comparison across orderings | Data-derived order beats random controls on a predeclared metric. |
| N3 | Split/data-mixture stability test | Improvement survives independent calibration splits. |
| N4 | Masked block-quality sweep | Better or comparable quality/energy Pareto frontier. |
| N5 | Add task signatures and gradients | Clear incremental value over simpler affinity signals. |
| N6 | Consider constrained learned lifting | N2–N4 provide stable positive evidence. |

Stop or redesign the track if data-derived orderings do not outperform random
controls across independent calibration splits. Do not proceed to learned
lifting merely because one ordering produces visually attractive blocks.

## Cost discipline

Start with offline activations from a small model and a limited, versioned
calibration set. Cache only the statistics needed for reproducibility, not
unbounded raw activation dumps. Run structural analysis before pruning,
adaptation, or kernel work. Use a second model only to confirm a positive result
from the first.

## Current results

No results yet.

## Current conclusion and next step

The first individual experiment should implement the N1 activation-only
ordering pipeline, including original-order and multiple random-permutation
controls. It should define the calibration data, correlation estimator,
clustering method, dyadic grouping rule, and structural metrics before seeing
the final result.
