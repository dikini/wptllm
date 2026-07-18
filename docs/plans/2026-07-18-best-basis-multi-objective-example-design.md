# Best-Basis Multi-Objective Example Design

## Goal

Add an intuitive worked example of multi-objective packet best-basis selection
to the WPT reference chapter.

## Example

Use a four-coordinate Haar packet tree and compare three candidate prunings:
the root as one block, two level-one blocks, and four singleton leaves. The
example values are illustrative and dimensionless, not measurements from a
model.

## Objective

For each pruning, show normalized cross-block energy, normalized calibration
KL surrogate, and a block-size regularizer. Combine them as

$$
C(\mathcal L)=
\lambda_{\rm off}\widehat E_{\rm off}(\mathcal L)+
\lambda_{\rm KL}\widehat G_{\rm KL}(\mathcal L)+
\lambda_{\rm size}\sum_{v\in\mathcal L}\operatorname{dim}(v)^2.
$$

The selected weights will make the two-block pruning preferable, illustrating
why an intermediate partition can trade approximation quality against useful
structure.

## Boundary

The table will separately report a hypothetical held-out true KL, showing that
the calibration surrogate is a search signal rather than a quality guarantee.
