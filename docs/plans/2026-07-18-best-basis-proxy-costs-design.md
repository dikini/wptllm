# Best-Basis Proxy Costs Design

## Goal

Extend the WPT reference chapter with a worked method for deriving additive
best-basis proxies for global objectives that are not themselves additive over
packet leaves.

## Teaching approach

The new material will first give a reusable recipe:

1. state the true held-out global objective;
2. declare the approximation introduced by a candidate packet pruning;
3. identify a packet-node-local quantity that estimates the approximation's
   contribution to that objective on calibration data;
4. sum that quantity over selected leaves for dynamic programming; and
5. evaluate the actual global objective on held-out data.

It will then work through a specific, practical approximation: retaining
within-packet-block entries of a transformed linear operator and masking its
cross-block entries.

## Worked costs

- **Off-block energy:** use the squared Frobenius energy of cross-node blocks;
  this is exact for a declared block partition, while the node-local allocation
  is a proxy when a pruning changes the partition.
- **Output KL divergence:** use calibration residuals and a first-order logit
  perturbation from the masked operator. A quadratic Fisher/softmax-curvature
  form yields a non-negative local surrogate; the actual token-distribution KL
  remains the held-out criterion.
- **Retention:** use a local-update or Fisher-weighted parameter-drift proxy on
  a fixed prior calibration set. Retention after adaptation remains a separate
  held-out measurement.

## Boundaries

The chapter will label assumptions explicitly: fixed calibration inputs,
small perturbations for the KL Taylor approximation, a declared mask and block
assignment, and no claim that the dynamic program globally optimizes the true
KL or retention objective.
