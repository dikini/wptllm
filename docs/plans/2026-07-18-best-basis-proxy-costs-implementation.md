# Best-Basis Proxy Costs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a rigorous, worked explanation of additive best-basis proxy costs
for off-block energy, output KL divergence, and retention.

**Architecture:** Insert one subsection in the WPT chapter after the existing
non-additivity warning. Start with a generic derivation recipe, then instantiate
it for a declared transformed-operator packet-block mask. Keep true held-out
objectives separate from calibration surrogates.

**Tech Stack:** Markdown, VS Code-compatible LaTex math, existing experiment
track terminology.

---

### Task 1: Add the generic surrogate recipe

**Files:**

- Modify: `docs/wavelet-packet-transform.md`

**Step 1:** Define a global objective $G(\mathcal L)$ over a packet pruning and
state why it cannot generally be optimized by the classical tree dynamic
program.

**Step 2:** Define a declared approximation and a node-local calibration cost
$C(v)$ whose sum is optimized by the tree dynamic program.

**Step 3:** State that the selected pruning is a candidate and that $G$ is
evaluated on a held-out split.

### Task 2: Work through the packet-block-mask example

**Files:**

- Modify: `docs/wavelet-packet-transform.md`

**Step 1:** Define transformed operator blocks, the within-block mask, and its
perturbation.

**Step 2:** Derive an off-block-energy proxy using squared Frobenius norms.

**Step 3:** Derive a calibration KL proxy from logit perturbations and softmax
curvature, with a small-perturbation boundary.

**Step 4:** Define a retention proxy using fixed-prior-data Fisher-weighted
parameter drift and distinguish it from post-adaptation retention evaluation.

### Task 3: Validate the chapter change

**Files:**

- Modify: `CHANGELOG.md`

**Step 1:** Record the educationally significant chapter extension under
`Unreleased/Changed`.

**Step 2:** Run `git diff --check`.

**Step 3:** Check that display-math delimiters are paired and that the new
subsection follows the stated global-objective versus surrogate distinction.

**Step 4:** Commit the owned documentation changes with message
`docs: derive WPT best-basis proxy costs`.
