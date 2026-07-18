# Polar Coordinates and WPT Exposition Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Explain the relationship between polar-coordinate coding and WPT,
including advantages, limits, and a staged future experiment recommendation.

**Architecture:** Insert a standalone section after the transform comparisons.
Describe the recommended WPT-then-polar composition before the nonlinear
polar-then-WPT alternative and the random-preconditioning control. Link the
future work to existing feature-compilation and channel-neighborhood gates.

**Tech Stack:** Markdown, VS Code-compatible LaTex math, PolarQuant reference.

---

### Task 1: Add polar-coordinate exposition

**Files:**

- Modify: `docs/wavelet-packet-transform.md`

**Step 1:** Distinguish linear WPT coordinate changes from nonlinear polar
coordinates and their implications for exact checkpoint compilation.

**Step 2:** Derive the recommended per-packet polar rate--distortion cost.

**Step 3:** Explain benefits and risks of applying WPT after polar conversion
and of PolarQuant-style random rotations.

**Step 4:** Specify the future `polar-packet-rate-distortion` track and staged
entry criteria.

### Task 2: Validate documentation

**Files:**

- Modify: `CHANGELOG.md`

**Step 1:** Record the reference-chapter extension under `Unreleased/Changed`.

**Step 2:** Run `git diff --check`, verify paired display math, and confirm the
PolarQuant reference remains present.

**Step 3:** Commit owned documentation changes with message
`docs: explain polar WPT research path`.
