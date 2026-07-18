# WPT Chapter Teaching Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the WPT chapter an educational introduction that directly supports the project's feature-space and sequence-axis experiment decisions.

**Architecture:** Reorder and bridge the existing mathematical sections; add a small Haar packet walkthrough and an experiment map tied to the current track documents. Keep claims classified as exact algebra, measurement target, approximation, or speculation.

**Tech Stack:** Markdown, VS Code-compatible LaTex math, project experiment documents.

---

### Task 1: Improve the WPT teaching progression

**Files:**

- Modify: `docs/wavelet-packet-transform.md`
- Modify: `CHANGELOG.md`

**Step 1:** Add a reader orientation that distinguishes coordinate change from demonstrated sparsity, speed, or quality preservation.

**Step 2:** Strengthen the DWT-to-WPT transition with a worked Haar packet tree showing both approximation and detail splitting, coefficient interpretation, and reconstruction.

**Step 3:** Explain best-basis dynamic programming as a choice within a fixed packet library; state what calibration data and controls are needed.

**Step 4:** Reframe the LLM tensor section around axes, geometry, and experiment scope: feature axis, token axis, head axis, and batch axis.

**Step 5:** Replace the project connection with an experiment map covering feature-space compilation, channel-neighborhood discovery, and time-scale/block-causal generation. Include transformed axis, hypothesis, baseline/control, and non-claim for each.

**Step 6:** Validate heading flow, math delimiters, links to tracks/references, and whitespace. Commit with message `docs: strengthen WPT experiment guide`.

