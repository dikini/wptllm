# SmolLM2-1.7B Category Diagrams Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an editable VS Code Markdown reference that explains SmolLM2-1.7B through category-theory-style diagrams and WPT compilation squares.

**Architecture:** A single documentation page will pair Mermaid flowcharts with nearby VS Code-compatible LaTex equations. It will introduce the categories and layer-state functor precisely, then distinguish exact feature-coordinate compilation from nonlinear and sequence-axis cases needing additional care.

**Tech Stack:** Markdown, Mermaid (VS Code Markdown Preview), LaTex math markup, Git.

---

### Task 1: Create the diagram reference

**Files:**

- Create: `docs/smollm2-category-diagrams.md`
- Modify: `CHANGELOG.md`

**Step 1: Define the mathematical scope**

Write a short opening identifying the concrete SmolLM2-1.7B configuration and
introducing `\mathbf{DiffVec}_{\mathbb R}` for nonlinear computation. State
that individual decoder layers are morphisms; define the 24-step path category
and the layer-state functor.

**Step 2: Add the whole-model Mermaid diagram**

Draw tokens, embeddings, the 24-layer decoder, final RMSNorm, tied output
projection, and softmax. Label principal spaces and add dimensions in
surrounding equations, not inside Mermaid math labels.

**Step 3: Add the decoder-layer diagram and equations**

Show the two pre-norm residual branches: Q/K/V, RoPE, causal attention, output
projection, then the SwiGLU MLP. Include equations identifying linear and
nonlinear maps.

**Step 4: Add causality and WPT naturality diagrams**

Draw a prefix-restriction square for causal layers and a WPT compilation square
with component maps `S_T`. State the naturality equation and explain the
attention-interface convention for RoPE and head partitioning.

**Step 5: Add source and rendering guidance**

Link the official Hugging Face configuration and state that current VS Code
Markdown Preview renders Mermaid; an extension is optional for older setups.

**Step 6: Update the changelog**

Under `## [Unreleased]` / `### Added`, record the category-theory-style
SmolLM2 reference diagrams, retaining the existing design-record entry.

### Task 2: Validate the Markdown source

**Files:**

- Verify: `docs/smollm2-category-diagrams.md`

**Step 1: Check equation delimiters**

Run `rg -n '\\\\[()]|\\\\[\[\]]' docs/smollm2-category-diagrams.md`.
Expected: no output; equations use `$...$` or `$$...$$`.

**Step 2: Check Mermaid fences and changed-file whitespace**

Run `rg -n '^```mermaid|^```$' docs/smollm2-category-diagrams.md` and
`git diff --check`.
Expected: each Mermaid opening fence has a closing fence and no whitespace
errors are reported.

**Step 3: Inspect the rendered preview**

Open the document with VS Code Markdown Preview and check diagram and equation
rendering. If the installed release lacks Mermaid support, install or enable a
Mermaid-capable Markdown preview extension and document that only if required.

**Step 4: Commit the completed reference**

Stage `docs/smollm2-category-diagrams.md` and `CHANGELOG.md`, then commit with
message `docs: add SmolLM2 category diagrams`.

