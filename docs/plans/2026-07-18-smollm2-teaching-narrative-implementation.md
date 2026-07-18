# SmolLM2 Teaching Narrative Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the SmolLM2 reference into a coherent explanation of Transformer data, computation, geometry, training, generation, and WPT coordinates.

**Architecture:** Reorder and bridge the existing content rather than replace its technical core. Begin with the concrete token-to-distribution map and explicit spaces; explain one forward pass before category theory; use the categorical/WPT material as a second, geometric reading of the established computation.

**Tech Stack:** Markdown, Mermaid, TikZJax/TikZ, VS Code-compatible LaTex math.

---

### Task 1: Build the data-to-computation foundation

**Files:**

- Modify: `docs/smollm2-category-diagrams.md`
- Modify: `CHANGELOG.md`

**Step 1:** Add a concise orientation and a `Data, spaces, and geometry` section before the current categorical notation. Define token IDs, one-hot vocabulary vectors, the row-vector embedding table, residual stream $H_T$, logits, and probabilities. Explain token axis versus feature axis, and computational versus geometric interpretation.

**Step 2:** Move or reframe categorical notation after this foundation so it summarizes an understood computation rather than introduces it.

**Step 3:** Add transition sentences that connect embeddings, residual stream, and output head to the whole-model diagram.

### Task 2: Make one causal forward pass the central thread

**Files:**

- Modify: `docs/smollm2-category-diagrams.md`

**Step 1:** Ensure the whole-model section explicitly states the input/output map and axes.

**Step 2:** Reframe the layer section around two kinds of transformation: feature-coordinate maps at each token and attention-mediated position mixing. Explain residual updates as maintaining a common working space.

**Step 3:** Add a short `Attention and MLP: what moves where` bridge defining Q/K/V, heads, RoPE, causal masking, MLP, and their geometric/computational roles before later causal/category squares.

### Task 3: Connect sequence, learning, and geometry lenses

**Files:**

- Modify: `docs/smollm2-category-diagrams.md`

**Step 1:** Retain the training/generation, fold/unfold, AD, and program sections as the time/learning continuation of one forward pass. Add transitions that identify their shared model state and parameter spaces.

**Step 2:** Present categories, functor, causality, and WPT naturality after the operational explanation. Make clear what each diagram abstracts and what it does not establish.

**Step 3:** Tighten the WPT sections around feature-axis geometry, coordinate change, exact general-map conjugation, architecture-preserving limitations, and sequence-axis distinction.

### Task 4: Validate the teaching note

**Files:**

- Verify: `docs/smollm2-category-diagrams.md`

**Step 1:** Verify section order matches the design: orientation/data → forward pass/layer mechanics → sequence/training → category/WPT.

**Step 2:** Check all new symbols have a first definition, headings are navigable, Mermaid and TikZ fences are balanced, Mermaid labels stay parser-safe, and equations use VS Code-compatible delimiters.

**Step 3:** Run `git diff --check`, inspect the Markdown Preview/TikZJax render, and commit with message `docs: restructure SmolLM2 teaching note`.
