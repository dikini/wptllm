# Functional Program Notation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add pseudo-functional programs that read the causal training, generation, fold/unfold, and reverse-mode AD equations operationally.

**Architecture:** Insert a compact `Programs and equations` subsection between the fold/unfold and no-magic AD subsections. Use language-neutral `text` fences and pair each program with existing mathematical notation.

**Tech Stack:** Markdown, language-neutral pseudocode, LaTex math.

---

### Task 1: Add pseudo-functional readings

**Files:**

- Modify: `docs/smollm2-category-diagrams.md`
- Modify: `CHANGELOG.md`

**Step 1: Write the failing content check**

Run a check for `Programs and equations`, `teacherForcedLoss`, `trainStep`, `generate`, `unfold`, `valueAndGrad`, `forwardTrace`, and `reverseVjp`. It must fail before the section exists.

**Step 2: Add the teacher-forced training program**

Use a language-neutral `text` block for per-token logits, cross-entropy, mean loss, `valueAndGrad`, and `optimizerStep`. State that targets are supplied, not generated. Pair it with the mean-loss equation.

**Step 3: Add the generation unfold program**

Define `step` from prefix to final logits to selected token to appended prefix. Define `generate` as unfold plus take-until-stop. Pair it with the existing state-transition equation and state that $\Theta$ remains fixed.

**Step 4: Add the AD program expansion**

Show `valueAndGrad` as a conceptual `forwardTrace` followed by a reverse fold of `reverseVjp`, seeded by the scalar-loss cotangent. Pair it with the VJP equation and say it need not be a literal framework tape.

**Step 5: Validate and commit**

Run content, code-fence, math-delimiter, and `git diff --check` validations. Update `CHANGELOG.md` under `Changed`, then commit with message `docs: add functional readings of training`.
