# Fold, Unfold, and Automatic Differentiation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give the causal training/generation section an intuitive operational account of loss folding, sequence unfolding, state building, and reverse-mode automatic differentiation.

**Architecture:** Add two subsections after the existing training/generation contrast. The first maps known-token training to map-plus-fold and generation to unfold-plus-build. The second traces a forward composition into reverse vector-Jacobian-product computation, with a simple local softmax-cross-entropy derivative and an optimizer connection.

**Tech Stack:** Markdown, Mermaid, TikZJax-compatible TikZ, VS Code-compatible LaTex math.

---

### Task 1: Add the fold/unfold and no-magic autodiff lens

**Files:**

- Modify: `docs/smollm2-category-diagrams.md`
- Modify: `CHANGELOG.md`

**Step 1: Write the failing content check**

Run a shell check that fails until the note contains headings `Fold, unfold,
and build` and `No magic: reverse-mode automatic differentiation`, the terms
`fold`, `unfold`, `vector-Jacobian product`, and `softmax-cross-entropy`, plus
expressions containing $\operatorname{VJP}$, $\bar{\mathcal L}$, and
$p-e_y$.

**Step 2: Verify the check fails**

Run it before editing. Expected: a nonzero exit status because the subsections
are absent.

**Step 3: Add the operational fold/unfold lens**

State that training maps known positions to per-token losses and folds them by
sum or mean into $\mathcal L$. State that generation unfolds a prefix state:
one step selects a token and build appends it into the next prefix. Include a
small Mermaid diagram using parser-safe plain labels, and clearly state that
these are intuitive operational names rather than a formal category-theory
claim.

**Step 4: Add the reverse-mode no-magic explanation**

Show a composed forward graph $h_0\to h_1\to\cdots\to h_n\to\mathcal L$ and
the reverse gradient flow. Use one TikZJax-compatible TikZ diagram or a
Mermaid diagram plus adjacent equation; avoid unsupported TikZJax macros.
Define the reverse seed $\bar{\mathcal L}=1$ and show the local VJP update
rule for both predecessor activations and operation parameters. Explain the
forward pass records/recomputes required values and reverse mode applies local
chain-rule actions; no independent learning magic is introduced.

**Step 5: Make a local derivative concrete**

For logits $z$, probability $p=\operatorname{softmax}(z)$, and one-hot target
$e_y$, show $\partial\ell/\partial z=p-e_y$. Explain these local gradient
contributions fold/add at shared parameters to form $\nabla_\Theta\mathcal L$,
which the existing optimizer update consumes. State ordinary generation runs
only the forward computation, with no gradient recording or optimizer update.

**Step 6: Validate and commit**

Run content, Mermaid/TikZ fence-balance, parser-safe Mermaid-label, math
delimiter, and `git diff --check` checks. Update `CHANGELOG.md` under
`Changed`, then commit with message
`docs: explain fold unfold and autodiff`.
