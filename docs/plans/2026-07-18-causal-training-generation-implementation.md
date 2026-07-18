# Causal Training and Generation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Explain how SmolLM2's causal forward pass supports parallel next-token training and sequential autoregressive generation.

**Architecture:** Add one section after the whole-model overview. A shared forward-pass explanation establishes causal logits; Mermaid process diagrams then separate training's loss/back-propagation/update path from generation's token-selection/append/repeat loop, with equations connecting both to the same model distribution.

**Tech Stack:** Markdown, Mermaid, VS Code-compatible LaTex math.

---

### Task 1: Add causal training and generation mechanics

**Files:**

- Modify: `docs/smollm2-category-diagrams.md`
- Modify: `CHANGELOG.md`

**Step 1: Write the failing content check**

Run a shell check that fails until the note contains the section heading, two
new Mermaid fences, the terms `shifted targets`, `cross-entropy`,
`back-propagation`, `optimizer`, `KV cache`, and `end-of-sequence`, plus
equations containing $p_\Theta$ and $\nabla_\Theta$.

**Step 2: Verify the check fails**

Run the check before editing. Expected: a nonzero exit status because the new
section is absent.

**Step 3: Explain the shared causal forward pass**

Define token sequence $x_{1:T}$ and model logits $z_t$ at each position.
Explain the causal restriction: $z_t$, hence $p_\Theta(x_{t+1}\mid x_{1:t})$,
depends only on the prefix $x_{1:t}$. State all positions can be computed in
parallel for a supplied sequence despite this restriction.

**Step 4: Add the training branch**

Add a Mermaid diagram from a dataset token sequence through shifted targets,
the causal forward pass, per-position cross-entropy, aggregated loss,
back-propagation, optimizer update, and updated parameters. Include a mean
negative log-likelihood and gradient-update equation. State targets are supplied
by the dataset, not sampled from the model.

**Step 5: Add the generation branch**

Add a Mermaid loop from prompt/prefix through causal forward pass and
last-position logits to sampling/argmax, append token, end-of-sequence or
length check, then repeat. Include a distribution and token-choice equation.
State ordinary generation does not compute gradients or update $\Theta$.
Explain the KV cache stores prior layers' key/value projections, so each
iteration computes the appended position rather than the whole prefix; results
match the un-cached causal forward pass up to numerical implementation details.

**Step 6: Validate and commit**

Run the content check, Mermaid-fence balance and parser-safe-label checks,
VS Code math-delimiter check, and `git diff --check`. Update the changelog
under `Changed`, then commit with message
`docs: explain causal training and generation`.
