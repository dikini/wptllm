# SmolLM2 Symbolic TikZ Diagrams Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a terminology bridge and mathematically precise TikZ diagrams to the SmolLM2-1.7B reference without replacing its Mermaid process diagrams.

**Architecture:** A compact glossary will connect Transformer words, Mermaid labels, and equations. Inline TikZJax blocks will then provide a functor square, residual-update diagram, causal-prefix square, and WPT naturality square, each adjacent to its existing plain-language explanation and formal equation.

**Tech Stack:** Markdown, LaTex, TikZJax for VS Code, `tikz-cd`, Mermaid.

---

### Task 1: Add the terminology bridge and symbolic diagrams

**Files:**

- Modify: `docs/smollm2-category-diagrams.md`
- Modify: `CHANGELOG.md`

**Step 1: Write the failing content check**

Run a shell check that fails when the note lacks `## Reading the model diagrams`,
four `tikz` code blocks, and the symbols `$\mathbf{L}_{24}$`, `$\mathcal{H}$`,
`$H_T$`, `$L_\ell$`, `$r_t$`, `$S_T$`, and `$\eta_\ell^S$` in the TikZ source.

**Step 2: Verify the check fails**

Run it before editing. Expected: a nonzero exit status because the glossary and
TikZ diagrams are absent.

**Step 3: Add the glossary**

After `## Notation and categorical scope`, introduce the residual stream as
$h\in H_T$ and a generic update $h_{\mathrm{next}}=h+\Delta(h)$. Add a compact
table mapping residual stream, residual connection, RMSNorm, projection,
Q/K/V, attention, RoPE, causal mask, MLP, SwiGLU, logits, category, functor,
and natural transformation to their diagram labels and symbols.

**Step 4: Add the TikZ companions**

Use complete inline `tikz` blocks. Load `tikz-cd` in the three commutative
diagrams and use math labels in TikZ nodes/arrows:

- the layer-index/functor square maps $\ell\to\ell+1$ to
  $L_\ell:H_T\to H_T$;
- the residual diagram makes $u=h+\Delta_{\mathrm{attn}}(h)$ and
  $h_{\ell+1}=u+\Delta_{\mathrm{mlp}}(u)$ visible;
- the causal square uses $H_T$, $H_t$, $r_t$, $L_\ell^{(T)}$, and
  $L_\ell^{(t)}$;
- the WPT naturality square uses $S_T$, $L_\ell$, $L_\ell^S$, and
  $\eta_\ell^S$.

Keep existing Mermaid diagrams unchanged. Precede each TikZ block with a short
reading sentence; place its equation directly after it.

**Step 5: Update preview guidance and changelog**

State that the installed TikZJax extension renders `tikz` fences in VS Code
Markdown Preview and that it uses `tikz-cd`. Add a concise `Changed` entry
under `Unreleased` for the glossary and symbolic companions.

**Step 6: Verify and commit**

Re-run the content check; verify each `tikz` block contains
`\begin{document}` and `\end{document}`, Mermaid checks remain clean, math
delimiters remain VS Code-compatible, and `git diff --check` passes. Inspect
the rendered preview in VS Code. Commit the note and changelog with message
`docs: add symbolic Transformer diagrams`.

