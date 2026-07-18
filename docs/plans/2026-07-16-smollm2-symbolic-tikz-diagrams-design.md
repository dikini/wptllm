# SmolLM2 symbolic TikZ diagrams design

## Purpose

Complement the existing Mermaid process diagrams with mathematically precise,
commutative-style diagrams. The new visuals should make symbols, categories,
functors, natural transformations, and residual additions visible rather than
leaving them only in surrounding prose.

## Chosen representation

Keep Mermaid for plain-language process flow. Add inline `tikz` code blocks for
the symbolic companions, using `tikz-cd` for commutative squares and ordinary
TikZ for the residual-addition data-flow diagram. This is supported by the
project user's installed TikZJax for VS Code extension and is readable in the
same Markdown Preview.

## Diagram set

1. A functor square that maps the layer-index arrow $\ell\to\ell+1$ in
   $\mathbf{L}_{24}$ to the layer morphism $L_\ell:H_T\to H_T$ in
   $\mathbf{DiffVec}_{\mathbb{R}}$.
2. A symbolic residual-update diagram that makes the attention update and MLP
   update entering their respective additions explicit.
3. A precise causal-prefix commuting square labelled with $H_T$, $H_t$,
   $L_\ell^{(T)}$, $L_\ell^{(t)}$, and $r_t$.
4. A WPT naturality square labelled with $S_T$, $L_\ell$, $L_\ell^S$, and the
   components $\eta_\ell^S$ of the natural isomorphism.

Each TikZ diagram will have a short reading sentence and an adjacent equation;
the diagrams do not replace the existing process diagrams or prose.

## Validation

Verify the TikZ block structure, required `tikz-cd` package declarations, and
the matching mathematical labels statically. Visual rendering is checked in VS
Code Markdown Preview using the installed extension.
