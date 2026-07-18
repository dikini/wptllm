# SmolLM2 teaching narrative design

## Goal

Restructure the SmolLM2 reference as a coherent teaching note for a reader who
is comfortable with linear algebra but new to Transformer mechanics. The note
must connect data, computation, vector-space transformations, geometry, and
the later WPT/category-theory perspective.

## Narrative order

1. Orient the reader with the model's input/output and concrete dimensions.
2. Define tokens, embeddings, residual streams, logits, and probabilities as
   data in explicit vector spaces, with computational and geometric meaning.
3. Trace one causal forward pass through the whole model and then one decoder
   layer.
4. Explain attention and MLP operations as coordinate changes, token-position
   mixing, and residual updates.
5. Show training and generation as different uses of the same causal forward
   computation, including fold/unfold and automatic differentiation.
6. Introduce the categorical diagrams as a compact reformulation of the
   already-explained computation.
7. Present WPT as a feature-coordinate change, then its limits and the distinct
   sequence-axis proposal.

## Editing principles

- Retain the existing concrete SmolLM2-1.7B facts, equations, and diagrams
  where accurate.
- Add short transitions and cross-references rather than duplicate derivations.
- Keep Mermaid for plain process flow and TikZ for symbolic commutative views.
- State vector orientation, tensor shape, and axis meaning wherever a new map
  first appears.
- Keep exact coordinate-change claims distinct from architecture-preserving
  approximations and empirical claims.
