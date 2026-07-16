# SmolLM2-1.7B category-diagram reference design

## Purpose

Create an editable visual reference for the concrete SmolLM2-1.7B decoder-only
architecture. It should help readers follow ordinary Transformer computation,
feature-space WPT compilation, and the boundary between exact coordinate-change
claims and nonlinear or sequence-axis operations.

## Chosen representation

Use Mermaid flowcharts in a Markdown document, with standard `$...$` and
`$$...$$` LaTex equations immediately adjacent to each diagram. This keeps the
diagrams inspectable and editable in VS Code Markdown Preview. Current VS Code
versions include Mermaid diagram support in the built-in Markdown preview; an
extension is not required for the initial document.

## Diagram set

The reference will contain:

1. a whole-model pipeline with the concrete SmolLM2-1.7B spaces;
2. a decoder-layer computation diagram for RMSNorm, attention/RoPE, residuals,
   and SwiGLU;
3. the layer-index category and its layer-state functor into a category of
   finite-dimensional real vector spaces and differentiable maps;
4. WPT naturality/compilation squares, including the attention-interface
   convention that leaves RoPE and conventional head partitioning in their
   original coordinates; and
5. a causality square showing why feature-axis coordinate maps preserve prefix
   restriction, plus the limitation for sequence-axis transforms.

## Correctness boundaries

A fixed decoder layer is a morphism, not a functor. RMSNorm, softmax, and
SwiGLU are nonlinear, so the diagrams use differentiable maps rather than only
linear maps. The WPT naturality square denotes an ideal exact compiled model;
it is not evidence that a numerically implemented, pruned, or quantized model
is equivalent. Those cases require the existing empirical evaluation process.

