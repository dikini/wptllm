# SmolLM2 diagram glossary design

## Purpose

Make the SmolLM2-1.7B category-diagram note readable for someone who does not
already connect standard Transformer terms with their symbols and computation.

## Chosen structure

Add a compact `Reading the model diagrams` section immediately after the
notation section. It will begin with the residual stream as the central object,
then use a three-column glossary to connect plain-language terms, diagram
labels, and the corresponding symbols or equations. A short translation key
under the decoder diagram will identify its two residual additions explicitly.

## Scope and limits

The addition explains terminology already used in the note; it does not add a
second tutorial, alter the categorical claims, or introduce new model behavior.
It will preserve Mermaid-safe plain-language labels and VS Code-compatible math
markup.
