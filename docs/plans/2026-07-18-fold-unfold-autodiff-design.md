# Fold, unfold, and automatic-differentiation design

## Purpose

Extend the causal training/generation explanation with an intuitive operational
view of sequence construction, loss aggregation, and back-propagation. The
reader should see that automatic differentiation transforms an ordinary forward
computation graph into reverse derivative computation by the chain rule; it
does not introduce a separate unexplained learning mechanism.

## Chosen structure

Add a short `Fold, unfold, and build` subsection after the training/generation
comparison. It treats training as mapping a known sequence to per-token losses
and folding them into one scalar, while generation unfolds a prefix state and
builds the next prefix by appending each chosen token.

Follow it with `No magic: reverse-mode automatic differentiation`. It presents
a composed forward graph, a reverse vector-Jacobian-product rule, and a small
forward/reverse diagram. A softmax-cross-entropy derivative makes one local
derivative concrete, then connects the accumulated gradient to the optimizer.

## Boundaries

The section is intuitive and operational, not a formal treatment of
catamorphisms or anamorphisms. It will distinguish an abstract reverse-mode
rule from implementation details such as memory-saving checkpointing, fused
kernels, or distributed training.

