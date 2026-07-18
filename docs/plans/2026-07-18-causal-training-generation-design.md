# Causal training and generation design

## Purpose

Complete the SmolLM2 diagram reference with the outer mechanics of causal
next-token modelling: how one shared decoder forward pass is used differently
for parallel training and autoregressive generation.

## Chosen structure

Insert a `Causal prediction: training and generation` section after the
whole-model overview and before the decoder-layer detail. It will first define
the shared causal forward pass, then split into a training branch and a
generation branch.

The training branch will show shifted targets, per-position cross-entropy,
loss aggregation, back-propagation, and optimizer updates to $\Theta$. The
generation branch will show last-position logits, token selection, append,
stop-or-repeat control flow, and KV-cache reuse. The text will make clear that
ordinary generation has no gradient or parameter update.

## Visual strategy

Use Mermaid for the readable process/control-flow views and nearby VS
Code-compatible equations for exact notation. Existing component and TikZ
diagrams remain unchanged; the new section explains how they are coordinated
across a sequence and across successive generation steps.
