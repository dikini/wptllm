# Functional program notation design

## Purpose

Make the training, generation, fold/unfold, and automatic-differentiation
sections executable in the reader's mind by pairing their equations with
small pseudo-functional programs.

## Chosen structure

Place a short `Programs and equations` subsection after the operational
fold/unfold explanation. It will give a teacher-forced training program, an
autoregressive generation unfold, and an explicit conceptual expansion of
`valueAndGrad` into a forward trace and reverse VJP pass. Each program is
followed by its matching equation and a note that it is explanatory
pseudocode, not a framework API.

## Boundaries

The programs will make evaluation order and data flow explicit but will not
specify batching, distributed execution, mixed precision, checkpointing
policy, or a concrete optimizer implementation. The existing diagrams and
equations remain the source of mathematical detail.

