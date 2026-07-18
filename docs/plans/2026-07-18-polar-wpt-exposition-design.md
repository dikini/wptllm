# Polar Coordinates and WPT Exposition Design

## Goal

Add a self-contained explanation of how polar-coordinate ideas can complement,
but not replace, WPT feature-space experiments.

## Structure

The section will distinguish an orthogonal WPT basis change from a nonlinear
polar coordinate representation. It will compare three compositions:

1. WPT followed by per-packet polar statistics or coding, recommended as a
   best-basis and quantization extension.
2. Polar representation followed by WPT, a new nonlinear architecture rather
   than exact checkpoint compilation.
3. PolarQuant-style random preconditioning, useful as a geometry-agnostic
   control but not evidence for feature-channel semantics.

## Experiment recommendation

Recommend a future `polar-packet-rate-distortion` track after feature-space
compilation and channel-neighborhood results. Its stages are Cartesian WPT
controls, per-packet rate--distortion measurements, equal-memory cache/activation
coding, and only then nonlinear polar-packet architecture experiments.

## Boundaries

The text will explicitly cover angular periodicity and zero-radius singularities,
pairing topology, transform/kernel overhead, and the lack of a fixed linear
weight conjugation through polar coordinates.
