# Time-scale block-causal generation

This track explores a different idea from feature compilation. A WPT acts on
the sequence axis, producing localized time/scale states. Generation remains
causal between completed blocks, while the active block may use masked,
bidirectional, or denoising-style inference internally.

This is a new architecture and training regime, not an exact conversion of an
existing causal checkpoint. The early work should be deliberately small: prove
that the masking and block-generation protocol is leakage-free and useful before
adding learned bases, routing, or custom kernels.

The detailed living track record is [experiment.md](experiment.md). Individual
runs will be added as subdirectories with their own documentation.

The previous root-level planning note is retained as a
[historical reference](../../wpt-time-scale-experiments.md).
