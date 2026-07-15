# Feature-space channel neighborhoods

This track asks a practical question that comes before learned lifting: can we
put residual channels into an order that gives a fixed Wavelet Packet Transform
(WPT) useful local neighborhoods?

The original channel order in a Transformer is not expected to be a meaningful
spatial order. A WPT nevertheless needs an ordering, because its fine-scale
operations act on nearby coordinates. We will derive candidate orders from
activation behavior, gradients, weight fingerprints, and optionally labelled
task-conditioned activation profiles. The goal is operational rather than
interpretive: find an ordering that makes the compiled model more nearly
block-structured than appropriate random controls.

This is a follow-up to feature-space checkpoint compilation. It does not yet
learn a lifting scheme, change the model architecture, or claim that individual
channels have fixed semantic meanings. If it finds stable, useful neighborhoods,
it can provide a grounded starting point for later constrained lifting work.

The detailed, living track record is in [experiment.md](experiment.md). Concrete
runs will live in subdirectories of this track and will each have their own
`README.md` and `experiment.md`.
