# Feature-space WPT compilation

This track tests the most conservative WPT hypothesis: an existing decoder-only
checkpoint can be re-expressed exactly in a feature-space WPT basis, and that
basis may reveal stable block structure useful for approximation, local
adaptation, or specialized kernels.

The first job is correctness, not speed. We must show that compilation preserves
the source model before measuring sparsity or applying masks. Only then can we
ask whether a WPT basis beats original and random-coordinate controls, and only
after that should we invest in pruning, continual-learning, or kernel work.

The detailed living track record is [experiment.md](experiment.md). Individual
runs will be added as subdirectories, each with its own `README.md` and
`experiment.md`.

The practical setup, model-selection, bootstrap, and decision guide is in the
[feature-compilation playbook](playbook.md).

The previous root-level planning note is retained as a
[historical reference](../../wpt-feature-compilation-experiments.md).
