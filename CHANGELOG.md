# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Design record for an editable SmolLM2-1.7B category-diagram reference.
- Category-theory-style SmolLM2-1.7B architecture and WPT commutation diagrams.
- Design record for a terminology glossary that connects the diagrams to their
  equations.
- Design record for symbolic TikZ companions to the SmolLM2 process diagrams.
- Design record for a causal training-versus-generation explanation.
- Design record for fold/unfold/build and reverse-mode automatic-differentiation
  intuition.

- Project guidance for reproducible, documented, cost-aware WPT-LLM research.
- Separate experiment plans for feature-space WPT checkpoint compilation and
  sparse-kernel feasibility, and for time-scale block-causal generation.
- A follow-on experiment track for discovering data-derived feature-channel
  neighborhoods before considering learned lifting.
- Living experiment tracks for feature-space WPT compilation and time-scale
  block-causal generation; their earlier root-level plans are now historical
  references.
- A feature-space compilation playbook and project-wide templates for preparing
  reproducible individual experiments.
- A reference chapter on wavelets, wavelet packets, best-basis optimization,
  multidimensional signals, and transform comparisons.

### Changed

- Expanded the SmolLM2 diagram reference with operational fold/unfold/build
  and reverse-mode automatic-differentiation explanations of training and
  generation.
- Expanded the SmolLM2 diagram reference with a causal training and
  autoregressive-generation explanation.
- Expanded the SmolLM2 category-diagram reference with a beginner-oriented
  terminology glossary and symbolic TikZ commutative diagrams.
- Added a critical LMWT related-work note, evaluation implications, and paper
  reference to the wavelet packet transform chapter.
- Reframed the WPT theory note around near block-diagonal structure, quality
  preservation, performance optimization, and localized continuous learning.

### Fixed

- Replaced Mermaid-sensitive mathematical punctuation in SmolLM2 diagram labels
  with plain-language labels that render in VS Code Markdown Preview.
- Removed display-math wrappers from `tikz-cd` blocks to match TikZJax's
  documented Markdown block form.
- Replaced TikZJax-sensitive `\text` and `\operatorname` labels in the functor
  and residual diagrams with basic math-mode labels.
- Corrected the residual-diagram shorthand so each attention and MLP update
  explicitly consumes its normalized input.
- Added explicit $x$ and $y$ nodes after the two RMSNorm operations in the
  residual diagram.

## [0.1.0] - 2026-07-15

### Added

- Initial theoretical documentation for WPT-based Transformer compilation,
  basis drift, local adaptation, and block-causal time-scale variants.
