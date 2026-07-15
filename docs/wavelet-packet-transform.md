# Wavelets and Wavelet Packet Transforms

## Purpose and scope

This chapter introduces wavelets as the foundation for the **Wavelet Packet
Transform** (WPT), then focuses on the properties of WPT that matter for this
project: localized subspaces, selectable bases, multichannel signals,
best-basis optimization, and structured coordinate systems.

The WPT is not universally better than other transforms. It is a constrained
library of localized bases. Its value appears when that library contains a basis
that makes a signal or operator more concentrated, less coupled, or easier to
approximate than suitable controls.

For this project, the question is:

$$
\text{Does a declared WPT basis reveal useful structure without unacceptable
loss of model quality?}
$$

## 1. Signals, bases, and coordinate changes

A finite real signal is a vector $x\in\mathbb R^N$. Given an orthonormal basis
matrix $B\in\mathbb R^{N\times N}$, its coefficients are

$$
c=Bx, \qquad x=B^\top c.
$$

Orthogonality preserves energy:

$$
\|c\|_2^2=\|x\|_2^2.
$$

A transform does not create information. It changes where information is
expressed. A useful basis can concentrate energy in a few coefficients, reduce
cross-block covariance, or make a linear operator nearly block diagonal.

The word *signal* includes audio, images, sensor streams, feature vectors, token
sequences, hidden states, and tensors with several transformable axes.

## 2. Wavelet foundation

Fourier basis functions are global sinusoids. They represent stationary periodic
components efficiently, but a local transient affects coefficients across the
whole domain. Wavelets use translated and scaled short waveforms, giving joint
localization in position/time and scale.

In continuous notation, a wavelet family is generated from a mother wavelet
$\psi$ by translation and dilation:

$$
\psi_{j,k}(t)=2^{-j/2}\psi(2^{-j}t-k),
$$

where $j$ indexes scale and $k$ position. A scaling function $\phi$ represents
coarse content. Multiresolution analysis builds nested approximation spaces:

$$
\cdots\subset V_{j+1}\subset V_j\subset\cdots,
\qquad V_j=V_{j+1}\oplus W_{j+1}.
$$

Here $V$ is an approximation space and $W$ a detail space. Mallat's work
connects this construction to fast discrete filter banks and exact reconstruction
[1].

## 3. Discrete wavelet transform (DWT)

In a critically sampled orthogonal DWT, low-pass and high-pass filters $h$ and
$g$ are convolved with a signal and downsampled by two:

$$
a_{j+1}[n]=\sum_k h[k-2n]a_j[k],
$$

$$
d_{j+1}[n]=\sum_k g[k-2n]a_j[k],
$$

with $a_0=x$. The sequence $a_{j+1}$ is a coarser approximation and $d_{j+1}$
is its detail. Synthesis upsamples and filters the two children to reconstruct
their parent exactly for a perfect-reconstruction filter bank.

The standard multilevel DWT decomposes **only the approximation**:

```text
x
├─ d1                  detail: stop
└─ a1
   ├─ d2               detail: stop
   └─ a2
      ├─ d3            detail: stop
      └─ a3            approximation: stop
```

### Haar example

The Haar transform is the simplest orthonormal wavelet. For a pair $(x_0,x_1)$:

$$
a=\frac{x_0+x_1}{\sqrt{2}},
\qquad d=\frac{x_0-x_1}{\sqrt{2}}.
$$

The approximation is a local average and the detail a local difference. Haar is
transparent and has minimal local support, making it a strong first experimental
baseline. Smoother wavelets trade longer support for smoother basis functions
and more vanishing moments.

## 4. Wavelet packet transform (WPT)

The WPT generalizes the DWT by allowing **both** children of every node to be
decomposed. Starting with the same filter pair, every node may split into a
low-pass and high-pass child:

```text
x
├─ d1
│  ├─ da
│  └─ dd
└─ a1
   ├─ aa
   └─ ad
```

The letters describe low/high filtering relative to the parent. They are not
exact physical frequency labels: real filters have transition bands, boundary
handling matters, and frequency ordering may differ from tree traversal order.

At depth $J$, a fully expanded 1D packet tree has $2^J$ leaves, each with about
$N/2^J$ coefficients. The full tree is not one basis; it is a **library of
bases**. Any pruning that covers the root with disjoint nodes defines a valid
packet basis. The ordinary DWT is one special pruning: continue splitting only
the approximation branch.

For an orthonormal packet filter bank, every complete pruning preserves total
energy and permits exact reconstruction. This comparability enables best-basis
selection [2].

```text
DWT: decompose the low-pass branch repeatedly.
WPT: permit any branch to be decomposed, then choose a pruning.
```

## 5. Packet nodes and structured operators

Each packet node represents a localized subspace. Selecting a node retains that
subspace as a unit; splitting it replaces it by finer subspaces. This gives a
natural block interpretation:

- selected leaves are candidate local blocks;
- coefficient concentration measures localized activity;
- off-block operator energy measures coupling between blocks;
- sparsity or block masking is an additional approximation, not a WPT guarantee.

For a linear operator $A$, an orthonormal change of basis gives

$$
A'=SAS^\top.
$$

One structural metric is

$$
E_{\rm offblock}(A')=
\frac{\|\operatorname{OffBlock}(A')\|_F^2}{\|A'\|_F^2}.
$$

This differs from coefficient sparsity. A basis can sparsify activations without
block-diagonalizing weights, or block-structure weights without sparse
activations.

## 6. Best-basis optimization

The full packet tree offers many bases. Best-basis optimization selects a
pruning relative to a declared cost; it does not discover an arbitrary optimal
transform.

Let $v$ be a node with coefficients $c_v$ and node cost $C(v)$. For an additive
cost, select the leaves $\mathcal L$ that minimize

$$
\min_{\mathcal L}\sum_{v\in\mathcal L} C(v).
$$

Coifman and Wickerhauser's bottom-up dynamic program is

$$
J(v)=\min\left(C(v),\;J(v_{\rm low})+J(v_{\rm high})\right).
$$

If the parent cost is lower, prune its descendants; otherwise retain the best
prunings of both children. The tree makes this search efficient [2].

### Useful costs

Energy is conserved by every complete orthonormal basis, so raw total energy is
not by itself a selection criterion. Useful costs reward concentration,
thresholdability, coding efficiency, or task-relevant structure.

For nonzero node energy, define $p_i=|c_i|^2/\|c_v\|_2^2$. Its entropy is

$$
H(v)=-\sum_i p_i\log p_i.
$$

Low entropy means energy is concentrated in fewer coefficients. For dynamic
programming, use an additive form such as energy-weighted entropy or

$$
C_{\rm Shannon}(v)=-\sum_i |c_i|^2\log(|c_i|^2),
$$

with a defined zero convention. Threshold-count and estimated coding-rate costs
are other common choices.

For this project, off-block energy, output KL divergence, and retention are
important but generally not additive over tree leaves. Use an additive proxy to
generate candidate bases, then evaluate the full global objective on held-out
data. Do not claim the classical best-basis optimum for a non-additive objective.

For a shared basis across signals $x^{(1)},\ldots,x^{(M)}$, aggregate the node
cost before search:

$$
C_{\rm group}(v)=\frac1M\sum_{m=1}^M C\bigl(c_v^{(m)}\bigr).
$$

Select on one calibration split and validate on another to detect basis
overfitting.

## 7. Multidimensional and multichannel signals

### 7.1 Transform axes

A tensor can have several axes, but a WPT is applied only to selected axes. For
$X\in\mathbb R^{T\times C}$, with time $T$ and channels $C$:

$$
Z_{\rm time}=\Psi X
$$

applies a time/sequence WPT independently to each channel, while

$$
Z_{\rm channel}=XS^\top
$$

applies a feature/channel WPT at each time position. A separable two-axis
transform is

$$
Z=\Psi X S^\top.
$$

Time and channel transforms make different assumptions and need different
calibration data, masks, and interpretations.

### 7.2 Images and higher-dimensional data

For an image $X\in\mathbb R^{H\times W}$, a separable DWT filters rows and
columns. One level produces low-low, low-high, high-low, and high-high subbands.
A 2D packet tree may recursively decompose all four, producing a quadtree; in
$D$ transformed dimensions, each node has $2^D$ children. PyWavelets documents
1D, 2D, and $N$-dimensional packet trees and node conventions [3, 4].

For multichannel sensor data there are two distinct choices:

1. apply a temporal WPT independently to each sensor, then model relations
   between coefficient streams;
2. apply a channel WPT across sensors/features, which requires a meaningful
   channel ordering or grouping.

The second choice is closer to the feature-space LLM proposal and is fragile
when channel order is arbitrary.

### 7.3 LLM tensors

A decoder residual stream can be written

$$
H\in\mathbb R^{B\times T\times d},
$$

where $B$ is batch, $T$ token position, and $d$ model width.

**Feature-space WPT** applies $S\in\mathbb R^{d\times d}$ on the last axis:

$$
Z=HS^\top.
$$

This is a coordinate change of each token's residual vector. It is not
intrinsically a time/scale representation. Its key difficulty is that residual
channel order has no obvious natural locality; channel grouping or an
empirically derived permutation may be necessary.

**Sequence-space WPT** applies $\Psi\in\mathbb R^{T\times T}$ along token
position:

$$
Z=\Psi H.
$$

This has a genuine time/scale interpretation, but a generic non-causal transform
does not preserve ordinary autoregressive attention masks. It is an architectural
and training problem, not simple checkpoint compilation.

## 8. Relation to other transforms

| Transform | Representation | Localization | Adaptation | Main trade-off |
|---|---|---|---|---|
| DFT/FFT | Global complex sinusoids | Frequency only; global in time/position | Fixed | Excellent for global periodic structure; poor transient localization. |
| STFT | Fourier transform of sliding fixed windows | Uniform time-frequency cells | Window and hop selected in advance | Fixed resolution trade-off. |
| Gabor | Gaussian-window STFT | Uniform cells with optimal Gaussian joint concentration | Gaussian width/lattice selected in advance | Still fixed resolution; often redundant depending on sampling. |
| DWT | Low-pass branch recursively refined | Multiscale and localized | Wavelet family/depth selected | Mostly refines approximation content. |
| WPT | Selectable pruning of a full subband tree | Multiscale localized packet subspaces | Wavelet family plus best-basis pruning | More choice and analysis cost; possible overfitting. |
| KLT/PCA | Covariance eigenvectors | Usually global and data-dependent | Learns from covariance | Dense, global, and costly to estimate/update. |

### DFT and FFT

The DFT maps $x[n]$ to sinusoidal coefficients:

$$
X[k]=\sum_{n=0}^{N-1}x[n]e^{-2\pi i kn/N}.
$$

The FFT is an efficient algorithm for computing the DFT, not a different basis.
The DFT is a good baseline for global periodic or stationary spectral structure;
it has no native local block tree.

### STFT and Gabor transform

For window $w$, the STFT is

$$
\operatorname{STFT}_x(\tau,\omega)=
\sum_n x[n]w[n-\tau]e^{-i\omega n}.
$$

It gives a uniform time-frequency grid. A longer window improves frequency
resolution and worsens time resolution; a shorter window reverses that tradeoff.
The Gabor transform is the Gaussian-window STFT. It has optimal joint Gaussian
time-frequency concentration, but still a fixed resolution. Wavelets instead
use scale-dependent resolution [6, 7].

### KLT / PCA

For zero-mean $x$ with covariance $\Sigma$, let

$$
\Sigma=U\Lambda U^\top.
$$

The KLT coefficients are $c=U^\top x$. For the estimated covariance this
decorrelates coefficients and is optimal among orthogonal linear transforms for
mean-square reconstruction after retaining a fixed number of components.

KLT is therefore an important data-adaptive energy-compaction baseline. Its
limitations for this project are equally important: $U$ is generally dense and
global, depends on calibration data, has no packet-tree locality, and can drift
as data change. WPT trades unconstrained covariance optimality for a structured,
local, efficiently implemented transform family. PCA is also called the KLT,
Hotelling transform, or eigenvector transform [8].

## 9. Practical choices and pitfalls

### Boundary handling

Finite signals need an extension convention: periodic, symmetric, zero padding,
or another scheme. It changes edge coefficients and may affect shapes. Record it
in every experiment. For feature-space compilation, do not silently pad residual
width; use an explicitly declared grouped transform for non-dyadic widths.

### Orthogonal and biorthogonal filters

Orthogonal transforms preserve Euclidean energy and have inverse $S^\top$.
Biorthogonal wavelets use distinct analysis and synthesis bases and can provide
desirable symmetry or smoothness, but require $S^{-1}$ rather than $S^\top$.
Orthogonality is the simpler initial choice for LLM compilation because it helps
conditioning and simplifies norm-related reasoning.

### Critical sampling and shift sensitivity

The ordinary DWT/WPT downsampled tree is critically sampled and can be shift
sensitive. Undecimated or stationary wavelet transforms reduce shift sensitivity
by retaining redundant coefficients, but increase memory and compute and are not
simple orthonormal basis changes.

### A best basis is not automatically deployable

Low entropy, sparse coefficients, or energy compaction are evidence, not final
deployment objectives. Validate separately:

1. exact compiled-model quality;
2. weight and activation block structure;
3. quality under declared masking or quantization;
4. retention under local adaptation;
5. realized kernel latency and memory.

## 10. Connection to this project's experiments

The feature-space compilation track uses WPT as a structured coordinate system.
For an orthonormal feature transform $S$, residual-to-residual maps use

$$
W'=SWS^\top,
$$

while projections from residual space to conventional Q/K/V head coordinates
use one-sided conversion, for example $W_q'=W_qS^\top$. These are exact
re-expressions before pruning or sparse execution.

The channel-neighborhood track asks whether a permutation $P$ can make feature
neighbors meaningful enough for a fixed packet transform:

$$
S=W_{\rm packet}P.
$$

Only after fixed WPT bases and data-derived orderings show stable value should
the project consider constrained learned lifting. Lifting can parameterize local,
invertible refinement, but must be constrained for conditioning, locality, and
reproducibility.

The time-scale block-causal track is different: it applies WPT to token position.
It is not a mere re-basing of a standard causal checkpoint because attention
masking and nonlinear operations do not generally commute with a temporal basis
change.

## 11. Reference reading

All links below are directly readable online; the first two are downloadable
PDFs of foundational papers.

1. S. Mallat, “A Theory for Multiresolution Signal Decomposition: The Wavelet
   Representation,” *IEEE Transactions on Pattern Analysis and Machine
   Intelligence*, 1989. [Author-hosted PDF](https://www.di.ens.fr/~mallat/papiers/MallatTheory89.pdf).
   Foundational multiresolution and filter-bank treatment, including 2D work.
2. R. R. Coifman and M. V. Wickerhauser, “Entropy-Based Algorithms for Best
   Basis Selection,” *IEEE Transactions on Information Theory*, 1992.
   [Downloadable PDF](https://bpb-us-e1.wpmucdn.com/sites.gatech.edu/dist/2/436/files/2011/04/coifman92en.pdf?bid=436).
   The primary best-basis paper.
3. PyWavelets, “Overview of multilevel wavelet decompositions.”
   [Open documentation](https://pywavelets.readthedocs.io/en/latest/ref/2d-decompositions-overview.html).
4. PyWavelets, “Wavelet Packets.”
   [Open documentation](https://pywavelets.readthedocs.io/en/latest/ref/wavelet-packets.html).
5. PyWavelets, “2D Wavelet Packets.”
   [Open worked example](https://pywavelets.readthedocs.io/en/stable/regression/wp2d.html).
6. MIT OpenCourseWare, “Short-Time Fourier Transform.”
   [Downloadable lecture notes](https://ocw.mit.edu/courses/hst-582j-biomedical-signal-and-image-processing-spring-2007/resources/ch7_stft/).
7. K. Gerhardt, S. F. Wittmann, and M. Gräfe, “Fourier, Gabor, Morlet or
   Wigner: Comparison of Time-Frequency Transforms,” 2021.
   [Open arXiv paper](https://arxiv.org/abs/2101.06707).
8. R. Wang, “The Principal Component Transform.”
   [Open tutorial](https://pages.hmc.edu/ruye/e161/lectures/pca/node3.html).
