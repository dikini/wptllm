# Wavelets and Wavelet Packet Transforms

## Purpose and scope

This chapter starts from the practical intuition behind the **Wavelet Packet
Transform** (WPT): it is a structured change of coordinates that can divide a
signal into localized subspaces at several resolutions. The hope in this project
is not that a transform makes an LLM inherently better, but that a well-chosen
coordinate system may reveal approximately independent blocks in its activations
or operators. Those blocks could later make approximation, local adaptation, or
specialized kernels more practical.

The reading path is deliberate: first see the difference between an ordinary
wavelet transform and a packet tree, then work through a small Haar example,
then connect basis selection to the axes of LLM tensors and to the project's
separate experiments. It is intended as a self-contained introduction and as a
reference for the experiment records.

The WPT is not universally better than other transforms. It is a constrained
library of localized bases. Its value appears only when that library contains a
basis that makes a signal or operator more concentrated, less coupled, or
easier to approximate than suitable controls.

For this project, the question is:

$$
\text{Does a declared WPT basis reveal useful structure without unacceptable
loss of model quality?}
$$

The answer cannot be assumed from the transform alone. Exact orthogonal
re-expression preserves information, but does **not** prove sparsity,
near-block-diagonality, preserved finite-precision model outputs, speedup, or
safer continual learning. Each is a separate empirical claim with its own
baseline and evaluation.

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

For four samples, first form two local averages and differences:

$$
a_0=\frac{x_0+x_1}{\sqrt2},\quad d_0=\frac{x_0-x_1}{\sqrt2},\qquad
a_1=\frac{x_2+x_3}{\sqrt2},\quad d_1=\frac{x_2-x_3}{\sqrt2}.
$$

The second DWT level splits only the approximation pair:

$$
a_{\mathrm a}=\frac{a_0+a_1}{\sqrt2},\qquad
a_{\mathrm d}=\frac{a_0-a_1}{\sqrt2}.
$$

Its four coefficients are therefore $\{a_{\mathrm a},a_{\mathrm d},d_0,d_1\}$. A full level-two
packet tree also splits the detail pair:

$$
d_{\mathrm a}=\frac{d_0+d_1}{\sqrt2},\qquad
d_{\mathrm d}=\frac{d_0-d_1}{\sqrt2}.
$$

The packet representation
$\{a_{\mathrm a},a_{\mathrm d},d_{\mathrm a},d_{\mathrm d}\}$ contains the same number of
coefficients and the same information, but it offers a different set of local
subspaces. This small example is the essential WPT move: detail content is
allowed to have its own multi-resolution structure instead of being treated as
one terminal remainder.

Reconstruction simply reverses the same local operations. For example,
$a_0=(a_{\mathrm a}+a_{\mathrm d})/\sqrt2$ and
$d_0=(d_{\mathrm a}+d_{\mathrm d})/\sqrt2$, after which
$x_0=(a_0+d_0)/\sqrt2$ and $x_1=(a_0-d_0)/\sqrt2$. The other pair is recovered
in the same way. This exact inverse is why the packet tree can be used as a
coordinate system before any approximation is introduced.

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
selection [2]. A basis is a pruning, not necessarily the fully expanded tree:
one may retain a parent where its children offer no useful extra structure.

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

### From a global objective to an additive proxy

The following recipe turns a global objective into a best-basis *candidate
generator*. Let a pruning $\mathcal L$ define disjoint coordinate blocks, with
$P_v$ the diagonal projector onto the coordinates in leaf $v$. First declare
the approximation that the pruning will enable, and write its true objective as
$G(\mathcal L)$. Next, assign to every possible leaf a calibration cost $C(v)$
that estimates its local contribution, and search with

$$
\widetilde G(\mathcal L)=\sum_{v\in\mathcal L} C(v).
$$

Finally, run the selected approximation and measure $G(\mathcal L)$ on a
held-out split. The important distinction is:

```text
global objective G: decides whether the approximation is useful;
additive surrogate G-tilde: makes a packet-tree search feasible.
```

The approximation must be declared before its surrogate. For a practical
example, let $A^{\prime}=SAS^\top$ be a transformed residual-to-residual linear
operator and retain only its within-leaf blocks:

$$
M_{\mathcal L}(A^{\prime})=
\sum_{v\in\mathcal L}P_v A^{\prime}P_v,
\qquad
\Delta_{\mathcal L}=A^{\prime}-M_{\mathcal L}(A^{\prime}).
$$

Thus $\Delta_{\mathcal L}$ is the cross-packet coupling removed by the declared
block mask. This is an *approximation* after an exact WPT coordinate change;
the coordinate change itself does not remove any entries.

#### Worked proxy: off-block energy

For a fixed pruning, off-block energy has an especially useful decomposition.
For each candidate leaf, define its outgoing cross-block energy

$$
C_{\rm off}(v)=
\sum_{\ell}\alpha_\ell
\left\|(I-P_v)A^{\prime}_{\ell}P_v\right\|_F^2,
$$

where $A^{\prime}_{\ell}$ is a transformed linear operator in layer $\ell$ and
$\alpha_\ell\geq0$ declares layer importance. Since the leaf projectors are
disjoint and sum to the identity,

$$
\sum_{v\in\mathcal L} C_{\rm off}(v)=
\sum_{\ell}\alpha_\ell\left\|\Delta_{\mathcal L,\ell}\right\|_F^2.
$$

This equality counts directed matrix blocks exactly once: the source coordinates
belong to one leaf $v$. Therefore, for this specific mask, squared off-block
Frobenius energy is additive over leaves, not merely a proxy. A pure minimization
is degenerate, however: it selects the root as one large block and removes
nothing. Add a declared complexity term or a maximum leaf width, for example

$$
C(v)=C_{\rm off}(v)+\lambda\,\operatorname{dim}(v)^2,
$$

where the second term makes one very large dense block expensive. The resulting
tree search optimizes this stated structure--error trade-off, not model quality.

#### Worked proxy: output KL divergence

Let $z(x)$ be original-model logits for a calibration input $x$, let
$p(x)=\operatorname{softmax}(z(x))$, and let $z_{\mathcal L}(x)$ be logits after
the declared packet-block mask. The true quality objective is

$$
G_{\rm KL}(\mathcal L)=\frac1M\sum_{m=1}^{M}
D_{\rm KL}\!\left(p(x_m)\,\middle\|\,
\operatorname{softmax}(z_{\mathcal L}(x_m))\right).
$$

It is not additive: a removed block may change later nonlinearities, attention,
and logits, and different removals can interact. To obtain a local calibration
surrogate, assign each leaf its masked-operator atom

$$
\Delta A_v=(I-P_v)A^{\prime}P_v
$$

and use the first-order logit response $r_v(x)$ obtained by propagating the
effect of $\Delta A_v$ through the unmasked network. For small perturbations,
the softmax KL has the quadratic approximation

$$
D_{\rm KL}\!\left(p\,\middle\|\,\operatorname{softmax}(z+\delta z)\right)
\approx \frac12\delta z^\top F(p)\delta z,
\qquad F(p)=\operatorname{diag}(p)-pp^\top.
$$

Dropping cross-leaf terms in
$\delta z\approx\sum_{v\in\mathcal L}r_v(x)$ gives the additive cost

$$
C_{\rm KL}(v)=\frac{1}{2M}\sum_{m=1}^{M}
r_v(x_m)^\top F\bigl(p(x_m)\bigr)r_v(x_m).
$$

This is useful when the local responses are small and weakly correlated. It is
not the KL of the jointly masked model: the omitted terms
$r_u^\top F r_v$ and higher-order nonlinear effects can be large. The selected
pruning must therefore be evaluated with the actual masked forward pass and
held-out $G_{\rm KL}$, including maximum-logit and loss differences.

#### Worked multi-objective example

Consider a width-four feature group with a Haar packet tree. Its possible
prunings include the root $\{r\}$, two level-one packets $\{a,d\}$, and four
level-two packets $\{a_{\mathrm a},a_{\mathrm d},d_{\mathrm a},d_{\mathrm d}\}$.
Suppose the declared approximation is the packet-block mask above. For a
calibration set, normalize the structural error and KL surrogate by fixed
reference values $E_{\rm ref}$ and $K_{\rm ref}$:

$$
\widehat E_{\rm off}(\mathcal L)=
\frac{\sum_{v\in\mathcal L}C_{\rm off}(v)}{E_{\rm ref}},
\qquad
\widehat G_{\rm KL}(\mathcal L)=
\frac{\sum_{v\in\mathcal L}C_{\rm KL}(v)}{K_{\rm ref}}.
$$

The normalized terms can then share a declared multi-objective search score:

$$
C(\mathcal L)=
0.45\,\widehat E_{\rm off}(\mathcal L)+
0.35\,\widehat G_{\rm KL}(\mathcal L)+
0.025\sum_{v\in\mathcal L}\operatorname{dim}(v)^2.
$$

The numbers below are illustrative dimensionless calibration values, not model
measurements. They show the calculation and the reason to normalize before
choosing weights.

| Pruning $\mathcal L$ | Blocks | $\widehat E_{\rm off}$ | $\widehat G_{\rm KL}$ | $\sum_v\operatorname{dim}(v)^2$ | $C(\mathcal L)$ | Held-out true KL |
|---|---:|---:|---:|---:|---:|---:|
| $\{r\}$ | 1 | 0.00 | 0.00 | 16 | $0.45(0)+0.35(0)+0.025(16)=0.400$ | 0.00 |
| $\{a,d\}$ | 2 | 0.20 | 0.10 | 8 | $0.45(0.20)+0.35(0.10)+0.025(8)=0.325$ | 0.12 |
| $\{a_{\mathrm a},a_{\mathrm d},d_{\mathrm a},d_{\mathrm d}\}$ | 4 | 0.80 | 0.70 | 4 | $0.45(0.80)+0.35(0.70)+0.025(4)=0.705$ | 0.85 |

The root has zero cross-block error and zero KL because it masks nothing, but
it leaves one large dense block. The four-leaf choice gives the smallest dense
blocks, but removes too much coupling. With the stated weights, the two-block
pruning is the calibration candidate because $0.325$ is smallest.

The final column is deliberately **not** used by the packet-tree dynamic
program. It stands for an actual forward-pass measurement on held-out inputs.
Here it supports the candidate, but a different held-out value could reject it;
the surrogate is a transparent search heuristic, not proof that the selected
pruning minimizes true KL or maximizes runtime benefit.

#### Worked proxy: retention under local adaptation

Suppose adaptation proposes a parameter change
$\delta\theta=\sum_{v\in\mathcal L}\delta\theta_v$, where
$\delta\theta_v$ is assigned to the parameters or packet blocks associated with
leaf $v$. On a fixed prior-data calibration set, let $F_{\rm prior}$ be an
empirical Fisher or another positive-semidefinite curvature estimate. The true
retention quantity is the post-adaptation increase in prior loss,

$$
G_{\rm retain}(\mathcal L)=
L_{\rm prior}(\theta+\delta\theta)-L_{\rm prior}(\theta).
$$

Its local quadratic approximation is

$$
G_{\rm retain}\approx
g_{\rm prior}^\top\delta\theta+
\frac12\delta\theta^\top F_{\rm prior}\delta\theta,
\qquad g_{\rm prior}=\nabla L_{\rm prior}(\theta).
$$

Dropping interactions between leaf updates produces the additive search cost

$$
C_{\rm retain}(v)=
g_{\rm prior}^\top\delta\theta_v+
\frac12\delta\theta_v^\top F_{\rm prior}\delta\theta_v.
$$

Here $\delta\theta_v$ can be a small pilot update, or a declared local
gradient-step model such as $-\eta R_v g$, where $R_v$ selects the parameters
assigned to packet leaf $v$. At a well-converged prior solution,
$g_{\rm prior}$ is often small, leaving the familiar Fisher-weighted drift
term. This proxy ranks bases by estimated interference with prior behavior; it
does not establish retention. The required test remains an actual adaptation run
followed by held-out prior-task and new-task evaluation.

For a shared basis across signals $x^{(1)},\ldots,x^{(M)}$, aggregate the node
cost before search:

$$
C_{\rm group}(v)=\frac1M\sum_{m=1}^M C\bigl(c_v^{(m)}\bigr).
$$

Select on one calibration split and validate on another to detect basis
overfitting.

A cost specifies what “best” means. Entropy can prefer concentrated
activations; off-block operator energy can prefer a useful matrix layout; a
downstream loss can prefer task quality. These objectives need not agree. In
particular, the project should select a candidate WPT basis using declared,
reproducible calibration data, then compare it on held-out data against original
coordinates, a random permutation, and a random orthogonal basis.

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

For multi-head attention it is also useful to expose a reshaped view
$Q\in\mathbb R^{B\times T\times h\times d_h}$, where $h$ is the number of
heads and $d=h d_h$. The axis is part of the hypothesis:

| Axis | Natural WPT interpretation | Main constraint in an LLM |
|---|---|---|
| Batch $B$ | None in ordinary inference | Examples are independent; do not mix them. |
| Token position $T$ | Time and scale | A transform or mask must prevent future-token leakage. |
| Residual feature $d$ | Channel/feature packets | Channel adjacency is not given by the architecture. |
| Attention head $h$ | Coarse groups of projections | Head partitioning and RoPE interfaces must remain well defined. |
| Per-head feature $d_h$ | Local coordinates within one head | May alter attention geometry unless all dependent operations are compiled. |

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

## 9. Polar coordinates and packet bases

Polar coordinates offer another view of a vector, but they are not an
orthogonal linear basis change. For $x\ne0$ in $\mathbb R^m$, a polar map has
the schematic form

$$
\operatorname{Polar}(x)=(r,\theta),
\qquad r=\|x\|_2,
\qquad \theta\in S^{m-1}.
$$

It can be invertible away from singular cases, but it does not preserve vector
addition:

$$
\operatorname{Polar}(x+y)\ne
\operatorname{Polar}(x)+\operatorname{Polar}(y).
$$

This is the decisive difference from $c=Sx$. An orthogonal WPT permits fixed
linear weights to be re-expressed with products such as $SWS^\top$. There is
no corresponding fixed matrix compilation through a polar map. A polar
representation therefore changes the model computation, not merely its linear
coordinates.

### Recommended use: polar statistics within WPT packets

The lowest-risk combination keeps WPT as the feature-space coordinate system.
For a candidate packet leaf $v$, first compute its linear coefficients $c_v$,
then represent only that local vector in polar form:

$$
x\xrightarrow{S_{\mathcal L}}\{c_v:v\in\mathcal L\}
\xrightarrow{\operatorname{Polar}\ \text{per leaf}}
\{(r_v,\theta_v):v\in\mathcal L\}.
$$

The polar representation can define a rate--distortion-aware best-basis cost,
without being inserted into the original model's forward pass. For a declared
per-packet quantizer $Q_v$, one example is

$$
C_{\rm polar}(v)=
\lambda_R\widehat R\bigl(Q_v(r_v,\theta_v)\bigr)+
\lambda_D\frac{\|c_v-\widehat c_v\|_2^2}{D_{\rm ref}}+
\lambda_S C_{\rm struct}(v),
$$

where $\widehat c_v$ is the decoded coefficient block and
$C_{\rm struct}$ can be an off-block, KL-surrogate, or block-size term. The
cost remains suitable for a packet-tree search when rate, distortion, and the
structural term are measured independently per candidate leaf. The true
end-to-end quality and memory objective must still be evaluated after selecting
a pruning.

This construction has three possible advantages:

1. it asks whether a proposed WPT block has a favorable local rate--distortion
   curve, rather than only whether its Cartesian coefficients are sparse;
2. it can guide bit allocation or codec selection inside selected blocks;
3. it separates an exact WPT compilation experiment from a later lossy storage
   or activation/KV-cache coding experiment.

PolarQuant is relevant here as an example of recursive polar coding and
quantization after preconditioning [10]. It groups coordinates recursively into
radii and angles, and uses the resulting angle distributions to design
quantizers. Its result motivates measuring polar rate and distortion within
packet blocks; it is not evidence that those blocks are semantically meaningful.

### Alternatives and concerns

The reverse composition,

$$
x\xrightarrow{\operatorname{Polar}}(r,\theta)
\xrightarrow{\operatorname{WPT}}\widetilde c,
$$

is a possible nonlinear architecture, but not feature-space checkpoint
compilation. It introduces angular periodicity, coordinate singularities near
$r=0$, and a new imposed notion of adjacency among angles. Its nonlinearities
also prevent a single precompiled replacement for the original Transformer
weights. Treat it as a train-or-adapt-and-evaluate proposal, with its own causal
and numerical tests.

PolarQuant-style random preconditioning supplies a useful control, not a
neighborhood-discovery method. A random orthogonal rotation can preserve norms
and inner products while deliberately making coordinate identity uninformative.
Applied across proposed WPT blocks, it can erase the locality the packet basis
is intended to test. Use whole-vector random rotations as a strong null
baseline; if a rotation is used in a codec, restrict it within an already
selected packet block and compare its benefit against the added kernel cost.

Other practical concerns are metadata and runtime overhead, angle-codebook
stability under distribution drift, the behavior of low-radius vectors, and
interactions between quantization errors from several blocks. A lower local
polar coding cost does not imply lower model KL, faster decode, or improved
continual learning.

### Recommended future track: polar-packet rate--distortion

Create a separate `polar-packet-rate-distortion` track only after the
feature-space compilation track demonstrates exact numerical re-expression and
the channel-neighborhood track demonstrates held-out structure beyond identity,
random-permutation, and random-orthogonal controls. It should proceed in stages:

| Stage | Smallest decision | Required comparison or gate |
|---|---|---|
| P0 | Can polar per-packet statistics be measured reproducibly? | Fixed calibration data; Cartesian entropy and rate--distortion controls. |
| P1 | Does a polar cost select a stable WPT pruning? | Held-out basis stability and all channel-order controls. |
| P2 | Does selected-block coding help at equal memory? | Cartesian per-block quantization, whole-head PolarQuant-style baseline, output KL, and cache/activation memory. |
| P3 | Is there realized systems value? | End-to-end prefill/decode latency and metadata/kernel accounting. |
| P4 | Is polar-before-WPT worth architectural work? | Only after P0--P3; train/adapt a small model with causal and numerical controls. |

The track's first claim would be about compression or approximation efficiency,
not semantic channel meaning. Any claim about local learning or better inference
must be tested separately against the project baselines.

## 10. Practical choices and pitfalls

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

## 11. Connection to this project's experiments

The project separates three questions because they make different assumptions.
They must not be combined into one claim without an experiment that demonstrates
their interaction.

| Track | WPT target | Immediate decision | Required gate and controls | It does not establish |
|---|---|---|---|---|
| [Feature-space compilation](../experiments/feature-space-compilation/experiment.md) | Residual feature coordinate $d$ | Can an existing checkpoint be re-expressed exactly and does a fixed basis expose measurable structure? | Fixed-input logit and loss equivalence before approximation; original, permutation, and random-orthogonal controls for structure. | A speedup, useful sparsity, or better continual learning. |
| [Feature-channel neighborhoods](../experiments/feature-space-channel-neighborhoods/experiment.md) | A permutation or grouping before feature WPT | Can data-derived adjacency make the packet tree more useful and stable? | Held-out stability and comparison with identity and random orders. | That an ordering has a unique or human-semantic interpretation. |
| [Time-scale block-causal generation](../experiments/time-scale-block-causal-generation/experiment.md) | Token position $T$ | Can a time/scale representation support a valid block-causal generation schedule? | Prefix-leakage tests and a matched strictly causal baseline. | Feature-space checkpoint rebasing or an automatic quality gain. |

The cost-aware progression is the same in every track:

```text
shape and algebra checks
  -> exact smoke test
  -> small calibration measurement
  -> targeted ablation
  -> small adaptation or training run
  -> larger confirmation or kernel work
```

Advance only when the previous stage supplies evidence worth the additional
cost. The track documents define the concrete stopping rules and metrics.

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

## 12. Related work: Learnable Multi-Scale Wavelet Transformer

Kiruluta, Burity, and Williams propose a *Learnable Multi-Scale Wavelet
Transformer* (LMWT) that replaces self-attention with a learned Haar-like
sequence-mixing module [9]. The module pairs adjacent token representations,
forms learned approximation and detail vectors, recursively processes the
approximation branch, and aggregates multi-scale coefficients. The paper reports
a linear-in-sequence-length module cost and a WMT16 English--German comparison
against a self-attention baseline.

This is useful related work, but it addresses a different problem from
feature-space checkpoint compilation:

| Question | This project's feature-space track | LMWT preprint |
|---|---|---|
| Starting point | Existing checkpoint | Model trained with a new mixing module |
| Transform axis | Residual feature/channel axis | Token/sequence axis |
| Main objective | Exact re-basing, then structured approximation | Replace quadratic attention |
| Transform family | Fixed WPT basis, packet pruning, later constrained lifting | Learned Haar-like decomposition |
| Perfect reconstruction | Important for exact compilation | Not required by the proposed forward feature extractor |
| Packet best basis | Central possible method | Not used; only the approximation branch is recursively processed |

The paper is therefore **not** evidence that an existing LLM can be compiled
into a useful WPT feature basis. Its learned forward and inverse parameters are
not constrained in the paper to be an orthogonal or perfect-reconstruction
transform, and its recursive approximation-only hierarchy is closer to a
learnable DWT-style mixer than a full WPT packet library.

It can still inform later work in three ways:

1. **Time-scale baseline.** The time-scale block-causal track can compare a
   fixed Haar mixer, a constrained learnable lifting/WPT mixer, and attention on
   matched small models.
2. **Ablation pattern.** Compare fixed versus learned filters, number of scales,
   coefficient aggregation, quality, memory, and measured prefill/decode cost.
3. **Coefficient diagnostics.** Its per-scale coefficient visualizations suggest
   useful summaries: scale energy, entropy, sparsity, cross-scale covariance,
   and task-conditioned activation profiles.

There are important requirements before using LMWT-style ideas in our
experiments. A decoder must be demonstrably causal: a recursive transform over a
teacher-forced full target sequence can expose future tokens unless it uses
causal filters, completed blocks, masked/noisy slots, or a valid block-generation
schedule. The preprint's attention-free decoder description does not itself
establish an autoregressive no-leakage guarantee. Any reproduction or extension
must include adversarial leakage tests, matched parameter/training budgets, and
wall-clock measurements over several sequence lengths; a theoretical $O(T)$
module cost alone is insufficient.

For learned basis work in the feature-space track, use the paper as motivation
for learning structured local transforms, but retain explicit constraints on
invertibility, conditioning, locality, and—when required—orthogonality. Those
constraints are what preserve the distinction between a learnable WPT basis and
a generic learned sequence mixer.

## 13. Reference reading

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
9. A. Kiruluta, P. Burity, and S. Williams, “Learnable Multi-Scale Wavelet
   Transformer: A Novel Alternative to Self-Attention,” arXiv:2504.08801,
   version 1, 2025. [Open HTML paper](https://arxiv.org/html/2504.08801v1).
10. I. Han, P. Kacham, A. Karbasi, V. Mirrokni, and A. Zandieh, “PolarQuant:
    Quantizing KV Caches with Polar Transformation,” arXiv:2502.02617,
    version 1, 2025. [Open HTML paper](https://arxiv.org/html/2502.02617v1).
