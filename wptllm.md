Transformer residual spaces are usually dense and entangled: a feature can
interact with many others, and the corresponding weight matrices need not have a
local or sparse organization. The central hypothesis is that a learned or
selected Wavelet Packet Transform (WPT) basis can expose a more useful
coordinate system—one in which activation energy and model couplings are more
localized and the important operators are close to block diagonal. The aim is
not perfect diagonalization, but a stable approximate structure that retains the
model's useful behavior.

An orthonormal WPT basis can first be treated as an exact change of coordinates.
The model's embeddings, projections, MLP interfaces, and output head can be
compiled into that basis once, so inference need not apply an additional WPT at
runtime. This exact re-expression does not itself make the model faster; its
value is that it lets us test whether a basis selected by energy, entropy,
decorrelation, or off-block criteria reveals structure that specialized kernels,
block sparsity, quantization, or memory layouts can exploit without an
unacceptable quality loss.

The same locality may also make continuous learning safer and cheaper. If new
information primarily affects a small set of WPT blocks, updates can be focused
there while the remaining blocks are protected, with rarer global consolidation
steps when cross-block changes are necessary. This is an empirical claim rather
than a guarantee: the basis, its stability under learning, the retained model
quality, and any inference or retention benefit must all be measured against
appropriate dense and random-basis baselines.


# Compiling a Transformer LLM into a Wavelet-Packet Basis

This note describes an exact change-of-basis compilation for a decoder-only
Transformer, followed by the constraints that matter when the basis is a
wavelet packet transform (WPT).

Let the residual-stream width be $d$, let
$S = W_{\mathrm{wpt}} \in \mathbb{R}^{d\times d}$ be a chosen WPT basis,
and define WPT-space coordinates by

$$
z = S h, \qquad h = S^{-1}z.
$$

For an orthonormal WPT, $S^{-1}=S^\top$. This is the preferred case since
it preserves Euclidean norms and attention dot products.

## General compilation rule

For a linear or affine map between residual-stream vectors,

$$
y = Ah+b,
$$

the equivalent WPT-space map is

$$
\boxed{A' = S A S^\top, \qquad b'=Sb.}
$$

It satisfies $z_y=A'z_x+b'=S(Ah+b)$. A more general map from a space
using basis $S_{\rm in}$ to one using $S_{\rm out}$ is

$$
A' = S_{\rm out} A S_{\rm in}^{-1}.
$$

## Two distinct constructions

The term “WPT Transformer” can mean either a model trained natively in WPT
coordinates or an existing checkpoint compiled into such coordinates. These
are related but have different meanings for a symbol such as $W_q$.

### Train a WPT-basis Transformer from scratch

Choose $z^{(W)}$ as the model's native residual-stream representation and
train its weights directly. For example,

$$
q=W_q^{(W)}z^{(W)}, \qquad
k=W_k^{(W)}z^{(W)}, \qquad
v=W_v^{(W)}z^{(W)}.
$$

There is no required “original” query matrix. $W_q^{(W)}$ is simply the
learned native parameter from WPT residual coordinates to the conventional
attention-head coordinates. The WPT basis has practical significance only if
training or execution preserves its intended structure, for example with
block-sparse layouts, WPT-local update rules, subband regularization, or a
basis-selection objective. A fully dense unconstrained model can re-entangle
its representations during training.

### Compile an existing Transformer checkpoint

Use a superscript $(O)$ for tensors in the source checkpoint's residual
coordinate system, and $(W)$ for tensors in the compiled WPT residual system:

$$
z^{(W)}=S h^{(O)}, \qquad h^{(O)}=S^\top z^{(W)}.
$$

The source-checkpoint matrices $W_q^{(O)}$, $W_k^{(O)}$, and $W_v^{(O)}$
are the actual tensors loaded from the original model. They map its original
residual activations to ordinary per-head coordinates, for example

$$
q=W_q^{(O)}h^{(O)}.
$$

Keeping query, key, and value coordinates conventional, the precompiled
runtime weights are therefore

$$
\boxed{
W_q^{(W)}=W_q^{(O)}S^\top, \qquad
W_k^{(W)}=W_k^{(O)}S^\top, \qquad
W_v^{(W)}=W_v^{(O)}S^\top.
}
$$

At runtime the compiled model evaluates only

$$
q=W_q^{(W)}z^{(W)},
$$

without reconstructing $h^{(O)}$ or applying a separate WPT transform. The
attention output projection is compiled in the opposite direction:

$$
W_o^{(W)}=S W_o^{(O)}.
$$

One may algebraically define $W_q^{(O)}=W_q^{(W)}S$ for a model trained from
scratch, but it has no privileged meaning unless interoperability with a
conventional-coordinate model is needed.

## Embedding and language-model head

Use column-vector activations. If the token embedding matrix is
$W_e\in\mathbb{R}^{d\times V}$, then

$$
h_0 = W_e\operatorname{onehot}(x),
\qquad
\boxed{W_e'=S W_e.}
$$

If the language-model head is

$$
\operatorname{logits}=W_h h+b_h,
\qquad W_h\in\mathbb{R}^{V\times d},
$$

then compile it as

$$
\boxed{W_h'=W_hS^\top, \qquad b_h'=b_h.}
$$

If input embeddings and output weights are tied, $W_h=W_e^\top$, tying is
preserved because $(W_e')^\top=W_e^\top S^\top$.

## Decoder blocks

A pre-normalization decoder block has the form

$$
u = h + \operatorname{Attn}(\operatorname{Norm}_1(h)),
$$

$$
h_{\rm next}=u+\operatorname{MLP}(\operatorname{Norm}_2(u)).
$$

In the WPT basis it remains

$$
\tilde z = z + \operatorname{Attn}'(\operatorname{Norm}_1'(z)),
$$

$$
z_{\rm next}=\tilde z+\operatorname{MLP}'(\operatorname{Norm}_2'(\tilde z)).
$$

Residual additions need no special handling: both operands are represented in
the same basis.

## Self-attention

For one attention head,

$$
q=W_qh, \qquad k=W_kh, \qquad v=W_vh.
$$

The preferred compilation keeps the residual stream in WPT coordinates but
leaves $q$, $k$, and $v$ in their existing per-head coordinates. All basis
conversion is absorbed into the stored weights:

$$
\boxed{
W_q'=W_qS^\top, \qquad
W_k'=W_kS^\top, \qquad
W_v'=W_vS^\top, \qquad
W_o'=S W_o.
}
$$

At inference time there is no explicit application of $S$ or $S^\top$:

$$
q=W_q'z, \qquad k=W_k'z, \qquad v=W_v'z,
\qquad z_{\rm attn}=W_o'\operatorname{Attn}(q,k,v).
$$

This is exactly the original attention computation, followed by conversion of
the attention result into WPT residual coordinates. Causal masking, the
softmax, RoPE, and the ordinary head split are therefore unchanged.

### Head constraint

The above preferred formulation has no head-preservation constraint on the
residual-stream WPT: $W_qS^\top$, $W_kS^\top$, and $W_vS^\top$ are
precompiled dense matrices with the original output-head layout.

Alternatively, queries, keys, and values may themselves be expressed in a new
head-space basis $T$. In that optional formulation,

$$
W_q'=T W_qS^\top, \qquad
W_k'=T W_kS^\top, \qquad
W_v'=T W_vS^\top, \qquad
W_o'=S W_oT^\top.
$$

Because multi-head attention applies independent softmaxes by head, $T$ must
be blockwise by head (apart from whole-head permutations) for exactness. The
preferred formulation avoids this complication by taking $T=I$.

## MLP

For a conventional MLP,

$$
\operatorname{MLP}(h)=W_{\rm down}\,\sigma(W_{\rm up}h),
$$

do not transform the nonlinear hidden width $d_{\rm ff}$. Transform only
the interfaces to and from the residual stream:

$$
\boxed{
W_{\rm up}'=W_{\rm up}S^\top,
\qquad
W_{\rm down}'=S W_{\rm down}.
}
$$

Then

$$
W_{\rm down}'\sigma(W_{\rm up}'z)
=S W_{\rm down}\sigma(W_{\rm up}h).
$$

For SwiGLU, apply the input-side conversion to both projections:

$$
W_{\rm gate}'=W_{\rm gate}S^\top,
\qquad
W_{\rm up}'=W_{\rm up}S^\top,
\qquad
W_{\rm down}'=S W_{\rm down}.
$$

This avoids trying to commute a basis transform through an elementwise
nonlinearity, which is not valid in general.

## RMSNorm and LayerNorm

For RMSNorm,

$$
\operatorname{RMSNorm}(h)
=\operatorname{diag}(\gamma)\frac{h}{\operatorname{rms}(h)}.
$$

Orthogonal $S$ preserves the RMS denominator, but its learned per-channel
scale is no longer diagonal in WPT coordinates:

$$
S\operatorname{RMSNorm}(S^\top z)
=\left[S\operatorname{diag}(\gamma)S^\top\right]
\frac{z}{\operatorname{rms}(z)}.
$$

The bracketed factor is generally dense. For RMSNorm without bias, a practical
exact representation uses unit-scale RMSNorm in WPT coordinates and folds the
original scale into all outgoing projections. For example,

$$
W_q'=W_q\operatorname{diag}(\gamma)S^\top,
$$

and likewise for $W_k$, $W_v$, or the MLP input projections.

LayerNorm is less convenient because it includes mean subtraction. Exact
compilation requires transforming that projection too, unless $S$ preserves
the all-ones direction.

## RoPE

With the preferred attention compilation above, $q$ and $k$ remain in their
original per-head coordinates, so the existing RoPE operation applies
unchanged. There is no runtime WPT transform.

If the optional head-space basis $T$ is used, a position-dependent RoPE
rotation $R_t$ instead becomes

$$
\boxed{R_t'=T R_t T^\top.}
$$

Using the ordinary pairwise RoPE kernel after an arbitrary nontrivial $T$
would change the model. Exact options are to implement $R_t'$, restrict $T$
so it commutes with RoPE, or use the preferred $T=I$ formulation.

## Exactness versus useful sparsity

The above transformations are an exact reparameterization, assuming the head,
normalization, and RoPE constraints are handled. Exact reparameterization alone
does **not** reduce cost: $SWS^\top$ is generally still dense.

Acceleration requires the selected WPT basis to expose exploitable structure,
such as approximate block diagonality,

$$
SWS^\top \approx
\begin{bmatrix}
B_1 & 0 & \cdots\\
0 & B_2 & \cdots\\
\vdots & & \ddots
\end{bmatrix},
$$

or sparse/low-entropy activations. One possible basis-selection objective is

$$
\min_S\sum_\ell
\left\|\operatorname{OffBlock}(S W_\ell S^\top)\right\|_F^2
+\lambda\,\mathbb{E}[\mathcal{H}(S h_\ell)]
+\mu\,D_{\rm KL}(p_{\rm original}\|p_{\rm compiled}).
$$

A single global basis will not generally jointly block-diagonalize all weights:
the layer operators need not commute or share invariant subspaces. Treat WPT
basis selection as structured approximation, validate it on calibration data,
and expect post-compilation fine-tuning if pruning or block-sparse execution is
introduced.

## Recommended starting point

1. Use one orthonormal, head-preserving WPT matrix $S$.
2. Apply the embedding, linear-map, MLP-interface, and LM-head conversions
   above.
3. Fold RMSNorm scales into subsequent projections.
4. Transform RoPE explicitly, or retain its original query/key coordinates.
5. Measure off-block energy, activation sparsity, and output-logit divergence
   before pruning or using sparse kernels.

## Operational goals after compilation

An exact WPT compilation preserves the model function. Its purpose is to expose
structure that can subsequently be used by kernels, local-learning policies,
and auxiliary routing/probing mechanisms.

### Structured kernels

Let the WPT residual stream be partitioned into $K$ blocks,

$$
z=(z_1,\ldots,z_K).
$$

For a compiled map $W_\ell'=S W_\ell S^\top$, a block mask can retain
diagonal blocks and selected cross-block couplings:

$$
\widehat W_\ell'=M_\ell\odot W_\ell'.
$$

The exact change of basis alone does not reduce computation: $W_\ell'$ is
normally dense. Kernel-level gains require measured low off-block energy and a
deliberate approximation such as masking, pruning, quantization by subband, or
a block-sparse execution format.

### Local continuous learning and rare global updates

If WPT blocks have low cross-block activation and gradient covariance, online
updates can be restricted to selected diagonal blocks:

$$
\Delta W_{\ell,kk}'=-\eta\nabla_{W_{\ell,kk}'}\mathcal L,
\qquad
\Delta W_{\ell,ij}'=0\quad(i\ne j).
$$

This makes locality an explicit learning policy rather than an assumed property
of the transform. Occasional global consolidation can update off-diagonal
couplings or relax the mask. The expected approximation error is controlled by
discarded off-block weights and cross-block covariance, so both should be
tracked on calibration and continual-learning data.

### Colored noise in WPT coordinates

Subband-specific noise can be defined by

$$
\widetilde z_k=z_k+\epsilon_k,
\qquad
\epsilon_k\sim\mathcal N(0,\sigma_k^2I).
$$

The spectrum $\{\sigma_k\}$ can be used as a training regularizer, a
robustness probe, or an uncertainty/diversity control. It should be calibrated
in WPT coordinates and normally disabled for deterministic inference, since
inference noise trades fidelity for stochasticity.

### WPT meta-pattern probes or experts

Auxiliary heads can operate on WPT activation statistics, such as subband
energy, sparsity, sign/phase behavior, temporal persistence, and cross-layer
coupling. A simple router is

$$
r=\operatorname{Pool}(\phi(z_1),\ldots,\phi(z_K)),
\qquad
g=\operatorname{softmax}(W_{\rm route}r).
$$

The routing vector $g$ may select an expert, select updateable blocks, or
produce probe scores for behaviors such as inductive, deductive, or abductive
reasoning. Semantic categories such as metaphor or allegory are not implied by
activation geometry alone: the probes require labels, weak supervision,
synthetic tasks, or behavior-derived pseudo-labels.

## Basis drift and local re-basing

Continual learning can make the original best WPT basis stale. Detect this by
periodically evaluating a basis-quality score on a calibration window, for
example

$$
J(S)=
\sum_\ell\left\|\operatorname{OffBlock}(S W_\ell S^\top)\right\|_F^2
+\lambda\,\mathbb E[\mathcal H(S h_\ell)]
+\rho\,\operatorname{CrossCov}(S h_\ell)
+\nu\,D_{\rm KL}(p_{\rm reference}\|p_S).
$$

Here the terms respectively measure off-block weight energy, activation
entropy/sparsity, residual cross-block dependence, and output drift. Trigger a
search only when the score degrades beyond a hysteresis threshold, rather than
reacting to normal minibatch variation.

Given an old basis $S_0$ and a nearby candidate basis $S_1$, define the
coordinate transition

$$
C=S_1S_0^\top.
$$

When both bases are orthonormal, re-basing every residual-stream map is an exact
function-preserving operation:

$$
W_{\rm residual}^{(1)}=C W_{\rm residual}^{(0)}C^\top,
\qquad
b^{(1)}=Cb^{(0)}.
$$

The corresponding boundary conversions are

$$
W_e^{(1)}=C W_e^{(0)},
\qquad
W_h^{(1)}=W_h^{(0)}C^\top.
$$

For MLPs, re-base only the residual interfaces:

$$
W_{\rm up}^{(1)}=W_{\rm up}^{(0)}C^\top,
\qquad
W_{\rm down}^{(1)}=C W_{\rm down}^{(0)}.
$$

The same normalisation, head-partition, and RoPE constraints described above
apply to every re-basing. Recompilation itself cannot undo forgetting because
it is only a coordinate change. It can help manage forgetting indirectly when
it restores a basis in which local updates and sparse blocks align better with
the newly learned structure; this claim must be tested against a fixed replay
or retention benchmark.

To avoid destabilizing kernels and expert identities, constrain the search to a
local WPT dictionary neighborhood, preserve block/head layout where possible,
and use canonical ordering/sign conventions for selected subbands. Evaluate a
candidate by exact re-basing first, verify near-zero logit drift, then apply any
new masks or local-learning rules as a separate approximate step.

## Time-scale variants: block-causal, locally parallel generation

The WPT compilation described above acts on the residual-feature axis. It
changes the representation of each token but does not by itself define a
time/scale representation. For hidden states
$H\in\mathbb{R}^{B\times T\times d}$, a temporal WPT acts on the sequence
axis:

$$
Z=\Psi_{\rm time}H.
$$

A separable time/feature construction can use both transforms:

$$
Z=\Psi_{\rm time}H S_{\rm feature}^\top.
$$

Here $\Psi_{\rm time}$ describes localized temporal windows and scales, while
$S_{\rm feature}$ is the feature-space basis used for the checkpoint
compilation. A temporal WPT is an architectural change, rather than a simple
function-preserving compilation of a causal Transformer: a generic temporal
change of basis does not preserve the lower-triangular causal mask or attention
softmax structure.

### Relation to causal and diffusion-style models

A decoder-only Transformer uses a total token order,

$$
x_1\prec x_2\prec\cdots\prec x_T.
$$

Diffusion is not intrinsically a spectral-domain model; its defining structure
is a denoising/noise-level process. A time-scale WPT can nevertheless provide
a useful space for scale-specific noise conditioning, local denoising, and
multiresolution planning.

### Block-causal factorization

Partition the sequence into temporal blocks $\mathcal B_1,\ldots,\mathcal B_R$.
Preserve causality between blocks while allowing a non-total dependency
structure inside an active block:

$$
p(x_{1:T})=
\prod_{r=1}^{R}
p_\theta(x_{\mathcal B_r}\mid x_{\mathcal B_{<r}}).
$$

This is a partial-order generation model. A block may contain a tree of WPT
intervals $\mathcal I_{\ell,k}$, with coarse parent states coordinating finer
child states. A possible schedule is

$$
\text{past blocks}
\rightarrow z_{\rm coarse\ block}
\rightarrow (z_{\rm left\ child},z_{\rm right\ child})
\rightarrow\cdots
\rightarrow x_{\rm block}.
$$

The sequential block frontier remains causal, while a node's child states or
masked positions can be refined in parallel.

### Hierarchical attention mask

Let $n_i$ and $n_j$ denote interval-tree nodes associated with query and key
states. One conceptual mask is

$$
M_{ij}=
\begin{cases}
0, & n_j\in\operatorname{Past}(n_i),\\
0, & n_i=n_j\text{ and the state is masked, noisy, or latent},\\
-\infty, & \text{otherwise}.
\end{cases}
$$

This permits access to completed predecessor blocks and local bidirectional
communication within the currently generated region. It must not expose
ground-truth future token embeddings within that region during training.

### What “adaptive future leakage” can safely mean

Observed future tokens or coefficients cannot be read by a causal generator.
The valid counterpart is a future *forecast* derived from available context:

$$
\widehat z_{\rm future}=f(z_{\leq t}).
$$

Equivalently, a jointly denoised current-block latent can coordinate its token
positions. A gate can select the permitted mode by layer, scale block, and
time/block index:

$$
g_{\ell,k,r}\in\{0,1\}.
$$

For example, it may select strictly causal processing, completed-block context,
past-derived forecasts, or parallel local denoising. This is an adaptive
parallelism policy, not access to actual unseen tokens.

### Internal block-generation mechanisms

Because a block is not generated token-by-token, its conditional distribution
cannot be represented solely by the usual next-token softmax. Candidate
mechanisms include:

- masked-token iterative refinement;
- denoising/diffusion over WPT coefficients or block latents;
- a discrete coarse block latent followed by fine reconstruction;
- semi-autoregressive token groups within each block.

Scale-specific colored noise is natural in this setting:

$$
\widetilde z_{k,r}=z_{k,r}+\sigma_{k,r}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

The noise schedule can target only selected scales, blocks, uncertainty states,
or routed inference modes instead of requiring universal denoising behavior.
