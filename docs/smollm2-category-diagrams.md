# SmolLM2-1.7B through commutative diagrams

This note presents the concrete [HuggingFaceTB/SmolLM2-1.7B configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/blob/main/config.json) as editable diagrams and equations. It is a coordinate-level aid for understanding the model and a precise vocabulary for later WPT compilation experiments; it does not claim that WPT improves this checkpoint.

The checkpoint is a `LlamaForCausalLM` configured in bf16, with vocabulary size $49{,}152$, hidden size $d=2{,}048$, MLP intermediate size $d_{ff}=8{,}192$, $24$ decoder layers, $32$ attention heads, and $32$ key/value heads. Its context limit is $8{,}192$ positions, RMSNorm epsilon is $10^{-5}$, RoPE theta is $130{,}000$, and its token embedding and language-model head are tied.

## Notation and categorical scope

For a sequence length $T$, write the residual-stream space as

$$
H_T = \mathbb{R}^{T \times 2048}.
$$

Let $\mathbf{DiffVec}_{\mathbb{R}}$ be the category whose objects are finite-dimensional real Euclidean spaces and whose morphisms are differentiable maps. This is the useful ambient category for a Transformer: RMSNorm, softmax, SiLU, and hence SwiGLU are nonlinear. A fixed decoder layer is therefore a morphism in this category, not a functor.

Let $\mathbf{L}_{24}$ be the path category with objects $0,1,\ldots,24$ and generating arrows $\ell \to \ell+1$ for $\ell=0,\ldots,23$. The sequence-level computation is represented by the layer-state functor

$$
\mathcal{H}:\mathbf{L}_{24}\longrightarrow\mathbf{DiffVec}_{\mathbb{R}},
\qquad
\mathcal{H}(\ell)=H_T,
\qquad
\mathcal{H}(\ell\to\ell+1)=L_\ell.
$$

Here $L_\ell$ is the complete decoder map at layer $\ell$. The token vocabulary can be regarded as a discrete category for diagrammatic purposes, but the learned embedding is most usefully treated as a map from one-hot token representations to $H_T$.

## Whole model

```mermaid
flowchart LR
  tok[Token IDs]
  emb[Embedding]
  h0[Initial residual stream]
  layers[Twenty four decoder layers]
  h24[Final residual stream]
  norm[Final RMSNorm]
  head[Tied LM head]
  softmax[Softmax]
  probs[Token probability simplex]
  tok --> emb --> h0 --> layers --> h24 --> norm --> head --> softmax --> probs
```

The embedding table has shape $W_E\in\mathbb{R}^{49152\times2048}$. With row-vector hidden states, the tied head is $W_H=W_E^\top\in\mathbb{R}^{2048\times49152}$; with column-vector conventions the displayed transposes reverse. The model computes

$$
z=\operatorname{RMSNorm}_{f}(L_{23}\circ\cdots\circ L_0(H_0)),
\qquad
p=\operatorname{softmax}(zW_H).
$$

For the row-vector embedding-table convention used here, a feature-coordinate change uses

$$
W_E^S=W_E S^\top,
\qquad
W_H^S=S W_H.
$$

Thus, when $W_H=W_E^\top$, tying is retained exactly: $(W_E^S)^\top=W_H^S$. These boundary transforms put the embedding output into WPT coordinates and map WPT-coordinate final residuals back to vocabulary logits.

## One pre-norm decoder layer

```mermaid
flowchart LR
  h[Input residual h]
  n1[Pre-attention RMSNorm]
  qkv[Q, K, V projections]
  rope[RoPE on Q and K]
  attn[Causal self-attention]
  wo[Output projection]
  add1[Add input residual]
  u[Intermediate residual u]
  n2[Pre-MLP RMSNorm]
  gu[Gate and up projections]
  swiglu[SwiGLU]
  down[Down projection]
  add2[Add intermediate residual]
  out[Output residual]
  h --> n1 --> qkv --> rope --> attn --> wo --> add1 --> u
  h --> add1
  u --> n2 --> gu --> swiglu --> down --> add2 --> out
  u --> add2
```

For layer $\ell$, suppressing sequence and head reshapes, let

$$
\begin{aligned}
x &= \operatorname{RMSNorm}_{\ell,1}(h),\\
Q &= xW_{q,\ell},\quad K=xW_{k,\ell},\quad V=xW_{v,\ell},\\
a &= \operatorname{Attn}_{\mathrm{causal}}(\operatorname{RoPE}(Q),\operatorname{RoPE}(K),V),\\
u &= h+aW_{o,\ell},\\
y &= \operatorname{RMSNorm}_{\ell,2}(u),\\
L_\ell(h) &= u+\bigl(\operatorname{SiLU}(yW_{\mathrm{gate},\ell})\odot(yW_{\mathrm{up},\ell})\bigr)W_{\mathrm{down},\ell}.
\end{aligned}
$$

Each attention head has dimension $2048/32=64$. Because the configuration has $32$ query heads and $32$ key/value heads, this is ordinary multi-head attention (MHA), not grouped-query attention (GQA). Causal attention is nonlinear because its weights are produced by a masked softmax of $QK^\top$.

## Causality as a commuting restriction square

Let $r_t:H_T\to H_t$ retain the first $t$ positions. For a causal decoder map, the output prefix cannot depend on later input positions. This property can be displayed as the following square.

```mermaid
flowchart TB
  HT[Full sequence residuals] -->|Layer map at full length| HTp[Next layer full sequence]
  HT -->|Prefix restriction| Ht[Prefix residuals]
  HTp -->|Prefix restriction| Htp[Next layer prefix]
  Ht -->|Layer map at prefix length t| Htp
```

It commutes when

$$
r_t\circ L_\ell^{(T)} = L_\ell^{(t)}\circ r_t.
$$

This is a statement about causal position computation, not a claim that the layer is linear or time-invariant.

## Feature-space WPT as a natural coordinate change

Let $S\in\mathbb{R}^{2048\times2048}$ be an orthogonal WPT/best-basis coordinate transform on the feature axis, and apply it independently at every position:

$$
S_T=I_T\otimes S: H_T\longrightarrow H_T.
$$

An ideal exact compilation defines a compiled layer-state functor $\mathcal{H}^{S}:\mathbf{L}_{24}\to\mathbf{DiffVec}_{\mathbb{R}}$ and a natural isomorphism

$$
\eta^S:\mathcal{H}\Rightarrow\mathcal{H}^{S},
\qquad
\eta^S_\ell=S_T.
$$

Its naturality condition for each decoder edge is the coordinate-change equation

$$
S_T\circ L_\ell=L_\ell^S\circ S_T.
$$

```mermaid
flowchart TB
  H0[Layer l original residuals] -->|Layer map| H1[Next layer original residuals]
  H0 -->|WPT coordinate map| HS0[Layer l WPT residuals]
  H1 -->|WPT coordinate map| HS1[Next layer WPT residuals]
  HS0 -->|Compiled layer map| HS1
```

This diagram describes the target of exact coordinate compilation. Actual implementations must establish the approximation quality numerically, including fixed-input logits and loss/perplexity differences at the stated precision.

## Attention interfaces: retain ordinary head and RoPE coordinates

The full conjugation $L_\ell^S=S_TL_\ell S_T^\top$ is a useful whole-layer idealization, but it should not be read as a license to push $S$ blindly through RMSNorm, softmax, RoPE, or SwiGLU. In particular, $S$ and $S^\top$ do not simply cancel through nonlinear operations.

A practical selective convention keeps $Q$, $K$, and $V$ in their ordinary head coordinates, so that existing head partitioning and RoPE act unchanged. For this interface calculation, use column vectors: $x^S=Sx$ and $q=W_qx$. The decoder equations above used row-vector notation only to make the sequence shapes compact. In this column-vector convention, compile the attention projections as

$$
W_q^S=W_qS^\top,
\qquad
W_k^S=W_kS^\top,
\qquad
W_v^S=W_vS^\top,
\qquad
W_o^S=SW_o.
$$

The first three maps accept a WPT-coordinate residual and emit the original $2048$-dimensional Q/K/V layout. The output projection maps the attention result back into WPT residual coordinates. This is an interface-specific compilation rule, distinct from claiming a full end-to-end layer conjugation.

### Exact coordinate change versus an unchanged Llama block

As a general differentiable-map coordinate change, an exact compiled layer can always be *defined* by

$$
L_\ell^S=S_TL_\ell S_T^\top.
$$

That definition does not mean that $L_\ell^S$ remains a Llama block with unchanged RMSNorm and SwiGLU structure. In particular, the learned diagonal RMSNorm scale $D_\gamma$ becomes $SD_\gamma S^\top$, which is generally dense for a nontrivial feature WPT. Likewise, conjugating elementwise SiLU, SwiGLU, and their gating produces cross-channel nonlinear maps. Consequently, merely precompiling $Q/K/V/O$ projections does not in general implement the exact $L_\ell^S$, nor can exact $L_\ell^S$ generally be represented using unchanged Llama RMSNorm/SwiGLU parameters.

The numerical equivalence checks in this note therefore apply to any implementation that claims an architecture-preserving exact compilation or a stated approximation; they are not implied by the diagram alone.

## Feature WPT and causal prefixes together

Because the feature transform acts independently at each position, it commutes with prefix restriction. With $S_t=I_t\otimes S$,

$$
r_t\circ S_T=S_t\circ r_t.
$$

```mermaid
flowchart TB
  HT[Full sequence original features] -->|Feature WPT| HST[Full sequence WPT features]
  HT -->|Prefix restriction| Ht[Original prefix]
  HST -->|Prefix restriction| HSt[WPT prefix]
  Ht -->|Feature WPT| HSt
```

Consequently, a feature-axis WPT does not relax the checkpoint's ordinary causal mask. A sequence-axis WPT or hierarchical block-causal method is a separate architectural proposal: it mixes positions and is not automatically causal, so its mask and generation semantics must be specified and tested independently.

## Limits and use in experiments

- These diagrams identify spaces and coordinate relations; they do not establish a speedup, quality preservation, or better continual learning.
- Exact coordinate compilation, approximate masking/pruning, quantization, and custom kernels are different claims and require separate measurements.
- A sparse or near-block-diagonal representation is useful only if the measured structure survives the chosen basis, thresholds, and hardware kernel.

## Viewing in VS Code

Modern VS Code Markdown Preview renders Mermaid fenced blocks. If a local build does not, install a Mermaid-capable Markdown Preview extension; the surrounding equations use standard `$...$` and `$$...$$` delimiters supported by the normal Markdown math workflow.
