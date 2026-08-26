# Learned Hermite VFI — v2.1

**Changes from v1:** CoeffNet rebuilt as a **~0.43M-param** head on frozen RAFT features (was a full-res U-Net). Blur cue added. Cross-lattice velocity consistency loss added. Claim restated.

**No U-Net anywhere in Phase 2.** AppNet is encoder-only (no decoder, no skips). CoeffHead is single-scale at $1/8$ res. Upsampling reuses RAFT's own convex mask rather than a learned one.

---

## Phase 0 — Notation

| Symbol | Shape | Lattice | Meaning |
|---|---|---|---|
| $I_0, I_1$ | $H\!\times\!W\!\times\!3$ | 0, 1 | input keyframes |
| $t$ | scalar | — | query time, $t\in(0,1)$ |
| $F$ | $H\!\times\!W\!\times\!2$ | **0** | $f_{0\to1}$, RAFT |
| $F'$ | $H\!\times\!W\!\times\!2$ | **1** | $f_{1\to0}$, RAFT |
| $c_i, h_i^{(N)}$ | $\tfrac H8\!\times\!\tfrac W8\!\times\!128$ | $i$ | RAFT context feats / final GRU state |
| $U$ | $H\!\times\!W\!\times\!1$ | 0 | fwd–bwd inconsistency (occlusion) |
| $Z$ | $H\!\times\!W\!\times\!1$ | 0 | splat importance |
| $\mathcal B_i$ | $H\!\times\!W\!\times\!3$ | $i$ | blur descriptor |
| $\mathcal S_i$ | $\tfrac H8\!\times\!\tfrac W8\!\times\!64$ | $i$ | appearance features (small CNN) |
| $\delta_0,\delta_1$ | $H\!\times\!W\!\times\!2$ | **0** | **velocity residuals** — what the head predicts |
| $m_0,m_1$ | $H\!\times\!W\!\times\!2$ | **0** | endpoint velocities, $m_i=F+\delta_i$ |
| $A,B$ | $H\!\times\!W\!\times\!2$ | **0** | basis coefficients |
| $\Phi(t)$ | $H\!\times\!W\!\times\!2$ | **0** | $f_{0\to t}$ |
| $\Phi'(t)$ | $H\!\times\!W\!\times\!2$ | **1** | $f_{1\to t}$ |
| $G_0,G_1$ | $H\!\times\!W\!\times\!2$ | **$t$** | $f_{t\to0}, f_{t\to1}$ |
| $M, R$ | $H\!\times\!W\!\times\!1,\,3$ | $t$ | fusion mask, residual |
| $\overleftarrow\omega,\overrightarrow\omega$ | — | — | backward warp / forward splat |

**Lattice discipline.** Tensors on different lattices cannot be added. Phase 4 exists solely to move a field from lattice 0 to lattice $t$.

**Learned:** AppNet (small CNN), CoeffHead, RefineNet, SynthNet. **Frozen:** RAFT.

**What velocity means here.** $F$ is the *average* velocity over $[0,1]$. $m_0, m_1$ are the *instantaneous* velocities at the endpoints. These differ only under non-uniform motion. $\delta_i = m_i - F$ is therefore a small, zero-centred correction, not a from-scratch prediction — which is why zero-init gives the linear baseline exactly.

---

## Phase 1 — Measure

**In:** $I_0, I_1$  **Out:** $F, F', U, Z$, $c_0, c_1, h_0^{(N)}, h_1^{(N)}$, $W_0, W_1$, $\mathcal B_0, \mathcal B_1$

Run RAFT both ways **inside `torch.no_grad()`** and keep the internals:

$$F,\; c_0,\; h_0^{(N)},\; W_0 = \text{RAFT}(I_0, I_1), \qquad F',\; c_1,\; h_1^{(N)},\; W_1 = \text{RAFT}(I_1, I_0)$$

$W_i \in \mathbb R^{\frac H8\times\frac W8\times 576}$ is RAFT's own convex-upsampling mask from the final iteration — reused verbatim in Phase 2.4, not re-predicted.

Freezing parameters stops updates but *not* backward traversal. `no_grad` is what removes the $N$-iteration GRU graph from memory. Keep it behind a flag in case RAFT is fine-tuned later.

$$U = \big\|F + \overleftarrow\omega(F', F)\big\|_1, \qquad Z = -\big\|I_0 - \overleftarrow\omega(I_1, F)\big\|_1$$

**Blur descriptor** (no parameters). From the structure tensor $J_i = \mathcal G_{5\times5} * (\nabla I_i \nabla I_i^\top)$ with eigenvalues $\lambda_1 \ge \lambda_2$ and dominant orientation $\theta$:

$$\mathcal B_i = \big[\; \lambda_1/(\lambda_2+\epsilon),\;\; \cos 2\theta,\;\; \sin 2\theta \;\big]$$

Anisotropy tracks blur extent; orientation tracks blur direction. Orientation is sign-ambiguous — $F$ resolves the sign. This is the **only** input carrying genuine sub-interval information: flow is path-independent, so $F, F', U$ contain no curvature signal.

---

## Phase 2 — Predict endpoint velocities · **no $t$** · all at $1/8$ res until 2.4

**In:** Phase-1 outputs  **Out:** $m_0, m_1$

**2.1 — AppNet (trainable, ~0.3M params, gateable).** Three stride-2 convs, channels $32{\to}64{\to}64$:

$$\mathcal S_0 = \text{AppNet}\big([\,I_0,\;\overleftarrow\omega(I_1,F),\;\mathcal B_0,\;\overleftarrow\omega(\mathcal B_1,F)\,]\big)$$

RAFT's encoder learned *which pixel matches which*. AppNet learns *what kind of thing this is and how such things move*. Different features; do not assume $c_0$ covers it.

**2.2 — Concatenate at $1/8$ res.**

$$\Psi_0 = \big[\; c_0,\;\; h_0^{(N)},\;\; \tfrac18 F^{\downarrow 8},\;\; \tfrac18\overleftarrow\omega(F',F)^{\downarrow 8},\;\; U^{\downarrow 8},\;\; \mathcal S_0 \;\big]$$

**Divide flow by 8.** RAFT's internal flow is in $1/8$-resolution pixel units. Mixing full-res magnitudes with $1/8$ features is a silent scale bug that presents as "the network refuses to learn."

**2.3 — CoeffHead (~0.37M params).** Single scale, no encoder/decoder:

$$\Xi_0 = \text{ResBlock}^{\times 2}_{96}\big(\text{Conv}_{1\times1}(\Psi_0)\big)$$

$$\delta_0^{\downarrow 8},\; \delta_1^{\downarrow 8} = \text{Conv}_{3\times3}(\Xi_0) \in \mathbb R^{\frac H8\times\frac W8\times 4}$$

**Zero-init the final conv.** Training starts exactly at linear.

**2.4 — Convex upsample using RAFT's mask.**

$$\delta_i = \text{ConvexUp}\big(8\cdot\delta_i^{\downarrow 8},\;\; \text{softmax}_9(W_0)\big)$$

No learned mask head. $\delta$ is a correction *to* $F$, so it shares $F$'s motion boundaries and should share $F$'s upsampler. Reusing $W_0$ is not just a saving — it is the more principled choice, and it guarantees $\delta$ cannot leak across an edge that $F$ respects. Bilinear would smear $\delta$ across motion boundaries, exactly where non-uniform motion lives. The swapped pass uses $W_1$.

**2.5 — Occlusion gate.**

$$\alpha = \sigma(w_1 - w_2 U), \qquad \delta_i \leftarrow \alpha\,\delta_i$$

$$m_0 = F + \delta_0, \qquad m_1 = F + \delta_1$$

**2.6 — Evaluation basis.**

$$A = -2\delta_0 - \delta_1, \qquad B = \delta_0 + \delta_1$$

**2.7 — Swapped pass.** Same weights, inputs $\Psi_1 = [c_1, h_1^{(N)}, \tfrac18 F'^{\downarrow8}, \tfrac18\overleftarrow\omega(F,F')^{\downarrow8}, U'^{\downarrow8}, \mathcal S_1]$ → $\delta_0', \delta_1', A', B'$ on lattice 1. **$c_1, h_1^{(N)}$ already exist from Phase 1 — no extra RAFT call.**

**Ablation gates** (same checkpoint, no retraining):

| Gate | Zeroed | Kept |
|---|---|---|
| full | — | all |
| no-appearance | $\mathcal S_i$ | $c_i, h_i^{(N)}, F, F', U$ |
| flow-only (strict) | $\mathcal S_i, c_i, h_i^{(N)}$ | $F, \overleftarrow\omega(F',F), U$ |
| no-blur | $\mathcal B_i$ channels only | all else |

$h^{(N)}$ is initialized from `cnet`, so it is appearance-contaminated — it must be zeroed in the strict gate or the ablation is uninterpretable. Log $\|\delta_0\|,\|\delta_1\|$ per gate alongside PSNR.

**One caveat for the input ablation.** $W_0$ (RAFT's upsample mask) is reused in 2.4 and is *not* gated, and it too derives from a `cnet`-initialized state. It only redistributes $\delta$ across the $8\times8$ block — it cannot create magnitude where $\delta^{\downarrow 8}=0$ — so the flow-only gate remains valid for the "does curvature appear at all" question. Note it in the paper rather than letting a reviewer find it.

---

## Phase 3 — Evaluate the trajectory · **no parameters**

**In:** $F, A, B, t$  **Out:** $\Phi(t)$

$$\beta_2(s) = s^2 - s, \qquad \beta_3(s) = s^3 - s$$

$$\boxed{\;\Phi(t) = t\,F + \beta_2(t)\,A + \beta_3(t)\,B\;}\qquad\text{(lattice 0)}$$

$$\boxed{\;\Phi'(t) = (1-t)\,F' + \beta_2(1-t)\,A' + \beta_3(1-t)\,B'\;}\qquad\text{(lattice 1)}$$

$\beta_2, \beta_3$ vanish at 0 and 1, so **for any network output**:

$$\Phi(0)=0,\quad \Phi(1)=F,\qquad \Phi'(0)=F',\quad \Phi'(1)=0$$

$$\dot\Phi(0) = F - A - B = m_0, \qquad \dot\Phi(1) = F + A + 2B = m_1$$

**Degree switch (config flag):** $B=0$ → quadratic, which is *identical* to IQ-VFI Eq. 4 with $A = a_\tau/2$ — not an approximation, an identity. $\delta_i=0$ → linear. Add $C\beta_4(t)$, $\beta_4(s)=s^4-s$ → quartic.

*Only stage that consumes $t$. Three tensor multiplies.*

---

## Phase 4 — Reverse: lattice 0 → lattice $t$

**In:** $\Phi(t), \Phi'(t), Z$  **Out:** $G_0, G_1$

$$\tilde G_0(\mathbf u) = \frac{\displaystyle\sum_{\mathbf x\,:\,\mathbf x+\Phi(t)(\mathbf x)\,\in\,\mathcal N(\mathbf u)} w(\mathbf x)\,e^{Z(\mathbf x)}\,\big(-\Phi(t)(\mathbf x)\big)}{\displaystyle\sum_{\mathbf x} w(\mathbf x)\,e^{Z(\mathbf x)}}$$

$\mathcal N(\mathbf u)$ = bilinear neighbours; $e^Z$ = softmax-splat importance; the negation flips the arrow to point from $t$ back to 0. Likewise $\tilde G_1$ from $\Phi'(t)$.

**Hole filling.** $H_i$ = accumulated splat weight; $H_i=0$ marks a pixel nothing landed on:

$$G_0, G_1 = \tilde G_0, \tilde G_1 + \text{RefineNet}\big(\tilde G_0, \tilde G_1, H_0, H_1\big)$$

---

## Phase 5 — Warp and fuse

$$I_{t\to i}(x,y) = I_i\big(x + G_i^h(x,y),\; y + G_i^v(x,y)\big), \quad i \in \{0,1\}$$

$$M, R = \text{SynthNet}\big(I_{t\to0}, I_{t\to1}, G_0, G_1\big)$$

$$\hat I_t = M \odot I_{t\to0} + (1-M) \odot I_{t\to1} + R$$

---

## Cost

| Phase | Frequency | Params |
|---|---|---|
| 1 — RAFT (×2, no_grad) | once per clip | frozen |
| 2 — AppNet + CoeffHead | **once per clip** | **0.43M** |
| 3 — basis eval | per $t$ | 0 |
| 4 — reversal | per $t$ | RefineNet |
| 5 — synthesis | per $t$ | SynthNet |

**Phase-2 budget:**

| Piece | Shape | Params |
|---|---|---|
| AppNet | 3× stride-2 conv, $32{\to}64{\to}64$ | 60K |
| $1\times1$ fusion | $325 \to 96$ | 31K |
| ResBlock ×2 | $96 \to 96$, 2 convs each | 332K |
| $\delta$ heads | $96 \to 4$ | 3.5K |
| Upsample mask | reused from RAFT | **0** |
| | | **427K** |

Every one of these runs at $1/8$ resolution — $64\times$ fewer pixels than full res — and RAFT contributes no backward graph. Compute, not parameter count, was the v1 bottleneck.

For $16\times$: one Phase-2 pass, 15 closed-form evaluations. Phases 4–5 still run per $t$ — **the amortization is on the trajectory model only.**

---

## Scope, precisely

| Property | Holds for | Does **not** hold for |
|---|---|---|
| Exact endpoints | $\Phi$, the flow field | $\hat I_t$ — Phases 4–5 run regardless, so $\hat I_0 \neq I_0$ |
| One smooth curve | the $K$ flow estimates | the $K$ output frames — SynthNet runs per $t$ |

**Claim, stated correctly.** Prior explicit trajectory models need extra temporal support — four frames (QVI, EQVI), point tracks (CT-VFI), or events (Time Lens++). Existing two-frame non-linear methods (IQ-VFI, GIMM) are implicit and yield no closed-form curve. We parameterize by *endpoint velocity*, which raises the expressible motion order without raising the frame count, at zero marginal cost per $t$.

Velocity is a **latent** the loss back-propagates into — it is learned from the frames, not computed from them by formula. Flow is path-independent and therefore curvature-blind, so the signal must come from appearance. The blur descriptor $\mathcal B_i$ is computed from RGB and stays inside the "RGB alone" claim; it is the one concrete physical mechanism by which two images carry sub-interval motion.

**Do not write that velocity is measured.** Write that it is predicted from RGB and flow features, and let the input ablation quantify how much each contributes.

---

## Training

1. **Multiple $t$ per clip.** At a single $t$ only $\beta_2 A + \beta_3 B$ is observable, so $\delta_0$ and $\delta_1$ never separate. Vimeo **septuplet**, $t \in \{1/6,\dots,5/6\}$; reuse one Phase-2 pass across all $t$ in the batch.

2. **Photometric-only is the default. Teacher is a fallback, not the plan.**

   The paper claims curvature from two RGB frames *without* a privileged teacher. IQ-VFI's Table 2(d) — acceleration net + reconstruction loss only, zero gain — looks like counter-evidence, but it likely does not transfer:

   IQ-VFI trains on Vimeo **triplets**, i.e. a single $t=0.5$. At one timestep only the scalar combination $\beta_2(t)A + \beta_3(t)B$ is observable, so $A$ and $B$ are confounded and $\delta_0, \delta_1$ are **unidentifiable**. Their null result is a statement about identifiability, not about the strength of photometric loss.

   Training on **septuplets** observes the curve at five distinct $t$, which separates the two coefficients. Different problem.

   Using septuplet frames as *targets* is not a privileged signal — privileged means $I_t$ enters a teacher branch as *input*. Ordinary reconstruction loss at multiple $t$ is plain supervision.

   **Order of attempts:**
   1. Photometric multi-$t$ + $\mathcal L_{\text{vel}}$ + blur cue. If $\|\delta_i\| \gg 0$, contribution 2 holds as written.
   2. If $\delta \to 0$: add the teacher (RIFE's privileged block, IQ-VFI's $\mathcal L_{IA}/\mathcal L_{IM}$, BiM-VFI's KDVCF) and **rewrite the claim** — drop the IQ-VFI teacher contrast, keep the RGB-vs-flow input ablation as the contribution.

   Report the teacher row either way; the gap between (1) and (2) is itself a result.

3. **Cross-lattice velocity consistency.** Differentiating $\Phi'$ at $t=1$ (with $\beta_2'(0)=\beta_3'(0)=-1$):

$$\frac{d\Phi'}{dt}\Big|_{t=1} = -\big[F' - A' - B'\big] = -m_0'$$

   The particle leaving $\mathbf x$ arrives at $\mathbf x + F(\mathbf x)$ on lattice 1, so the two models must agree there:

$$m_1(\mathbf x) = -\,\overleftarrow\omega\big(m_0',\,F\big)(\mathbf x), \qquad m_0 = -\,\overleftarrow\omega(m_1',\,F')$$

$$\mathcal L_{\text{vel}} = \big\|(1-\alpha_{\text{occ}}) \odot \big(m_1 + \overleftarrow\omega(m_0', F)\big)\big\|_1, \qquad \lambda \approx 0.1$$

   **This is a regularizer, not a driver.** $\delta \equiv 0$ satisfies it trivially. It prunes inconsistent non-linear solutions and forces both passes to describe one physical curve; it does not create non-linearity. State this in the paper before a reviewer does.

4. **Optional curriculum.** Where 4 frames exist, initialize toward Catmull-Rom: $m_0 = \frac{1}{2}(F - f_{0\to-1})$, $m_1 = \frac{1}{2}(F + G')$ with $G' = \overleftarrow\omega(f_{1\to2}, F)$.

5. **Falsification test.** The framework *predicts* that the flow-only gate collapses to $\delta \approx 0$, because flow is path-independent. If flow-only matches the full model, the appearance hypothesis is dead — and one run tells you.

   **Run this at full capacity first.** A small head can suppress $\delta$ on its own, which makes a null result ambiguous: "appearance carries no dynamics signal" and "the head was too small to extract it" look identical. Confirm $\|\delta_i\| \gg 0$ with the 3×128 head, *then* shrink to 2×96 and verify the effect survives. Report both sizes. Capacity is an efficiency knob, not part of the claim — Phase 3 is where the contribution lives, and it has no parameters at all.

---

# Appendix A — Module I/O

Shapes: $H,W$ = full resolution. $h = H/8$, $w = W/8$. Batch omitted.

**Runs once per clip:** A1–A9. **Runs per $t$:** A10–A15.

---

### A1 · RAFT — *frozen, `no_grad`, called twice*

| | Tensor | Shape | Lattice |
|---|---|---|---|
| **in** | $I_0$ | $H\times W\times 3$ | 0 |
| | $I_1$ | $H\times W\times 3$ | 1 |
| **out** | $F$ | $H\times W\times 2$ | 0 |
| | $c_0$ (context, ReLU half) | $h\times w\times 128$ | 0 |
| | $h_0^{(N)}$ (final GRU state) | $h\times w\times 128$ | 0 |
| | $W_0$ (convex-upsample mask) | $h\times w\times 576$ | 0 |

Second call swaps the arguments → $F', c_1, h_1^{(N)}, W_1$ on lattice 1.
Note $F$ is emitted at full res but RAFT's *internal* flow is $h\times w\times 2$ in $1/8$-pixel units. Phase 2 needs the internal scale — divide by 8 after downsampling.

---

### A2 · Occlusion & importance — *no parameters*

| | Tensor | Shape |
|---|---|---|
| **in** | $F$, $F'$, $I_0$, $I_1$ | — |
| **out** | $U = \|F + \overleftarrow\omega(F',F)\|_1$ | $H\times W\times 1$ |
| | $Z = -\|I_0 - \overleftarrow\omega(I_1,F)\|_1$ | $H\times W\times 1$ |

$U$ → occlusion gate (A7) and head input. $Z$ → splat weights (A11) only.

---

### A3 · Blur descriptor — *no parameters*

| | Tensor | Shape |
|---|---|---|
| **in** | $I_i$ | $H\times W\times 3$ |
| **out** | $\mathcal B_i = [\lambda_1/(\lambda_2{+}\epsilon),\, \cos2\theta,\, \sin2\theta]$ | $H\times W\times 3$ |

From the smoothed structure tensor $\mathcal G_{5\times5} * (\nabla I_i \nabla I_i^\top)$.

---

### A4 · AppNet — *trainable, 60K, gateable*

| | Tensor | Shape |
|---|---|---|
| **in** | $[\,I_0,\; \overleftarrow\omega(I_1,F),\; \mathcal B_0,\; \overleftarrow\omega(\mathcal B_1,F)\,]$ | $H\times W\times 12$ |
| **out** | $\mathcal S_0$ | $h\times w\times 64$ |

Three stride-2 convs, $12\to32\to64\to64$. Encoder only — no decoder, no skips. Every input is pre-warped to lattice 0 before entering.

---

### A5 · Fusion $1\times1$ conv — *trainable, 31K*

| | Tensor | Shape | Ch |
|---|---|---|---|
| **in** | $c_0$ | $h\times w$ | 128 |
| | $h_0^{(N)}$ | $h\times w$ | 128 |
| | $\tfrac18 F^{\downarrow8}$ | $h\times w$ | 2 |
| | $\tfrac18 \overleftarrow\omega(F',F)^{\downarrow8}$ | $h\times w$ | 2 |
| | $U^{\downarrow8}$ | $h\times w$ | 1 |
| | $\mathcal S_0$ | $h\times w$ | 64 |
| | **concat** | | **325** |
| **out** | $\Psi_0 \to$ fused | $h\times w\times 96$ | |

---

### A6 · CoeffHead + $\delta$ heads — *trainable, 336K*

| | Tensor | Shape |
|---|---|---|
| **in** | fused | $h\times w\times 96$ |
| | ↓ 2× ResBlock$_{96}$ | $h\times w\times 96$ |
| **out** | $\delta_0^{\downarrow8},\,\delta_1^{\downarrow8}$ (one conv, split) | $h\times w\times 4$ |

Final conv **zero-initialized** → training starts at $\delta = 0$, i.e. exactly linear motion.

---

### A7 · Occlusion gate — *trainable, 2 scalars*

| | Tensor | Shape |
|---|---|---|
| **in** | $\delta_i^{\downarrow8}$, $U^{\downarrow8}$ | $h\times w$ |
| **out** | $\delta_i^{\downarrow8} \leftarrow \sigma(w_1 - w_2 U)\odot\delta_i^{\downarrow8}$ | $h\times w\times 2$ |

Where RAFT is unreliable the correction dies and the trajectory reverts to linear.

---

### A8 · ConvexUp — *no parameters*

| | Tensor | Shape |
|---|---|---|
| **in** | $8\cdot\delta_i^{\downarrow8}$ | $h\times w\times 2$ |
| | $\text{softmax}_9(W_0)$ — reshaped $(64,9)$ | $h\times w\times 576$ |
| **out** | $\delta_i$ | $H\times W\times 2$ |

Each output pixel is a convex combination of the $3\times3$ low-res neighbours. Mask comes from RAFT (A1), not predicted.

$$m_0 = F + \delta_0, \qquad m_1 = F + \delta_1 \qquad (H\times W\times 2 \text{ each, lattice } 0)$$

---

### A9 · Basis conversion — *no parameters*

| | Tensor | Shape |
|---|---|---|
| **in** | $\delta_0, \delta_1$ | $H\times W\times 2$ |
| **out** | $A = -2\delta_0 - \delta_1$ | $H\times W\times 2$ |
| | $B = \delta_0 + \delta_1$ | $H\times W\times 2$ |

**A4–A9 run a second time** on lattice 1 (inputs $c_1, h_1^{(N)}, F', W_1, \mathcal S_1$) → $A', B'$. Same weights, no extra RAFT call.

*Everything above is $t$-free. This is the amortization point.*

---

### A10 · Trajectory evaluation — *no parameters, per $t$*

| | Tensor | Shape | Lattice |
|---|---|---|---|
| **in** | $F, A, B$ | $H\times W\times 2$ | 0 |
| | $t$ | scalar | — |
| **out** | $\Phi(t) = tF + \beta_2(t)A + \beta_3(t)B$ | $H\times W\times 2$ | 0 |
| | $\Phi'(t) = (1{-}t)F' + \beta_2(1{-}t)A' + \beta_3(1{-}t)B'$ | $H\times W\times 2$ | 1 |

Three multiplies and two adds per field. This is the entire per-frame cost of the motion model.

---

### A11 · Softmax splat — *no parameters, per $t$*

| | Tensor | Shape | Lattice |
|---|---|---|---|
| **in** | $\Phi(t)$ | $H\times W\times 2$ | 0 |
| | $Z$ | $H\times W\times 1$ | 0 |
| **out** | $\tilde G_0$ | $H\times W\times 2$ | **$t$** |
| | $H_0$ (accumulated weight) | $H\times W\times 1$ | **$t$** |

Scatters $-\Phi(t)$ to bilinear neighbours of $\mathbf x + \Phi(t)(\mathbf x)$, weighted by $e^Z$. The negation reverses the arrow. Same for $\Phi'(t) \to \tilde G_1, H_1$.

**This is the lattice change.** Input indexed by frame-0 pixels, output indexed by frame-$t$ pixels.

---

### A12 · RefineNet — *trainable, per $t$*

| | Tensor | Shape | Ch |
|---|---|---|---|
| **in** | $\tilde G_0, \tilde G_1, H_0, H_1$ | $H\times W$ | 6 |
| **out** | residual | $H\times W$ | 4 |
| | $G_0, G_1 = \tilde G_0,\tilde G_1 + \text{residual}$ | $H\times W\times 2$ each | |

$H_i = 0$ marks pixels nothing splatted onto — holes to fill.

---

### A13 · Backward warp — *no parameters, per $t$*

| | Tensor | Shape |
|---|---|---|
| **in** | $I_i$ ($H\times W\times 3$), $G_i$ ($H\times W\times 2$) | |
| **out** | $I_{t\to i}$ | $H\times W\times 3$ |

Bilinear `grid_sample`. $G_i$ lives on lattice $t$, which is why A11 was necessary.

---

### A14 · SynthNet — *trainable, per $t$*

| | Tensor | Shape | Ch |
|---|---|---|---|
| **in** | $I_{t\to0}, I_{t\to1}, G_0, G_1$ | $H\times W$ | 10 |
| **out** | $M$ (sigmoid) | $H\times W\times 1$ | |
| | $R$ | $H\times W\times 3$ | |

---

### A15 · Fusion — *no parameters, per $t$*

$$\hat I_t = M \odot I_{t\to0} + (1-M)\odot I_{t\to1} + R \qquad (H\times W\times 3)$$

---

### Summary of trainable modules

| Module | Params | Frequency | Gateable |
|---|---|---|---|
| AppNet (A4) | 60K | per clip ×2 | **yes** |
| Fusion conv (A5) | 31K | per clip ×2 | partial ($c_i, h_i^{(N)}$) |
| CoeffHead (A6) | 336K | per clip ×2 | no |
| Occlusion gate (A7) | 2 | per clip ×2 | no |
| RefineNet (A12) | — | per $t$ | no |
| SynthNet (A14) | — | per $t$ | no |

For $16\times$ interpolation: A1–A9 run **once**, A10–A15 run **15 times**, and of those only A10 is the trajectory model.
