# Learned Hermite VFI — Algorithm

Cubic Hermite trajectory model for video frame interpolation, parameterized by endpoint velocities predicted from two RGB frames.

**Learned:** CoeffNet, RefineNet, SynthNet.
**Frozen:** RAFT (or FlowFormer).

---

## Phase 0 — Notation

| Symbol | Shape | Lattice | Meaning |
|---|---|---|---|
| $I_0, I_1$ | $H\!\times\!W\!\times\!3$ | 0, 1 | input keyframes at $t=0,1$ |
| $t$ | scalar | — | query time, $t\in(0,1)$ |
| $F$ | $H\!\times\!W\!\times\!2$ | **0** | $f_{0\to1}$, measured by RAFT |
| $F'$ | $H\!\times\!W\!\times\!2$ | **1** | $f_{1\to0}$, measured by RAFT |
| $U$ | $H\!\times\!W\!\times\!1$ | 0 | fwd–bwd inconsistency (occlusion signal) |
| $Z$ | $H\!\times\!W\!\times\!1$ | 0 | splat importance (brightness consistency) |
| $\delta_0,\delta_1$ | $H\!\times\!W\!\times\!2$ | **0** | **velocity residuals** — what the network predicts |
| $m_0,m_1$ | $H\!\times\!W\!\times\!2$ | **0** | endpoint velocities, $m_i=F+\delta_i$ |
| $A,B$ | $H\!\times\!W\!\times\!2$ | **0** | basis coefficients, derived from $\delta_i$ |
| $\Phi(t)$ | $H\!\times\!W\!\times\!2$ | **0** | $f_{0\to t}$ — forward trajectory |
| $\Phi'(t)$ | $H\!\times\!W\!\times\!2$ | **1** | $f_{1\to t}$ |
| $G_0,G_1$ | $H\!\times\!W\!\times\!2$ | **$t$** | $f_{t\to0}, f_{t\to1}$ — what warping needs |
| $M, R$ | $H\!\times\!W\!\times\!1,\,3$ | $t$ | fusion mask, residual |
| $\overleftarrow\omega,\overrightarrow\omega$ | — | — | backward warp (sample) / forward warp (splat) |

**Lattice discipline.** A tensor on lattice $i$ holds one value per pixel *of frame $i$*. Tensors on different lattices cannot be added. Phase 4 exists solely to move a field from lattice 0 to lattice $t$.

---

## Dataflow

```
I₀, I₁
  │
  ├─[1]─ RAFT ───────────────────► F, F', U, Z
  │                                 │
  ├─[2]─ CoeffNet ─────────────────► δ₀, δ₁ → m₀, m₁ → A, B     (no t)
  │                                 │
  └─[3]─ Φ(t) = tF + β₂(t)A + β₃(t)B                            (per t)
                                    │
     [4]─ reversal + hole fill ────► G₀, G₁
                                    │
     [5]─ warp + fuse ─────────────► Î_t
```

---

## Phase 1 — Measure

**In:** $I_0, I_1$ **Out:** $F, F', U, Z$

$$F, F' = \text{RAFT}(I_0, I_1)$$

$$U = \big\|F + \overleftarrow\omega(F', F)\big\|_1, \qquad Z = -\big\|I_0 - \overleftarrow\omega(I_1, F)\big\|_1$$

$U$: travel forward then back; failure to return means occlusion. $Z$: how well a pixel matches its correspondence — used to resolve collisions during splatting.

*This is the only measurement in the pipeline. Everything downstream is inference about the unobserved interval.*

---

## Phase 2 — Predict endpoint velocities

**In:** $I_0, I_1, F, F', U$ **Out:** $m_0, m_1$ · **no $t$**

### 2.1 Encode

Align every input to lattice 0, then encode through two branches:

$$\Xi = \text{CoeffNet}\big(\underbrace{F,\;\overleftarrow\omega(F',F),\;U}_{\text{flow branch}},\;\; \underbrace{I_0,\;\overleftarrow\omega(I_1,F)}_{\text{RGB branch}}\big)$$

**Keep the RGB branch separately gateable** — zeroing it must not require retraining the flow branch. This is the ablation the headline claim rests on.

### 2.2 Two heads

$$\delta_0, \delta_1 = \text{heads}(\Xi)$$

$$m_0 = F + \delta_0, \qquad m_1 = F + \delta_1$$

$m_0$ = velocity as the pixel leaves frame 0; $m_1$ = velocity as it arrives at frame 1. $\delta_i = 0$ means constant velocity $F$ throughout, i.e. linear motion.

**Zero-init the final conv of both heads.** Training begins at exactly the linear baseline.

### 2.3 Occlusion gate

$$\alpha = \sigma(w_1 - w_2 U), \qquad \delta_i \leftarrow \alpha\,\delta_i$$

Where RAFT is unreliable, the correction dies and the trajectory reverts to linear.

### 2.4 Convert to the evaluation basis

$$A = -2\delta_0 - \delta_1, \qquad B = \delta_0 + \delta_1$$

### 2.5 Frame-1 side

Run the same weights on the swapped pair $(I_1, I_0, F', F, U')$ to obtain $A', B'$ on lattice 1.

> **Diagnostic:** log $\|\delta_0\|, \|\delta_1\|$ from the first few thousand steps. If they stay near zero, the RGB claim is dead and you need to know immediately.

---

## Phase 3 — Evaluate the trajectory

**In:** $F, A, B$ and $t$ **Out:** $\Phi(t)$ · **no parameters**

$$\beta_2(s) = s^2 - s, \qquad \beta_3(s) = s^3 - s$$

$$\boxed{\;\Phi(t) = t\,F + \beta_2(t)\,A + \beta_3(t)\,B\;}\qquad\text{(lattice 0)}$$

$$\boxed{\;\Phi'(t) = (1-t)\,F' + \beta_2(1-t)\,A' + \beta_3(1-t)\,B'\;}\qquad\text{(lattice 1)}$$

Since $\beta_2,\beta_3$ vanish at $0$ and $1$:

$$\Phi(0)=0,\quad \Phi(1)=F,\qquad \Phi'(0)=F',\quad \Phi'(1)=0$$

**for any network output.** Velocities are recovered as intended:

$$\dot\Phi(0) = F - A - B = F + \delta_0 = m_0, \qquad \dot\Phi(1) = F + A + 2B = F + \delta_1 = m_1$$

**Degree switch (config flag):**

| variant | setting | equivalent to |
|---|---|---|
| linear | $\delta_i = 0$ | RIFE-style linear |
| quadratic | $B = 0$ | IQ-VFI |
| cubic | full | ours |
| quartic | add $C\,\beta_4(t)$, $\beta_4(s)=s^4-s$ | ablation upper end |

*This is the only stage that consumes $t$. Three tensor multiplies.*

---

## Phase 4 — Reverse: lattice 0 → lattice $t$

**In:** $\Phi(t), \Phi'(t), Z$ **Out:** $G_0, G_1$

$\Phi(t)$ answers "where does frame-0 pixel $\mathbf x$ go?" Warping needs "where does frame-$t$ pixel $\mathbf u$ come from?" — a different grid.

$$\tilde G_0(\mathbf u) = \frac{\displaystyle\sum_{\mathbf x\,:\,\mathbf x+\Phi(t)(\mathbf x)\,\in\,\mathcal N(\mathbf u)} w(\mathbf x)\,e^{Z(\mathbf x)}\,\big(-\Phi(t)(\mathbf x)\big)}{\displaystyle\sum_{\mathbf x} w(\mathbf x)\,e^{Z(\mathbf x)}}$$

- $\mathcal N(\mathbf u)$ = bilinear neighbors of $\mathbf u$
- $w$ = scatter weight
- $e^Z$ = softmax-splat importance, resolves collisions
- the negation flips the arrow to point from $t$ back to $0$

Likewise $\tilde G_1$ from $\Phi'(t)$.

**Hole filling.** Let $H_i$ = accumulated splat weight; $H_i=0$ marks a pixel nothing landed on:

$$G_0, G_1 = \tilde G_0, \tilde G_1 + \text{RefineNet}\big(\tilde G_0, \tilde G_1, H_0, H_1\big)$$

---

## Phase 5 — Warp and fuse

**In:** $I_0, I_1, G_0, G_1$ **Out:** $\hat I_t$

$$I_{t\to i}(x,y) = I_i\big(x + G_i^h(x,y),\; y + G_i^v(x,y)\big), \quad i \in \{0,1\}$$

$$M, R = \text{SynthNet}\big(I_{t\to0}, I_{t\to1}, G_0, G_1\big)$$

$$\hat I_t = M \odot I_{t\to0} + (1-M) \odot I_{t\to1} + R$$

---

## Cost

| Phase | Frequency |
|---|---|
| 1 — RAFT | once per clip |
| 2 — CoeffNet | **once per clip** |
| 3 — basis eval | per $t$ (3 tensor ops) |
| 4 — reversal | per $t$ |
| 5 — synthesis | per $t$ |

For $16\times$: one CoeffNet pass, 15 closed-form evaluations. Phases 4–5 still run per $t$ — **the amortization is on the trajectory model only**, and the paper must say so.

---

## Scope of the structural guarantees

| Property | Holds for | Does **not** hold for |
|---|---|---|
| Exact endpoints | $\Phi$, the flow field | $\hat I_t$ — Phases 4–5 run regardless, so $\hat I_0 \neq I_0$ |
| One smooth curve | the $K$ flow estimates | the $K$ output frames — SynthNet runs per $t$ |

Both claims must be scoped to the trajectory/flow in the paper. Stating them at the image level is an overclaim a reviewer will catch by pointing at SynthNet.

---

## Training

1. **Multiple $t$ per clip.** At a single $t$ only $\beta_2 A + \beta_3 B$ is observable (the ratio $\beta_2/\beta_3$ is fixed), so $\delta_0$ and $\delta_1$ never separate. Use Vimeo **septuplet** with $t \in \{1/6,\dots,5/6\}$; reuse one Phase-2 pass across all $t$ in the batch.
2. **Photometric loss alone is insufficient.** Time-to-location ambiguity drives $\delta_i \to 0$ (the network averages over trajectories). Add a teacher with access to $I_t^{GT}$ and supervise $m_0, m_1$ directly — RIFE's privileged block, IQ-VFI's $\mathcal L_{IA}/\mathcal L_{IM}$, or BiM-VFI's KDVCF.
3. **Optional curriculum.** Where 4 frames exist, initialize toward Catmull-Rom:

$$m_0 = \tfrac{1}{2}\big(F - f_{0\to-1}\big), \qquad m_1 = \tfrac{1}{2}\big(F + G'\big), \qquad G' = \overleftarrow\omega(f_{1\to2},\,F)$$

---

## Experiments that gate the contribution list

| # | Experiment | Decides |
|---|---|---|
| ① | CoeffNet input ablation: flow-only vs. flow+RGB | whether "curvature from RGB" is real |
| ② | Degree ablation: linear / quadratic / cubic / quartic, EPE only, no synthesis net | whether cubic beats IQ-VFI's quadratic |

Run ① first — the headline claim depends on it. Benchmarks: SNU-FILM hard/extreme, X-TEST, X4K1000FPS at $8\times$ and $16\times$. Vimeo triplets will show nothing (near-linear motion, and no amortization gain at $2\times$).
