You're right, and the $\\mathcal C\_\\psi$ thing was just bad notation on my part — it's a CNN, "C" for CNN, $\\psi$ for its weights. I'll drop the Greek. Let me redo it with the dataflow made explicit.

---

# **The chain**

I₀, I₁  
  │  
  ├─\[1\]─ RAFT ──────────────► F, F'          (flow between the two real frames)  
  │                            │  
  ├─\[2\]─ CoeffNet ────────────► A, B          (how curved is each pixel's path)  
  │                            │  
  │                            ▼  
  └─\[3\]─ formula(t, F, A, B) ─► Φ(t)          (flow from frame 0 to time t)  
                               │  
     \[4\]─ reversal ────────────► G₀, G₁       (flow from time t back to 0 and 1\)  
                               │  
     \[5\]─ warp \+ fuse ─────────► Î\_t

Every phase's output is the next phase's input. Phase 2 makes $A,B$; **Phase 3 plugs $A,B$ into the formula.** That's the link that wasn't visible before.

---

## **Why Phase 2 and Phase 3 are separate**

Because one depends on $t$ and one doesn't.

* **The shape of a pixel's path** — is the ball accelerating? — is a property of the *scene*. It doesn't change when you ask for a different $t$. That's Phase 2\.  
* **Where along that path the pixel is at time $t$** — that's Phase 3\.

Fit the curve once (Phase 2), then evaluate it at $t=0.1, 0.2, \\ldots$ (Phase 3). If you merged them you'd re-run a CNN for every frame you generate, and nothing would guarantee the 15 outputs lie on the same curve.

---

# **Phase 1 — Measure the motion you can actually see**

**In:** $I\_0, I\_1$ · **Out:** $F, F', U$

$$F, F' \= \\text{RAFT}(I\_0, I\_1)$$

* $F \= f\_{0\\to1}$: for each pixel of frame 0, how far it moved to reach frame 1\. Shape $H\\times W\\times2$.  
* $F' \= f\_{1\\to0}$: same, from frame 1's point of view.  
* $U \= |F \+ \\overleftarrow\\omega(F', F)|\_1$: go forward then come back; if you don't land where you started, that pixel is occluded. Used later to distrust it.

RAFT is frozen. This is the only real measurement in the pipeline — everything after is inference about the unseen middle.

---

# **Phase 2 — Predict the curvature**

**In:** $I\_0, I\_1, F, F', U$ · **Out:** $A, B$

$$A, B \= \\text{CoeffNet}(I\_0,; I\_1,; F,; F',; U)$$

A small CNN. Two output maps, $H\\times W\\times2$ each.

**What it's for.** RAFT tells you the ball ended up 40 px right. It cannot tell you whether the ball moved at constant speed, or started slow and sped up. $A$ and $B$ are that missing information:

* $A$ \= **acceleration** at that pixel. $A=0$ ⟹ constant speed.  
* $B$ \= **jerk** (rate of change of acceleration). $B=0$ ⟹ constant acceleration.  
* Both zero ⟹ straight line at constant speed \= plain linear VFI.

**Where the information comes from.** The CNN looks at the image: motion blur streaks reveal speed, semantics reveal what kind of object it is, flow-field structure reveals object boundaries. This is why RGB is an input and not just flow.

**Zero-init the last layer** so training starts at $A=B=0$, i.e. exactly the linear baseline. It can only improve from there.

**Gate on occlusion:** $A \\leftarrow \\sigma(w\_1 \- w\_2 U),A$. Where RAFT is unreliable, kill the correction and fall back to linear.

Run the same CNN on the swapped inputs to get $A', B'$ for frame 1's side.

---

# **Phase 3 — Evaluate the curve at time $t$**

**In:** $F, A, B$ (from Phases 1 and 2\) and $t$ · **Out:** $\\Phi(t)$

$$\\boxed{;\\Phi(t) ;=; t,F ;+; (t^2-t),A ;+; (t^3-t),B;}$$

$\\Phi(t) \= f\_{0\\to t}$: displacement of each frame-0 pixel at time $t$.

Pure arithmetic — no network, no parameters. Three tensor multiplies.

**Read it as three pieces:**

| term | what it does |
| ----- | ----- |
| $tF$ | the linear guess — straight line, constant speed |
| $(t^2-t)A$ | bends it for acceleration |
| $(t^3-t)B$ | bends it further for jerk |

**Why those exact polynomials.** Both $t^2-t$ and $t^3-t$ equal zero at $t=0$ and $t=1$. So no matter what the CNN outputs, $\\Phi(0)=0$ and $\\Phi(1)=F$. **The curve is nailed to the two frames you actually measured.** The network can only change what happens in between — which is the only place it's allowed an opinion.

**Numbers**, $F=(40,0)$, $A=(15,0)$, $B=0$, $t=0.5$:

$$\\Phi(0.5) \= 0.5(40,0) \+ (-0.25)(15,0) \+ 0 \= (20,0) \- (3.75,0) \= (16.25,,0)$$

Linear said 20 px. Curve says 16.25 px — the ball was slower early on.

Frame 1's side, same formula with $s \= 1-t$:

$$\\Phi'(t) \= (1-t)F' \+ \\big((1-t)^2-(1-t)\\big)A' \+ \\big((1-t)^3-(1-t)\\big)B'$$

---

# **Phase 4 — Flip the flow around**

**In:** $\\Phi(t), \\Phi'(t)$ · **Out:** $G\_0, G\_1$

**The problem.** $\\Phi(t)$ is stored one-value-per-*frame-0*\-pixel: "pixel $\\mathbf x$ of frame 0 moves by this much." But to build frame $t$ you need the opposite: "pixel $\\mathbf u$ of frame $t$ — where in frame 0 do I copy from?" Different pixels, different grid.

**The fix.** Send each frame-0 pixel to its destination, negate, and record what lands where:

$$G\_0(\\mathbf u) \= \\frac{\\sum\_{\\mathbf x ,\\to, \\mathbf u} w(\\mathbf x)\\big(-\\Phi(t)(\\mathbf x)\\big)}{\\sum\_{\\mathbf x ,\\to, \\mathbf u} w(\\mathbf x)}$$

* $\\mathbf x \\to \\mathbf u$ means frame-0 pixel $\\mathbf x$ lands near frame-$t$ pixel $\\mathbf u$  
* $w$ \= bilinear scatter weight  
* negation because the arrow now points from $t$ back to $0$  
* $G\_0 \= f\_{t\\to0}$, $G\_1 \= f\_{t\\to1}$

**Then patch the holes.** Some target pixels receive nothing (two objects both moved away). A small CNN fills them:

$$G\_0, G\_1 ;\\leftarrow; G\_0, G\_1 \+ \\text{RefineNet}(G\_0, G\_1, \\text{hole mask})$$

This whole phase is what RIFE argued you should skip. Keeping it is the price of having a real trajectory model.

---

# **Phase 5 — Build the frame**

**In:** $I\_0, I\_1, G\_0, G\_1$ · **Out:** $\\hat I\_t$

$$I\_{t\\to0} \= \\overleftarrow\\omega(I\_0, G\_0), \\qquad I\_{t\\to1} \= \\overleftarrow\\omega(I\_1, G\_1)$$

Two candidate frames — one made by pulling pixels from $I\_0$, one from $I\_1$. Both are complete images; they disagree at occlusions.

$$M, R \= \\text{SynthNet}(I\_{t\\to0}, I\_{t\\to1}, G\_0, G\_1)$$

$$\\hat I\_t \= M\\odot I\_{t\\to0} \+ (1-M)\\odot I\_{t\\to1} \+ R$$

$M\\in\[0,1\]$ picks which source to trust per pixel; $R$ is a residual for whatever neither source could supply.

---

# **Summary table**

| Phase | In | Out | Learned? | Runs |
| ----- | ----- | ----- | ----- | ----- |
| 1 measure | $I\_0,I\_1$ | $F,F',U$ | frozen RAFT | once/clip |
| 2 curvature | $I\_0,I\_1,F,F',U$ | $A,B$ | **yes, CNN** | once/clip |
| 3 evaluate | $F,A,B,t$ | $\\Phi(t)$ | no — formula | per $t$ |
| 4 reverse | $\\Phi(t)$ | $G\_0,G\_1$ | small CNN | per $t$ |
| 5 synthesize | $I\_i,G\_i$ | $\\hat I\_t$ | yes, CNN | per $t$ |

Phase 3 is the only place $t$ enters. Phases 1–2 run once for a clip; 3–5 run once per output frame.

