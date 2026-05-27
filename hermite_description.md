Here's a clean, copy-pasteable instruction set for Antigravity (Google's agentic IDE). I've structured it as a sequence of tasks an agent can execute, with enough specificity that it won't guess wrong on the parts that matter (tangent scaling, sign convention, the EPE test).

One framing note first: I've written this for the **M2M base** (simplest flow-scaling entry point, fastest to the decision experiment). If you're using AMT instead, the same instructions apply — only the file/function names change, which the agent can locate.

---

## Instructions for Antigravity

### Context / goal (give this to the agent first)

> We are implementing **HermiteFlow**, a video frame interpolation method. The core idea: instead of scaling the two endpoint optical flows **linearly** in time to get the intermediate flow, we use a **cubic Hermite interpolant** whose **tangents are the optical flows themselves**. The motion model is closed-form and training-free. Base codebase: M2M (Many-to-Many Splatting). Do NOT change the flow estimator, warping, losses, or eval scripts. The only change is the flow-scaling step. Before building the full pipeline, we will run an isolated EPE experiment to decide whether to proceed.

### Task 1 — Locate the flow-scaling step

> Find where the codebase computes the intermediate flow at timestep t from the bidirectional flows. Look for the line that scales the flow by t, schematically `F_t = t * F_0to1` and `F_t = (1-t) * F_1to0` (or similar). Report the file, function, the exact variable names for the two bidirectional flows, the timestep variable, and the tensor shapes. Do not modify anything yet — just report what you found and confirm this is the linear-scaling step we will replace.

(Wait for its report before continuing — this is where bases differ.)

### Task 2 — Add a Hermite interpolation function

> Create a new function `hermite_flow_scale(F_0to1, F_1to0, t)` that returns the intermediate forward and backward flows using cubic Hermite interpolation. Requirements:
>
> - The two **endpoint positions** are 0 (no displacement at the source frame).
> - The two **tangents** are the optical flows. Use this sign convention: tangent at the start = `F_0to1`; tangent at the end = `-F_1to0` (the backward flow points in the opposite temporal direction, so it must be negated to act as a forward-time tangent). This matches GIMM's normalization.
> - **Tangent scaling:** the Hermite basis includes a factor for the time-interval length. Since our interval is [0,1] (length 1), the tangents are used as-is, but implement the scaling factor explicitly as a variable `dt = 1.0` so it is visible and correct — do not silently drop it.
> - Use the standard cubic Hermite basis functions of s where s = t:
>   - h00 = 2s³ − 3s² + 1
>   - h10 = s³ − 2s² + s
>   - h01 = −2s³ + 3s²
>   - h11 = s³ − s²
> - The displacement of a pixel from the source frame to time t is: `pos_t = h00*pos_0 + h10*dt*tangent_0 + h01*pos_1 + h11*dt*tangent_1`, with pos_0 = pos_1 = 0. So it reduces to `pos_t = h10*dt*tangent_0 + h11*dt*tangent_1`.
> - Compute the intermediate **forward** flow F_{t→0} and **backward** flow F_{t→1} consistent with how the base method consumes them. Match the existing tensor shapes and dtypes exactly.
> - Keep it fully vectorized (per-pixel, on the flow tensors). No Python loops over pixels.
> - Add a docstring explaining each step.

### Task 3 — Add a synthetic unit test (do NOT skip — this catches the silent bug)

> Before wiring into the pipeline, write a standalone test that validates `hermite_flow_scale` on a known curved trajectory. Construct a synthetic case: a single point moving along a known quadratic path (e.g., projectile motion) where the true intermediate position at t=0.5 is analytically known. Feed the corresponding start/end flows as tangents and assert the Hermite output matches the analytic intermediate position within a small tolerance. Also test the degenerate case: if both tangents equal the straight-line displacement, the Hermite output must equal the linear-interpolation output (this confirms correct reduction to linear). Report whether both tests pass. If they fail, the most likely cause is the tangent scaling (dt factor) or a sign error in the end tangent — check those first.

### Task 4 — Wire in behind a config flag

> Integrate `hermite_flow_scale` into the flow-scaling step found in Task 1, but gate it behind a config/argument flag `--motion_model` with options `linear` (default, unchanged original behavior) and `hermite`. When `hermite` is selected, replace the linear scaling with `hermite_flow_scale`; otherwise use the original code path untouched. This lets us run identical experiments swapping only the motion model. Do not retrain anything yet.

### Task 5 — Build the isolated EPE experiment (the decision test)

> Create a standalone evaluation script that does NOT run the full frame-synthesis pipeline. For the SNU-FILM-arb-Hard split:
>
> 1. For each sample, take the two input frames and compute bidirectional flows using the existing flow estimator.
> 2. Compute the intermediate flow at the target t **twice**: once with `linear`, once with `hermite`.
> 3. Obtain a pseudo-ground-truth intermediate flow by running the flow estimator between an input frame and the *real* intermediate frame (the same protocol GIMM uses for its VTF/VSF benchmarks).
> 4. Compute **EPE** (End-Point Error) of each method's intermediate flow against the pseudo-ground-truth.
> 5. Report mean EPE for `linear` vs `hermite` over the whole split, plus a per-sample breakdown so we can see where Hermite wins or loses (especially large-motion samples).
>
> Run this and report the two EPE numbers and the win/loss distribution. Do not proceed to building the full system until I review these numbers.

### Task 6 (conditional — only if Task 5 shows Hermite wins) — Confidence fallback

> If and only if I confirm the EPE result is favorable: add an occlusion/low-confidence fallback. Compute a forward-backward flow consistency check; where consistency is poor (likely occlusion or unreliable flow), blend the Hermite intermediate flow toward the linear result, weighted by confidence. Expose the blend strength as a config parameter. This prevents Hermite from confidently producing wrong motion where the input flow is unreliable.

---

## Critical things to tell the agent NOT to do

Add these as guardrails:

> - Do **not** modify the flow estimator, the warping operation, the loss functions, or the existing evaluation scripts.
> - Do **not** retrain any network in Tasks 1–5.
> - Do **not** "optimize" or "improve" the Hermite math beyond what's specified — keep it the standard cubic Hermite basis so it's interpretable and matches the paper.
> - Do **not** skip Task 3 (the synthetic test). The tangent-scaling and sign conventions are the most common silent bugs; the test must pass before wiring in.
> - Keep `linear` as the default so the original method is always reproducible for comparison.

---

## Two honest notes for you (not for the agent)

1. **The whole project hinges on Task 5.** Everything before it is ~a day of work; the EPE numbers tell you if HermiteFlow is real. Insist the agent stops there for your review — don't let it build the full system before you've seen linear-vs-Hermite EPE.

2. **Antigravity will likely need to fetch the M2M (or AMT) repo and read its actual flow-scaling code first.** Task 1 is deliberately a "report, don't modify" step for exactly this reason — let the agent ground itself in the real code before it edits, since I'm describing the structure schematically and the real variable names will differ.

Want the actual ~10-line Hermite function written out (correct dt scaling and sign handling already baked in) so you can paste it as a reference for the agent to match against, rather than relying on it to derive the reduction itself?