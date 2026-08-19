# HermiteFlow-VFI

Cubic Hermite trajectory model for video frame interpolation, parameterized by
endpoint velocities predicted from two RGB frames. Built on the infrastructure
of [GIMM-VFI](https://github.com/GSeanCDAT/GIMM-VFI).

**Learned:** CoeffNet, RefineNet, SynthNet. **Frozen:** RAFT (or FlowFormer).

The derivation is in [`learned_hermite_vfi.md`](learned_hermite_vfi.md); this
file is the operating manual.

## The chain

```
I0, I1
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

| Phase | In | Out | Learned? | Runs | Cost* |
| --- | --- | --- | --- | --- | --- |
| 1 measure | `I0, I1` | `F, F', U, Z` | frozen RAFT | once/clip | 186 ms |
| 2 velocities | `I0, I1, F, F', U` | `δ₀, δ₁` | **yes, CNN** | once/clip | 51 ms |
| 3 evaluate | `F, A, B, t` | `Φ(t)` | no — formula | per `t` | <1 ms |
| 4 reverse | `Φ(t), Φ'(t), Z` | `G₀, G₁` | small CNN | per `t` | 12 ms |
| 5 synthesize | `I0, I1, G₀, G₁` | `Î_t` | yes, CNN | per `t` | 26 ms |

\* 256×256, batch 1, RTX 3050 laptop. For 16×: one CoeffNet pass and 15
closed-form evaluations. Phases 4–5 still run per `t` — **the amortization is
on the trajectory model only.**

### Lattice discipline

A tensor on lattice *i* holds one value per pixel *of frame i*. `F` and `Z` live
on lattice 0, `F'` and `Z'` on lattice 1, `G₀/G₁` on lattice *t*. Tensors on
different lattices cannot be added. Phase 2 backwarps every input onto the
lattice it is encoding (`F, ω(F',F), U | I0, ω(I1,F)`); Phase 4 exists solely to
move a field from lattice 0 to lattice *t*.

### Phase 3, the core

$$\Phi(t) = t\,F + \beta_2(t)\,A + \beta_3(t)\,B,\qquad \beta_2(s)=s^2-s,\ \ \beta_3(s)=s^3-s$$

The network predicts **velocity residuals** `δ₀, δ₁`, giving endpoint velocities
`m_i = F + δ_i`, converted by `A = −2δ₀ − δ₁`, `B = δ₀ + δ₁`. Because every
basis function vanishes at 0 and 1, for *any* network output:

$$\Phi(0)=0,\quad \Phi(1)=F,\qquad \dot\Phi(0)=m_0,\quad \dot\Phi(1)=m_1$$

`δ = 0` is exactly linear interpolation, and that is where training starts.

**Scope.** Those guarantees hold for **Φ, the flow field** — not for `Î_t`.
Phases 4–5 run at every `t` regardless, so `Î_0 ≠ I_0`, and "one smooth curve"
describes the K *flow* estimates, not the K rendered frames, because SynthNet
runs per `t`. Stating either at the image level is an overclaim.

Source map: `src/models/hermite_vfi/modules/phase{1..5}_*.py`, assembled in
`hermiteflow_base.py`.

## Setup

```bash
pip install -r requirements.txt        # skip torch/torchvision on Kaggle/Colab
```

Pretrained flow estimators (Phase 1, frozen) go in `pretrained/`:
`raft-things.pth` for `hermiteflow_r`, `flowformer_sintel.pth` for
`hermiteflow_f`. Both paths are overridable at launch, see below.

Verify the install — 76 checks on the algorithm's invariants (endpoint pinning,
the dΦ(0)=m₀ / dΦ(1)=m₁ velocity identities, linear-baseline init, the degree
switch, the RGB-branch switch, gate behaviour, splat correctness and importance,
one-curve-per-clip, gradient reach, and DDP unused-parameter safety across every
ablation setting). Needs no data and no weights:

```bash
python others/verify_hermiteflow.py
```

## Training

Every path is a command-line argument; nothing is hard-coded to a machine.
Run from the repository root — the launcher snapshots `./src` next to the
checkpoints, so the working directory matters.

**Kaggle, 2×T4** — one plain `python` call, no launcher:

```python
!cd /kaggle/working/HermiteFlow-Code && python src/main.py \
    --model-config configs/hermiteflow/hermiteflow_r.yaml \
    --result-path  /kaggle/working/runs \
    --data-path    /kaggle/input/vimeo90k/vimeo_septuplet \
    --raft-ckpt    /kaggle/input/hermiteflow-weights/raft-things.pth \
    experiment.batch_size=8
```

With a single process the model is wrapped in `torch.nn.DataParallel`, which
uses **every visible GPU** — both T4s — and splits each batch between them.
`batch_size=8` therefore means 4 samples per card. Drop to 6 if GPU 0 runs out
(it carries the gathered outputs on top of its own half). The batch that the
optimizer actually sees is `experiment.total_batch_size`, reached by gradient
accumulation, so the LR schedule is unaffected by how you launch.

This is the recommended path: nothing to configure, no ports, no orphaned
worker processes if you interrupt the cell.

**Optional — DDP instead of DataParallel**, worth roughly 10-25% on two cards:

```python
!cd /kaggle/working/HermiteFlow-Code && python -m torch.distributed.run \
    --nproc_per_node=2 src/main.py \
    --model-config configs/hermiteflow/hermiteflow_r.yaml \
    --result-path  /kaggle/working/runs \
    --data-path    /kaggle/input/vimeo90k/vimeo_septuplet \
    --raft-ckpt    /kaggle/input/hermiteflow-weights/raft-things.pth
```

`python -m torch.distributed.run` is the `torchrun` launcher invoked as a
module; `--nproc_per_node=2` spawns one process per GPU. Keep
`experiment.batch_size` at 4 here — under DDP it is per process, not per node.

**Single GPU** — identical to the first command; DataParallel over one device
is a no-op wrapper.

| Flag | Overrides | Notes |
| --- | --- | --- |
| `--data-path` | `dataset.path` | training root (Vimeo `sequences/`, or X4K `encoded_train`) |
| `--val-path` | `dataset.val_path` | validation root (X4K keeps it in a separate tree) |
| `--num-timesteps` | `dataset.num_timesteps` | K middle frames supervised per clip |
| `--raft-ckpt` | `arch.pretrained_raft_ckpt` | |
| `--flowformer-ckpt` | `arch.pretrained_flowformer_ckpt` | |
| `--result-path` | — | output dir (default `./results.tmp`) |
| `--resume --load-path <ckpt>` | — | resume optimizer, scheduler and epoch |

Any other field can be set with a trailing dotlist argument, e.g.
`experiment.batch_size=2 arch.coeff_net_channels=48 arch.splat_impl=auto`.
These flags work identically in train, resume and `--eval` mode.

### X4K1000FPS — the dataset this model is for

Vimeo motion is near-linear: `F` alone explains it, so `δ₀, δ₁` have nothing to
learn and the curvature claim cannot be demonstrated. X4K is captured at
1000 fps, so a 32-frame window contains genuinely non-linear motion.

```python
!cd /kaggle/working/HermiteFlow-Code && python src/main.py \
    --model-config configs/hermiteflow/hermiteflow_r_x4k.yaml \
    --result-path  /kaggle/working/runs \
    --data-path    /kaggle/input/datasets/mdrifaturrahman33/x4k1000fps/encoded_train \
    --val-path     /kaggle/input/datasets/mdrifaturrahman33/x4k1000fps/val \
    --raft-ckpt    /kaggle/input/<your-weights>/raft-things.pth
```

**Training reads the `.mp4` files directly — do not decode them.** X-TRAIN
expands to ~240 GB of PNG, which does not fit a Kaggle working disk, and the
input mount is read-only anyway. The loader decodes only the frames each sample
needs (`grab()` past the rest), measured at **27 ms per clip** for 65 frames at
768×768 — invisible behind the dataloader workers. `mp4_decoding.py` is still
there if you want frames on disk for the ~6 GB test split or for inspection;
it now takes `<encoded-root> <output-root>` plus `--limit` / `--dry-run`.

Both path layouts resolve — the search is recursive, so `.../encoded_train` and
the doubled `.../encoded_train/encoded_train` behave identically, as do the
already-decoded `val/val/<Type>/<clip>/*.png` frames. Source is auto-detected
(`mp4` if any are present, else `png`); force it with `dataset.source`.

**Temporal protocol** mirrors X-TEST exactly as `src/X4K.py`'s `getXVFI`
evaluates it: endpoints `frame_gap: 32` apart, middles on the `t = k/8` grid.
Training therefore sees the structure the benchmark measures.

| `num_timesteps` | conditioning | peak GiB (256², B=1, fp32) |
| --- | --- | --- |
| 3 | 31.3× | 2.01 |
| 5 (default) | 26.5× | 2.82 |
| 7 (full grid) | **18.6×** | 3.64 |

Decoding cost is set by the *last* frame needed, not by how many are kept, so
raising `K` costs no extra I/O — only the per-`t` compute of phases 3–5. `K=7`
is the best-conditioned setting either dataset offers.

### The two things that make curvature learnable at all

**Multiple `t` per clip.** At a single `t` only the combination
`β₂(t)A + β₃(t)B` is observable — the ratio `β₂/β₃` is fixed — so `δ₀` and `δ₁`
never separate. The default loader (`vimeo_septuplet_multi_t`) returns `K`
ground-truth middle frames per clip, all on one trajectory between the same
endpoints, with `t ∈ {1/6…5/6}` (`span_mode: full`). One Phase-2 pass is reused
across all `K`. `num_timesteps: 3` by default; maximum 5 for 7-frame septuplets.

Identifiable is not the same as well-conditioned. Since `β₃(t)/β₂(t) = t+1`
exactly, the two basis functions are near-collinear and recovering `(A, B)` from
samples of `Φ(t)` amplifies error by 20–45× depending on **which** `t` are drawn.
The sampler therefore anchors both extremes rather than drawing uniformly, which
bounds the amplification at 24.8–27.7× for `K=3` instead of a 43.8× worst case;
`num_timesteps: 5` reaches 21.6×. Numbers and derivation in
`src/datasets/septuplet_multi_t.py`. **If you can afford it, train with
`num_timesteps=5`.**

**Trajectory distillation.** Photometric loss alone is *insufficient*:
time-to-location ambiguity lets the network average over trajectories and drive
`δ → 0` while the rendered frame still looks right. `loss.flow_distill_weight`
enables a privileged teacher that has seen `I_t^GT` — here the frozen flow
estimator itself, run on the ground-truth middle frames:

```
f_{0→t} = RAFT(I0, I_t^GT)   → supervises Φ(t)    (both on lattice 0)
f_{1→t} = RAFT(I1, I_t^GT)   → supervises Φ'(t)   (both on lattice 1)
```

No extra parameters and no second training stage. Supervising `Φ` at `K`
different `t` pins `A` and `B`, hence `m₀` and `m₁`, because the basis
conversion inverts: `δ₀ = −A − B`, `δ₁ = A + 2B`. Costs `2K` extra frozen flow
passes per step; set the weight to `0.0` to disable.

> **Watch `velocity_residual/delta_0` and `delta_1` in TensorBoard.** If they
> stay near zero through the first few thousand steps, the network has
> collapsed to the linear baseline and the curvature claim is dead — you want
> to know immediately, not after a full run.

The single-`t` triplet loaders (`vimeo_arb`, `vimeo_rgb_with_flow`) still work
and the trainer handles them as `K = 1`, but a model trained that way cannot
learn curvature.

**Not implemented: the optional Catmull-Rom curriculum.** The algorithm document
suggests initialising `m₀ = ½(F − f_{0→−1})`, `m₁ = ½(F + ω(f_{1→2}, F))` "where
4 frames exist". Under the recommended protocol there are none: `span_mode:
full` puts the endpoints at `im1` and `im7`, the extremes of the septuplet, so
no frame exists outside the span. It would only apply under `span_mode: random`
with a narrowed span, and it is marked optional, so it is left out rather than
half-wired. Enabling it would need the loader to return the two neighbour frames
and two more frozen flow passes.

### Ablations

Both gating experiments are runtime switches — no code edits, no reshaping, and
one checkpoint serves every arm because the unused parts are frozen rather than
removed.

| Experiment | Switch | Decides |
| --- | --- | --- |
| ① CoeffNet inputs | `arch.use_rgb_branch=false` | whether "curvature from RGB" is real |
| ② Trajectory degree | `arch.degree=linear\|quadratic\|cubic\|quartic` | whether cubic beats IQ-VFI's quadratic |

The two branches are fused by **addition**, not concatenation, so switching the
RGB branch off leaves the flow branch's input distribution untouched and needs
no retraining — which is the whole point of ①. Run it first.

```bash
python src/main.py -m configs/hermiteflow/hermiteflow_r.yaml \
    --data-path <septuplet> arch.use_rgb_branch=false
```

`degree=linear` freezes CoeffNet entirely (4.55M trainable vs 8.95M) since a
straight line has nothing to predict; `quadratic` and `cubic` share identical
parameters and differ only in Phase 3's basis conversion; `quartic` activates a
third head. Benchmarks: SNU-FILM hard/extreme, X-TEST, X4K1000FPS at 8× and
16×. Vimeo triplets show nothing — near-linear motion, and no amortization gain
at 2×.

### Memory and precision

At 256×256 crops the model needs ~1.9 GiB per sample, so `batch_size: 4` fits a
T4 with room to spare. Mixed precision is on by default; Phase 4's scatter and
its normalising division are forced to fp32 inside `forward_splat` regardless
of autocast, which is the one place half precision would bite. Set
`experiment.amp=False` if a run ever produces non-finite losses.

`arch.splat_impl` selects the Phase 4 scatter backend: `torch` (default,
portable, no extra dependency, ~2 ms) or `cupy`/`auto` (the CUDA
softmax-splatting kernel). Run `others/verify_hermiteflow.py` on the target
machine before switching — it compares the two backends whenever cupy is
importable.

## Evaluation

All single-process; no launcher needed.

```bash
python src/SNU_FILM_arb.py -m configs/hermiteflow/hermiteflow_r.yaml --eval \
    -l <checkpoint> --data-root /path/to/SNU-FILM -p ./eval_output/snu_film_arb

python src/X4K.py -m configs/hermiteflow/hermiteflow_f.yaml --eval \
    -l <checkpoint> --data-root /path/to/x4k/test -p ./eval_output/x4k

python src/video_Nx.py -m configs/hermiteflow/hermiteflow_r.yaml --eval \
    -l <checkpoint> --source-path <frames> --output-path <out> --N 8 --ds-factor 1.0
```

Each also accepts `--raft-ckpt` / `--flowformer-ckpt`. To score a trained run
on its own validation split:

```bash
python src/main.py --eval -m <run-dir>/config.yaml -r <run-dir> \
    -l <run-dir>/epoch60_model.pt --data-path /path/to/vimeo_septuplet
```

The `scripts/*.sh` wrappers exist for convenience on Linux and only forward
these same arguments.

## Layout

```
configs/hermiteflow/       hermiteflow_r.yaml      Vimeo septuplet, RAFT
                           hermiteflow_f.yaml      Vimeo septuplet, FlowFormer
                           hermiteflow_r_x4k.yaml  X4K1000FPS, RAFT
src/models/hermite_vfi/
    hermiteflow_base.py    the five phases, assembled
    hermiteflow_r.py       RAFT backbone for Phase 1
    hermiteflow_f.py       FlowFormer backbone for Phase 1
    modules/phase1..5_*.py one file per phase
    raft/, flowformer/     frozen flow estimators (from GIMM-VFI)
src/datasets/
    x4k_multi_t.py         X4K loader, reads .mp4 directly
    septuplet_multi_t.py   Vimeo septuplet K-timestep loader
src/trainers/
    trainer_hermiteflow.py multi-t loss, gradient accumulation
others/verify_hermiteflow.py   invariant checks
```

`src/VSF.py` and `src/VTF.py` are GIMM-era flow-supervision scripts. They target
a model this repository no longer contains and are not part of the HermiteFlow
pipeline.
