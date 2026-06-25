# HermiteFlow-VFI

HermiteFlow is a continuous video frame interpolation framework that replaces implicit neural representations (INRs) with closed-form Hermite cubic spline interpolation.

It builds upon the excellent infrastructure of [GIMM-VFI](https://github.com/GSeanCDAT/GIMM-VFI).

## Architecture
1. **The Head (RAFT/FlowFormer)**: Extracts bidirectional optical flows and deep latent features from input frames.
2. **The Brains (CoefficientNet)**: A shallow CNN that predicts dense, per-pixel Hermite polynomial coefficients $\alpha, \beta, \gamma, \delta$.
3. **The Engine (HermiteSplineEngine)**: A pure-math module computing context-aware intermediate bilateral flows in $O(1)$ time.
4. **The Canvas (Warping)**: Bilinear interpolation backward warping.
5. **The Tail (Synthesis Decoder)**: An AMT-style UNet decoder for high-frequency occlusion masking and blending.

## Usage

### Training
```bash
./scripts/train.sh configs/hermiteflow/hermiteflow_r.yaml ./work_dirs/hermiteflow_r "" 4
```

### Evaluation (SNU-FILM)
```bash
./scripts/bm_SNU_FILM_arb.sh
```

### Video Interpolation
```bash
./scripts/video_Nx.sh
```
