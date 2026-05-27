# HermiteFlow Implementation Walkthrough

The implementation of the Hermite-based motion model has been integrated into the `M2M_VFI` codebase following your exact mathematical and architectural specifications. As per the instructions, execution is halted here so that you can run the standalone End-Point Error (EPE) experiment before any further complexity (like confidence-based blending) is added.

## Summary of Changes

### 1. `M2M_VFI/Test/model/m2m.py`
- **[MODIFY]** Added the `hermite_flow_scale(F_0to1, F_1to0, t)` function.
- **[MODIFY]** Fixed the underlying mathematical contradiction: to ensure the Hermite polynomial reduces to `t * D` when the motion equals constant velocity `D` (as mandated by Task 3), the endpoint constraints for the standard Hermite curve were set to `pos_1 = F_0to1` (and respectively `F_1to0`).
- **[MODIFY]** Wired the Hermite scaling into `M2M_PWC.forward` under a new `motion_model='linear'` flag to easily toggle between linear interpolation and the new cubic Hermite interpolation.

### 2. `M2M_VFI/Test/test_hermite.py`
- **[NEW]** Created a standalone synthetic unit test.
- Features two tests:
  - `test_hermite_reduces_to_linear`: Validates that when providing constant velocity $D$ (meaning $F_{0\to1} = D$ and $F_{1\to0} = -D$), the output exactly matches linear scaling.
  - `test_hermite_quadratic_trajectory`: Validates that the Hermite equation correctly models a theoretical projectile/quadratic trajectory whose intermediate position and tangents are known.

### 3. `M2M_VFI/Test/epe_snu_film.py`
- **[NEW]** Created the standalone EPE decision experiment.
- The script iterates through triplets (`0.png`, `2.png` as inputs, `1.png` as pseudo-GT intermediate frame) assuming they exist inside the provided dataset folder. 
- It uses the loaded `pwcnet` component from M2M to estimate base bidirectional flows.
- It computes the intermediate flows using both models (`linear` and `hermite`), then calculates the End-Point Error (EPE) against a pseudo-ground-truth flow from $I_0 \to I_t$.
- At the end, it reports the mean EPEs and how many times Hermite "won" or "lost" vs the Linear model.

## Next Steps for You

Please run the test scripts on your machine:
```bash
conda activate autoeval

# 1. Run the synthetic unit tests to verify the math
python c:\REGULATA\Research2026\HermiteFlow\M2M_VFI\Test\test_hermite.py

# 2. Run the EPE experiment (replace with your exact SNU-FILM path if needed)
python c:\REGULATA\Research2026\HermiteFlow\M2M_VFI\Test\epe_snu_film.py --dataset /path/to/SNU-FILM-arb-Hard
```

> [!IMPORTANT]
> Once you have run the `epe_snu_film.py` experiment, please review the mean EPE metrics. If the results show that HermiteFlow outperforms the linear model, share the results with me, and we will proceed to **Task 6** (adding the occlusion/low-confidence fallback mechanism).
