# Growth-method comparison

**Recommendation: marker-controlled watershed with the existing lung seeds.**

The tested watershed implementation improves both mean Dice and leakage compared
with threshold propagation. Stronger blur alone does not solve overgrowth.

## Original-resolution scores

| Method | Tuning Dice (20) | Evaluation Dice (5) | Evaluation precision | Evaluation recall | Evaluation FP/reference area |
|---|---:|---:|---:|---:|---:|
| Previous threshold | 0.8839 | 0.8359 | 0.8099 | 0.8690 | 0.2220 |
| Threshold + extra blur | 0.8754 | 0.8242 | 0.8334 | 0.8215 | 0.1847 |
| Lower threshold without extra blur | 0.8794 | 0.8342 | 0.8581 | 0.8162 | 0.1507 |
| random walker | 0.8123 | 0.7840 | 0.9217 | 0.6831 | 0.0607 |
| watershed | 0.9284 | 0.8928 | 0.9566 | 0.8396 | 0.0378 |

Watershed reduces mean false-positive/reference area by **83.0%** on
the five evaluation patients. Dice improves on **4/5**. This metric counts all
predicted pixels outside the supplied lung masks, not only shoulder leakage.
The remaining recall deficit shows that parts of the lungs are still missed.

## Why change the method?

- Threshold propagation accepts every reachable pixel below one intensity cutoff.
  Once a low-intensity path reaches the shoulder, there is no competing background label.
- Marker-controlled watershed uses image gradients as barriers and floods from lung
  and background markers. The background can claim shoulder tissue before the lungs do.
  See the [official watershed example](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_marked_watershed.html).
- Random walker also uses competing markers, with diffusion discouraged across strong
  gradients. With these automatic markers it under-segments more severely; its lower
  Dice does not justify switching to it here. See the [official random-walker example](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_random_walker_segmentation.html).
- Active contours are another possible next experiment, but were not benchmarked in
  this run. There is no evidence here that they outperform the tested watershed.

## Selected watershed settings

```json
{
  "method": "watershed",
  "working_max_size": 384,
  "blur_sigma": 0,
  "background_threshold": 190,
  "boundary_margin": 100,
  "beta": 130
}
```

`beta` is used only by random walker. Extra blur is disabled for the selected watershed;
the original normalized sigma-30 seed preprocessing remains in effect.

## Implementation and fair comparison

- Both alternative methods use exactly the existing native lung seeds.
- Automatic background markers: valid perimeter/margin, intensities above the
  background threshold, and the chest-centre column. Seed labels take precedence.
  No reference mask enters marker construction or inference.
- The longest working-grid side is at most 384 pixels. Predictions are restored by
  nearest-neighbour sampling; original seeds are restored exactly, new invalid pixels
  are excluded, and disconnected additions are discarded. Native output shape is preserved.
- This working resolution is part of the method, not just a display reduction. It bounds
  graph memory and introduces small boundary quantization. All final Dice and leakage
  measurements compare restored masks with every original-resolution reference pixel.
- Screened 90 configurations on the same 20 tuning patients: 72 random-walker and
  18 watershed settings. Background thresholds 130/160/190, additional sigma 0/30/60,
  margins 100/265; random-walker beta 30/130/500/2000. No alternative seed tuning.
- Rechecked the top three settings per method at original output resolution. The method
  and configuration were chosen by mean tuning Dice, then frozen before evaluation.
- The same five evaluation images were inspected during earlier experiments and prompted
  changes. Their scores are a repeated comparison, not an unbiased fresh test estimate.

## Per-patient evaluation Dice

| Patient | Previous threshold | Watershed | Change |
|---|---:|---:|---:|
| MCUCXR_0029_0 | 0.8554 | 0.9303 | +0.0748 |
| MCUCXR_0081_0 | 0.8795 | 0.8749 | -0.0047 |
| MCUCXR_0102_0 | 0.8696 | 0.9345 | +0.0649 |
| MCUCXR_0309_1 | 0.6883 | 0.8001 | +0.1118 |
| MCUCXR_0352_1 | 0.8866 | 0.9244 | +0.0378 |

## Reproduce

`python scripts/compare_growth_methods.py --phase all` uses the cached native
preprocessing from `data/.cache/growth_tuning/`. Regenerate that cache with
`python scripts/tune_growth.py --phase prepare` if needed. The blur-comparison
summary is also read for the historical controls.

The notebook retains the existing sigma-5, first-40 baseline experiment and adds
separate blur and recommended watershed cells using saved sigma-30 settings.
