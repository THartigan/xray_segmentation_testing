# Additional blur before lung growth

The seed pipeline stays unchanged (sigma 30). Growth receives an additional
normalized Gaussian blur with **sigma 60 pixels**, excluding NaNs
and invalid background from both numerator and weights. Invalid pixels stay
invalid. This sigma is additional, not the total; away from missing data the
combined sigma is about 67 pixels. Seeds remain exactly the same.

## Native-resolution comparison

| Split | Settings | Dice | Precision | Recall | False-positive / reference area |
|---|---|---:|---:|---:|---:|
| Tuning (20) | Previous | 0.8839 | 0.8695 | 0.9015 | 0.1401 |
| Tuning (20) | Extra blur + lower threshold | 0.8754 | 0.8976 | 0.8569 | 0.1010 |
| Tuning (20) | Same settings, extra blur disabled | 0.8794 | 0.9149 | 0.8485 | 0.0812 |
| Evaluation (5) | Previous | 0.8359 | 0.8099 | 0.8690 | 0.2220 |
| Evaluation (5) | Extra blur + lower threshold | 0.8242 | 0.8334 | 0.8215 | 0.1847 |
| Evaluation (5) | Same settings, extra blur disabled | 0.8342 | 0.8581 | 0.8162 | 0.1507 |

The selected blurred settings reduce mean false-positive area by **27.9%**
on the tuning set relative to the previous settings, while reducing lung
coverage and Dice. False-positive area means predicted pixels outside the
manual lung mask divided by reference lung area; it is not a shoulder-only measure.

The matched control keeps the new threshold and all spatial settings, but
disables extra blur. This separates the effect of blur from the lower threshold.
A smoother image is not automatically a better anatomical segmentation.

## Configuration

```json
{
  "intensity_threshold": 130,
  "connection_radius": 235,
  "max_distance": null,
  "boundary_margin": 265,
  "min_seed_area": 1,
  "fill_holes": true,
  "blur_sigma": 60
}
```

## Method and limitations

- Reused the same 20 tuning / 5 evaluation patients and unchanged native seed arrays.
- Screened 180 configurations at stride 8, with blur sigmas 0, 60, 120, 180, 240;
  thresholds 95, 110, 125, 140, 148.2, 160; connection radii 100 or 235;
  maximum distances 800, 1400, or unlimited. Other controls stayed fixed.
- Selected the highest screening Dice among blurred candidates with less
  false-positive area than the previous configuration, then checked the
  candidate and threshold neighbours at native resolution on all 20 patients.
- Selected the best native Dice among those blurred candidates that reduced leakage.
  This deliberately tests the requested blur approach, not an unrestricted best-Dice model.
- Evaluation scores do not select parameters. These five cases have been inspected
  in earlier work, so this is a repeated comparison, not a fresh untouched test set.
- Seeds use no manual masks; annotations are used only for parameter scoring.

Run `python scripts/tune_growth_blur.py --phase all` from the project root.
The original experiment and settings remain in `results/growth_tuning/`.
Use `blur_sigma=0` to disable extra blur. Restart the notebook kernel after
updating the source to load the new configuration field.
