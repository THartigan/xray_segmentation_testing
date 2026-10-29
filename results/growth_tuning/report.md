# Growth-parameter tuning

One global configuration was selected using **20 tuning patients** (12 normal,
8 abnormal). The **5 evaluation patients** (3 normal, 2 abnormal) were held out
from growth-parameter selection. The three examples inspected in earlier experiments were
forced into the tuning set. Split seed: 20261026.

## Native-resolution results

| Split | Patients | Baseline mean Dice | Tuned mean Dice | Change |
|---|---:|---:|---:|---:|
| Tuning | 20 | 0.815233 | 0.883901 | +0.068667 |
| Held-out evaluation | 5 | 0.774705 | 0.835889 | +0.061183 |

Dice is computed on the union of the two lungs against the union of the supplied
manual masks, using every native-resolution pixel. Each patient has equal weight.
Missing seed pairs score zero and remain in the denominator.

## Selected configuration

```json
{
  "intensity_threshold": 148.2,
  "connection_radius": 235,
  "max_distance": null,
  "boundary_margin": 265,
  "min_seed_area": 1,
  "fill_holes": true
}
```

## Search procedure

- 2570 distinct configurations tested, including the original baseline.
- Initial Sobol screening of 2,048 proposals on all 20 tuning cases at stride 16,
  plus the baseline; 512 local proposals around the leading candidates.
- Recheck the best 96 screening entries plus the baseline at stride 8.
- Compare six finalists and the baseline at full resolution, then test local
  parameter neighbours at full resolution. Select by tuning Dice only.
- Keep Gaussian smoothing, initial threshold/morphology, and nearest-blob seed
  selection with missing-side recovery fixed. Spatial lengths are scaled during screening
  and reported in original pixels. No evaluation scores select parameters.

The broad search covers thresholds 70–180, connection radii 0–300 pixels,
maximum distances about 150–2,000 pixels or unlimited, margins 0–200 pixels,
and both hole-filling settings. Local searches can extend these ranges.
The result is the best found within this search, not a proven global optimum.
Five evaluation cases give only a small-sample estimate of generalization.

## Seed failures

None.

## Seed verification and evaluation scope

The first 40 Trial patients and the previously failing `MCUCXR_0309_1` all have
two nonempty, connected seeds, one on each image side, verified at native
resolution. See `seed_verification.json` and `seeds_first40_page1.png` /
`seeds_first40_page2.png`. This checks seed coverage, not perfect anatomical masks.

Missing-side recovery was added after observing a missing seed in an evaluation
image. It uses image intensities and morphology only, without manual masks or
evaluation Dice. Existing pairs are preserved; all 20 tuning seed arrays were
verified bit-for-bit unchanged, so the completed screening remains applicable.
The five evaluation cases were held out from growth-parameter selection, but
were not completely untouched during seed development. Both baseline and tuned
growth use the same improved seeds.

Recovered seeds:

- `MCUCXR_0309_1`: ['cleaned', 'fallback: threshold=55, opening_radius=100']

See `patient_scores.csv` for every individual result,
`split.json` for patient IDs, and the screening/native JSON files for all scores.

## Reproduce

From the project root, run `python scripts/tune_growth.py --phase all`.
Use `--phase prepare`, `search`, `native`, or `evaluate` to run an individual stage.
The exact preprocessing cache is stored under `data/.cache/growth_tuning/`
(ignored by Git, approximately 1.2 GB for this run); deleting it only requires
rerunning preparation. The search does not overwrite the library defaults.
