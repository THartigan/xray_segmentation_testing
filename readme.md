# MTGP — Montgom Project

Place reusable Python code in `src/mtgp/` and import it from notebooks with
`import mtgp`.

## Structure

- `src/mtgp/`: reusable Python package
- `notebooks/`: exploratory analysis and experiments
- `data/`: local datasets (ignored by Git)
- `results/`: experiment outputs
- `tests/`: package tests

## Setup

Use Python 3.12 or newer. Dependencies match the Myocardia Prediction Project
and are declared in `pyproject.toml`.

Run these commands from this project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m ipykernel install --user --name mtgp --display-name "Python (MTGP)"
```

Select **Python (MTGP)** as your notebook kernel, then open
`notebooks/01_getting_started.ipynb`. The editable installation makes changes
in `src/mtgp/` available without reinstalling the package; restart the kernel
when necessary to reload imported code.

The `.gitignore` is copied from the source project, including its Python,
macOS, dataset, and generated-output exclusions.

## Patient and trial loader

```python
from mtgp import Trial
trial = Trial()  # defaults to this project's data/MontgomerySet
trial.data      # one metadata row per patient, without loading pixel arrays
patient = trial[0]  # or trial["MCUCXR_0001_0"]
patient.clinical_reading  # complete original text
patient.age_years, patient.sex, patient.findings
patient.label, patient.classification  # 0 normal, 1 abnormal (filename label)
patient.image        # original pixel values as a NumPy array
patient.left_mask    # boolean anatomical-left mask
patient.right_mask   # boolean anatomical-right mask
patient.lung_mask    # union of both masks
patient.matrix_size, patient.pixel_spacing_mm  # both (row, column)
patient.metadata     # image/mask headers, clinical fields, and source information
```

`Trial(path)` accepts a MontgomerySet folder or its parent data folder.
`DataSet` and `Dataset` are aliases for `Trial`. Integer indexing means list
position; string indexing uses the complete filename stem. Iterate over
`trial.patients` or `trial` to access records. No random split is made implicitly.

Arrays load on demand and are not cached. Assign an array to a variable to reuse
it. Images are neither resized nor normalized. Mask dimensions are checked
against the image during initialization; mismatches are recorded in
`patient.issues` and `trial.data.masks_match_image` without resizing arrays. Missing clinical text or masks raise
by default; use `Trial(strict=False)` to expose missing values as `None`.
`patient.missing_files` and `trial.orphan_files` report unmatched files.

The supplied `NLM-MontgomeryCXRSet-ReadMe.pdf` documents **0.0875 mm** spacing
in both directions. This is a dataset-level assumption for original images,
not a value extracted from PNG headers. If you resize an image, update its
spacing accordingly. `field_of_view_mm` is derived from dimensions and spacing.
The documentation describes 12-bit gray levels; `stored_bit_depth` reports the
actual PNG bit depth separately. The abnormal label denotes the dataset class,
not an inference of active disease from the clinical text. Masks retain the
provided anatomical left/right names, not viewer-left/viewer-right.

Run package tests with `python -m unittest discover -s tests`.

## Threshold-based lung masks

```python
from mtgp import SegmentationConfig, segment_lungs
from mtgp.plotting import plot_segmentation_batch

config = SegmentationConfig(
    sigma=30, threshold=65, min_valid_intensity=10,
    opening_radius=100, closing_radius=100, max_invalid_contacts=100,
)
result = segment_lungs(trial[0].image, config)
lung_mask = result.cleaned  # Full-resolution boolean array.
figures = plot_segmentation_batch(trial[:30], config)
```

`thresholding.py` owns configuration, normalized Gaussian smoothing and pipeline
composition. `morphology.py` supplies independently usable mask operations;
`plotting.py` produces the diagnostic grids without retaining all full-resolution
results. Call `plt.show()` or save the returned figures yourself. Choose specific
plots with `stages=("candidate", "opened", "repaired", "cleaned")`.

The order matters: **open → relabel → close each component → fill holes → reject
boundary components**. Opening cuts thin white connections to boundary strips.
Closing fills narrow black clefts within each remaining component. Separate
components do not jointly participate in closing. The output may contain several
blobs; the pipeline does not select exactly two or force convex shapes.

All radii and Gaussian sigma are in original-image pixels. Increase
`opening_radius` for wider connections, at the cost of rounding tips and losing
small fragments. A zero radius disables that operation. Larger closing radii
repair wider clefts but can distort outlines; repaired regions can still touch.
Opening may leave detached border remnants, and thick boundary connections may
survive. Inspect the intermediate masks; these are experimental intensity masks.

`SegmentationResult` exposes `smoothed`, `valid`, `candidate`, `opened`, `repaired`,
`mask_valid`, and `cleaned`. Invalid pixels do not contribute to Gaussian
intensities and remain NaN in `smoothed`. Only enclosed invalid holes can be filled;
exterior invalid areas cannot serve as closing bridges. `mask_valid` includes the
filled holes for display and boundary checks. Contact tolerance counts distinct
invalid pixels in an 8-neighbourhood and is inclusive. Image-edge contact always
rejects a component. The notebook shows each stage with invalid pixels in red.

## Central seeds and region growth

```python
from mtgp import GrowthConfig, grow_lungs
from mtgp.plotting import plot_lung_growth

growth = grow_lungs(result, GrowthConfig(
    intensity_threshold=100,
    connection_radius=100,
    max_distance=800,
    boundary_margin=30,
))
left_mask = growth.left_mask    # Image-left, not anatomical left.
right_mask = growth.right_mask
lung_mask = growth.mask
fig = plot_lung_growth(trial[0].image, result, growth)
```

`seeding.py` computes the centroid of the original valid chest pixels,
then selects the closest cleaned-blob centroid on each side. Distance is
Euclidean; ties prefer the larger blob, then the smaller connected-component ID.
`min_seed_area` can exclude tiny candidates; its default of 1 applies the requested
nearest-blob rule without a size preference. `find_lung_seeds(result)` preserves
existing seeds and recovers missing sides by retrying image thresholds and
opening radii, followed by the same repair and boundary rejection. It first tries
darker interiors to break connections to boundary tissue. Recovery uses no manual
annotations and still raises if no usable blob exists. `growth.seeds.sources`
records how each seed was found. `select_lung_seeds(result.cleaned, result.valid)`
exposes strict selection without recovery.
The complete chosen blobs are retained, rather than reduced to seed points.

`python scripts/verify_seeds.py` checks the first 40 patients at full resolution,
plus the previously failing `MCUCXR_0309_1`. All 41 have nonempty, connected seeds
on both image sides. The first 40 retain their original seed pairs; the extra
case recovers its missing image-right seed at threshold 55. See the
[verification record](results/growth_tuning/seed_verification.json) and overlays
for [patients 1–20](results/growth_tuning/seeds_first40_page1.png) and
[patients 21–40](results/growth_tuning/seeds_first40_page2.png).

The region grower uses [SciPy binary propagation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_propagation.html)
to expand through 8-connected admissible pixels until no further growth is
possible. It opens the more permissive intensity mask first to block narrow
routes back to border strips. New growth is restricted to its image half, a
distance from its seed, and clearance from invalid regions and image edges.
Labels remain distinct: 0 background, 1 image-left, 2 image-right.
The centre is exposed as `growth.seeds.chest_center` in `(row, column)` order.

Seed pixels are never removed, including holes filled by the earlier pipeline.
New invalid pixels are never admitted. Optional final hole filling admits bright
enclosed valid pixels while respecting the same spatial limits. `admissible`
exposes the opened intensity mask before per-side spatial constraints.

The threshold-growth diagnostic plot shows the candidates, the two seeds and chest centre,
the grown labels, and their outlines on the X-ray. All computation remains at
the original resolution; only plot previews are reduced. Increase
`intensity_threshold` to include brighter tissue or `max_distance` to reach
farther from a small seed. `None` disables the distance limit, and zero disables
the connection opening or boundary margin. These controls can also allow
leakage into surrounding tissue. Complete anatomical lung coverage is not
guaranteed by intensity-based growth. Manual annotations are used only for
external comparisons, never by the algorithm.

## Growth tuning

`python scripts/tune_growth.py --phase all` runs a reproducible search using
20 tuning patients and 5 separate evaluation patients. It screens thousands of
configurations at reduced resolution, then selects the winner using full-resolution
Dice on the tuning set. The evaluation scores never select growth parameters.
Use `--phase prepare`, `search`, `native`, or `evaluate` to resume a stage.
The preprocessing cache is under `data/.cache/growth_tuning/` (ignored by Git).

The previous selected settings are saved in `results/growth_tuning/best_config.json`;
library defaults remain available for comparison. See the [tuning report](results/growth_tuning/report.md)
for scores, search ranges, split details, and limitations, and
[patient scores](results/growth_tuning/patient_scores.csv) for individual results.
The seed-recovery change was prompted by a missing side in an evaluation image,
without using its manual annotation or Dice. Thus evaluation is held out from
growth-parameter selection, but was not untouched during seed-development work.

This run tested **2,570 distinct configurations**. Mean full-resolution Dice:

| Split | Baseline | Tuned |
|---|---:|---:|
| Tuning (20) | 0.8152 | 0.8839 |
| Evaluation (5) | 0.7747 | 0.8359 |

Both comparisons use the improved seeds. Every evaluation patient improved,
but the recovered `MCUCXR_0309_1` remains difficult (Dice 0.6883). These are the
best settings found in this search; five evaluation cases give a limited estimate.

## Additional blur and alternative growing methods

`GrowthConfig(blur_sigma=60)` applies an additional normalized Gaussian blur to
the seed-preprocessing intensities before threshold growth. Invalid pixels stay
invalid and never darken the intensity average; seed selection is unchanged.
`blur_sigma=0` disables the extra blur. `growth.growth_image` exposes the exact
intensity field used for growth, and `plot_lung_growth(..., show_growth_image=True)`
displays it. Sigma is in original pixels, additional to the initial sigma 30.

Testing extra sigma 0/60/120/180/240 showed that stronger blur alone does not solve
leakage. Lowering the threshold helps more; extra blur weakens useful boundaries.
See the [blur experiment](results/growth_blur/report.md), reproduced with
`python scripts/tune_growth_blur.py --phase all`.

**Marker-controlled watershed is now recommended.** Both lungs and automatic
background markers compete across image gradients, so a dark connection does not
automatically admit the whole shoulder. `competitive_growing.py` also implements
random walker for comparison. Background markers use only image intensities,
valid-image boundaries and the chest centre; no manual annotations are used.

```python
from mtgp import CompetitiveGrowthConfig, grow_competitive
from mtgp.plotting import plot_competitive_growth

growth = grow_competitive(result, CompetitiveGrowthConfig(
    method="watershed", working_max_size=384,
    background_threshold=190, boundary_margin=100, blur_sigma=0,
))
lung_mask = growth.mask  # Original image shape; the original seeds are retained.
fig = plot_competitive_growth(patient.image, growth)
```

The working grid has a longest side of at most 384 pixels. Masks are restored by
nearest-neighbour sampling, and only valid additions connected to each original
seed are retained. This grid is part of inference, not just plot reduction.
`growth.markers`, `working_image`, and `stride` expose that computation; returned
lung masks have the original image dimensions. `beta` affects random walker only.

| Method | Tuning Dice (20) | Evaluation Dice (5) |
|---|---:|---:|
| Previous tuned threshold growth | 0.8839 | 0.8359 |
| Lower cutoff + extra blur | 0.8754 | 0.8242 |
| Random walker | 0.8123 | 0.7840 |
| **Watershed** | **0.9284** | **0.8928** |

Watershed reduces evaluation false-positive/reference area by about 83%, with
higher Dice on four of five patients. Some lung bases remain under-segmented.
All final scores use restored masks against full-resolution manual masks.
These five cases were inspected in earlier experiments, so they are a repeated
comparison rather than a fresh untouched test set. Settings were selected on the
20 tuning patients. See the [method assessment](results/growth_methods/report.md)
and run `python scripts/compare_growth_methods.py --phase all` to reproduce it.

The notebook preserves the existing sigma-5 / first-40 baseline experiment and
adds separate blur and recommended watershed sections. The new sections load
the saved sigma-30 preprocessing so their settings match the measured experiment.
The watershed section inspects the first 40 patients. Restart the kernel
after updating the source to load the new functions.

The final **Dice score over all images** section uses `mtgp.evaluation.evaluate_lung_dice`
to score every patient in `trial` with the saved watershed settings. It reports
mean per-image and pooled-pixel Dice, keeps segmentation failures at Dice zero,
and lists the loader's excluded patients. Per-image scores and the configuration
summary are saved to `results/all_images_dice/`. This includes tuning patients,
so the result describes the available dataset rather than held-out performance.

### Local dataset exception

`MCUCXR_0354_1` has a 1024 × 1024 image but 4020 × 4892 masks.
The trial loader excludes this patient and prints a skip message each time it
is encountered. Its sidecars are not counted as orphan files. The excluded ID
is recorded in `trial.excluded_patient_ids`. For any nonstandard
image dimensions, `pixel_spacing_mm` and `field_of_view_mm` are `None` because
the original-image spacing cannot be assumed. Resolve the image/mask geometry
before using this record for segmentation. Inspect `trial.data`'s `issues`
column for all detected exceptions.
