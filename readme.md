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

### Local dataset exception

`MCUCXR_0354_1` has a 1024 × 1024 image but 4020 × 4892 masks.
The trial loader excludes this patient and prints a skip message each time it
is encountered. Its sidecars are not counted as orphan files. The excluded ID
is recorded in `trial.excluded_patient_ids`. For any nonstandard
image dimensions, `pixel_spacing_mm` and `field_of_view_mm` are `None` because
the original-image spacing cannot be assumed. Resolve the image/mask geometry
before using this record for segmentation. Inspect `trial.data`'s `issues`
column for all detected exceptions.
