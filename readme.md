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

Run loader tests with `python -m unittest discover -s tests`.

### Local dataset exception

`MCUCXR_0354_1` has a 1024 × 1024 image but 4020 × 4892 masks.
The loader preserves these arrays and flags the mismatch. For any nonstandard
image dimensions, `pixel_spacing_mm` and `field_of_view_mm` are `None` because
the original-image spacing cannot be assumed. Resolve the image/mask geometry
before using this record for segmentation. Inspect `trial.data`'s `issues`
column for all detected exceptions.
