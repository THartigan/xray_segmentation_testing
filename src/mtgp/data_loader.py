"""Montgomery CXR records. Array axes and spacing are always (row, column)."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image

README_NAME = 'NLM-MontgomeryCXRSet-ReadMe.pdf'
NAME_PATTERN = re.compile(r'MCUCXR_(\d+)_([01])')


def _image_metadata(path: Path) -> dict:
    with Image.open(path) as image:
        metadata = dict(format=image.format, mode=image.mode,
                        matrix_size=(image.height, image.width),
                        info=dict(image.info), file_size_bytes=path.stat().st_size)
    with path.open('rb') as stream:
        header = stream.read(29)
    if header[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f'Expected PNG: {path}')
    metadata['bit_depth'] = header[24]
    metadata['color_type'] = header[25]
    return metadata


class Patient:
    """One CXR and its paired clinical text and anatomical lung masks.

    Arrays are read on access (not cached), preserving the stored image values.
    Missing sidecars raise by default; strict=False exposes them as None.
    The documented spacing is a dataset-level value, not a PNG measurement.
    """

    def __init__(self, image_path: str | Path, *, strict: bool = True):
        self.image_path = Path(image_path).expanduser().resolve()
        match = NAME_PATTERN.fullmatch(self.image_path.stem)
        if not match:
            raise ValueError(f'Unexpected Montgomery filename: {self.image_path.name}')
        self.patient_id = self.image_path.stem
        self.patient_num = int(match[1])
        self.label = int(match[2])
        self.classification = {0: 'normal', 1: 'abnormal'}[self.label]
        self.directory = self.image_path.parent.parent
        self.clinical_path = self.directory / 'ClinicalReadings' / f'{self.patient_id}.txt'
        self.left_mask_path = self.directory / 'ManualMask' / 'leftMask' / self.image_path.name
        self.right_mask_path = self.directory / 'ManualMask' / 'rightMask' / self.image_path.name
        sidecars = [self.clinical_path, self.left_mask_path, self.right_mask_path]
        self.missing_files = [p for p in sidecars if not p.is_file()]
        if strict and self.missing_files:
            raise FileNotFoundError(f'{self.patient_id}: missing {self.missing_files}')
        self.clinical_reading = (self.clinical_path.read_text(encoding='utf-8-sig')
                                 if self.clinical_path.is_file() else None)
        self.clinical_fields = {}
        findings = []
        for line in (self.clinical_reading or '').splitlines():
            match_field = re.match(r"Patient['’]s\s+(Sex|Age):\s*(.*)", line, re.I)
            if match_field:
                self.clinical_fields[match_field[1].lower()] = match_field[2].strip()
            elif line.strip():
                findings.append(line.strip())
        self.sex = self.clinical_fields.get('sex')
        self.age_raw = self.clinical_fields.get('age')
        age_match = re.fullmatch(r'(\d+)Y', self.age_raw or '', re.I)
        self.age_years = int(age_match[1]) if age_match else None
        self.findings = '\n'.join(findings) if self.clinical_reading is not None else None
        self.image_metadata = _image_metadata(self.image_path)
        self.matrix_size = self.image_metadata['matrix_size']
        self.mask_metadata = {}
        self.issues = []
        for side, path in [('left', self.left_mask_path), ('right', self.right_mask_path)]:
            meta = _image_metadata(path) if path.is_file() else None
            if meta is not None and meta['matrix_size'] != self.matrix_size:
                self.issues.append(f'{side} mask shape {meta["matrix_size"]} does not match image {self.matrix_size}')
            self.mask_metadata[side] = meta
        self.pixel_spacing_mm = ((0.0875, 0.0875) if self.matrix_size in
                                 {(4020, 4892), (4892, 4020)} else None)
        if self.pixel_spacing_mm is None:
            self.issues.append('Nonstandard image dimensions: physical pixel spacing unknown')
        self.pixel_spacing_source = f'{README_NAME}: dataset-level image parameters'
        self.documented_gray_level_bits = 12
        self.metadata = {
            'image': self.image_metadata, 'masks': self.mask_metadata,
            'pixel_spacing_mm': self.pixel_spacing_mm,
            'pixel_spacing_source': self.pixel_spacing_source,
            'documented_gray_level_bits': self.documented_gray_level_bits,
            'clinical_fields': self.clinical_fields,
            'issues': self.issues,
            'documentation_path': self.directory / README_NAME,
            'label_source': 'filename suffix: 0 normal, 1 abnormal',
        }

    @property
    def image(self) -> np.ndarray:
        """Original PNG pixels, without resizing or intensity normalization."""
        with Image.open(self.image_path) as image:
            return np.array(image)

    def _mask(self, path: Path) -> np.ndarray | None:
        if not path.is_file():
            return None
        with Image.open(path) as image:
            return np.array(image) != 0

    @property
    def left_mask(self) -> np.ndarray | None:
        """Boolean annotation from leftMask (patient's anatomical left)."""
        return self._mask(self.left_mask_path)

    @property
    def right_mask(self) -> np.ndarray | None:
        return self._mask(self.right_mask_path)

    @property
    def lung_mask(self) -> np.ndarray | None:
        left, right = self.left_mask, self.right_mask
        if left is None or right is None:
            return None
        if left.shape != right.shape:
            raise ValueError('Left and right masks have different shapes')
        return left | right

    @property
    def field_of_view_mm(self) -> tuple[float, float] | None:
        if self.pixel_spacing_mm is None:
            return None
        return tuple(n * s for n, s in zip(self.matrix_size, self.pixel_spacing_mm))

    def to_dict(self) -> dict:
        """Tabular metadata only; does not load any pixel arrays."""
        return dict(patient_id=self.patient_id, patient_num=self.patient_num,
                    label=self.label, classification=self.classification,
                    sex=self.sex, age_years=self.age_years, age_raw=self.age_raw,
                    findings=self.findings, clinical_reading=self.clinical_reading,
                    rows=self.matrix_size[0], columns=self.matrix_size[1],
                    matrix_size=self.matrix_size, pixel_spacing_mm=self.pixel_spacing_mm,
                    pixel_spacing_source=self.pixel_spacing_source,
                    field_of_view_mm=self.field_of_view_mm,
                    stored_bit_depth=self.image_metadata['bit_depth'],
                    documented_gray_level_bits=self.documented_gray_level_bits,
                    image_mode=self.image_metadata['mode'], image_path=self.image_path,
                    clinical_path=self.clinical_path, left_mask_path=self.left_mask_path,
                    right_mask_path=self.right_mask_path,
                    issues=list(self.issues),
                    masks_match_image=all(m is not None and m["matrix_size"] == self.matrix_size
                                          for m in self.mask_metadata.values()),
                    has_clinical=self.clinical_path.is_file(),
                    has_left_mask=self.left_mask_path.is_file(),
                    has_right_mask=self.right_mask_path.is_file())

    def __repr__(self):
        return f'Patient({self.patient_id!r}, {self.classification}, shape={self.matrix_size})'


class Trial:
    """Patient collection with a metadata DataFrame and deterministic ID lookup.

    Accepts the MontgomerySet directory or its containing data directory.
    With no path, finds data/MontgomerySet relative to this editable package.
    No train/test split or image preprocessing is performed implicitly.
    """

    def __init__(self, directory: str | Path | None = None, *, strict: bool = True):
        root = (Path(directory).expanduser() if directory is not None else
                Path(__file__).resolve().parents[2] / 'data' / 'MontgomerySet')
        if not (root / 'CXR_png').is_dir() and (root / 'MontgomerySet').is_dir():
            root = root / 'MontgomerySet'
        self.directory = root.resolve()
        self.name = self.directory.name
        images = sorted((self.directory / 'CXR_png').glob('*.png'))
        if not images:
            raise FileNotFoundError(f'No PNG images in {self.directory / "CXR_png"}')
        self.excluded_patient_ids = [p.stem for p in images if p.stem == 'MCUCXR_0354_1']
        for patient_id in self.excluded_patient_ids:
            print(f'Skipping {patient_id}: image and lung mask dimensions do not match.')
        images = [p for p in images if p.stem not in self.excluded_patient_ids]
        self.patients = [Patient(path, strict=strict) for path in images]
        self.num_patients = len(self.patients)
        self._by_id = {p.patient_id: p for p in self.patients}
        numbers = [p.patient_num for p in self.patients]
        if len(set(numbers)) != len(numbers):
            raise ValueError('Duplicate patient numbers found; check image filenames')
        self.data = pd.DataFrame(p.to_dict() for p in self.patients)
        self.metadata = {'documentation_path': self.directory / README_NAME,
                         'pixel_spacing_mm': (0.0875, 0.0875),
                         'label_mapping': {0: 'normal', 1: 'abnormal'}}
        image_ids = set(self._by_id) | set(self.excluded_patient_ids)
        self.orphan_files = []
        for folder, pattern in [('ClinicalReadings', '*.txt'),
                                ('ManualMask/leftMask', '*.png'),
                                ('ManualMask/rightMask', '*.png')]:
            self.orphan_files.extend(sorted(p for p in (self.directory / folder).glob(pattern)
                                            if p.stem not in image_ids))

    def __len__(self):
        return self.num_patients

    def __iter__(self):
        return iter(self.patients)

    def __getitem__(self, key):
        """Integer/slice indexes collection order; string looks up full patient ID."""
        return self._by_id[key] if isinstance(key, str) else self.patients[key]

    def __repr__(self):
        return f'Trial({str(self.directory)!r}, patients={len(self)})'


# Familiar spelling from the original project's loader.
DataSet = Trial
Dataset = Trial
