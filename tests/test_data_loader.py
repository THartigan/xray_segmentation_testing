import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from mtgp import Patient, Trial


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'MontgomerySet'
        for folder in ['CXR_png', 'ClinicalReadings', 'ManualMask/leftMask', 'ManualMask/rightMask']:
            (self.root / folder).mkdir(parents=True)
        self.name = 'MCUCXR_0001_0'
        self.image = np.arange(12, dtype=np.uint8).reshape(3, 4)
        self.path = self.root / 'CXR_png' / f'{self.name}.png'
        Image.fromarray(self.image).save(self.path)
        left = np.zeros((3, 4), dtype=np.uint8)
        left[:, 0] = 255
        right = np.zeros((3, 4), dtype=np.uint8)
        right[:, -1] = 255
        for side, mask in [('left', left), ('right', right)]:
            Image.fromarray(mask).save(self.root / 'ManualMask' / f'{side}Mask' / f'{self.name}.png')
        (self.root / 'ClinicalReadings' / f'{self.name}.txt').write_text(
            "Patient's Sex: F \nPatient's Age: 027Y\nnormal\nsecond line\n")

    def test_pairing_and_pixels(self):
        trial = Trial(self.root.parent)
        p = trial[self.name]
        self.assertIs(trial[0], p)
        self.assertEqual(len(trial), 1)
        self.assertEqual((p.age_years, p.sex, p.label), (27, 'F', 0))
        self.assertEqual(p.findings, 'normal\nsecond line')
        self.assertEqual(p.matrix_size, (3, 4))
        self.assertIsNone(p.pixel_spacing_mm)
        self.assertIsNone(p.field_of_view_mm)
        self.assertEqual(p.image_metadata['bit_depth'], 8)
        np.testing.assert_array_equal(p.image, self.image)
        self.assertEqual(p.left_mask.dtype, np.dtype(bool))
        self.assertEqual(p.lung_mask.sum(), 6)
        self.assertEqual(trial.data.loc[0, 'columns'], 4)

    def test_missing_sidecar(self):
        path = self.root / 'ManualMask/leftMask' / f'{self.name}.png'
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            Trial(self.root)
        p = Trial(self.root, strict=False)[0]
        self.assertIsNone(p.left_mask)
        self.assertIsNone(p.lung_mask)

    def test_bad_mask_shape(self):
        Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(
            self.root / 'ManualMask/rightMask' / f'{self.name}.png')
        trial = Trial(self.root)
        self.assertFalse(trial.data.loc[0, 'masks_match_image'])
        self.assertTrue(any('shape' in issue for issue in trial[0].issues))
        with self.assertRaisesRegex(ValueError, 'different shapes'):
            _ = trial[0].lung_mask

    def test_missing_clinical_and_orphan(self):
        (self.root / 'ClinicalReadings' / f'{self.name}.txt').unlink()
        (self.root / 'ClinicalReadings/orphan.txt').write_text('orphan')
        trial = Trial(self.root, strict=False)
        self.assertIsNone(trial[0].age_years)
        self.assertIsNone(trial[0].clinical_reading)
        self.assertEqual(len(trial.orphan_files), 1)

    def test_unknown_age_and_invalid_name(self):
        (self.root / 'ClinicalReadings' / f'{self.name}.txt').write_text("Patient's Age: unknown\nnormal")
        self.assertIsNone(Patient(self.path).age_years)
        with self.assertRaises(ValueError):
            Patient(self.path.with_name('bad.png'))


if __name__ == '__main__':
    unittest.main()
