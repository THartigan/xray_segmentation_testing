"""Regression checks for gaps, border tethers, and invalid-pixel semantics."""
import unittest

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage import measure

from mtgp import SegmentationConfig, segment_lungs
from mtgp.morphology import open_mask, repair_components, remove_boundary_components
from mtgp.thresholding import smooth_valid_image


class SegmentationTests(unittest.TestCase):
    def test_smoothing_does_not_darken_valid_pixels(self):
        image = np.full((31, 31), 50, dtype=np.float32)
        image[:, :9] = 0
        image[12, 12] = np.nan
        image[13, 13] = np.inf
        original = image.copy()
        smoothed, valid = smooth_valid_image(image, sigma=3)
        np.testing.assert_allclose(smoothed[valid], 50, atol=1e-5)
        self.assertTrue(np.isnan(smoothed[~valid]).all())
        np.testing.assert_array_equal(image, original)

    def test_opening_separates_border_tether_before_rejection(self):
        image = np.full((100, 140), 150, dtype=np.float32)
        image[:, :5] = 0
        image[:, 5:12] = 30  # Invalid-boundary strip.
        image[20:80, 40:85] = 30  # Lung blob.
        image[48:51, 10:42] = 30  # Thin foreground tether.
        image[40:45, 60:65] = np.nan  # Enclosed invalid hole.
        result = segment_lungs(image, SegmentationConfig(
            sigma=0, opening_radius=4, closing_radius=3, max_invalid_contacts=0,
        ))
        self.assertTrue(result.cleaned[35, 60])
        self.assertTrue(result.cleaned[42, 62])
        self.assertFalse(result.cleaned[:, :20].any())
        self.assertFalse(result.cleaned[49, 25])
        np.testing.assert_array_equal(result.cleaned, binary_fill_holes(result.cleaned))
        no_opening = segment_lungs(image, SegmentationConfig(
            sigma=0, opening_radius=0, closing_radius=3, max_invalid_contacts=0,
        ))
        self.assertFalse(no_opening.cleaned.any())

    def test_close_open_cleft_and_fill_enclosed_invalid_hole(self):
        valid = np.ones((60, 60), dtype=bool)
        mask = np.zeros_like(valid)
        mask[10:50, 10:50] = True
        mask[10:30, 28:31] = False  # Narrow cleft open to outside.
        mask[35:39, 35:39] = False
        valid[35:39, 35:39] = False
        valid[:, :3] = False
        repaired, mask_valid = repair_components(mask, valid, closing_radius=3)
        self.assertTrue(repaired[20, 29])
        self.assertTrue(repaired[36, 36])
        self.assertTrue(mask_valid[36, 36])
        self.assertFalse(mask_valid[:, :3].any())
        np.testing.assert_array_equal(repaired, binary_fill_holes(repaired))

    def test_closing_does_not_jointly_bridge_separate_blobs(self):
        mask = np.zeros((70, 90), dtype=bool)
        mask[15:55, 10:40] = True
        mask[15:55, 43:73] = True
        repaired, _ = repair_components(mask, np.ones_like(mask), closing_radius=8)
        self.assertEqual(measure.label(repaired, connectivity=2).max(), 2)
        self.assertFalse(repaired[:, 40:43].any())

    def test_invalid_channel_cannot_be_used_as_closing_bridge(self):
        mask = np.zeros((60, 60), dtype=bool)
        mask[10:50, 10:50] = True
        mask[10:30, 28:31] = False
        valid = np.ones_like(mask)
        valid[:30, 28:31] = False
        repaired, mask_valid = repair_components(mask, valid, closing_radius=5)
        self.assertFalse(repaired[20, 29])
        self.assertFalse(mask_valid[20, 29])

    def test_boundary_rejection_and_inclusive_contact_tolerance(self):
        mask = np.zeros((30, 40), dtype=bool)
        mask[:6, 2:8] = True  # Edge always rejected.
        mask[10:15, 10:15] = True
        valid = np.ones_like(mask)
        valid[9, 9] = False  # A single diagonal contact.
        rejected = remove_boundary_components(mask, valid, max_invalid_contacts=0)
        accepted = remove_boundary_components(mask, valid, max_invalid_contacts=1)
        self.assertFalse(rejected.any())
        self.assertEqual(accepted.sum(), 25)

    def test_zero_radii_and_empty_or_full_inputs(self):
        for value in (False, True):
            mask = np.full((20, 30), value)
            np.testing.assert_array_equal(open_mask(mask, radius=0), mask)
            repaired, _ = repair_components(mask, np.ones_like(mask), closing_radius=0)
            np.testing.assert_array_equal(repaired, mask)
        for value in (0, np.nan, np.inf, 150):
            result = segment_lungs(np.full((25, 25), value))
            self.assertFalse(result.cleaned.any())
        full = segment_lungs(np.full((25, 25), 30), SegmentationConfig(
            sigma=0, opening_radius=0, closing_radius=0,
        ))
        self.assertFalse(full.cleaned.any())

    def test_opening_cannot_create_foreground_or_mutate_input(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[2, 2] = True
        original = mask.copy()
        opened = open_mask(mask, radius=8)
        self.assertFalse(opened.any())
        np.testing.assert_array_equal(mask, original)

    def test_invalid_parameters_fail_early(self):
        for kwargs in ({'sigma': -1}, {'opening_radius': np.inf},
                       {'closing_radius': -1}, {'threshold': np.nan},
                       {'max_invalid_contacts': 1.5}, {'max_invalid_contacts': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SegmentationConfig(**kwargs)
        for image in (np.zeros((3, 3, 3)), np.zeros((0, 3))):
            with self.assertRaises(ValueError):
                segment_lungs(image)
        with self.assertRaises(ValueError):
            repair_components(np.zeros((2, 2), bool), np.ones((3, 3), bool), closing_radius=1)
        with self.assertRaises(ValueError):
            open_mask(np.zeros((3, 3)), radius=1)


if __name__ == '__main__':
    unittest.main()
