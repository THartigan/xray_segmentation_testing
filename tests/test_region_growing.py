"""Geometric seed choice and constrained propagation regression tests."""
from dataclasses import replace
import unittest

import numpy as np
from skimage.measure import label

from mtgp import (
    GrowthConfig, SegmentationConfig, SegmentationResult,
    grow_lungs, select_lung_seeds,
)
from mtgp.region_growing import smooth_growth_image


def segmentation(image, cleaned):
    image = np.asarray(image, dtype=np.float32)
    valid = np.isfinite(image) & (image >= 10)
    return SegmentationResult(
        SegmentationConfig(), image, valid, cleaned.copy(), cleaned.copy(),
        cleaned.copy(), valid | cleaned, cleaned.copy(),
    )


class SeedSelectionTests(unittest.TestCase):
    def test_centre_uses_valid_chest_and_chooses_closest_not_largest(self):
        valid = np.zeros((100, 160), dtype=bool)
        valid[10:90, 60:160] = True  # Centre is (49.5, 109.5), not image centre.
        cleaned = np.zeros_like(valid)
        cleaned[20:70, 65:85] = True  # Large, farther image-left component.
        cleaned[45:55, 97:104] = True  # Closest image-left component.
        cleaned[45:55, 118:128] = True
        cleaned[15:30, 140:150] = True
        seeds = select_lung_seeds(cleaned, valid)
        self.assertEqual(seeds.chest_center, (49.5, 109.5))
        self.assertEqual(seeds.labels[49, 100], 1)
        self.assertEqual(seeds.labels[49, 120], 2)
        self.assertEqual(seeds.labels[49, 75], 0)
        self.assertEqual(seeds.labels.dtype, np.uint8)
        self.assertEqual(label(seeds.labels == 1, connectivity=2).max(), 1)
        self.assertEqual(label(seeds.labels == 2, connectivity=2).max(), 1)

    def test_missing_side_or_empty_chest_is_explicit(self):
        valid = np.ones((30, 50), dtype=bool)
        cleaned = np.zeros_like(valid)
        cleaned[10:20, 3:10] = True
        with self.assertRaisesRegex(ValueError, 'image-right'):
            select_lung_seeds(cleaned, valid)
        with self.assertRaisesRegex(ValueError, 'no valid pixels'):
            select_lung_seeds(cleaned, ~valid)
        with self.assertRaises(ValueError):
            select_lung_seeds(cleaned, valid, min_area=0)

    def test_min_area_and_tie_break_are_deterministic(self):
        valid = np.ones((51, 51), dtype=bool)
        cleaned = np.zeros_like(valid)
        cleaned[25, 20] = True  # Nearest tiny blob.
        cleaned[14:17, 14:17] = True
        cleaned[33:38, 13:18] = True  # Same centroid distance, larger area.
        cleaned[20:30, 35:45] = True
        seeds = select_lung_seeds(cleaned, valid, min_area=2)
        self.assertEqual(seeds.labels[35, 15], 1)
        self.assertEqual(seeds.labels[15, 15], 0)
        self.assertEqual(seeds.labels[25, 20], 0)


class RegionGrowingTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((80, 120), 200, dtype=np.float32)
        self.image[10:70, 10:50] = 80
        self.image[10:70, 70:110] = 80
        self.seeds = np.zeros_like(self.image, dtype=bool)
        self.seeds[30:50, 25:35] = True
        self.seeds[30:50, 85:95] = True
        self.config = GrowthConfig(
            intensity_threshold=100, connection_radius=0,
            max_distance=None, boundary_margin=0, fill_holes=False,
        )

    def test_expands_to_connected_regions_without_mutating_inputs(self):
        original = self.image.copy()
        result = segmentation(self.image, self.seeds)
        growth = grow_lungs(result, self.config)
        np.testing.assert_array_equal(growth.mask, self.image == 80)
        self.assertFalse((growth.left_mask & growth.right_mask).any())
        self.assertTrue(growth.mask[self.seeds].all())
        np.testing.assert_array_equal(self.image, original)
        np.testing.assert_array_equal(result.cleaned, self.seeds)
        self.assertEqual(label(growth.left_mask, connectivity=2).max(), 1)

    def test_unconnected_dark_island_is_not_added(self):
        self.image[2:6, 2:6] = 30
        growth = grow_lungs(segmentation(self.image, self.seeds), self.config)
        self.assertFalse(growth.mask[2:6, 2:6].any())

    def test_nan_wall_is_not_crossed(self):
        self.image[20, 10:50] = np.nan
        growth = grow_lungs(segmentation(self.image, self.seeds), self.config)
        self.assertFalse(growth.mask[10:20, 10:50].any())
        self.assertFalse(growth.mask[~np.isfinite(self.image)].any())
        self.assertTrue(growth.left_mask[25, 25])

    def test_distance_limit_is_measured_from_seed(self):
        self.image[:] = 80
        growth = grow_lungs(segmentation(self.image, self.seeds), replace(self.config, max_distance=2))
        self.assertTrue(growth.left_mask[28, 30])
        self.assertFalse(growth.left_mask[27, 30])
        self.assertTrue(growth.right_mask[51, 90])
        unchanged = grow_lungs(segmentation(self.image, self.seeds), replace(self.config, max_distance=0))
        np.testing.assert_array_equal(unchanged.mask, self.seeds)

    def test_midline_and_boundary_margin_constrain_growth(self):
        self.image[:] = 80
        growth = grow_lungs(segmentation(self.image, self.seeds), replace(self.config, boundary_margin=3))
        self.assertFalse(growth.left_mask[:, 60:].any())
        self.assertFalse(growth.right_mask[:, :60].any())
        self.assertFalse(growth.mask[:3].any())
        self.assertFalse(growth.mask[:, -3:].any())
        self.assertTrue(growth.mask[4, 4])

    def test_opening_admission_mask_blocks_thin_border_tether(self):
        self.image[10:70, 1:5] = 80
        self.image[39:41, 4:12] = 80
        result = segmentation(self.image, self.seeds)
        unguarded = grow_lungs(result, self.config)
        guarded = grow_lungs(result, replace(self.config, connection_radius=3))
        self.assertTrue(unguarded.mask[25, 2])
        self.assertFalse(guarded.mask[25, 2])
        self.assertTrue(guarded.mask[25, 25])

    def test_hole_filling_respects_invalid_pixels(self):
        self.image[20:23, 20:23] = 200
        self.image[57:60, 20:23] = np.nan
        result = segmentation(self.image, self.seeds)
        hollow = grow_lungs(result, self.config)
        filled = grow_lungs(result, replace(self.config, fill_holes=True))
        self.assertFalse(hollow.mask[21, 21])
        self.assertTrue(filled.mask[21, 21])
        self.assertFalse(filled.mask[58, 21])

    def test_existing_seed_holes_and_bright_seed_pixels_are_preserved(self):
        self.image[35, 30] = np.nan
        self.image[36, 30] = 250
        growth = grow_lungs(segmentation(self.image, self.seeds), self.config)
        self.assertTrue(growth.left_mask[35, 30])
        self.assertTrue(growth.left_mask[36, 30])
        self.assertTrue(growth.mask_valid[35, 30])
        self.assertTrue(growth.mask[self.seeds].all())

    def test_invalid_configuration(self):
        for kwargs in ({'intensity_threshold': np.nan}, {'connection_radius': -1},
                       {'blur_sigma': -1}, {'blur_sigma': np.inf},
                       {'max_distance': np.inf}, {'boundary_margin': -1},
                       {'min_seed_area': 0}, {'min_seed_area': 1.5}, {'fill_holes': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GrowthConfig(**kwargs)

    def test_blur_preserves_constant_intensity_next_to_missing_data(self):
        image = np.full((40, 50), 80, dtype=np.float32)
        valid = np.ones(image.shape, dtype=bool)
        valid[:, :15] = False
        image[:, :15] = np.nan
        image[:, :2] = 0
        before = image.copy()
        blurred = smooth_growth_image(image, valid, sigma=12)
        np.testing.assert_allclose(blurred[valid], 80, atol=2e-5)
        self.assertTrue(np.isnan(blurred[~valid]).all())
        np.testing.assert_array_equal(image, before)

    def test_stronger_growth_blur_reduces_texture_without_changing_seeds(self):
        texture = np.indices(self.image.shape).sum(axis=0) % 2
        self.image += (texture * 20 - 10).astype(np.float32)
        result = segmentation(self.image, self.seeds)
        original = grow_lungs(result, self.config)
        blurred = grow_lungs(result, replace(self.config, blur_sigma=3))
        np.testing.assert_array_equal(blurred.seeds.labels, original.seeds.labels)
        interior = np.s_[25:55, 20:40]
        self.assertLess(blurred.growth_image[interior].std(), original.growth_image[interior].std() / 2)
        self.assertTrue(blurred.mask[self.seeds].all())
        np.testing.assert_array_equal(original.growth_image, result.smoothed)


if __name__ == '__main__':
    unittest.main()
