"""Competing background markers must constrain growth without damaging seeds."""
from dataclasses import replace
import unittest

import numpy as np
from skimage.measure import label

from mtgp import CompetitiveGrowthConfig, SegmentationConfig, SegmentationResult, grow_competitive
from mtgp.competitive_growing import _restore_prediction
from mtgp.seeding import select_lung_seeds
from mtgp.tuning import dice_score


class CompetitiveGrowthTests(unittest.TestCase):
    def setUp(self):
        image = np.full((80, 120), 180, np.float32)
        image[10:70, 10:50] = 80
        image[10:70, 70:110] = 80
        image[:, :4] = np.nan
        valid = np.isfinite(image)
        seeds = np.zeros(image.shape, bool)
        seeds[30:50, 25:35] = True
        seeds[30:50, 85:95] = True
        self.seg = SegmentationResult(SegmentationConfig(), image, valid, seeds.copy(),
                                      seeds.copy(), seeds.copy(), valid.copy(), seeds.copy())
        self.config = CompetitiveGrowthConfig(working_max_size=80, background_threshold=160,
                                              boundary_margin=2, beta=500)

    def test_both_methods_keep_seeds_connected_and_respect_invalid_data(self):
        original = self.seg.smoothed.copy()
        for method in ('random_walker','watershed'):
            with self.subTest(method=method):
                result = grow_competitive(self.seg, replace(self.config, method=method))
                self.assertEqual(result.labels.shape, original.shape)
                self.assertEqual(result.stride, 2)
                self.assertLessEqual(max(result.working_image.shape), 80)
                self.assertTrue(result.mask[self.seg.cleaned].all())
                self.assertFalse(result.mask[~self.seg.valid].any())
                self.assertFalse(result.left_mask[:, 60:].any())
                self.assertFalse(result.right_mask[:, :60].any())
                self.assertEqual(label(result.left_mask, connectivity=2).max(), 1)
                self.assertEqual(label(result.right_mask, connectivity=2).max(), 1)
                self.assertTrue((result.markers == 3).any())
                self.assertTrue((result.markers == -1).any())
                self.assertGreater(int(result.mask.sum()), int(self.seg.cleaned.sum()))
                # Allow the one-pixel quantization from the stride-two grid.
                self.assertGreater(dice_score(result.mask, original == 80), .94)
        np.testing.assert_array_equal(original, self.seg.smoothed)

    def test_restore_preserves_filled_seed_holes_but_not_disconnected_islands(self):
        self.seg.smoothed[35,30] = np.nan
        self.seg.valid[35,30] = False
        seeds = select_lung_seeds(self.seg.cleaned, self.seg.valid)
        working = np.zeros(self.seg.valid.shape, np.uint8)
        working[28:52, 23:37] = 1
        working[2:5, 10:13] = 1  # Not connected to the seed.
        working[28:52, 83:97] = 2
        restored = _restore_prediction(working, self.seg.valid, seeds, 1)
        self.assertEqual(restored[35,30], 1)
        self.assertFalse(restored[2:5,10:13].any())
        self.assertTrue((restored[seeds.labels>0] == seeds.labels[seeds.labels>0]).all())

    def test_invalid_configuration_is_rejected(self):
        for kwargs in ({'method':'flood'}, {'working_max_size':0}, {'working_max_size':True},
                       {'beta':-1}, {'blur_sigma':np.nan}, {'background_threshold':np.inf},
                       {'boundary_margin':-1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CompetitiveGrowthConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
