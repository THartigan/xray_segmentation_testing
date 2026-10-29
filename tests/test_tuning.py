"""Search scoring must match inference and must not hide failed cases."""
from dataclasses import replace
import unittest

import numpy as np

from mtgp import GrowthConfig, SegmentationConfig, grow_lungs, segment_lungs
from mtgp.morphology import open_mask
from mtgp.region_growing import _grow_from_seeds, _growth_distances, select_lung_seeds, smooth_growth_image
from mtgp.tuning import (
    TuningCase, config_key, dice_score, evaluate_configs,
    sample_configs, scaled_config, score_case,
)


class TuningTests(unittest.TestCase):
    def setUp(self):
        image = np.full((80, 120), 170, dtype=np.float32)
        image[:, :3] = 0
        image[10:70, 10:50] = 80
        image[10:70, 70:110] = 80
        image[25:55, 20:40] = 40
        image[25:55, 80:100] = 40
        self.seg = segment_lungs(image, SegmentationConfig(
            sigma=0, opening_radius=0, closing_radius=0,
        ))
        self.seeds = select_lung_seeds(self.seg.cleaned, self.seg.valid)
        self.case = TuningCase('synthetic', self.seg.smoothed, self.seg.valid,
                               self.seeds, (image < 100) & (image > 0))
        self.config = GrowthConfig(connection_radius=2, boundary_margin=3, max_distance=25)

    def test_dice_empty_masks_and_shape_validation(self):
        empty = np.zeros((3, 3), dtype=bool)
        self.assertEqual(dice_score(empty, empty), 1)
        self.assertEqual(dice_score(empty, ~empty), 0)
        self.assertAlmostEqual(dice_score(np.array([1, 1, 0]), np.array([0, 1, 1])), .5)
        with self.assertRaises(ValueError):
            dice_score(empty, np.zeros((2, 2)))

    def test_cached_and_uncached_growth_are_identical(self):
        distances = _growth_distances(self.seg.valid, self.seeds)
        for config in (self.config, replace(self.config, fill_holes=False),
                       replace(self.config, blur_sigma=3),
                       replace(self.config, boundary_margin=0, max_distance=None)):
            expected = grow_lungs(self.seg, config)
            growth_image = smooth_growth_image(self.seg.smoothed, self.seg.valid, config.blur_sigma)
            admitted = open_mask(self.seg.valid & (growth_image <= config.intensity_threshold),
                                 radius=config.connection_radius)
            actual = _grow_from_seeds(self.seg.smoothed, self.seg.valid, self.seeds,
                                      config, distances, admitted, growth_image)
            np.testing.assert_array_equal(actual.labels, expected.labels)
            self.case.distances = distances
            self.assertEqual(score_case(self.case, config), dice_score(expected.mask, self.case.reference))

    def test_failed_seed_pair_counts_in_macro_mean(self):
        failed = replace(self.case, patient_id='missing_seed', seed_error='no right seed')
        valid_score = score_case(self.case, self.config)
        rows = evaluate_configs([self.case, failed], [self.config], workers=1)
        self.assertEqual(rows[0]['per_patient']['missing_seed'], 0)
        self.assertAlmostEqual(rows[0]['mean_dice'], valid_score / 2)
        with self.assertRaises(ValueError):
            score_case(self.case, replace(self.config, min_seed_area=100))

    def test_scaling_preserves_intensity_and_physical_distance(self):
        config = replace(self.config, blur_sigma=120)
        scaled = scaled_config(config, 8)
        self.assertEqual(scaled.blur_sigma * 8, config.blur_sigma)
        self.assertEqual(scaled.intensity_threshold, self.config.intensity_threshold)
        self.assertEqual(scaled.connection_radius * 8, self.config.connection_radius)
        self.assertEqual(scaled.max_distance * 8, self.config.max_distance)
        self.assertIsNone(scaled_config(replace(self.config, max_distance=None), 8).max_distance)
        with self.assertRaises(ValueError):
            scaled_config(self.config, 0)

    def test_search_candidates_are_deterministic_and_include_baseline(self):
        first = sample_configs(power=4, seed=42)
        second = sample_configs(power=4, seed=42)
        self.assertEqual(first, second)
        self.assertIn(GrowthConfig(), first)
        self.assertEqual(len(first), len({config_key(c) for c in first}))


if __name__ == '__main__':
    unittest.main()
