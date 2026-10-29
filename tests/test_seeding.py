"""Recover a missing lung interior without replacing good existing seeds."""
import unittest

import numpy as np
from skimage.measure import label

from mtgp import (
    GrowthConfig, SegmentationConfig, find_lung_seeds, grow_lungs,
    segment_lungs, select_lung_seeds,
)


class SeedRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.config = SegmentationConfig(sigma=0, opening_radius=0, closing_radius=0)
        self.image = np.full((100, 140), 170, dtype=np.float32)
        self.image[30:70, 20:45] = 40
        self.image[30:70, 95:120] = 40

    def test_existing_pair_is_unchanged(self):
        segmentation = segment_lungs(self.image, self.config)
        strict = select_lung_seeds(segmentation.cleaned, segmentation.valid)
        recovered = find_lung_seeds(segmentation)
        np.testing.assert_array_equal(recovered.labels, strict.labels)
        self.assertEqual(recovered.sources, ('cleaned', 'cleaned'))

    def test_darker_interior_recovers_lung_joined_to_top_border(self):
        self.image[:35, 105:110] = 60  # A brighter connection to the image edge.
        segmentation = segment_lungs(self.image, self.config)
        with self.assertRaisesRegex(ValueError, 'image-right'):
            select_lung_seeds(segmentation.cleaned, segmentation.valid)
        recovered = find_lung_seeds(segmentation)
        self.assertEqual(recovered.labels[50, 105], 2)
        np.testing.assert_array_equal(recovered.labels == 1, segmentation.cleaned)
        self.assertEqual(recovered.sources[0], 'cleaned')
        self.assertIn('threshold=55', recovered.sources[1])
        self.assertFalse(recovered.labels[0].any())
        self.assertEqual(label(recovered.labels == 2, connectivity=2).max(), 1)
        self.assertFalse(np.any((recovered.labels > 0) & ~segmentation.valid))

    def test_blank_image_does_not_get_fabricated_seeds(self):
        blank = segment_lungs(np.full((30, 40), 170), self.config)
        with self.assertRaisesRegex(ValueError, 'after image-based recovery'):
            find_lung_seeds(blank)

    def test_both_missing_sides_can_be_recovered(self):
        self.image[:35, 30:35] = 60
        self.image[:35, 105:110] = 60
        segmentation = segment_lungs(self.image, self.config)
        self.assertFalse(segmentation.cleaned.any())
        recovered = find_lung_seeds(segmentation)
        self.assertEqual(recovered.labels[50, 30], 1)
        self.assertEqual(recovered.labels[50, 105], 2)
        self.assertTrue(all(source.startswith('fallback:') for source in recovered.sources))
        self.assertFalse(recovered.labels[0].any())

    def test_public_grower_uses_recovered_pair(self):
        self.image[:35, 105:110] = 60
        segmentation = segment_lungs(self.image, self.config)
        growth = grow_lungs(segmentation, GrowthConfig(
            connection_radius=0, boundary_margin=0, max_distance=0,
        ))
        self.assertTrue(growth.left_mask.any())
        self.assertTrue(growth.right_mask.any())
        np.testing.assert_array_equal(growth.labels, growth.seeds.labels)
        self.assertIn('threshold=55', growth.seeds.sources[1])


if __name__ == '__main__':
    unittest.main()
