"""Check that notebook diagnostics reflect the actual configuration."""
from types import SimpleNamespace
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mtgp import GrowthConfig, SegmentationConfig, grow_lungs, segment_lungs, grow_competitive
from mtgp.plotting import plot_lung_growth, plot_segmentation_batch, plot_competitive_growth


class PlottingTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_grid_hides_empty_panels_and_uses_actual_threshold(self):
        patients = [SimpleNamespace(patient_id=str(i), image=np.full((40, 50), 50))
                    for i in range(3)]
        figures = plot_segmentation_batch(
            patients, SegmentationConfig(sigma=0, threshold=61, opening_radius=0, closing_radius=0),
            columns=2, stages=('histogram', 'cleaned'), display_max_size=10,
        )
        self.assertFalse(figures['cleaned'].axes[-1].get_visible())
        self.assertEqual(figures['cleaned'].axes[0].get_title(), '0')
        self.assertLessEqual(max(figures['cleaned'].axes[0].images[0].get_array().shape), 10)
        self.assertEqual(list(figures['histogram'].axes[0].lines[0].get_xdata()), [61, 61])

    def test_invalid_plot_request_does_not_leave_figures_open(self):
        patient = SimpleNamespace(patient_id='bad', image=np.zeros((2, 2, 3)))
        with self.assertRaises(ValueError):
            plot_segmentation_batch([patient], stages=('cleaned',))
        self.assertEqual(plt.get_fignums(), [])
        with self.assertRaises(ValueError):
            plot_segmentation_batch([], stages=('cleaned',))

    def test_growth_plot_shows_the_same_seed_centre_and_labels(self):
        image = np.full((50, 80), 150, dtype=np.float32)
        image[15:35, 10:25] = 40
        image[15:35, 55:70] = 40
        segmentation = segment_lungs(image, SegmentationConfig(
            sigma=0, opening_radius=0, closing_radius=0,
        ))
        growth = grow_lungs(segmentation, GrowthConfig(
            connection_radius=0, boundary_margin=0,
        ))
        fig = plot_lung_growth(image, segmentation, growth, display_max_size=40)
        self.assertEqual(len(fig.axes), 4)
        cy, cx = growth.seeds.chest_center
        cross = fig.axes[1].lines[0]
        self.assertEqual(list(cross.get_xdata()), [cx])
        self.assertEqual(list(cross.get_ydata()), [cy])
        np.testing.assert_array_equal(
            fig.axes[2].images[0].get_array(), growth.labels[::2, ::2],
        )
        detailed = plot_lung_growth(image, segmentation, growth, show_growth_image=True)
        self.assertEqual(len(detailed.axes), 5)
        np.testing.assert_array_equal(detailed.axes[2].images[0].get_array(), growth.growth_image)
        with self.assertRaises(ValueError):
            plot_lung_growth(image[:2], segmentation, growth)

        competing = grow_competitive(segmentation)
        marked = plot_competitive_growth(image, competing, display_max_size=40)
        self.assertEqual(len(marked.axes), 4)
        np.testing.assert_array_equal(marked.axes[1].images[0].get_array(), competing.markers + 1)
        np.testing.assert_array_equal(marked.axes[2].images[0].get_array(), competing.labels[::2, ::2])


if __name__ == '__main__':
    unittest.main()
