"""Check that notebook diagnostics reflect the actual configuration."""
from types import SimpleNamespace
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mtgp import SegmentationConfig
from mtgp.plotting import plot_segmentation_batch


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


if __name__ == '__main__':
    unittest.main()
