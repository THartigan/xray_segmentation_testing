"""Dataset scoring must retain failed patients and reject invalid geometry."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mtgp.evaluation import evaluate_lung_dice


class EvaluationTests(unittest.TestCase):
    def patient(self, pid='example'):
        return SimpleNamespace(patient_id=pid, image=np.zeros((2, 2)),
                               lung_mask=np.array([[True, True], [False, False]]))

    def test_scores_and_retains_failed_case_in_denominator(self):
        growth = SimpleNamespace(mask=np.array([[True, False], [True, False]]))
        progress = []
        with patch('mtgp.evaluation.segment_lungs', return_value=object()), patch(
            'mtgp.evaluation.grow_competitive', side_effect=[growth, ValueError('missing seed')],
        ):
            table = evaluate_lung_dice([self.patient('one'), self.patient('two')],
                                      progress=lambda done,total: progress.append((done,total)))
        self.assertEqual(table['dice'].tolist(), [.5, 0])
        self.assertEqual(table['dice'].mean(), .25)
        self.assertEqual(table['reference_pixels'].tolist(), [2, 2])
        self.assertEqual(table['prediction_pixels'].tolist(), [2, 0])
        self.assertEqual(table['intersection_pixels'].tolist(), [1, 0])
        self.assertEqual(table.iloc[1]['status'], 'segmentation_failed')
        self.assertEqual(table.iloc[1]['error'], 'missing seed')
        self.assertEqual(progress, [(1,2),(2,2)])

    def test_bad_reference_or_empty_input_does_not_silently_disappear(self):
        with self.assertRaises(ValueError):
            evaluate_lung_dice([])
        patient = self.patient()
        patient.lung_mask = np.zeros((3,3), bool)
        with self.assertRaisesRegex(ValueError, 'mismatched reference'):
            evaluate_lung_dice([patient])


if __name__ == '__main__':
    unittest.main()
