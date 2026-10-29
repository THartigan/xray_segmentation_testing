"""Full-resolution, per-patient evaluation of the fixed lung segmentation pipeline."""
from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from .competitive_growing import CompetitiveGrowthConfig, grow_competitive
from .data_loader import Patient
from .thresholding import SegmentationConfig, segment_lungs
from .tuning import dice_score


def evaluate_lung_dice(
    patients: Iterable[Patient],
    segmentation_config: SegmentationConfig | None = None,
    growth_config: CompetitiveGrowthConfig | None = None,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Score the union of both predicted lungs against each native reference mask.

    Process one image at a time without retaining pixel arrays. Segmentation
    ValueErrors (including missing seeds) remain in the table with Dice zero;
    data-loading errors or mismatched reference geometry raise rather than being
    silently skipped. Manual masks are passed only to scoring, never inference.
    Count columns allow pooled-pixel Dice as well as mean per-patient Dice.
    """
    patients = list(patients)
    if not patients:
        raise ValueError('patients must not be empty')
    segmentation_config = segmentation_config or SegmentationConfig()
    growth_config = growth_config or CompetitiveGrowthConfig()
    rows = []
    for index, patient in enumerate(patients, 1):
        image = patient.image
        reference = patient.lung_mask
        if reference is None or reference.shape != image.shape:
            raise ValueError(f'{patient.patient_id}: missing or mismatched reference mask')
        row = dict(patient_id=patient.patient_id, status='ok', error='', dice=0.0,
                   intersection_pixels=0, prediction_pixels=0,
                   reference_pixels=int(np.count_nonzero(reference)))
        segmentation = None
        try:
            segmentation = segment_lungs(image, segmentation_config)
            growth = grow_competitive(segmentation, growth_config)
        except ValueError as exc:
            row.update(status='segmentation_failed', error=str(exc))
        else:
            prediction = growth.mask
            row.update(dice=dice_score(prediction, reference),
                       intersection_pixels=int(np.count_nonzero(prediction & reference)),
                       prediction_pixels=int(np.count_nonzero(prediction)))
            del prediction, growth
        rows.append(row)
        # Do not retain the preceding patient's intermediate arrays during the next run.
        del segmentation, image, reference
        if progress is not None:
            progress(index, len(patients))
    return pd.DataFrame.from_records(rows)
