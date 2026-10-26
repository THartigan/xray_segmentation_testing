"""Compact, memory-bounded diagnostic plots for threshold segmentation."""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .thresholding import SegmentationConfig, segment_lungs

if TYPE_CHECKING:
    from .data_loader import Patient

Stage = Literal["histogram", "smoothed", "candidate", "opened", "repaired", "cleaned"]
_STAGE_TITLES = {
    "histogram": "Smoothed intensity histogram",
    "smoothed": "Smoothed image",
    "candidate": "Threshold mask",
    "opened": "After separating thin connections",
    "repaired": "After individual closing and filling",
    "cleaned": "After boundary removal",
}


def plot_segmentation_batch(
    patients: Iterable[Patient],
    config: SegmentationConfig | None = None,
    *,
    stages: Iterable[Stage] = ("histogram", "smoothed", "candidate", "opened", "repaired", "cleaned"),
    columns: int = 3,
    display_max_size: int = 600,
) -> dict[str, Figure]:
    """Return one patient grid per stage; the caller controls show/save/close.

    Process one image at a time. Display arrays are copied at reduced resolution
    so Matplotlib does not retain every full-resolution intermediate. Histograms
    use all finite smoothed pixels, and segmentation always uses the full image.
    Unused axes are hidden; patient IDs identify every panel.
    """
    patients, stages = list(patients), tuple(stages)
    if not patients:
        raise ValueError("patients must not be empty")
    if not stages or len(set(stages)) != len(stages) or any(s not in _STAGE_TITLES for s in stages):
        raise ValueError(f"stages must be unique names from {tuple(_STAGE_TITLES)}")
    for name, value in (("columns", columns), ("display_max_size", display_max_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    config = config if config is not None else SegmentationConfig()
    columns = min(columns, len(patients))
    rows = (len(patients) + columns - 1) // columns
    figures, grids = {}, {}
    cmap = plt.colormaps["gray"].with_extremes(bad="red")

    try:
        for stage in stages:
            fig, axes = plt.subplots(
                rows, columns, figsize=(4 * columns, 3.8 * rows),
                squeeze=False, layout="constrained",
            )
            fig.suptitle(_STAGE_TITLES[stage])
            figures[stage], grids[stage] = fig, axes
            for ax in axes.flat[len(patients):]:
                ax.set_visible(False)

        for i, patient in enumerate(patients):
            result = segment_lungs(patient.image, config)
            step = max(1, int(np.ceil(max(result.valid.shape) / display_max_size)))
            sample = np.s_[::step, ::step]
            height, width = result.valid.shape
            for stage in stages:
                ax = grids[stage].flat[i]
                ax.set_title(patient.patient_id, fontsize=9)
                if stage == "histogram":
                    values = result.smoothed[np.isfinite(result.smoothed)]
                    ax.hist(values, bins=256)
                    ax.axvline(config.threshold, color="red", linestyle="--")
                    ax.set_xlabel("Intensity")
                    ax.set_ylabel("Pixels")
                    del values
                else:
                    pixels = getattr(result, stage)[sample].astype(np.float32, copy=True)
                    valid = result.mask_valid if stage in ("repaired", "cleaned") else result.valid
                    pixels[~valid[sample]] = np.nan
                    limits = {} if stage == "smoothed" else {"vmin": 0, "vmax": 1}
                    ax.imshow(
                        pixels, cmap=cmap, interpolation="nearest",
                        extent=(-0.5, width - 0.5, height - 0.5, -0.5), **limits,
                    )
            del result
    except Exception:
        for fig in figures.values():
            plt.close(fig)
        raise
    return figures
