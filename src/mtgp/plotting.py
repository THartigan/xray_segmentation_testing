"""Compact, memory-bounded diagnostic plots for threshold segmentation."""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike

from .thresholding import SegmentationConfig, SegmentationResult, segment_lungs
from .region_growing import LungGrowthResult
from .competitive_growing import CompetitiveGrowthResult

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


def plot_lung_growth(
    image: ArrayLike,
    segmentation: SegmentationResult,
    growth: LungGrowthResult,
    *,
    title: str = "Seeded lung growth",
    display_max_size: int = 800,
    show_growth_image: bool = False,
) -> Figure:
    """Show candidate blobs, chosen seeds, grown labels and an X-ray overlay.

    Cyan is image-left and orange is image-right. The cross and dashed vertical
    line show the centroid of valid chest pixels. Only plot data is downsampled.
    """
    image = np.asarray(image)
    if image.ndim != 2 or image.shape != growth.labels.shape or image.shape != segmentation.valid.shape:
        raise ValueError("image, segmentation and growth must have matching 2-D shapes")
    if not isinstance(display_max_size, int) or isinstance(display_max_size, bool) or display_max_size < 1:
        raise ValueError("display_max_size must be a positive integer")
    step = max(1, int(np.ceil(max(image.shape) / display_max_size)))
    sample = np.s_[::step, ::step]
    height, width = image.shape
    extent = (-0.5, width - 0.5, height - 0.5, -0.5)
    gray = plt.colormaps["gray"].with_extremes(bad="red")
    colors = ("#24cce0", "#ffad42")
    label_cmap = ListedColormap(["black", *colors]).with_extremes(bad="red")
    count = 5 if show_growth_image else 4
    fig, axes = plt.subplots(1, count, figsize=(4 * count, 5), layout="constrained")
    fig.suptitle(title)
    panels = [
        (segmentation.cleaned, segmentation.mask_valid, gray, 1, "Candidate blobs"),
        (growth.seeds.labels, growth.mask_valid, label_cmap, 2, "Selected seeds"),
        (growth.labels, growth.mask_valid, label_cmap, 2, "Grown regions"),
    ]
    if show_growth_image:
        panels.insert(2, (growth.growth_image, segmentation.valid, gray, None,
                          f"Growth image (extra σ={growth.config.blur_sigma:g})"))
    for ax, (pixels, valid, cmap, vmax, panel_title) in zip(axes, panels):
        display = np.where(valid[sample], pixels[sample], np.nan).astype(np.float32)
        ax.imshow(display, cmap=cmap, vmin=0, vmax=vmax, extent=extent, interpolation="nearest")
        ax.set_title(panel_title)
    cy, cx = growth.seeds.chest_center
    axes[1].plot(cx, cy, marker="+", color="magenta", markersize=12)
    axes[1].axvline(cx, color="magenta", linestyle="--", linewidth=0.8)
    axes[-1].imshow(image[sample], cmap="gray", extent=extent, interpolation="nearest")
    rows, columns = np.arange(height)[::step], np.arange(width)[::step]
    for side, color in enumerate(colors, start=1):
        mask = growth.labels[sample] == side
        if min(mask.shape) >= 2 and mask.any() and not mask.all():
            axes[-1].contour(columns, rows, mask, levels=[0.5], colors=[color], linewidths=1)
    axes[-1].set_title("Grown contours on X-ray")
    axes[-1].legend(
        handles=[Line2D([0], [0], color=color, label=side)
                 for color, side in zip(colors, ("Image-left", "Image-right"))],
        loc="lower center", fontsize=8,
    )
    for ax in axes:
        ax.set_axis_off()
    return fig


def plot_competitive_growth(
    image: ArrayLike, growth: CompetitiveGrowthResult, *,
    title: str = "Lung/background competition", display_max_size: int = 800,
) -> Figure:
    """Show working intensities, automatic markers, native labels and contours.

    Markers are diagnostic inputs, not reference annotations. Cyan/orange mark
    the two lungs; gray marks background; black is undecided; red is invalid.
    """
    image = np.asarray(image)
    if image.ndim != 2 or image.shape != growth.labels.shape:
        raise ValueError('image and growth must have matching 2-D shapes')
    if not isinstance(display_max_size, int) or isinstance(display_max_size, bool) or display_max_size < 1:
        raise ValueError('display_max_size must be a positive integer')
    step = max(1, int(np.ceil(max(image.shape) / display_max_size)))
    sample = np.s_[::step, ::step]
    colors = ('#24cce0', '#ffad42')
    marker_colors = ListedColormap(['red', 'black', *colors, '#aaaaaa'])
    label_colors = ListedColormap(['black', *colors])
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), layout='constrained')
    fig.suptitle(title)
    axes[0].imshow(growth.working_image, cmap=plt.colormaps['gray'].with_extremes(bad='red'))
    axes[0].set_title('Working image')
    axes[1].imshow(growth.markers + 1, cmap=marker_colors, vmin=0, vmax=4, interpolation='nearest')
    axes[1].set_title('Seeds + automatic background (gray)')
    axes[2].imshow(growth.labels[sample], cmap=label_colors, vmin=0, vmax=2, interpolation='nearest')
    axes[2].set_title(f'{growth.config.method.replace("_", " ").title()} regions')
    axes[3].imshow(image[sample], cmap='gray')
    for side, color in enumerate(colors, 1):
        mask = growth.labels[sample] == side
        if min(mask.shape) >= 2 and mask.any() and not mask.all():
            axes[3].contour(mask, levels=[.5], colors=[color], linewidths=1)
    axes[3].set_title('Contours on X-ray')
    for ax in axes:
        ax.axis('off')
    return fig
