"""Seeded lung/background competition on an explicitly bounded working grid.

Unlike threshold propagation, both lung and background markers compete for
unlabelled pixels. Random walker penalizes transitions across image gradients;
watershed floods gradient basins from the same markers. Neither uses annotations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_propagation, distance_transform_edt, sobel
from skimage.segmentation import random_walker, watershed

from .region_growing import smooth_growth_image
from .seeding import LungSeeds, find_lung_seeds
from .thresholding import SegmentationResult


@dataclass(frozen=True)
class CompetitiveGrowthConfig:
    """Original-pixel spatial scales, original-intensity background threshold.

    ``working_max_size`` bounds graph size; the prediction is restored with
    nearest-neighbour sampling, then original seeds and validity are enforced.
    ``beta`` affects random walker only. Markers: 1/2=image-left/right, 3=background,
    0=unknown, -1=invalid. Seeds take precedence over background suggestions.
    """

    method: Literal['random_walker', 'watershed'] = 'watershed'
    working_max_size: int = 384
    blur_sigma: float = 0.0
    background_threshold: float = 190.0
    boundary_margin: float = 100.0
    beta: float = 130.0

    def __post_init__(self) -> None:
        if self.method not in ('random_walker', 'watershed'):
            raise ValueError('method must be random_walker or watershed')
        if (not isinstance(self.working_max_size, int) or isinstance(self.working_max_size, bool)
                or self.working_max_size < 16):
            raise ValueError('working_max_size must be an integer >= 16')
        for name in ('blur_sigma', 'boundary_margin', 'beta'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not np.isfinite(self.background_threshold):
            raise ValueError('background_threshold must be finite')


@dataclass(frozen=True)
class CompetitiveGrowthResult:
    config: CompetitiveGrowthConfig
    seeds: LungSeeds
    labels: NDArray[np.uint8]  # Native resolution: 0=background, 1/2=lung sides.
    markers: NDArray[np.int32]  # Working grid, including background and invalid.
    working_image: NDArray[np.float32]
    stride: int

    @property
    def mask(self) -> NDArray[np.bool_]:
        return self.labels > 0

    @property
    def left_mask(self) -> NDArray[np.bool_]:
        return self.labels == 1

    @property
    def right_mask(self) -> NDArray[np.bool_]:
        return self.labels == 2


def grow_competitive(
    segmentation: SegmentationResult, config: CompetitiveGrowthConfig | None = None,
) -> CompetitiveGrowthResult:
    """Grow from the same full-resolution seeds, with explicit background rivals."""
    config = config if config is not None else CompetitiveGrowthConfig()
    seeds = find_lung_seeds(segmentation)
    return _grow_competitive_from_seeds(segmentation.smoothed, segmentation.valid, seeds, config)


def _working_prediction(
    smoothed: NDArray[np.float32], valid: NDArray[np.bool_], seeds: LungSeeds,
    config: CompetitiveGrowthConfig, pixel_scale: int,
) -> tuple[NDArray[np.uint8], NDArray[np.int32], NDArray[np.float32]]:
    """Shared inference/search core on a sampled grid; lengths remain native units."""
    image = smooth_growth_image(smoothed, valid, config.blur_sigma / pixel_scale)
    support = valid & np.isfinite(image)
    if not support.any() or any(not np.any((seeds.labels == side) & support) for side in (1, 2)):
        raise ValueError('Working grid must contain valid pixels and both lung seeds')
    # Keep an explicit valid perimeter even when no extra margin is requested.
    boundary = distance_transform_edt(np.pad(support, 1))[1:-1, 1:-1]
    background = support & ((boundary <= max(1, config.boundary_margin / pixel_scale))
                            | (image >= config.background_threshold))
    # A background column separates the lungs through the mediastinum.
    center_column = int(np.clip(round(seeds.chest_center[1]), 0, image.shape[1] - 1))
    background[:, center_column] |= support[:, center_column]
    markers = np.zeros(image.shape, dtype=np.int32)
    markers[background] = 3
    markers[~support] = -1
    for side in (1, 2):
        markers[(seeds.labels == side) & support] = side
    if not np.any(markers == 3):
        raise ValueError('No background markers remain outside lung seeds')
    # Robust image-only scaling keeps graph penalties comparable across patients.
    low, high = np.percentile(image[support], (1, 99))
    normalized = np.clip((np.where(support, image, low) - low) / max(float(high-low), 1e-6), 0, 1)
    if config.method == 'random_walker':
        if np.any(markers == 0):
            # Direct solve on a bounded 2-D grid avoids unconverged CG results.
            prediction = random_walker(normalized, markers, beta=config.beta, mode='bf')
        else:
            prediction = markers
    else:
        gradient = np.hypot(sobel(normalized, axis=0), sobel(normalized, axis=1))
        prediction = watershed(gradient, np.maximum(markers, 0), mask=support)
    labels = np.where((prediction == 1) | (prediction == 2), prediction, 0).astype(np.uint8)
    return labels, markers, image


def _restore_prediction(
    working_labels: NDArray[np.uint8], valid: NDArray[np.bool_], seeds: LungSeeds, stride: int,
) -> NDArray[np.uint8]:
    """Restore native geometry, preserve seeds and discard disconnected additions."""
    height, width = valid.shape
    # Samples lie at 0, stride, ...; restore using the nearest sampled coordinate.
    rows = np.minimum((np.arange(height) + stride // 2) // stride, working_labels.shape[0] - 1)
    columns = np.minimum((np.arange(width) + stride // 2) // stride, working_labels.shape[1] - 1)
    sampled = working_labels[rows[:, None], columns[None, :]]
    labels = seeds.labels.copy()
    for side in (1, 2):
        seed = seeds.labels == side
        half = (np.arange(width)[None, :] < seeds.chest_center[1])
        if side == 2:
            half = ~half
        allowed = valid & half & (sampled == side) & ((seeds.labels == 0) | seed)
        grown = binary_propagation(seed, structure=np.ones((3, 3), bool), mask=allowed | seed)
        labels[grown] = side
    return labels


def _grow_competitive_from_seeds(
    smoothed: NDArray[np.float32], valid: NDArray[np.bool_], seeds: LungSeeds,
    config: CompetitiveGrowthConfig,
) -> CompetitiveGrowthResult:
    """Use native cached inputs without repeating segmentation during tuning."""
    stride = max(1, int(np.ceil(max(valid.shape) / config.working_max_size)))
    sample = np.s_[::stride, ::stride]
    seed_labels = seeds.labels[sample].copy()
    for side in (1, 2):
        if not np.any(seed_labels == side):
            raise ValueError('Working grid loses a lung seed; increase working_max_size')
    sampled_seeds = LungSeeds(seed_labels, tuple(c / stride for c in seeds.chest_center),
                             seeds.component_ids, seeds.sources)
    working, markers, image = _working_prediction(smoothed[sample], valid[sample], sampled_seeds, config, stride)
    labels = _restore_prediction(working, valid, seeds, stride)
    return CompetitiveGrowthResult(config, seeds, labels, markers, image, stride)
