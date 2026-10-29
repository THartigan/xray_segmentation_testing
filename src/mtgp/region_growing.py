"""Select central lung blobs and grow them through connected image regions.

Left/right always refer to image columns, not anatomical sides. Growth uses
normalized smoothed intensities from the thresholding pipeline; manual lung
annotations are never used to select seeds or determine the grown masks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_fill_holes, binary_propagation, distance_transform_edt, gaussian_filter

from .morphology import open_mask
from .thresholding import SegmentationResult
from .seeding import LungSeeds, find_lung_seeds, select_lung_seeds

_NEIGHBOURS = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class GrowthConfig:
    """Admission and spatial limits for monotone seeded region growth.

    Intensities use the source image scale; lengths are in original pixels.
    ``connection_radius`` opens the admission mask to block thin paths into
    border strips. ``max_distance`` limits Euclidean distance from the seed;
    None disables that limit. ``boundary_margin`` excludes new pixels close
    to invalid areas or image edges. Existing seed pixels are always retained.
    ``blur_sigma`` applies additional normalized Gaussian smoothing before
    growth, in original pixels. It does not change seed selection. Zero retains
    the existing seed-preprocessing intensities without additional smoothing.
    """

    intensity_threshold: float = 20
    connection_radius: float = 60.0
    max_distance: float | None = None
    boundary_margin: float = 2000.0
    min_seed_area: int = 1
    fill_holes: bool = True
    blur_sigma: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.intensity_threshold):
            raise ValueError("intensity_threshold must be finite")
        for name in ("connection_radius", "boundary_margin", "max_distance", "blur_sigma"):
            value = getattr(self, name)
            if name == "max_distance" and value is None:
                continue
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (not isinstance(self.min_seed_area, (int, np.integer))
                or isinstance(self.min_seed_area, (bool, np.bool_))
                or self.min_seed_area < 1):
            raise ValueError("min_seed_area must be a positive integer")
        if not isinstance(self.fill_holes, (bool, np.bool_)):
            raise ValueError("fill_holes must be boolean")


@dataclass(frozen=True)
class LungGrowthResult:
    """Full-resolution seeds, grown labels, and masks used for diagnostics.

    ``admissible`` is the opened intensity-admission mask before per-side
    distance and boundary constraints. Hole filling can admit brighter interior
    pixels, but cannot add invalid pixels or exceed those spatial constraints.
    """

    config: GrowthConfig
    seeds: LungSeeds
    labels: NDArray[np.uint8]
    admissible: NDArray[np.bool_]
    mask_valid: NDArray[np.bool_]
    growth_image: NDArray[np.float32]

    @property
    def mask(self) -> NDArray[np.bool_]:
        """Union of the grown image-left and image-right regions."""
        return self.labels > 0

    @property
    def left_mask(self) -> NDArray[np.bool_]:
        return self.labels == 1

    @property
    def right_mask(self) -> NDArray[np.bool_]:
        return self.labels == 2


def grow_lungs(
    segmentation: SegmentationResult, config: GrowthConfig | None = None,
) -> LungGrowthResult:
    """Select two seeds and propagate each through admissible neighbours.

    Propagation repeats until no eligible connected pixels remain. New growth
    stays on its side of the valid-chest centre, cannot enter the other seed,
    and cannot enter invalid pixels. Selected seeds are never shrunk, including
    previously filled invalid holes or parts crossing the centre column.
    This is an intensity-based estimate, not a guarantee of anatomical borders.
    """
    config = config if config is not None else GrowthConfig()
    seeds = find_lung_seeds(segmentation, min_area=config.min_seed_area)
    valid = segmentation.valid & np.isfinite(segmentation.smoothed)
    return _grow_from_seeds(segmentation.smoothed, valid, seeds, config)


def _growth_distances(
    valid: NDArray[np.bool_], seeds: LungSeeds,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Cache parameter-independent geometry when evaluating many settings."""
    boundary = distance_transform_edt(np.pad(valid | (seeds.labels > 0), 1))
    return (
        boundary[1:-1, 1:-1].copy(),
        distance_transform_edt(seeds.labels != 1),
        distance_transform_edt(seeds.labels != 2),
    )


def smooth_growth_image(
    smoothed: NDArray[np.float32], valid: NDArray[np.bool_], sigma: float,
) -> NDArray[np.float32]:
    """Further smooth valid intensities without admitting NaNs or dark padding.

    This is an additional blur of the seed-preprocessing image, not a new
    total sigma. Away from missing data, the effective sigma is approximately
    sqrt(seed_sigma**2 + sigma**2). Invalid pixels remain NaN. Inputs are not
    modified; sigma=0 returns the existing image without allocating a copy.
    """
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    if sigma == 0:
        return smoothed
    support = valid & np.isfinite(smoothed)
    numerator = gaussian_filter(np.where(support, smoothed, 0), sigma=sigma)
    weights = gaussian_filter(support.astype(np.float32), sigma=sigma)
    result = np.full_like(smoothed, np.nan)
    np.divide(numerator, weights, out=result, where=support & (weights > 0))
    return result


def _grow_from_seeds(
    smoothed: NDArray[np.float32],
    valid: NDArray[np.bool_],
    seeds: LungSeeds,
    config: GrowthConfig,
    distances: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None = None,
    admissible: NDArray[np.bool_] | None = None,
    growth_image: NDArray[np.float32] | None = None,
) -> LungGrowthResult:
    """Shared production/search implementation; cached distances are optional."""
    seed_pixels = seeds.labels > 0
    if growth_image is None:
        growth_image = smooth_growth_image(smoothed, valid, config.blur_sigma)
    if admissible is None:
        admissible = open_mask(
            valid & (growth_image <= config.intensity_threshold),
            radius=config.connection_radius,
        )
    # Previously filled seed holes are interior, not new exclusion boundaries.
    safe = valid.copy()
    if config.boundary_margin > 0:
        if distances is None:
            distance = distance_transform_edt(np.pad(valid | seed_pixels, 1))
            safe &= distance[1:-1, 1:-1] > config.boundary_margin
            del distance
        else:
            safe &= distances[0] > config.boundary_margin
    columns = np.arange(valid.shape[1])[None, :]
    labels = seeds.labels.copy()
    for side in (1, 2):
        seed = seeds.labels == side
        half = (columns < seeds.chest_center[1] if side == 1
                else columns >= seeds.chest_center[1])
        domain = safe & half & ((seeds.labels == 0) | seed)
        if config.max_distance is not None:
            distance = (distance_transform_edt(~seed) if distances is None
                        else distances[side])
            domain &= distance <= config.max_distance
            del distance
        grown = binary_propagation(
            seed, structure=_NEIGHBOURS, mask=(admissible & domain) | seed,
        )
        if config.fill_holes:
            filled = grown | (binary_fill_holes(grown) & domain)
            # Keep only additions connected to the seed after domain clipping.
            grown = binary_propagation(seed, structure=_NEIGHBOURS, mask=filled)
        labels[grown] = side
    return LungGrowthResult(config, seeds, labels, admissible, valid | seed_pixels, growth_image)
