"""Invalid-aware intensity thresholding for the Montgomery experiments.

Scales are in input pixels; intensities use the original image scale.
Inputs are not resized, normalized, or mutated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter

from .morphology import open_mask, repair_components, remove_boundary_components


@dataclass(frozen=True)
class SegmentationConfig:
    """Smoothing, thresholding, morphology and boundary-rejection parameters.

    Set a radius to zero to disable that transformation. Opening removes thin
    foreground connections; closing repairs narrow background clefts. Neither
    scale is an anatomical constraint or a guarantee of exactly two lungs.
    """

    sigma: float = 30.0
    threshold: float = 65.0
    min_valid_intensity: float = 10.0
    opening_radius: float = 100.0
    closing_radius: float = 100.0
    max_invalid_contacts: int = 100

    def __post_init__(self) -> None:
        for name in ("sigma", "opening_radius", "closing_radius"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("threshold", "min_valid_intensity"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if (not isinstance(self.max_invalid_contacts, (int, np.integer))
                or isinstance(self.max_invalid_contacts, (bool, np.bool_))
                or self.max_invalid_contacts < 0):
            raise ValueError("max_invalid_contacts must be a nonnegative integer")


@dataclass(frozen=True)
class SegmentationResult:
    """Full-resolution processing stages from a single image.

    ``valid`` describes the original image. ``mask_valid`` additionally admits
    enclosed invalid holes filled during repair. It is for mask displays and
    boundary filtering only; invalid intensities remain NaN in ``smoothed``.
    """

    config: SegmentationConfig
    smoothed: NDArray[np.float32]
    valid: NDArray[np.bool_]
    candidate: NDArray[np.bool_]
    opened: NDArray[np.bool_]
    repaired: NDArray[np.bool_]
    mask_valid: NDArray[np.bool_]
    cleaned: NDArray[np.bool_]


def smooth_valid_image(
    image: ArrayLike, *, sigma: float = 30.0, min_valid_intensity: float = 10.0,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Return normalized Gaussian smoothing and the original validity mask.

    Divide blurred valid intensities by blurred validity weights so zero, NaN,
    and infinite background pixels do not darken their neighbours.
    """
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    if not np.isfinite(min_valid_intensity):
        raise ValueError("min_valid_intensity must be finite")
    pixels = np.asarray(image, dtype=np.float32)
    if pixels.ndim != 2 or 0 in pixels.shape:
        raise ValueError("image must be a nonempty 2-D grayscale array")
    valid = np.isfinite(pixels) & (pixels >= min_valid_intensity)
    blurred = gaussian_filter(np.where(valid, pixels, 0), sigma=sigma)
    weights = gaussian_filter(valid.astype(np.float32), sigma=sigma)
    smoothed = np.full_like(pixels, np.nan)
    np.divide(blurred, weights, out=smoothed, where=valid & (weights > 0))
    return smoothed, valid


def segment_lungs(
    image: ArrayLike, config: SegmentationConfig | None = None,
) -> SegmentationResult:
    """Disconnect thin tethers, repair individual blobs, then reject borders.

    Label components after opening, so closing does not use the original
    boundary strip to bridge a connection that opening just removed.
    """
    config = config if config is not None else SegmentationConfig()
    smoothed, valid = smooth_valid_image(
        image, sigma=config.sigma, min_valid_intensity=config.min_valid_intensity,
    )
    candidate = valid & (smoothed < config.threshold)
    opened = open_mask(candidate, radius=config.opening_radius)
    repaired, mask_valid = repair_components(
        opened, valid, closing_radius=config.closing_radius,
    )
    cleaned = remove_boundary_components(
        repaired, mask_valid, max_invalid_contacts=config.max_invalid_contacts,
    )
    return SegmentationResult(
        config=config, smoothed=smoothed, valid=valid, candidate=candidate,
        opened=opened, repaired=repaired, mask_valid=mask_valid, cleaned=cleaned,
    )
