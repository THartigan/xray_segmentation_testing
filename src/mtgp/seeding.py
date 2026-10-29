"""Annotation-free central lung seed selection with missing-side recovery."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import binary_dilation
from skimage import measure

from .morphology import open_mask, repair_components, remove_boundary_components
from .thresholding import SegmentationResult


@dataclass(frozen=True)
class LungSeeds:
    """Two complete blobs: labels 1=image-left, 2=image-right, 0=background."""

    labels: NDArray[np.uint8]
    chest_center: tuple[float, float]  # (row, column), in original pixels.
    component_ids: tuple[int, int]  # Labels in the source mask for each side.
    sources: tuple[str, str] = ("cleaned", "cleaned")


def _available_seeds(
    cleaned: ArrayLike, valid: ArrayLike, *, min_area: int = 1,
) -> LungSeeds:
    """Select the closest component centroid on each side of the chest centre.

    The centre is the centroid of all original valid chest pixels, excluding
    NaNs and the low-intensity padding already rejected by preprocessing.
    Label the cleaned mask with 8-connectivity and compare Euclidean centroid
    distances. Equal distances prefer larger blobs, then smaller label IDs.
    A centroid exactly on the centre column is eligible for neither side.
    Blobs are retained whole, including any invalid holes previously filled.
    Missing sides retain label zero so the caller can recover them explicitly.
    """
    cleaned, valid = np.asarray(cleaned), np.asarray(valid)
    for name, array in (("cleaned", cleaned), ("valid", valid)):
        if array.ndim != 2 or 0 in array.shape or array.dtype != np.bool_:
            raise ValueError(f"{name} must be a nonempty 2-D boolean array")
    if cleaned.shape != valid.shape:
        raise ValueError("cleaned and valid must have the same shape")
    if (not isinstance(min_area, (int, np.integer))
            or isinstance(min_area, (bool, np.bool_)) or min_area < 1):
        raise ValueError("min_area must be a positive integer")
    count = np.count_nonzero(valid)
    if count == 0:
        raise ValueError("Cannot locate the chest centre: no valid pixels")
    center = (
        float(np.dot(np.arange(valid.shape[0]), valid.sum(axis=1)) / count),
        float(np.dot(np.arange(valid.shape[1]), valid.sum(axis=0)) / count),
    )
    components = measure.label(cleaned, connectivity=2)
    candidates: list[list[tuple[float, float, int]]] = [[], []]
    for region in measure.regionprops(components):
        if region.area < min_area:
            continue
        if not np.any(valid[region.slice] & region.image):
            continue
        row, column = region.centroid
        if column == center[1]:
            continue
        side = 0 if column < center[1] else 1
        distance = np.hypot(row - center[0], column - center[1])
        candidates[side].append((distance, -region.area, region.label))
    chosen = tuple(int(min(options)[2]) if options else 0 for options in candidates)
    seed_labels = np.zeros(cleaned.shape, dtype=np.uint8)
    for side, component in enumerate(chosen, start=1):
        if component:
            seed_labels[components == component] = side
    return LungSeeds(seed_labels, center, chosen)


def _missing_sides(seeds: LungSeeds) -> list[str]:
    return [name for name, label in zip(('image-left', 'image-right'), seeds.component_ids)
            if label == 0]


def select_lung_seeds(
    cleaned: ArrayLike, valid: ArrayLike, *, min_area: int = 1,
) -> LungSeeds:
    """Strict nearest-centroid selection; raise if either side has no blob.

    Left/right are image sides. See ``find_lung_seeds`` for automatic recovery
    from the original smoothed image when a cleaned side is absent.
    """
    seeds = _available_seeds(cleaned, valid, min_area=min_area)
    missing = _missing_sides(seeds)
    if missing:
        raise ValueError(f"No seed blob found on {', '.join(missing)}; review the cleaned mask")
    return seeds


def find_lung_seeds(
    segmentation: SegmentationResult, *, min_area: int = 1,
) -> LungSeeds:
    """Keep existing seeds and recover only a missing side from image data.

    A lung may be rejected because a thin connection joins it to the image
    boundary. Retry lower intensity thresholds to isolate its dark interior,
    then stronger opening and moderately higher thresholds if needed. Every
    fallback undergoes the same per-component repair and boundary rejection.
    Only missing sides are replaced. No annotations, forced rectangles, or
    arbitrary seed points are used. Images without plausible blobs still fail.
    ``sources`` records the parameters that produced each selected seed.
    """
    selected = _available_seeds(segmentation.cleaned, segmentation.valid, min_area=min_area)
    if not _missing_sides(selected):
        return selected
    config = segmentation.config
    labels = selected.labels.copy()
    component_ids = list(selected.component_ids)
    sources = list(selected.sources)
    # Start with darker interiors to avoid admitting more boundary tissue.
    trials = [
        (config.threshold - 10, config.opening_radius),
        (config.threshold - 20, config.opening_radius),
        (config.threshold, max(config.opening_radius * 1.5, 1)),
        (config.threshold + 10, config.opening_radius),
        (config.threshold, max(config.opening_radius * 2, 1)),
        (config.threshold + 25, max(config.opening_radius * 1.5, 1)),
        (config.threshold + 40, max(config.opening_radius * 2, 1)),
    ]
    for threshold, radius in trials:
        if threshold <= config.min_valid_intensity:
            continue
        candidate = segmentation.valid & (segmentation.smoothed < threshold)
        opened = open_mask(candidate, radius=radius)
        repaired, mask_valid = repair_components(
            opened, segmentation.valid, closing_radius=config.closing_radius,
        )
        cleaned = remove_boundary_components(
            repaired, mask_valid, max_invalid_contacts=config.max_invalid_contacts,
        )
        # Prevent recovery from overlapping or joining the retained seed.
        cleaned &= ~binary_dilation(labels > 0, structure=np.ones((3, 3), dtype=bool))
        fallback = _available_seeds(cleaned, segmentation.valid, min_area=min_area)
        for index, side in enumerate((1, 2)):
            if component_ids[index] == 0 and fallback.component_ids[index] != 0:
                labels[fallback.labels == side] = side
                component_ids[index] = fallback.component_ids[index]
                sources[index] = f'fallback: threshold={threshold:g}, opening_radius={radius:g}'
        if all(component_ids):
            return LungSeeds(labels, selected.chest_center, tuple(component_ids), tuple(sources))
    missing = [name for name, component in zip(('image-left', 'image-right'), component_ids)
               if component == 0]
    raise ValueError(f"No usable seed on {', '.join(missing)} after image-based recovery")
