"""Binary mask repair with explicit foreground and invalid-background rules."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import binary_dilation, binary_fill_holes
from skimage import measure
from skimage.morphology import isotropic_closing, isotropic_opening


def _mask_array(mask: ArrayLike) -> NDArray[np.bool_]:
    array = np.asarray(mask)
    if array.ndim != 2 or 0 in array.shape or array.dtype != np.bool_:
        raise ValueError("mask must be a nonempty 2-D boolean array")
    return array


def _mask_pair(
    mask: ArrayLike, valid: ArrayLike,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    mask, valid = _mask_array(mask), _mask_array(valid)
    if mask.shape != valid.shape:
        raise ValueError("mask and valid must have the same shape")
    if np.any(mask & ~valid):
        raise ValueError("foreground pixels must be valid")
    return mask, valid


def _check_radius(radius: float) -> None:
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")


def open_mask(mask: ArrayLike, *, radius: float) -> NDArray[np.bool_]:
    """Remove thin white bridges using disk opening (erosion then dilation).

    This also closes narrow gaps in the black background. It can separate a
    lung from a boundary strip, but also rounds tips and removes small objects.
    Zero disables opening. False padding treats outside-image pixels as black.
    """
    mask = _mask_array(mask)
    _check_radius(radius)
    if radius == 0 or not mask.any():
        return mask.copy()
    pad = int(np.ceil(radius)) + 1
    padded = np.pad(mask, pad, mode="constant")
    opened = isotropic_opening(padded, radius=radius)[pad:-pad, pad:-pad]
    # Enforce anti-extensivity, including the empty-eroded-mask EDT edge case.
    return opened & mask


def repair_components(
    mask: ArrayLike, valid: ArrayLike, *, closing_radius: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Close each 8-connected component, then fill its enclosed holes.

    Closing adds pixels only in valid image regions. Enclosed invalid holes
    may be filled and are admitted by the returned mask-validity array.
    Exterior invalid regions stay excluded. Components are closed separately
    so separate objects cannot jointly bridge a gap. Their repaired unions
    can still touch: this is not a topology-preserving transformation.
    """
    mask, valid = _mask_pair(mask, valid)
    _check_radius(closing_radius)
    labels = measure.label(mask, connectivity=2)
    repaired = np.zeros_like(mask)
    mask_valid = valid.copy()
    pad = int(np.ceil(closing_radius)) + 1

    for region in measure.regionprops(labels):
        crop = region.slice
        component = labels[crop] == region.label
        if closing_radius > 0:
            padded = np.pad(component, pad, mode="constant")
            closed = isotropic_closing(padded, radius=closing_radius)
            component |= closed[pad:-pad, pad:-pad] & valid[crop]
        filled = binary_fill_holes(component)
        repaired[crop] |= filled
        mask_valid[crop] |= filled

    return repaired, mask_valid


def remove_boundary_components(
    mask: ArrayLike, valid: ArrayLike, *, max_invalid_contacts: int = 100,
) -> NDArray[np.bool_]:
    """Reject components at the image edge or touching too many invalid pixels.

    Count distinct invalid pixels in each component's 8-neighbourhood, not
    foreground boundary pixels or contact pairs. The tolerance is inclusive.
    """
    mask, valid = _mask_pair(mask, valid)
    if (not isinstance(max_invalid_contacts, (int, np.integer))
            or isinstance(max_invalid_contacts, (bool, np.bool_))
            or max_invalid_contacts < 0):
        raise ValueError("max_invalid_contacts must be a nonnegative integer")
    labels = measure.label(mask, connectivity=2)
    keep = np.zeros(labels.max() + 1, dtype=bool)
    height, width = mask.shape
    for region in measure.regionprops(labels):
        y0, x0, y1, x1 = region.bbox
        if y0 == 0 or x0 == 0 or y1 == height or x1 == width:
            continue
        crop = np.s_[y0 - 1:y1 + 1, x0 - 1:x1 + 1]
        component = labels[crop] == region.label
        neighbours = binary_dilation(component, structure=np.ones((3, 3), dtype=bool))
        contacts = np.count_nonzero(neighbours & ~valid[crop])
        keep[region.label] = contacts <= max_invalid_contacts
    return keep[labels]
