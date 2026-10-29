"""Reproducible Dice scoring and multi-resolution growth-parameter search.

Search uses the same growth implementation as inference. Preprocessing and
seed selection are done once at native resolution; only screening is reduced.
Final model selection and held-out evaluation must use stride=1.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

from .data_loader import Patient
from .region_growing import (
    GrowthConfig, LungSeeds, _grow_from_seeds, _growth_distances,
)
from .seeding import find_lung_seeds
from .thresholding import SegmentationConfig, segment_lungs


@dataclass
class TuningCase:
    patient_id: str
    smoothed: NDArray[np.float32]
    valid: NDArray[np.bool_]
    seeds: LungSeeds
    reference: NDArray[np.bool_]
    stride: int = 1
    seed_error: str = ""
    distances: tuple[NDArray, NDArray, NDArray] | None = None


def dice_score(prediction: NDArray, reference: NDArray) -> float:
    """Binary Dice; two empty masks score 1. Reject mismatched geometry."""
    prediction, reference = np.asarray(prediction, bool), np.asarray(reference, bool)
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference must have matching shapes")
    total = np.count_nonzero(prediction) + np.count_nonzero(reference)
    return float(2 * np.count_nonzero(prediction & reference) / total) if total else 1.0


def config_key(config: GrowthConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True)


def scaled_config(config: GrowthConfig, stride: int) -> GrowthConfig:
    """Keep original pixel units in reports while scaling screening geometry."""
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError("stride must be a positive integer")
    return replace(
        config, connection_radius=config.connection_radius / stride,
        boundary_margin=config.boundary_margin / stride,
        blur_sigma=config.blur_sigma / stride,
        max_distance=None if config.max_distance is None else config.max_distance / stride,
    )


def prepare_case_cache(
    patient: Patient, directory: Path, config: SegmentationConfig,
) -> dict:
    """Cache exact native preprocessing; invalidate after code/data changes.

    Missing seed pairs are recorded, not dropped. Corrupt data or unexpected
    preprocessing failures still raise. Reference masks never affect seeds.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = [patient.image_path, patient.left_mask_path, patient.right_mask_path]
    source_files = [Path(__file__).with_name(name) for name in
                    ("thresholding.py", "morphology.py", "region_growing.py", "seeding.py")]
    signature = json.dumps({
        "segmentation": asdict(config),
        "data": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files],
        "code": [sha256(p.read_bytes()).hexdigest() for p in source_files],
        "cache_version": 2,
    }, sort_keys=True)
    fingerprint = sha256(signature.encode()).hexdigest()
    path = directory / f"{patient.patient_id}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if str(old["fingerprint"]) == fingerprint:
                return {"patient_id": patient.patient_id, "seed_error": str(old["seed_error"]),
                        "seed_sources": old["seed_sources"].tolist(), "seeds_changed": False, "cached": True}
    result = segment_lungs(patient.image, config)
    error = ""
    try:
        seeds = find_lung_seeds(result)
    except ValueError as exc:
        error = str(exc)
        seeds = LungSeeds(np.zeros(result.valid.shape, np.uint8), (0.0, 0.0), (0, 0))
    reference = patient.lung_mask
    if reference is None or reference.shape != result.valid.shape:
        raise ValueError(f"Missing or mismatched reference mask for {patient.patient_id}")
    temporary = path.with_suffix(".tmp.npz")
    changed = None
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            changed = not np.array_equal(old["seeds"], seeds.labels)
    np.savez_compressed(
        temporary, patient_id=patient.patient_id, fingerprint=fingerprint,
        smoothed=result.smoothed, shape=result.valid.shape,
        valid=np.packbits(result.valid), seeds=seeds.labels,
        center=seeds.chest_center, component_ids=seeds.component_ids,
        reference=np.packbits(reference), seed_error=error,
        seed_sources=seeds.sources,
    )
    temporary.replace(path)
    return {"patient_id": patient.patient_id, "seed_error": error, "cached": False,
            "seed_sources": list(seeds.sources), "seeds_changed": changed}


def load_case(path: Path, *, stride: int = 1, cache_distances: bool = True) -> TuningCase:
    scaled_config(GrowthConfig(), stride)  # Validate the sampling stride.
    with np.load(path, allow_pickle=False) as data:
        shape = tuple(data["shape"])
        size = int(np.prod(shape))
        valid = np.unpackbits(data["valid"], count=size).reshape(shape).astype(bool)
        reference = np.unpackbits(data["reference"], count=size).reshape(shape).astype(bool)
        seeds = LungSeeds(
            data["seeds"][::stride, ::stride].copy(),
            tuple(float(x / stride) for x in data["center"]),
            tuple(int(x) for x in data["component_ids"]),
            tuple(str(x) for x in data["seed_sources"]) if 'seed_sources' in data else ('cleaned', 'cleaned'),
        )
        case = TuningCase(
            str(data["patient_id"]), data["smoothed"][::stride, ::stride].copy(),
            valid[::stride, ::stride].copy(), seeds,
            reference[::stride, ::stride].copy(), stride, str(data["seed_error"]),
        )
    if not case.seed_error and cache_distances:
        case.distances = _growth_distances(case.valid, case.seeds)
    return case


def score_case(
    case: TuningCase, config: GrowthConfig, *, admissible: NDArray[np.bool_] | None = None,
    growth_image: NDArray[np.float32] | None = None,
) -> float:
    """Count a failed seed pair as Dice=0 rather than inflating the mean."""
    if config.min_seed_area != 1:
        raise ValueError("This search keeps the requested nearest-blob seed selection fixed")
    if case.seed_error:
        return 0.0
    growth = _grow_from_seeds(
        case.smoothed, case.valid, case.seeds,
        scaled_config(config, case.stride), case.distances, admissible, growth_image,
    )
    return dice_score(growth.mask, case.reference)


def evaluate_configs(
    cases: list[TuningCase], configs: Iterable[GrowthConfig], *,
    workers: int = 4, progress: Callable[[int, int, float], None] | None = None,
) -> list[dict]:
    """Score configurations on an explicitly supplied tuning set only."""
    if not cases:
        raise ValueError("cases must not be empty")
    configs = list(configs)
    def evaluate(config):
        scores = [score_case(case, config) for case in cases]
        return {"config": asdict(config), "mean_dice": float(np.mean(scores)),
                "per_patient": dict(zip((c.patient_id for c in cases), scores))}
    results = []
    best = 0.0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, row in enumerate(pool.map(evaluate, configs), 1):
            results.append(row)
            best = max(best, row["mean_dice"])
            if progress is not None:
                progress(done, len(configs), best)
    return sorted(results, key=lambda row: (-row["mean_dice"], json.dumps(row["config"], sort_keys=True)))


def sample_configs(*, power: int = 11, seed: int = 20261026) -> list[GrowthConfig]:
    """Broad Sobol search of five growth controls, plus the original baseline."""
    points = qmc.Sobol(d=5, scramble=True, rng=np.random.default_rng(seed)).random_base2(power)
    configs = [GrowthConfig()]
    for threshold, radius, distance, margin, fill in points:
        configs.append(GrowthConfig(
            intensity_threshold=round(70 + threshold * 110, 1),
            connection_radius=0 if radius < 0.05 else round(radius * 300 / 5) * 5,
            max_distance=None if distance > 0.95 else round((150 + distance * 1850) / 25) * 25,
            boundary_margin=0 if margin < 0.1 else round(margin * 200 / 5) * 5,
            fill_holes=bool(fill >= 0.2),
        ))
    return list({config_key(c): c for c in configs}.values())


def refine_configs(
    leaders: list[GrowthConfig], *, count: int = 512, seed: int = 20261027,
) -> list[GrowthConfig]:
    """Sample local neighbourhoods of several leaders, without evaluation data."""
    if not leaders:
        raise ValueError("leaders must not be empty")
    rng = np.random.default_rng(seed)
    configs = {}
    for index in range(count):
        leader = leaders[index % len(leaders)]
        distance = leader.max_distance
        if rng.random() < 0.1:
            distance = None
        else:
            distance = round(np.clip((distance or 1800) + rng.normal(0, 200), 0, 2400) / 10) * 10
        config = replace(
            leader,
            intensity_threshold=round(float(np.clip(leader.intensity_threshold + rng.normal(0, 8), 65, 220)), 1),
            connection_radius=round(np.clip(leader.connection_radius + rng.normal(0, 35), 0, 400) / 5) * 5,
            max_distance=distance,
            boundary_margin=round(np.clip(leader.boundary_margin + rng.normal(0, 25), 0, 300) / 5) * 5,
            fill_holes=not leader.fill_holes if rng.random() < 0.1 else leader.fill_holes,
        )
        configs[config_key(config)] = config
    return list(configs.values())
