"""Spatially grouped splits and independent leakage checks."""

from __future__ import annotations

import hashlib
import logging
from typing import Sequence

import numpy as np
import pandas as pd
from rasterio.crs import CRS

logger = logging.getLogger(__name__)
BOUND_COLUMNS = ["minx", "miny", "maxx", "maxy"]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return float(np.hypot(dx, dy))


def _common_bounds(frame: pd.DataFrame) -> np.ndarray:
    # A global equal-area CRS does not preserve distances. Fail closed rather
    # than applying a regional buffer to distorted mixed-CRS coordinates.
    if frame.crs.nunique() != 1:
        raise ValueError("Split checks require a common projected metre CRS; reproject all patch metadata "
                         "and rasters to a suitable shared regional grid before splitting/evaluation")
    crs = CRS.from_user_input(frame.crs.iloc[0])
    if not crs.is_projected or crs.linear_units != "metre":
        raise ValueError("Split bounds must be in a projected metre CRS")
    return frame[BOUND_COLUMNS].to_numpy(dtype=float)


def assign_geographic_splits(
    frame: pd.DataFrame, *, seed: int = 42, block_size_m: float = 10000,
    buffer_m: float = 500, fractions: Sequence[float] = (.6, .2, .2),
) -> pd.DataFrame:
    """Assign connected spatial blocks, merging overlapping/nearby patches first.

    No acquisition in one spatial component can cross splits. Small fixtures
    receive at least one component per split; larger datasets use shuffled
    component counts. Component sizes can make patch-count fractions unequal.
    """
    if block_size_m <= 0 or buffer_m < 0:
        raise ValueError("Spatial block size must be positive and buffer nonnegative")
    if len(fractions) != 3 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError("Need positive train/validation/test fractions summing to one")
    result = frame.copy().reset_index(drop=True)
    if result.empty:
        result["block_id"] = pd.Series(dtype=str)
        result["split"] = pd.Series(dtype=str)
        return result
    bounds = _common_bounds(result)
    centers = (bounds[:, :2] + bounds[:, 2:]) / 2
    # Integer block coordinates avoid a region name supplied by the caller
    # bypassing geographical grouping.
    cells = [tuple(np.floor(xy / block_size_m).astype(np.int64)) for xy in centers]
    parent = list(range(len(result)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def merge(i: int, j: int) -> None:
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    first: dict[tuple[int, int], int] = {}
    for i, cell in enumerate(cells):
        if cell in first:
            merge(i, first[cell])
        else:
            first[cell] = i
    # Sweep in x: only footprints close enough in x can violate the buffer.
    order = np.argsort(bounds[:, 0], kind="stable")
    for pos, i in enumerate(order):
        for j in order[pos + 1:]:
            if bounds[j, 0] - bounds[i, 2] > buffer_m:
                break
            if _distance(bounds[i], bounds[j]) <= buffer_m:
                merge(int(i), int(j))
    components: dict[int, list[int]] = {}
    for i in range(len(result)):
        components.setdefault(find(i), []).append(i)
    if len(components) < 3:
        raise ValueError(f"Only {len(components)} independent geographic components: need at least 3 for "
                         "train/validation/test. Add separated regions or reduce block size/buffer.")
    stable = sorted(components, key=lambda key: tuple(np.min(bounds[components[key]], axis=0)))
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(stable))
    count = len(shuffled)
    n_train = min(max(1, int(round(count * fractions[0]))), count - 2)
    n_val = min(max(1, int(round(count * fractions[1]))), count - n_train - 1)
    assignments = ["train"] * n_train + ["validation"] * n_val + ["test"] * (count - n_train - n_val)
    for key, split in zip(shuffled, assignments):
        members = components[key]
        footprint = np.min(bounds[members], axis=0).round(3).tobytes()
        component_id = "geo-" + hashlib.sha256(footprint).hexdigest()[:12]
        result.loc[members, "block_id"] = component_id
        result.loc[members, "split"] = split
    validate_splits(result, buffer_m=buffer_m)
    logger.info("Geographic split counts: %s; components=%d buffer_m=%s",
                result["split"].value_counts().to_dict(), len(components), buffer_m)
    return result


def validate_splits(frame: pd.DataFrame, buffer_m: float = 500) -> None:
    """Reject duplicated pair IDs, shared spatial blocks and cross-split proximity."""
    required = {"pair_id", "block_id", "split", "crs", *BOUND_COLUMNS}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Split manifest missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("No paired patches available for split validation")
    if frame[list(required)].isna().any().any():
        raise ValueError("Split manifest contains missing geographic metadata")
    if frame.pair_id.duplicated().any():
        raise ValueError("Duplicate pair IDs cause split leakage or duplicate examples")
    if not set(frame.split).issubset({"train", "validation", "test"}):
        raise ValueError("Split must be train, validation or test")
    if (frame.groupby("block_id")["split"].nunique() > 1).any():
        raise ValueError("Spatial block leakage: block appears in multiple splits")
    bounds = _common_bounds(frame)
    if not np.isfinite(bounds).all() or np.any(bounds[:, 2:] <= bounds[:, :2]):
        raise ValueError("Invalid geographic patch bounds")
    splits = frame.split.to_numpy()
    order = np.argsort(bounds[:, 0], kind="stable")
    for pos, i in enumerate(order):
        for j in order[pos + 1:]:
            if bounds[j, 0] - bounds[i, 2] > buffer_m:
                break
            if splits[i] != splits[j] and _distance(bounds[i], bounds[j]) <= buffer_m:
                raise ValueError(f"Geographic leakage: pairs {frame.iloc[i].pair_id} and {frame.iloc[j].pair_id} "
                                 f"cross splits within {buffer_m} metres")
