"""Source-aware raster validation, reprojection and masked normalization."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RasterInfo:
    """Validated georeferencing and quality summary for a local raster."""

    path: str
    crs: str
    bounds: tuple[float, float, float, float]
    resolution: tuple[float, float]
    width: int
    height: int
    bands: int
    nodata: float | None
    invalid_fraction: float


def inspect_raster(path: str | Path, bands: Sequence[int] | None = None) -> RasterInfo:
    """Open a raster and validate CRS, dimensions, selected bands and data masks."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Raster not found: {path}. See docs/data_sources.md for preparation steps.")
    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError("missing CRS")
            if src.width < 1 or src.height < 1 or src.count < 1:
                raise ValueError("empty raster")
            if src.transform.is_identity:
                raise ValueError("missing georeferenced transform")
            chosen = list(bands or range(1, src.count + 1))
            if not chosen or min(chosen) < 1 or max(chosen) > src.count:
                raise ValueError(f"requested bands {chosen}, but raster has {src.count} bands")
            arr = src.read(chosen, masked=True)
            valid = ~np.ma.getmaskarray(arr) & np.isfinite(arr.data)
            info = RasterInfo(str(path.resolve()), src.crs.to_string(), tuple(src.bounds), tuple(src.res),
                              src.width, src.height, src.count, src.nodata, float(1 - valid.mean()))
            logger.info("Raster inspected: %s", asdict(info))
            return info
    except Exception as exc:
        logger.exception("Raster validation failed for %s", path)
        raise ValueError(f"Cannot inspect raster {path}: {exc}") from exc


def projected_bounds(info: RasterInfo, target_crs: str) -> tuple[float, float, float, float]:
    """Transform a raster's extent to a projected metric common CRS."""
    crs = CRS.from_user_input(target_crs)
    if not crs.is_projected or crs.linear_units != "metre":
        raise ValueError(f"Common grid CRS must use projected metres, got {target_crs}")
    if CRS.from_user_input(info.crs) == crs:
        return info.bounds
    return transform_bounds(info.crs, crs, *info.bounds, densify_pts=21)


def read_on_grid(
    path: str | Path,
    bands: Sequence[int],
    target_crs: str,
    bounds: tuple[float, float, float, float],
    resolution_m: float,
    *,
    categorical: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample selected bands onto one grid, excluding nodata from interpolation."""
    left, bottom, right, top = bounds
    width = int(round((right - left) / resolution_m))
    height = int(round((top - bottom) / resolution_m))
    if width < 1 or height < 1:
        raise ValueError("Common raster grid has no pixels")
    dst_transform = from_origin(left, top, resolution_m, resolution_m)
    dst = np.full((len(bands), height, width), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        arr = src.read(list(bands), masked=True).astype(np.float32)
        valid = ~np.ma.getmaskarray(arr) & np.isfinite(arr.data)
        data = np.where(valid, arr.data, np.nan)
        for channel in range(len(bands)):
            reproject(data[channel], dst[channel], src_transform=src.transform, src_crs=src.crs,
                      src_nodata=np.nan, dst_transform=dst_transform, dst_crs=target_crs,
                      dst_nodata=np.nan, resampling=Resampling.nearest if categorical else Resampling.bilinear,
                      init_dest_nodata=True)
        # A nearest-neighbour validity check keeps nodata holes out of the result.
        validity = np.zeros((height, width), dtype=np.uint8)
        reproject(valid.all(axis=0).astype(np.uint8), validity,
                  src_transform=src.transform, src_crs=src.crs, dst_transform=dst_transform,
                  dst_crs=target_crs, src_nodata=None, dst_nodata=0, resampling=Resampling.nearest)
    mask = validity.astype(bool) & np.isfinite(dst).all(axis=0)
    return dst, mask


def normalize_sar(
    array: np.ndarray, valid: np.ndarray, *, units: str = "db", clip_db: Sequence[float] = (-30, 5)
) -> tuple[np.ndarray, np.ndarray]:
    """Clip calibrated SAR dB, or convert positive calibrated linear sigma0 to dB."""
    arr = np.asarray(array, dtype=np.float32).copy()
    good = np.asarray(valid, dtype=bool) & np.isfinite(arr).all(axis=0)
    if units == "linear":
        good &= (arr > 0).all(axis=0)
        arr = 10 * np.log10(np.maximum(arr, np.finfo(np.float32).tiny))
    elif units != "db":
        raise ValueError("SAR units must be 'db' or calibrated 'linear'; raw digital numbers are unsupported")
    low, high = clip_db
    if high <= low:
        raise ValueError("SAR clipping upper bound must exceed lower bound")
    arr = (np.clip(arr, low, high) - low) / (high - low)
    arr[:, ~good] = 0
    return np.nan_to_num(arr, copy=False).astype(np.float32), good


def normalize_optical(
    array: np.ndarray, valid: np.ndarray, *, scale: float = 10000,
    offset: float = 0, clip: Sequence[float] = (0, 1), cloud_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale source digital numbers as (DN + offset)/scale and mask clouds/nodata."""
    if scale <= 0:
        raise ValueError("Optical reflectance scale must be positive")
    arr = np.asarray(array, dtype=np.float32)
    good = np.asarray(valid, dtype=bool) & np.isfinite(arr).all(axis=0)
    if cloud_mask is not None:
        if cloud_mask.shape != good.shape:
            raise ValueError("Cloud mask shape differs from optical grid")
        good &= ~cloud_mask.astype(bool)
    arr = np.clip((arr + offset) / scale, clip[0], clip[1])
    arr[:, ~good] = 0
    return np.nan_to_num(arr).astype(np.float32), good


def patch_windows(height: int, width: int, patch_size: int, stride: int) -> list[tuple[int, int]]:
    """Return full patch top-left coordinates; incomplete edge patches are excluded."""
    if min(height, width, patch_size, stride) <= 0:
        raise ValueError("Raster dimensions, patch size and stride must be positive")
    return [(row, col) for row in range(0, height - patch_size + 1, stride)
            for col in range(0, width - patch_size + 1, stride)]
