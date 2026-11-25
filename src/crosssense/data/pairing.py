"""Geographic/time pairing and patch preparation on a documented metric grid."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crosssense.data.dataset import REQUIRED_COLUMNS
from crosssense.data.preprocess import (
    inspect_raster,
    normalize_optical,
    normalize_sar,
    patch_windows,
    projected_bounds,
    read_on_grid,
)
from crosssense.evaluation.splits import assign_geographic_splits

logger = logging.getLogger(__name__)


def _get(config: Any, name: str, default: Any) -> Any:
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def _present(value: Any) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _true(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _band_name(value: Any) -> str:
    """Canonicalize Sentinel band labels, including B02/B2 and B8a/B8A."""
    name = str(value).strip().upper()
    match = re.fullmatch(r"B0*(\d+)(A?)", name)
    return f"B{int(match[1])}{match[2]}" if match else name


def overlap_fraction(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection divided by smaller footprint area, allowing nested scenes."""
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / smaller if smaller > 0 else 0


def _validate_source(row: pd.Series, config: Any) -> list[int]:
    source, sensor = row.source, row.sensor
    if source not in {"synthetic", "sen12ms", "copernicus_prepared"}:
        raise ValueError(f"Scene {row.scene_id}: unsupported source {source!r}; use synthetic, sen12ms or copernicus_prepared")
    bands = list(_get(config, "sar_bands", [1, 2]) if sensor == "sar"
                 else _get(config, "optical_bands", [2, 3, 4, 8]))
    try:
        names = json.loads(row.band_names) if isinstance(row.band_names, str) else list(row.band_names)
    except Exception as exc:
        raise ValueError(f"Scene {row.scene_id}: band_names must be a JSON array") from exc
    if not isinstance(names, list) or len(names) < max(bands) or len(set(names)) != len(names):
        raise ValueError(f"Scene {row.scene_id}: missing/duplicate band names for selection {bands}")
    names = [_band_name(value) for value in names]
    if len(set(names)) != len(names):
        raise ValueError(f"Scene {row.scene_id}: duplicate semantic band names")
    if source == "sen12ms" and sensor == "optical" and names != ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]:
        raise ValueError(f"Scene {row.scene_id}: SEN12MS optical band_names must describe the official full 13-band order")
    if sensor == "sar" and not {"VV", "VH"}.issubset(set(names)):
        raise ValueError(f"Scene {row.scene_id}: missing polarization; VV and VH are required")
    if sensor == "sar" and [str(names[i - 1]).upper() for i in bands] != ["VV", "VH"]:
        raise ValueError(f"Scene {row.scene_id}: sar_bands must select VV then VH")
    if sensor == "optical":
        selected = [str(names[i - 1]).upper() for i in bands]
        if any(not name.startswith("B") for name in selected):
            raise ValueError(f"Scene {row.scene_id}: optical selection does not describe spectral bands")
    if source == "copernicus_prepared":
        if sensor == "sar" and not (_true(row.get("calibrated", False)) and _true(row.get("terrain_corrected", False))):
            raise ValueError(f"Scene {row.scene_id}: raw Sentinel-1 GRD is unsupported. Apply calibration and terrain correction "
                             "externally, export VV/VH georeferenced sigma0, then set calibrated/terrain_corrected=true.")
        if sensor == "optical" and (str(row.get("processing_level", "")).upper() != "L2A"
                                    or not _present(row.get("reflectance_scale"))
                                    or not _present(row.get("reflectance_offset"))):
            raise ValueError(f"Scene {row.scene_id}: prepared Sentinel-2 requires processing_level=L2A and explicit "
                             "reflectance_scale/reflectance_offset from product metadata")
    return bands


def read_scene_manifest(path: str | Path, config: Any) -> pd.DataFrame:
    """Validate local scene files and per-source metadata before pairing."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Scene manifest not found: {path}. Obtain SEN12MS or prepare Copernicus data "
                                "using docs/data_sources.md; use crosssense synthetic --output data/synthetic for a labelled fixture.")
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"scene_id", "sensor", "path", "timestamp", "source", "band_names"}
    if required - set(frame):
        raise ValueError(f"Scene manifest missing columns: {sorted(required - set(frame))}")
    if frame.empty or frame.scene_id.duplicated().any():
        raise ValueError("Scene manifest must have nonempty, unique scene IDs")
    if not set(frame.sensor).issubset({"sar", "optical"}):
        raise ValueError("Scene sensor must be sar or optical")
    # Empty timestamps remain unknown. Invalid nonempty timestamps still fail.
    frame["timestamp"] = pd.to_datetime(frame.timestamp.replace("", None), utc=True, errors="raise")
    for column in ["path", "cloud_mask_path"]:
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].map(lambda item: str((path.parent / item).resolve())
                                         if _present(item) and not Path(item).is_absolute() else item)
    infos, band_selections = [], []
    for _, row in frame.iterrows():
        try:
            bands = _validate_source(row, config)
            infos.append(inspect_raster(row.path, bands))
            band_selections.append(bands)
            if _present(row.cloud_mask_path):
                inspect_raster(row.cloud_mask_path, [1])
        except Exception:
            logger.exception("Scene validation failed: scene_id=%s sensor=%s", row.scene_id, row.sensor)
            raise
    frame["_raster_info"] = infos
    frame["_bands"] = band_selections
    logger.info("Validated scene manifest %s scene_counts=%s", path, frame.sensor.value_counts().to_dict())
    return frame


def _reject(rejections: list[dict[str, Any]], sar_id: str, optical_id: str, reason: str,
            **details: Any) -> None:
    rejections.append({"sar_scene_id": sar_id, "optical_scene_id": optical_id, "reason": reason, **details})


def prepare_pairs(scene_csv: str | Path, output_dir: str | Path, data_config: Any, seed: int = 42) -> Path:
    """Create normalized local patches, stable pair IDs, rejection log and geographic splits.

    Patch positives share one snapped intersection grid and satisfy time/cloud/
    validity thresholds. Every other pair is potentially a negative, subject to
    the training objective's geographic exclusion rule.
    """
    scenes = read_scene_manifest(scene_csv, data_config)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = asdict(data_config) if is_dataclass(data_config) else (dict(data_config) if isinstance(data_config, dict) else vars(data_config))
    (output_dir / "preprocessing_config.json").write_text(json.dumps({"schema_version": 1,
        "seed": seed, "data": resolved_config}, indent=2, default=str), encoding="utf-8")
    patch_dir = output_dir / "patches"
    patch_dir.mkdir(exist_ok=True)
    crs = str(_get(data_config, "target_crs", "EPSG:32643"))
    res = float(_get(data_config, "resolution_m", 10))
    size, stride = int(_get(data_config, "patch_size", 32)), int(_get(data_config, "stride", 32))
    max_gap = float(_get(data_config, "max_time_gap_days", 7))
    min_overlap = float(_get(data_config, "min_overlap", .9))
    max_cloud = float(_get(data_config, "max_cloud_fraction", .2))
    min_valid = float(_get(data_config, "min_valid_fraction", .8))
    if res <= 0 or min(size, stride) <= 0:
        raise ValueError("Resolution, patch size and stride must be positive")
    rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    sar_scenes = scenes.loc[scenes.sensor == "sar"]
    optical_scenes = scenes.loc[scenes.sensor == "optical"]
    if sar_scenes.empty or optical_scenes.empty:
        raise ValueError("Scene manifest needs both SAR and optical scenes")
    for _, sar in sar_scenes.iterrows():
        sar_bounds = projected_bounds(sar._raster_info, crs)
        for _, optical in optical_scenes.iterrows():
            opt_bounds = projected_bounds(optical._raster_info, crs)
            time_verified = not (pd.isna(sar.timestamp) or pd.isna(optical.timestamp))
            gap = abs((sar.timestamp - optical.timestamp).total_seconds()) / 86400 if time_verified else -1.
            overlap = overlap_fraction(sar_bounds, opt_bounds)
            if overlap < min_overlap or overlap == 0:
                _reject(rejections, sar.scene_id, optical.scene_id, "geographic_overlap", overlap=overlap)
                continue
            if sar.source == "sen12ms" and _present(sar.get("pairing_key")) and sar.get("pairing_key") != optical.get("pairing_key"):
                _reject(rejections, sar.scene_id, optical.scene_id, "author_pair_key_mismatch")
                continue
            if not time_verified:
                allowed = (_get(data_config, "allow_author_pairs_without_time", False) and sar.source == "sen12ms"
                           and _present(sar.get("pairing_key")) and sar.get("pairing_key") == optical.get("pairing_key"))
                if not allowed:
                    _reject(rejections, sar.scene_id, optical.scene_id, "missing_acquisition_time")
                    continue
                logger.warning("Author-paired SEN12MS samples %s/%s have unknown acquisition times; "
                               "time-window threshold is UNVERIFIED (explicit configuration opt-in)",
                               sar.scene_id, optical.scene_id)
            if gap > max_gap:
                _reject(rejections, sar.scene_id, optical.scene_id, "time_gap", time_gap_days=gap)
                continue
            if sar.source != optical.source:
                _reject(rejections, sar.scene_id, optical.scene_id, "source_mismatch")
                continue
            intersection = (max(sar_bounds[0], opt_bounds[0]), max(sar_bounds[1], opt_bounds[1]),
                            min(sar_bounds[2], opt_bounds[2]), min(sar_bounds[3], opt_bounds[3]))
            common = (np.ceil(intersection[0] / res) * res, np.ceil(intersection[1] / res) * res,
                      np.floor(intersection[2] / res) * res, np.floor(intersection[3] / res) * res)
            if common[2] - common[0] < size * res or common[3] - common[1] < size * res:
                _reject(rejections, sar.scene_id, optical.scene_id, "intersection_smaller_than_patch")
                continue
            sar_raw, sar_valid = read_on_grid(sar.path, sar._bands, crs, common, res)
            opt_raw, opt_valid = read_on_grid(optical.path, optical._bands, crs, common, res)
            clouds = np.zeros(opt_valid.shape, dtype=bool)
            cloud_available = _present(optical.cloud_mask_path)
            if cloud_available:
                cloud_data, cloud_valid = read_on_grid(optical.cloud_mask_path, [1], crs, common, res, categorical=True)
                clouds = (cloud_data[0] != 0) | ~cloud_valid
            elif _get(data_config, "require_cloud_mask", False):
                _reject(rejections, sar.scene_id, optical.scene_id, "missing_cloud_mask")
                continue
            else:
                logger.warning("Optical scene %s has no pixel cloud mask; cloud_fraction=-1 (unknown), "
                               "cloud threshold cannot be verified", optical.scene_id)
            sar_units = str(sar.get("sar_units", "") or _get(data_config, "sar_units", "db"))
            # SEN12MS is already sigma0 in dB: reject accidental double logging.
            if sar.source == "sen12ms" and sar_units != "db":
                raise ValueError("SEN12MS SAR is already in dB; set sar_units=db")
            sar_arr, sar_valid = normalize_sar(sar_raw, sar_valid, units=sar_units,
                                             clip_db=_get(data_config, "sar_clip_db", [-30, 5]))
            scale = float(optical.get("reflectance_scale", "") or _get(data_config, "optical_scale", 10000))
            offset = float(optical.get("reflectance_offset", "") or 0)
            opt_arr, opt_valid = normalize_optical(opt_raw, opt_valid, scale=scale, offset=offset,
                                                 clip=_get(data_config, "optical_clip", [0, 1]), cloud_mask=clouds)
            for row, col in patch_windows(*opt_valid.shape, size, stride):
                sl = np.s_[row:row + size, col:col + size]
                mask = sar_valid[sl] & opt_valid[sl]
                cloud_fraction = float(clouds[sl].mean()) if cloud_available else -1.
                valid_fraction = float(mask.mean())
                if cloud_available and cloud_fraction > max_cloud:
                    _reject(rejections, sar.scene_id, optical.scene_id, "cloud_fraction", row=row, col=col,
                            cloud_fraction=cloud_fraction)
                    continue
                if valid_fraction < min_valid:
                    _reject(rejections, sar.scene_id, optical.scene_id, "valid_fraction", row=row, col=col,
                            valid_fraction=valid_fraction)
                    continue
                minx, maxy = common[0] + col * res, common[3] - row * res
                miny, maxx = maxy - size * res, minx + size * res
                identity = f"{sar.source}|{sar.scene_id}|{optical.scene_id}|{crs}|{minx:.6f}|{maxy:.6f}|{size}|{res}"
                pair_id = "pair-" + hashlib.sha256(identity.encode()).hexdigest()[:20]
                files = {kind: patch_dir / f"{pair_id}_{kind}.npy" for kind in ["sar", "optical", "mask"]}
                sp, op = sar_arr[:, sl[0], sl[1]].copy(), opt_arr[:, sl[0], sl[1]].copy()
                sp[:, ~mask], op[:, ~mask] = 0, 0
                np.save(files["sar"], sp, allow_pickle=False)
                np.save(files["optical"], op, allow_pickle=False)
                np.save(files["mask"], mask, allow_pickle=False)
                record = {"pair_id": pair_id, "sar_path": str(files["sar"].relative_to(output_dir)),
                          "optical_path": str(files["optical"].relative_to(output_dir)),
                          "mask_path": str(files["mask"].relative_to(output_dir)), "crs": crs,
                          "x": (minx + maxx) / 2, "y": (miny + maxy) / 2,
                          "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy,
                          "sar_time": sar.timestamp.isoformat() if not pd.isna(sar.timestamp) else "",
                          "optical_time": optical.timestamp.isoformat() if not pd.isna(optical.timestamp) else "",
                          "time_gap_days": gap, "cloud_fraction": cloud_fraction, "valid_fraction": valid_fraction,
                          "time_verified": time_verified,
                          "cloud_mask_available": cloud_available, "source": sar.source,
                          "land_cover": str(optical.get("land_cover", "unknown") or "unknown"),
                          "sar_scene_id": sar.scene_id, "optical_scene_id": optical.scene_id,
                          "resolution_m": res, "patch_size": size, "sar_units": sar_units,
                          "sar_clip_db": json.dumps(list(_get(data_config, "sar_clip_db", [-30, 5]))),
                          "optical_scale": scale, "optical_offset": offset,
                          "sar_bands": json.dumps(sar._bands), "optical_bands": json.dumps(optical._bands)}
                for sensor_name, scene in [("sar", sar), ("optical", optical)]:
                    names = json.loads(scene.band_names)
                    record[f"{sensor_name}_band_names"] = json.dumps([_band_name(names[index - 1]) for index in scene._bands])
                for sensor_name, scene in [("sar", sar), ("optical", optical)]:
                    for key in ["product_id", "tile_id", "processing_level", "orbit_direction", "relative_orbit", "license", "attribution"]:
                        if _present(scene.get(key)):
                            record[f"{sensor_name}_{key}"] = scene[key]
                rows.append(record)
    rejected = pd.DataFrame(rejections, columns=sorted({key for item in rejections for key in item}) or ["reason"])
    rejected.to_csv(output_dir / "rejections.csv", index=False)
    manifest = pd.DataFrame(rows)
    if len(manifest):
        manifest = assign_geographic_splits(manifest, seed=seed,
                                           block_size_m=float(_get(data_config, "block_size_m", 10000)),
                                           buffer_m=float(_get(data_config, "split_buffer_m", 500)),
                                           fractions=_get(data_config, "split_fractions", [.6, .2, .2]))
    else:
        manifest = pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))
    manifest.to_csv(output_dir / "pairs.csv", index=False)
    file_entries = []
    for path in sorted(patch_dir.glob("*.npy")):
        file_entries.append({"path": str(path.relative_to(output_dir)), "bytes": path.stat().st_size,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (output_dir / "files.json").write_text(json.dumps({"files": file_entries,
        "total_bytes": sum(item["bytes"] for item in file_entries), "source_scenes": str(Path(scene_csv).resolve()),
        "seed": seed, "target_crs": crs, "resolution_m": res,
        "preprocessing_config": "preprocessing_config.json"}, indent=2), encoding="utf-8")
    logger.info("Pair generation completed scenes=%d pairs=%d rejected=%s manifest=%s",
                len(scenes), len(manifest), dict(Counter(item["reason"] for item in rejections)), output_dir / "pairs.csv")
    return output_dir / "pairs.csv"


def import_sen12ms(root_dir: str | Path, scene_csv: str | Path) -> Path:
    """Index extracted official SEN12MS patches using author scene/patch identifiers.

    Acquisition times and pixel clouds are unknown in the standard distribution;
    supply verified metadata or explicitly opt into author pairs without a time
    check. This importer never infers an acquisition date from a season name.
    """
    root_dir, scene_csv = Path(root_dir).resolve(), Path(scene_csv).resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"SEN12MS extraction directory does not exist: {root_dir}")
    pattern = re.compile(r"(?P<season>ROIs\d+_[a-z]+)_(?P<sensor>s1|s2)_(?P<roi>\d+)_p(?P<patch>\d+)\.tif$", re.IGNORECASE)
    found: dict[str, dict[str, Path]] = {}
    for path in sorted(root_dir.rglob("*.tif")):
        match = pattern.fullmatch(path.name)
        if match:
            key = f"{match['season']}_{match['roi']}_p{match['patch']}"
            if match["sensor"].lower() in found.setdefault(key, {}):
                raise ValueError(f"Duplicate SEN12MS patch identifier {key}: {path}")
            found[key][match["sensor"].lower()] = path
    complete = {key: items for key, items in found.items() if {"s1", "s2"}.issubset(items)}
    if not complete:
        raise ValueError(f"No author-matched SEN12MS S1/S2 GeoTIFFs found in {root_dir}. "
                         "Extract matching seasonal s1 and s2 archives; see docs/data_sources.md.")
    records = []
    optical_names = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
    for key, items in complete.items():
        for modality, sensor in [("s1", "sar"), ("s2", "optical")]:
            inspect_raster(items[modality], list(range(1, 3 if sensor == "sar" else 14)))
            records.append({"scene_id": f"{key}_{modality}", "sensor": sensor, "path": str(items[modality]),
                            "timestamp": "", "source": "sen12ms", "pairing_key": key,
                            "band_names": json.dumps(["VV", "VH"] if sensor == "sar" else optical_names),
                            "cloud_mask_path": "", "land_cover": "unknown", "sar_units": "db",
                            "reflectance_scale": 10000, "reflectance_offset": 0,
                            "processing_level": "L1C" if sensor == "optical" else "calibrated_GRD",
                            "license": "CC-BY; verify official record terms", "attribution": "Schmitt et al. (2019), SEN12MS"})
    scene_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(scene_csv, index=False)
    logger.warning("SEN12MS indexed author_pairs=%d incomplete=%d at %s; acquisition times/cloud masks unknown",
                   len(complete), len(found) - len(complete), scene_csv)
    return scene_csv
