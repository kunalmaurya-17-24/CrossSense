"""Manifest-backed normalized paired patches, shared by scripts and notebooks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
REQUIRED_COLUMNS = {
    "pair_id", "sar_path", "optical_path", "mask_path", "block_id", "split", "crs", "x", "y",
    "minx", "miny", "maxx", "maxy", "sar_time", "optical_time", "time_gap_days", "cloud_fraction",
    "valid_fraction", "source", "land_cover",
}


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load and validate a CSV manifest, resolving local paths against its directory."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pair manifest not found: {path}. Run 'crosssense synthetic --output data/synthetic' "
                                "or prepare local real scenes; see docs/data_sources.md.")
    frame = pd.read_csv(path, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(frame)
    if missing:
        raise ValueError(f"Manifest {path} missing metadata columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Pair manifest {path} contains no accepted pairs. Inspect rejections.csv and logs.")
    if frame.pair_id.duplicated().any() or (frame.pair_id.astype(str).str.len() == 0).any():
        raise ValueError("Pair IDs must be unique and nonempty")
    for col in ["sar_path", "optical_path", "mask_path"]:
        frame[col] = frame[col].map(lambda item: str((path.parent / item).resolve()) if not Path(item).is_absolute()
                                   else str(Path(item).resolve()))
        missing_files = [item for item in frame[col] if not Path(item).is_file()]
        if missing_files:
            raise FileNotFoundError(f"Missing {col} file: {missing_files[0]}; restore/preprocess local data")
    for column in ["crs", "block_id", "split", "source"]:
        if (frame[column].astype(str).str.len() == 0).any():
            raise ValueError(f"Missing {column} metadata")
    for column in ["x", "y", "minx", "miny", "maxx", "maxy", "time_gap_days", "cloud_fraction", "valid_fraction"]:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Manifest contains invalid numeric {column} metadata") from exc
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Manifest contains missing/nonfinite {column} metadata")
    if not frame.valid_fraction.between(0, 1).all() or not ((frame.cloud_fraction == -1) | frame.cloud_fraction.between(0, 1)).all():
        raise ValueError("Manifest valid/cloud fractions must be in [0,1]; unknown cloud fraction is -1")
    for center, lower, upper in [("x", "minx", "maxx"), ("y", "miny", "maxy")]:
        if not np.allclose(frame[center], (frame[lower] + frame[upper]) / 2, rtol=0, atol=1e-3):
            raise ValueError(f"Manifest {center} coordinates do not match patch footprint centers")
    logger.info("Loaded manifest %s: pairs=%d split_sizes=%s sources=%s", path, len(frame),
                frame.split.value_counts().to_dict(), sorted(frame.source.unique()))
    frame.attrs["manifest_path"] = str(path)
    prep_config_path = path.parent / "preprocessing_config.json"
    if prep_config_path.is_file():
        import json
        frame.attrs["preprocessing_config"] = json.loads(prep_config_path.read_text(encoding="utf-8"))
    return frame


def load_pair_arrays(row: pd.Series | dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read normalized SAR/optical arrays and their common valid-pixel mask."""
    sar = np.load(row["sar_path"], allow_pickle=False).astype(np.float32)
    optical = np.load(row["optical_path"], allow_pickle=False).astype(np.float32)
    mask = np.load(row["mask_path"], allow_pickle=False).astype(bool)
    if sar.ndim != 3 or optical.ndim != 3 or mask.ndim != 2:
        raise ValueError(f"Malformed patch array dimensions for pair {row['pair_id']}")
    if sar.shape[1:] != mask.shape or optical.shape[1:] != mask.shape:
        raise ValueError(f"SAR/optical/mask shape mismatch for pair {row['pair_id']}")
    if not np.isfinite(sar).all() or not np.isfinite(optical).all() or not mask.any():
        raise ValueError(f"Nonfinite or all-invalid patch for pair {row['pair_id']}")
    return sar, optical, mask


class PairedPatchDataset(Dataset):
    """Load already normalized arrays without requiring a geospatial cloud account."""

    def __init__(self, manifest: str | Path | pd.DataFrame, split: str | None = None) -> None:
        frame = load_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest.copy()
        if split is not None:
            frame = frame.loc[frame.split == split]
        self.frame = frame.reset_index(drop=True)
        self.manifest = self.frame
        if self.frame.empty:
            raise ValueError(f"No paired patches in requested split {split!r}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        sar, optical, mask = load_pair_arrays(row)
        return {"sar": torch.from_numpy(sar), "optical": torch.from_numpy(optical),
                "mask": torch.from_numpy(mask),
                "coords": torch.tensor([row.x, row.y], dtype=torch.float32),
                "pair_id": str(row.pair_id), "block_id": str(row.block_id)}
