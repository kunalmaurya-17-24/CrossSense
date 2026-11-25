"""Manifest and array validation used at training/evaluation boundaries."""

import numpy as np
import pandas as pd
import pytest

from crosssense.config import DataConfig
from crosssense.data.dataset import PairedPatchDataset, load_manifest, load_pair_arrays
from crosssense.data.pairing import prepare_pairs
from crosssense.data.synthetic import create_synthetic_fixture


def test_dataset_channels_shapes_masks_and_missing_metadata(tmp_path):
    cfg = DataConfig()
    scenes = create_synthetic_fixture(tmp_path / "scenes", cfg)
    manifest = prepare_pairs(scenes, tmp_path / "pairs", cfg)
    frame = load_manifest(manifest)
    dataset = PairedPatchDataset(frame, split="train")
    sample = dataset[0]
    assert sample["sar"].shape == (2, 32, 32)
    assert sample["optical"].shape == (4, 32, 32)
    assert sample["coords"].shape == (2,) and sample["mask"].all()
    assert dataset.frame is dataset.manifest
    with pytest.raises(ValueError, match="No paired patches"):
        PairedPatchDataset(frame, split="unavailable")
    damaged = tmp_path / "damaged.csv"
    frame.drop(columns="crs").to_csv(damaged, index=False)
    with pytest.raises(ValueError, match="missing metadata columns"):
        load_manifest(damaged)
    frame.loc[0, "sar_path"] = str(tmp_path / "missing.npy")
    frame.to_csv(damaged, index=False)
    with pytest.raises(FileNotFoundError, match="Missing sar_path"):
        load_manifest(damaged)


def test_manifest_rejects_missing_numeric_metadata(tmp_path):
    cfg = DataConfig()
    scenes = create_synthetic_fixture(tmp_path / "scenes", cfg)
    manifest = prepare_pairs(scenes, tmp_path / "pairs", cfg)
    frame = pd.read_csv(manifest, keep_default_na=False)
    frame["x"] = frame["x"].astype(object)
    frame.loc[0, "x"] = ""
    frame.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="invalid numeric x|nonfinite x"):
        load_manifest(manifest)


def test_manifest_rejects_wrong_geographic_centers(tmp_path):
    cfg = DataConfig()
    scenes = create_synthetic_fixture(tmp_path / "scenes", cfg)
    manifest = prepare_pairs(scenes, tmp_path / "pairs", cfg)
    frame = pd.read_csv(manifest, keep_default_na=False)
    frame.loc[0, "x"] += 100
    frame.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="coordinates do not match"):
        load_manifest(manifest)


def test_pair_array_shape_and_nan_validation(tmp_path):
    paths = {"pair_id": "toy"}
    for name, arr in [("sar", np.ones((2, 8, 8))), ("optical", np.ones((4, 8, 8))), ("mask", np.ones((8, 8), bool))]:
        path = tmp_path / f"{name}.npy"
        np.save(path, arr)
        paths[f"{name}_path"] = str(path)
    load_pair_arrays(paths)
    np.save(paths["sar_path"], np.full((2, 8, 8), np.nan))
    with pytest.raises(ValueError, match="Nonfinite"):
        load_pair_arrays(paths)
    np.save(paths["sar_path"], np.ones((2, 4, 8)))
    with pytest.raises(ValueError, match="shape mismatch"):
        load_pair_arrays(paths)
