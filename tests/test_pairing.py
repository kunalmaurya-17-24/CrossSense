"""Scene acceptance, rejection, source provenance and synthetic end-to-end pairing."""

import json
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import rasterio

from crosssense.config import DataConfig
from crosssense.data.dataset import load_manifest
from crosssense.data.pairing import import_sen12ms, overlap_fraction, prepare_pairs, read_scene_manifest
from crosssense.data.synthetic import create_synthetic_fixture
from crosssense.evaluation.splits import validate_splits


@pytest.fixture
def scenes(tmp_path):
    cfg = DataConfig()
    return create_synthetic_fixture(tmp_path / "raw", cfg), cfg


def test_matching_scenes_create_deterministic_pairs(scenes, tmp_path):
    csv, cfg = scenes
    manifest = prepare_pairs(csv, tmp_path / "pairs", cfg)
    frame = load_manifest(manifest)
    assert len(frame) == 36 and set(frame.source) == {"synthetic"}
    assert frame.time_gap_days.eq(2).all() and frame.time_verified.all()
    assert json.loads(frame.iloc[0].sar_band_names) == ["VV", "VH"]
    assert json.loads(frame.iloc[0].optical_band_names) == ["B2", "B3", "B4", "B8"]
    assert set(frame.split) == {"train", "validation", "test"}
    validate_splits(frame, buffer_m=cfg.split_buffer_m)
    assert overlap_fraction((0, 0, 10, 10), (5, 0, 15, 10)) == .5
    assert overlap_fraction((0, 0, 10, 10), (10, 0, 20, 10)) == 0
    repeated = load_manifest(prepare_pairs(csv, tmp_path / "again", cfg))
    assert repeated.pair_id.to_list() == frame.pair_id.to_list()
    files = json.loads((manifest.parent / "files.json").read_text())
    assert len(files["files"]) == 108 and files["total_bytes"] > 0
    prep = json.loads((manifest.parent / "preprocessing_config.json").read_text())
    assert prep["data"]["optical_clip"] == [0, 1] and prep["seed"] == 42


def test_time_mismatch_rejected(scenes, tmp_path):
    csv, cfg = scenes
    manifest = prepare_pairs(csv, tmp_path / "time_gap", replace(cfg, max_time_gap_days=1))
    assert pd.read_csv(manifest).empty
    rejects = pd.read_csv(manifest.parent / "rejections.csv")
    assert "time_gap" in set(rejects.reason)


def test_all_cloud_and_nodata_rejected(scenes, tmp_path):
    csv, cfg = scenes
    scene_frame = pd.read_csv(csv)
    for cloud in scene_frame.loc[scene_frame.sensor == "optical", "cloud_mask_path"]:
        with rasterio.open(csv.parent / cloud, "r+") as dst:
            dst.write(np.ones((1, dst.height, dst.width), dtype=np.float32))
    manifest = prepare_pairs(csv, tmp_path / "cloudy", cfg)
    assert pd.read_csv(manifest).empty
    assert "cloud_fraction" in set(pd.read_csv(manifest.parent / "rejections.csv").reason)
    # Clearing clouds but removing SAR pixels triggers the independent validity filter.
    for cloud in scene_frame.loc[scene_frame.sensor == "optical", "cloud_mask_path"]:
        with rasterio.open(csv.parent / cloud, "r+") as dst:
            dst.write(np.zeros((1, dst.height, dst.width), dtype=np.float32))
    for sar_path in scene_frame.loc[scene_frame.sensor == "sar", "path"]:
        with rasterio.open(csv.parent / sar_path, "r+") as dst:
            dst.write(np.full((2, dst.height, dst.width), -9999, dtype=np.float32))
    manifest = prepare_pairs(csv, tmp_path / "nodata", cfg)
    assert pd.read_csv(manifest).empty
    assert "valid_fraction" in set(pd.read_csv(manifest.parent / "rejections.csv").reason)


def test_missing_band_polarization_and_raw_grd_errors(scenes):
    csv, cfg = scenes
    frame = pd.read_csv(csv, keep_default_na=False)
    frame.loc[0, "band_names"] = '["VV"]'
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="missing/duplicate band"):
        read_scene_manifest(csv, cfg)
    frame.loc[0, "band_names"] = '["HH", "HV"]'
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="missing polarization"):
        read_scene_manifest(csv, cfg)
    frame.loc[0, "band_names"] = '["VV", "VH"]'
    frame.loc[0, "source"] = "copernicus_prepared"
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="raw Sentinel-1 GRD"):
        read_scene_manifest(csv, cfg)


def test_unknown_time_never_silently_satisfies_window(scenes, tmp_path):
    csv, cfg = scenes
    frame = pd.read_csv(csv, keep_default_na=False)
    frame["timestamp"] = ""
    frame.to_csv(csv, index=False)
    manifest = prepare_pairs(csv, tmp_path / "unknown_time", cfg)
    assert pd.read_csv(manifest).empty
    assert "missing_acquisition_time" in set(pd.read_csv(manifest.parent / "rejections.csv").reason)


def test_no_scene_files_explains_preparation(tmp_path):
    with pytest.raises(FileNotFoundError, match="docs/data_sources.md"):
        prepare_pairs(tmp_path / "absent.csv", tmp_path / "out", DataConfig())


def test_sen12ms_import_requires_explicit_unknown_time_optin(scenes, tmp_path):
    csv, cfg = scenes
    official_layout = tmp_path / "sen12ms"
    official_layout.mkdir()
    # Test the author's filename grammar using synthetic GeoTIFF contents.
    for region in range(9):
        for sensor, code in [("sar", "s1"), ("optical", "s2")]:
            filename = f"ROIs1158_spring_{code}_{region}_p1.tif"
            shutil.copyfile(csv.parent / f"synthetic_region{region}_{sensor}.tif", official_layout / filename)
    scene_csv = import_sen12ms(official_layout, tmp_path / "sen12ms_scenes.csv")
    imported = pd.read_csv(scene_csv, keep_default_na=False)
    assert imported.timestamp.eq("").all() and imported.pairing_key.nunique() == 9
    rejected = prepare_pairs(scene_csv, tmp_path / "strict", cfg)
    assert pd.read_csv(rejected).empty
    assert "missing_acquisition_time" in set(pd.read_csv(rejected.parent / "rejections.csv").reason)
    opted = load_manifest(prepare_pairs(scene_csv, tmp_path / "opted",
                                       replace(cfg, allow_author_pairs_without_time=True)))
    assert len(opted) == 36 and opted.time_gap_days.eq(-1).all()
    assert not opted.time_verified.any() and opted.cloud_fraction.eq(-1).all()
    # A malicious overlapping extra SAR author key must not pair to another key.
    extra = imported.loc[imported.sensor == "sar"].iloc[0].copy()
    extra["scene_id"], extra["pairing_key"] = "different-author-key", "unmatched"
    pd.concat([imported, extra.to_frame().T], ignore_index=True).to_csv(scene_csv, index=False)
    with_extra = load_manifest(prepare_pairs(scene_csv, tmp_path / "keys",
                                            replace(cfg, allow_author_pairs_without_time=True)))
    assert len(with_extra) == 36
    assert "author_pair_key_mismatch" in set(pd.read_csv(tmp_path / "keys" / "rejections.csv").reason)
