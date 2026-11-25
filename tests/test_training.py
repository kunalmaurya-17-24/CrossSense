from pathlib import Path

import pytest
import torch

from crosssense.config import AppConfig
from crosssense.data.dataset import load_manifest
from crosssense.data.pairing import prepare_pairs
from crosssense.data.synthetic import create_synthetic_fixture
from crosssense.evaluation.reports import evaluate_experiment
from crosssense.models.training import (
    build_model,
    load_checkpoint,
    resolve_device,
    train_model,
    validate_checkpoint_holdout,
    validate_manifest_preprocessing,
    validate_preprocessing,
)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    output = tmp_path_factory.mktemp("training")
    config = AppConfig()
    config.data.patch_size = config.data.stride = 8
    config.model.width = 4
    config.model.embedding_dim = 8
    config.training.batch_size = 8
    config.training.epochs = 1
    scenes = create_synthetic_fixture(output / "scenes", config.data, config.seed)
    manifest = prepare_pairs(scenes, output / "pairs", config.data, config.seed)
    return output, config, manifest


def test_checkpoint_resume_matches_uninterrupted_cpu_and_reports(prepared):
    output, config, manifest = prepared
    first = train_model(config, manifest, output / "resumed")
    assert first.is_file()
    loaded = load_checkpoint(first, build_model(config))
    assert loaded["epoch"] == 1
    assert loaded["config"]["training"]["device"] == "cpu"
    config.training.epochs = 2
    train_model(config, manifest, output / "resumed", output / "resumed" / "last.pt")
    train_model(config, manifest, output / "continuous")
    resumed = load_checkpoint(output / "resumed" / "last.pt", build_model(config))
    continuous = load_checkpoint(output / "continuous" / "last.pt", build_model(config))
    assert resumed["epoch"] == continuous["epoch"] == 2
    for key in resumed["model"]:
        torch.testing.assert_close(resumed["model"][key], continuous["model"][key], rtol=0, atol=0)
    report = evaluate_experiment(config, manifest, output / "resumed" / "best.pt", output / "reports")
    assert report["data_label"] == "SYNTHETIC PIPELINE TEST"
    assert report["gallery_pairs"] > 1
    for method in ["statistical_baseline", "dual_encoder"]:
        assert set(report["methods"][method]) == {"optical_to_sar", "sar_to_optical"}
    for filename in ["metrics.json", "summary.csv", "retrievals.csv", "report.html", "retrieval_examples.png"]:
        assert (output / "reports" / filename).is_file()


def test_device_selection_and_missing_checkpoint():
    assert str(resolve_device("cpu")) == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="unavailable"):
            resolve_device("cuda")
    with pytest.raises(FileNotFoundError, match="Checkpoint missing"):
        load_checkpoint(Path("does-not-exist.pt"), build_model(AppConfig()))


def test_evaluation_preprocessing_mismatch(prepared):
    _, config, _ = prepared
    snapshot = config.to_dict()
    config.data.optical_scale *= 2
    try:
        with pytest.raises(ValueError, match="preprocessing mismatch.*optical_scale"):
            validate_preprocessing(config, snapshot)
    finally:
        config.data.optical_scale /= 2


def test_external_manifest_overlap_and_disjoint_geography(prepared):
    _, config, manifest = prepared
    frame = load_manifest(manifest)
    train = frame.loc[frame.split == "train"].copy()
    payload = {"manifest_sha256": "original", "fit_geography": train.to_dict("records")}
    test = train.copy()
    test["split"] = "test"
    test["pair_id"] = "external-" + test.pair_id
    with pytest.raises(ValueError, match="External evaluation overlaps"):
        validate_checkpoint_holdout(payload, test, "test", "different", config.data.split_buffer_m)
    disjoint = frame.loc[frame.split == "test"].copy()
    validate_checkpoint_holdout(payload, disjoint, "test", "different", config.data.split_buffer_m)


@pytest.mark.parametrize("field,value", [("optical_clip", [0., .8]), ("sar_clip_db", [-20., 5.]),
                                         ("patch_size", 16), ("optical_bands", [2, 3, 4, 7])])
def test_prepared_manifest_rejects_stale_config(prepared, field, value):
    _, config, manifest = prepared
    frame = load_manifest(manifest)
    previous = getattr(config.data, field)
    setattr(config.data, field, value)
    try:
        with pytest.raises(ValueError, match=f"Prepared manifest preprocessing mismatch.*{field}"):
            validate_manifest_preprocessing(config, frame)
        with pytest.raises(ValueError, match="Prepared manifest preprocessing mismatch"):
            train_model(config, manifest, manifest.parent / "should-not-train")
    finally:
        setattr(config.data, field, previous)


def test_prepared_manifest_missing_provenance_or_inconsistent_band_order(prepared):
    _, config, manifest = prepared
    frame = load_manifest(manifest)
    frame.attrs.pop("preprocessing_config")
    with pytest.raises(ValueError, match="provenance missing"):
        validate_manifest_preprocessing(config, frame)
    frame = load_manifest(manifest)
    frame.loc[0, "optical_band_names"] = '["B4", "B3", "B2", "B8"]'
    with pytest.raises(ValueError, match="Inconsistent.*semantic channel order"):
        validate_manifest_preprocessing(config, frame)


def test_unknown_metadata_remains_explicit_in_report_strata(prepared):
    import numpy as np

    from crosssense.evaluation.reports import _stratify
    _, _, manifest = prepared
    frame = load_manifest(manifest).head(2).copy()
    frame["cloud_fraction"] = -1.
    frame["time_gap_days"] = -1.
    strata = _stratify(frame, np.ones(2, dtype=int))
    assert strata["cloud_fraction"]["unknown"]["queries"] == 2
    assert strata["time_gap_days"]["unknown"]["queries"] == 2
