"""Independent tests for block, footprint and duplicate split leakage."""

import pandas as pd
import pytest

from crosssense.evaluation.splits import assign_geographic_splits, validate_splits


def spatial_frame():
    return pd.DataFrame([{"pair_id": f"p{i}", "block_id": f"b{i}", "split": "train", "crs": "EPSG:32643",
                          "minx": i * 30000., "miny": 1000., "maxx": i * 30000. + 320, "maxy": 1320.,
                          "x": i * 30000. + 160, "y": 1160.} for i in range(6)])


def test_deterministic_spatial_assignment():
    frame = spatial_frame()
    a = assign_geographic_splits(frame, seed=42)
    b = assign_geographic_splits(frame, seed=42)
    pd.testing.assert_frame_equal(a, b)
    assert set(a.split) == {"train", "validation", "test"}
    validate_splits(a)


def test_duplicate_pair_and_block_leakage_detected():
    frame = spatial_frame()
    frame.loc[1, "pair_id"] = frame.loc[0, "pair_id"]
    with pytest.raises(ValueError, match="Duplicate pair"):
        validate_splits(frame)
    frame = spatial_frame()
    frame.loc[1, "block_id"] = frame.loc[0, "block_id"]
    frame.loc[1, "split"] = "test"
    with pytest.raises(ValueError, match="block leakage"):
        validate_splits(frame)


def test_boundary_overlaps_cannot_cross_splits():
    frame = spatial_frame()
    # Same location with a dishonest different block identifier must still fail.
    frame.loc[1, ["minx", "maxx"]] = [200., 520.]
    frame.loc[1, "split"] = "test"
    with pytest.raises(ValueError, match="Geographic leakage"):
        validate_splits(frame)
    assigned = assign_geographic_splits(frame)
    assert assigned.loc[0, "block_id"] == assigned.loc[1, "block_id"]
    assert assigned.loc[0, "split"] == assigned.loc[1, "split"]


def test_missing_and_insufficient_spatial_metadata():
    with pytest.raises(ValueError, match="missing columns"):
        validate_splits(pd.DataFrame({"pair_id": ["x"]}))
    with pytest.raises(ValueError, match="at least 3"):
        assign_geographic_splits(spatial_frame().iloc[:2])


def test_mixed_and_geographic_crs_fail_closed():
    frame = spatial_frame()
    frame.loc[0, "crs"] = "EPSG:32644"
    with pytest.raises(ValueError, match="common projected metre CRS"):
        validate_splits(frame)
    frame["crs"] = "EPSG:4326"
    with pytest.raises(ValueError, match="projected metre CRS"):
        validate_splits(frame)
