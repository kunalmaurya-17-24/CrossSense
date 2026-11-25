"""Geospatial metadata, nodata, masks and boundary regressions."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from crosssense.data.preprocess import (
    inspect_raster,
    normalize_optical,
    normalize_sar,
    patch_windows,
    projected_bounds,
    read_on_grid,
)


def make_raster(path, data, crs="EPSG:32643", nodata=-9999):
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
                       count=data.shape[0], dtype="float32", crs=crs,
                       transform=from_origin(450000, 1450000, 10, 10), nodata=nodata) as dst:
        dst.write(data.astype(np.float32))
    return path


def test_inspection_masks_metadata_and_missing_bands(tmp_path):
    data = np.ones((2, 8, 8), dtype=np.float32)
    data[:, 0, 0] = -9999
    path = make_raster(tmp_path / "sar.tif", data)
    info = inspect_raster(path, [1, 2])
    assert info.crs == "EPSG:32643" and info.resolution == (10, 10)
    assert info.invalid_fraction == pytest.approx(1 / 64)
    with pytest.raises(ValueError, match="requested bands"):
        inspect_raster(path, [1, 3])


def test_malformed_raster_and_no_crs(tmp_path, caplog):
    broken = tmp_path / "broken.tif"
    broken.write_bytes(b"not a raster")
    with pytest.raises(ValueError, match="Cannot inspect raster"):
        inspect_raster(broken)
    assert "Raster validation failed" in caplog.text
    assert any(record.exc_info for record in caplog.records)
    path = make_raster(tmp_path / "no_crs.tif", np.ones((1, 8, 8)), crs=None)
    with pytest.raises(ValueError, match="missing CRS"):
        inspect_raster(path)


def test_common_grid_resampling_preserves_nodata(tmp_path):
    data = np.ones((2, 8, 8), dtype=np.float32)
    data[:, 2:4, 2:4] = -9999
    path = make_raster(tmp_path / "r.tif", data)
    info = inspect_raster(path)
    arr, valid = read_on_grid(path, [1, 2], info.crs, info.bounds, 5)
    assert arr.shape == (2, 16, 16) and valid.shape == (16, 16)
    assert not valid[5, 5] and valid[0, 0]
    with pytest.raises(ValueError, match="projected metres"):
        projected_bounds(info, "EPSG:4326")


def test_normalization_nan_linear_units_and_clouds():
    arr = np.array([[[1., 0.], [np.nan, .1]], [[1., 1.], [.1, .1]]])
    sar, mask = normalize_sar(arr, np.ones((2, 2), dtype=bool), units="linear")
    assert np.isfinite(sar).all() and not mask[0, 1] and not mask[1, 0]
    assert sar[0, 0, 0] == pytest.approx(30 / 35)
    with pytest.raises(ValueError, match="raw digital numbers"):
        normalize_sar(arr, mask, units="DN")
    optical, mask = normalize_optical(np.full((4, 2, 2), 5000), np.ones((2, 2), bool),
                                      cloud_mask=np.ones((2, 2), bool))
    assert not mask.any() and not optical.any()
    offset_data, _ = normalize_optical(np.full((1, 2, 2), 5000), np.ones((2, 2), bool), offset=-1000)
    assert offset_data[0, 0, 0] == pytest.approx(.4)


def test_patch_boundaries_exclude_partial_edges():
    assert patch_windows(65, 70, 32, 32) == [(0, 0), (0, 32), (32, 0), (32, 32)]
    assert patch_windows(16, 16, 32, 32) == []
    with pytest.raises(ValueError, match="positive"):
        patch_windows(32, 32, 0, 1)


def test_geographic_source_reprojects_to_metric_grid(tmp_path):
    path = tmp_path / "geographic.tif"
    with rasterio.open(path, "w", driver="GTiff", height=8, width=8, count=1,
                       dtype="float32", crs="EPSG:4326", nodata=-9999,
                       transform=from_origin(73.8, 13.1, .0001, .0001)) as dst:
        dst.write(np.ones((1, 8, 8), dtype=np.float32))
    info = inspect_raster(path)
    metric = projected_bounds(info, "EPSG:32643")
    arr, mask = read_on_grid(path, [1], "EPSG:32643", metric, 10)
    assert arr.shape[1] > 0 and arr.shape[2] > 0 and mask.any()
    assert np.allclose(arr[:, mask], 1)
