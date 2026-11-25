# Local data and manifests

Do not commit imagery. `raw/`, `synthetic/`, `processed/`, `downloads/`, arrays and rasters are ignored. Small catalogue/acquisition metadata and the sanitized bounded-download audit belong in `manifests/`. `samples/` reserves a local fixture location; fixture files are generated, not claimed to be real Sentinel data.

The runnable example is `crosssense synthetic --output data/synthetic`, followed by `crosssense build-pairs --scenes data/synthetic/scenes.csv`. It writes a scene CSV with paths relative to its directory. Existing data requires a CSV with one row per sensor scene and these columns:

| Field | Meaning |
|---|---|
| `scene_id`, `sensor`, `path` | Unique stable ID, `sar` or `optical`, local georeferenced raster |
| `timestamp`, `source` | Timezone-aware ISO time; source `sen12ms`, `copernicus_prepared`, or `synthetic` |
| `band_names` | JSON ordered band names; SAR must provide VV/VH for the selected channels |
| `cloud_mask_path` | Optional georeferenced single-band binary mask, nonzero means invalid/cloud; convert S2 SCL into a binary mask first |
| `land_cover` | Optional category; unknown remains unknown |
| `sar_units` | `db` for SEN12MS; declared `db` or calibrated `linear` for prepared Copernicus |
| `calibrated`, `terrain_corrected` | Required true for prepared Copernicus SAR |
| `processing_level`, `reflectance_scale`, `reflectance_offset` | Prepared optical must be L2A, with explicit source-specific `(DN+offset)/scale` |
| `pairing_key` | SEN12MS author matching key when exact times are unavailable |
| `product_id`, `tile_id`, `orbit_direction`, `relative_orbit`, `license`, `attribution` | Source provenance when available |

Example prepared optical band ordering can be `["B2","B3","B4","B8"]`. Band selections are 1-based indices into each local raster. For a prepared four-band raster in this order, set data.optical_bands=[1,2,3,4] instead of the 13-band SEN12MS default [2,3,4,8]. Selected semantic band names are preserved and verified against model checkpoints. Check `pairing.py` and `configs/default.yaml` before changing ordering. All rasters need a CRS, transform and nodata/mask values. Metadata that describes raw, uncalibrated GRD is not sufficient to convert it into calibrated imagery.

Pair output paths resolve relative to `pairs.csv`. Normalized float32 arrays are `C,H,W`; shared validity masks are boolean `H,W`. The manifest records common projected CRS, resolution, pixel bounds, centroids, source IDs, times/gap, cloud and validity fractions, normalization settings, bands, spatial component and split. Rejected candidates remain in `rejections.csv`. All patch arrays have checksums/byte counts in `files.json`. Changes in preprocessing should use a fresh output directory to avoid stale patch files; manifests determine which files are used.

A full three-way experiment needs at least three independent spatial components. A tiny single-location real subset may be useful for inspection but is insufficient for valid train/validation/test holdouts. For SEN12MS choose separated ROIs; for India expand the study beyond one contiguous AOI before claiming held-out generalization.
