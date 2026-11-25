# Methodology

## Task and relevance

Given an optical patch, rank a gallery of SAR patches, and vice versa. The default benchmark defines exactly one relevant gallery item by the common `pair_id`. This operational label establishes geographical correspondence under a preparation rule; sensors observe different physical responses. It does not establish identical pixels, semantic equivalence, pixel reconstruction or image translation.

Other overlapping acquisitions can be unlabelled positives. The training exclusion mask reduces this false-negative problem, while exact-ID evaluation deliberately retains a narrow, inspectable label definition. F1@K is omitted because the collection does not establish exhaustive relevance labels. Metrics use all queries; missing labels or malformed arrays raise errors rather than silently dropping difficult cases.

## Source-aware preparation

`inspect_raster` verifies readability, CRS, transform, selected channels, dimensions, nodata and finite values, with a logged diagnostic summary. Source scene CSVs describe what physical quantities the raster contains.

| Source | SAR handling | Optical handling | Caveat |
|---|---|---|---|
| SEN12MS | Already geocoded calibrated VV/VH sigma0 in dB; clip [-30,5], map to [0,1] | 13-band L1C TOA values; default select B2/B3/B4/B8, divide by 10000, clip [0,1] | Exact times and cloud QA are not recovered from season names |
| Copernicus prepared | Require externally calibrated, terrain-corrected VV/VH sigma0; declared dB or positive linear values, linear converted using 10 log10 | Require L2A and explicit product-specific scale/offset; reflectance=(DN+offset)/scale | Raw GRD SAFE measurement values require SNAP or equivalent processing outside CrossSense |
| Synthetic | Artificial patterns mapped to the same configured numeric ranges | Artificial correlated multi-band patterns | Fixture assumptions create easier correspondence than real sensors |

Cloud masks must be binary invalid-pixel masks, with nonzero invalid. Raw SCL codes must be converted first, e.g. mark nodata/saturated/cloud-shadow/cloud/cirrus/snow classes according to the experiment. CrossSense does not automatically interpret all QA conventions. Missing QA is recorded as unknown (`cloud_fraction=-1`); use `require_cloud_mask` for strict experiments. Missing exact SEN12MS times reject candidates unless `allow_author_pairs_without_time` explicitly permits the identical author pair key; such pairs have `time_gap_days=-1` and `time_verified=false`. Those settings do not certify that the thresholds were met.

The user selects a regional projected metre CRS and resolution (default 10 m, EPSG:32643 for the India study). Footprints transform only when needed. Candidate overlap is intersection area divided by the smaller scene footprint. Default minimum is 0.9. Intersections are snapped inward to the common resolution. Rasterio reads/resamples both modalities onto this exact grid: bilinear continuous quantities, nearest-neighbor cloud/validity masks, excluding source nodata from interpolation. Whole-scene reading is suitable for small inputs; large SAFE scenes should first be cropped to the study AOI.

A positive candidate needs a timezone-aware time difference at most 7 days, sufficient overlap, both polarizations and configured bands. Full 32×32 patches at stride 32 are extracted; incomplete edge patches are dropped. Accepted patches have cloud fraction at most 0.2 when QA exists and joint valid fraction at least 0.8. SAR and optical use a common validity mask; invalid samples are set to zero after normalization, and the boolean mask is saved. CNNs receive normalized image channels, without a separate mask channel. Valid zero reflectance can consequently resemble filled pixels; quality filtering reduces this ambiguity but does not eliminate it.

Stable pair IDs hash source/scene IDs, CRS, grid location, patch size and resolution. Normalization and quality settings are stored in `preprocessing_config.json`, per-pair metadata, file checksums and experiment snapshots. Rejections include scene IDs, reason and available context in `rejections.csv`. Changing a normalization configuration requires rebuilding patches in a fresh output directory; IDs encode geographic sample identity, not the version of its pixel values.

## Geographic holdouts

Spatial cells use floor(projected centroid / block size), with default 10 km cells. A union-find graph merges every acquisition in the same cell and any footprints overlapping or within the 500 m buffer, including across cell boundaries. Entire connected components are assigned with a seeded permutation to train/validation/test at approximately 60/20/20 percent of **components**, not exact patch counts. A minimum of three independent components is required. IDs record geographic components; all acquisitions in a connected block stay together.

`validate_splits` independently checks pair-ID uniqueness, allowed split names, shared blocks, finite bounds/CRS and cross-split footprint distances. Training and evaluation call it before computation. For externally supplied evaluation manifests, checkpoint provenance also checks against the training/validation geography used to fit or select the model. A contaminated external gallery must not be accepted merely because its own internal split labels look valid.

Randomly splitting adjacent patches would expose the same roads, texture and overlapping pixels during training and testing. It can make location memorization look like cross-modal generalization. These checks intentionally fail on highly contiguous single-location datasets rather than creating a misleading split. The default CRS should be changed for another study region; globally distorted or mixed projected coordinates are not a substitute for a careful geodesic design. Mixed-CRS manifests and geographic degree coordinates are rejected; first prepare all samples on the same appropriate regional metre grid.

## Baseline and learned encoders

The non-trained baseline averages normalized channels and summarizes each patch with a common eight-dimensional intensity/quantile/edge feature vector, then normalizes it. This creates comparable feature dimensions without assuming that independently trained RGB embeddings align with SAR. Backscatter and reflectance distributions are still different quantities, so this is a weak, honest diagnostic baseline. It uses no geographic coordinates to retrieve matches.

The learned model has separate compact CNN branches for two SAR channels and four optical channels by default, a configurable width and shared embedding dimension (32). Each branch produces an L2-normalized vector. `freeze_backbones` freezes convolutional feature layers while leaving projection heads trainable. All default weights are initialized from scratch; freezing is a controlled option, not an undocumented pretrained feature extractor. The default architecture has 30,784 parameters. It is deliberately small enough for CPU smoke tests; batch size/image size are configurable for an 8 GB consumer GPU, but that hardware claim still requires a measured VRAM run.

For normalized embeddings zS and zO, Lij = dot(zSi,zOj)/temperature. The loss is the mean of cross entropy over SAR-to-optical logits and its transpose, with diagonal targets. Default temperature is 0.1. Other valid samples are negatives except geographic exclusions. For projected centre differences, rectangular patch extent is removed to estimate edge-to-edge gaps; off-diagonal overlapping or sufficiently nearby patches are masked to negative infinity. Direct coordinate differences avoid catastrophic squared-distance cancellation at large UTM coordinates. Positives remain unmasked. A batch containing only spatially excluded negatives has finite zero loss; users should sample multiple separated regions to provide learning signal. There is no region-balanced sampler yet.

AdamW, gradient clipping, configurable learning rate, fixed maximum epochs and validation Recall@1 early stopping are implemented. The best checkpoint maximizes the mean of the two validation Recall@1 directions; test data does not select epochs. The last checkpoint includes optimizer state, Python/NumPy/Torch/CUDA RNG, data-loader generator state, epoch/history, configuration, geographic provenance and data fingerprints. Resume continues after the last complete epoch and rejects incompatible objectives or input provenance. CPU resume equivalence is tested against uninterrupted training; this is not a claim of bit-identical CUDA execution.

## Retrieval, metrics and diagnostics

Exact cosine similarity ranks normalized vectors, using stable manifest ordering for ties. Top-k results contain ID, similarity and metadata. The index validates nonempty gallery, unique IDs, nonzero finite vectors, correct dimensions and metadata. There is no FAISS approximation or LLM in the core pipeline.

Recall@K is mean(positive rank <= K) over queries for K=1,5,10. MRR is mean(1/positive rank). When K exceeds gallery size, it covers the entire gallery; Recall@10 on an eight-item gallery is therefore trivial. Both directions are reported separately. Seeded percentile query bootstrap (200 resamples) provides conditional 95% intervals. Adjacent queries are correlated, so these intervals are not spatial-population or repeated-training-seed uncertainty estimates.

Latency is wall-clock mean of one exact query search with sorting and metadata return, after one warmup; it excludes raster loading and encoding. Encoding time is recorded separately, synchronized for CUDA when used. Index size counts float32 vectors plus serialized ID/metadata bytes, excludes Python object overhead and is not peak process memory. Small-gallery timing is noisy and not a scalability benchmark.

Reports include per-query positive rank and top-1 failure, full top-k scores/metadata, geographic/land-cover/cloud/time strata when available, a joint PCA diagnostic, and query/retrieved/manifest-positive image triplets. Unknown fields do not become known labels. Synthetic labels resembling urban/rural/water are generator categories only. Real urban/rural, cloud, noise, season and time-gap failure patterns require suitable real data and are not established by the smoke test.
