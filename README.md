# CrossSense

Cross-modal satellite patch retrieval: query optical imagery to retrieve SAR patches, or query SAR to retrieve optical imagery. A match means the **same geographic patch under the manifest pairing rule**, not pixel identity or image reconstruction.

The workspace was empty before implementation. This repository now contains a reusable Python package, four connected notebooks, a CLI, geographic leakage checks, tests, and measured synthetic experiments. **Real satellite training/evaluation has not occurred.** The bounded official SEN12MS attempt stopped before transfer because a single SAR archive was 14.38 GB, above the 50 MiB cap. A real CDSE metadata query returned four product records for Pune; rasters require further authenticated access and source-specific preparation. See [data sources](docs/data_sources.md), [measured experiments](docs/experiments.md), and [limitations](docs/limitations.md).

## Install

Use Python **3.11–3.13** (tested here: Python 3.12.16, Windows, CPU). A project environment with roughly 1 GB of free disk is sufficient for the smoke test; real archives require substantially more. Dependencies are constrained in `pyproject.toml`; `requirements-tested.txt` records the actual tested environment. No credentials are needed for synthetic or already-local data.

Windows PowerShell:

```powershell
uv venv .venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"
```

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

For a matching Python/platform, an exact replay can use `uv pip install -r requirements-tested.txt` followed by `uv pip install -e . --no-deps`. The snapshot is from Windows; portable version constraints remain the recommended cross-platform route. For CUDA install the appropriate PyTorch wheel using the [official selector](https://pytorch.org/get-started/locally/) and set `training.device: cuda` or `auto`. CUDA has **not** been tested on this host. Start with batch 8, 32-pixel inputs, width 16 and 32-dimensional embeddings. The model has about 31k parameters; larger images/batches still require measuring VRAM on the target GPU.

## Run the working synthetic experiment

Run all commands from the repository root after installation:

```bash
crosssense smoke --output outputs/smoke --epochs 4
```

This creates nine separated mock regions as GeoTIFFs, aligns 36 SAR/optical patch pairs, assigns spatial train/validation/test holdouts, trains the compact dual encoder on CPU, evaluates a statistical baseline and the selected checkpoint in both directions, and writes `outputs/smoke/evaluation/report.html`. Generated scenes and results explicitly identify themselves as **synthetic pipeline tests**. They establish software behavior only.

The equivalent steps expose each intermediate artifact:

```bash
crosssense synthetic --output data/synthetic
crosssense build-pairs --scenes data/synthetic/scenes.csv --output data/processed
crosssense check-splits --manifest data/processed/pairs.csv
crosssense train --manifest data/processed/pairs.csv --output outputs/train --epochs 4
crosssense evaluate --manifest data/processed/pairs.csv --checkpoint outputs/train/best.pt --output outputs/evaluation
```

Query an existing patch ID from the test portion of `pairs.csv`:

```bash
crosssense retrieve --manifest data/processed/pairs.csv --checkpoint outputs/train/best.pt --query-id PAIR_ID --direction optical-to-sar --k 5
```

Results contain IDs, cosine similarities and geographic metadata. Scores are **not calibrated probabilities**. The CLI embeds the chosen split and builds an exact in-memory index; the Python `CosineIndex` accepts arbitrary valid embeddings for reuse. A manifest query is the implemented CLI interface; a standalone new-image query needs matching preprocessing plus the Python encoder API.

Resume from the last saved epoch (epochs is the **total** target, not extra epochs):

```bash
crosssense train --manifest data/processed/pairs.csv --output outputs/train --resume outputs/train/last.pt --epochs 6
```

A graceful process interruption preserves the last fully saved epoch. Mid-batch recovery is not implemented. Only load checkpoints you trust because optimizer/RNG checkpoint files use Python serialization.

## Prepare real data

The exact procedures, terms, attribution, metadata schema, and manual steps after the bounded download stop are in [docs/data_sources.md](docs/data_sources.md).

```bash
crosssense download-sen12ms --max-mib 50 --time-limit 90
crosssense import-sen12ms --root data/raw/SEN12MS --output data/manifests/sen12ms_scenes.csv
crosssense catalog-india --output data/manifests/india_catalogue.json
```

The downloader preflights sizes and disk space, enforces joint byte/time limits, audits failures, and refuses to repeat a previous attempt in the same output directory. The already-recorded official attempt is in `data/manifests/sen12ms_attempt.json`. Do not increase the download cap or fetch a full archive without deciding to do so explicitly.

SEN12MS author filenames establish paired patches but standard files do not provide reliable acquisition timestamps or a pixel cloud QA mask. By default unknown acquisition times are rejected. To use author pairing without a verified time threshold, copy `configs/default.yaml`, set `data.allow_author_pairs_without_time: true`, and keep the limitation in your experiment. `time_verified=false`, `time_gap_days=-1`, `cloud_mask_available=false` and `cloud_fraction=-1` remain inspectable. Set `require_cloud_mask: true` to reject unknown cloud QA. Never infer exact acquisition dates from season names.

```bash
crosssense --config configs/sen12ms.yaml build-pairs --scenes data/manifests/sen12ms_scenes.csv --output data/processed/sen12ms
```

Prepared Copernicus scenes use a CSV containing local calibrated, terrain-corrected SAR and reflectance optical GeoTIFFs plus binary cloud masks. **Raw GRD measurement TIFFs are not calibrated backscatter** and are rejected as a source. Use documented SNAP processing outside this package; encode the provenance, polarization order, and L2A reflectance offsets explicitly. Global SEN12MS needs an appropriate per-study projected CRS; the default EPSG:32643 is intended for the small India study/fixture, not the whole world. At least three separated spatial components are needed for a full three-way split. See [data/README.md](data/README.md).

## Configuration and outputs

`configs/default.yaml` configures the common grid, channel selections, normalization, quality/time thresholds, spatial blocks and exclusion distance, model dimensions, freeze policy, batch size, seed, device and log location. Unknown settings and invalid values fail before computation. Paths in configuration are resolved from the working directory; scene/patch paths in manifests resolve from the manifest directory. Source metadata can supply reflectance scale/offset and SAR units; these are recorded per pair.

- `data/processed/pairs.csv`: stable pair IDs, source scenes, arrays, coordinates, quality, time and split metadata.
- `data/processed/rejections.csv`: discarded scene/patch pairs and reasons.
- `data/processed/files.json`: patch file sizes and SHA-256 checksums.
- `outputs/train/`: resolved config, software versions, epoch history, `best.pt`, `last.pt`.
- `outputs/evaluation/`: metrics JSON, summary/query/ranked-result CSVs, embedding/failure PNGs, HTML report.
- `outputs/logs/`: rotating JSON event logs with run IDs and exception traces.

Datasets, weights, runtime environments, logs and generated reports are ignored by Git. Small credential-free source manifests and measured experiment summaries are included.

## Notebooks and modules

The four notebooks are narratives around the package, runnable top-to-bottom from the repository or notebook directory. They show paired images, metadata, rejection decisions, baseline rankings, learned embeddings and observed failures. They use the synthetic fixture unless replaced by a prepared local manifest. Notebook result cells are produced only by execution.

```bash
python scripts/execute_notebooks.py
jupyter lab notebooks
```

Reusable implementation lives in `src/crosssense/`: `data/` handles access, inspection, normalization, pairing and datasets; `models/` implements the compact dual CNN, geographic InfoNCE and resumable training; `retrieval/` implements statistical features and cosine search; `evaluation/` implements splits, metrics, uncertainty and reports. `cli.py` connects them. `scripts/prepare_data.py`, `build_pairs.py`, `train.py`, and `evaluate.py` are thin CLI adapters. No Earth Engine dependency is needed for local data; optional catalog/EE routes are described separately.

## Verify and troubleshoot

```bash
crosssense --help
python -m pytest -q
python scripts/run_scenarios.py
```

Tests cover rasters, masks, time/overlap pairing, bad metadata/configuration, spatial leakage, loss/encoder shapes, exact toy retrieval/metrics, checkpoint resume equivalence, logging, CLI and synthetic end-to-end behavior. The scenario runner writes actual outcomes to `outputs/scenarios/results.json`. [docs/experiments.md](docs/experiments.md) records commands and observed outcomes, including unavailable GPU testing.

Logs go to `outputs/logs/<run-id>.log`, rotate at 5 MB with three backups, and also stream to the console. Increase detail with:

```bash
crosssense --log-level DEBUG --run-id diagnose build-pairs --scenes data/synthetic/scenes.csv
```

Use `crosssense inspect PATH.tif` to inspect a suspect raster. A missing manifest error points to preparation instructions; an empty pair manifest means inspect `rejections.csv` for time, cloud, valid-pixel, author-key or overlap filters. A split error often means the AOI is too contiguous for independent holdouts; obtain separated areas instead of random patch splitting. Expected input/configuration errors and stopped data acquisition return CLI status 2; unexpected errors return 1 and write a full traceback. No exception is silently skipped. Credentials and tokens must stay in environment variables, never in manifests, commands pasted into issues, or logs.

## Research interpretation and license

The measured smoke experiment does not show that learned representations outperform a baseline, identify land cover, reconstruct pixels or generalize across sensors. See [methodology](docs/methodology.md) for equations and pairing definitions, and [limitations](docs/limitations.md) for what remains before a real-data study.

Code is under the [MIT license](LICENSE). Third-party data remains under its own terms; the code license grants no dataset rights. Cite Schmitt et al. (2019), *SEN12MS – A Curated Dataset of Georeferenced Multi-spectral Sentinel-1/2 Imagery for Deep Learning and Data Fusion*, DOI [10.5194/isprs-annals-IV-2-W7-153-2019](https://doi.org/10.5194/isprs-annals-IV-2-W7-153-2019), and attribute Copernicus Sentinel products according to their legal notice. See `CITATION.cff` and the source documentation.
