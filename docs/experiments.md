# Executed experiments and verification

These are measured development results from 2026-11-25. The workspace initially contained no files or Git repository. No existing implementation was overwritten. All model results below are **SYNTHETIC PIPELINE TESTS**; no real satellite raster training/evaluation occurred.

## Environment and commands

Tested on Windows with Python 3.12.16, PyTorch 2.10.0+cpu, Rasterio 1.4.4, CPU. CUDA is unavailable. `requirements-tested.txt` records the installed versions; `pyproject.toml` contains portable constraints. Both the editable package and development dependencies were actually installed. The Windows host sandbox generated a profile path that initially broke setuptools and prevented Jupyter localhost startup; installation/kernel execution outside that sandbox succeeded. These are execution-host restrictions, not extra project authentication requirements.

```bash
uv pip install -e ".[dev]"
crosssense --log-level ERROR smoke --output outputs/smoke --epochs 4
crosssense --log-level ERROR check-splits --manifest outputs/smoke/processed/pairs.csv
crosssense --help
python -m pytest -q --junitxml=outputs/tests.xml
python scripts/run_scenarios.py
python scripts/execute_notebooks.py --in-place
ruff check src tests scripts
```

The retrieval CLI was also actually exercised with a manifest test ID, the trained checkpoint and k=3, returning three ranked IDs, cosine scores and metadata. `scripts/build_pairs.py --help` was checked. Local logs, checkpoint weights, arrays, exact timing and reports remain under ignored `outputs/`; compact measured summaries are retained in [results/verification.json](results/verification.json) and [results/synthetic_summary.csv](results/synthetic_summary.csv).

## Data access actually attempted

The single official SEN12MS FTP attempt used the documented mediaTUM/author route and a joint 50 MiB, 90-second cap. The server reported 14,379,439,087 bytes for ROIs2017_winter_s1.tar.gz. It stopped before either archive body: **0 raster bytes and 0 retries**. The optical size was not queried after the SAR size failed. See the durable audit `data/manifests/sen12ms_attempt.json` and [source procedures](data_sources.md). No larger transfer was started. The official browser record also returned HTTP 429 during discovery.

The official CDSE catalogue query for Pune bbox [73.80,18.45,73.90,18.55], 2024-01-01 inclusive through 2024-01-16 exclusive returned four real product records: two optical acquisitions and original/COG forms of one SAR acquisition. This is a metadata candidate set, not four independent paired scenes or an acquired evaluation set. Product IDs, times, footprints, orbit/tile/level/cloud fields and source attribution are saved in `data/manifests/india_catalogue.json`. No CDSE pixels were downloaded or evaluated.

## Synthetic experiment

The fixture contains nine separated artificial regions, 18 sensor scenes and 36 accepted patches. Split counts: {'train': 20, 'validation': 8, 'test': 8}. The preparation recorded 72 rejected non-overlapping scene candidates. Seed=42, patch size=32, grid=10 m, shared embedding dimension=32, batch=8, AdamW lr=0.001, temperature=0.1. Four epochs completed; epoch 2 was selected using mean bidirectional validation Recall@1. Test data did not select the checkpoint.

| Method | Direction | Recall@1 | Recall@5 | Recall@10 | MRR | Search ms/query | Index bytes |
|---|---|---:|---:|---:|---:|---:|---:|
| statistical_baseline | optical_to_sar | 0.625 | 1.000 | 1.000 | 0.812500 | 0.02172 | 12567 |
| statistical_baseline | sar_to_optical | 0.625 | 1.000 | 1.000 | 0.791667 | 0.01722 | 12567 |
| dual_encoder | optical_to_sar | 0.250 | 1.000 | 1.000 | 0.531250 | 0.01671 | 13335 |
| dual_encoder | sar_to_optical | 0.250 | 1.000 | 1.000 | 0.541667 | 0.01625 | 13335 |

The gallery contains only eight items, so Recall@10 is automatically 1.0. Timings are noisy small-gallery exact-search wall times, excluding loading and image encoding. Index bytes include serialized metadata but exclude Python object overhead. The statistical baseline outperformed the four-epoch CNN in this fixture. The embedding plot shows persistent modality separation; failure triplets show incorrect texture matches. No accuracy improvement or real-world performance claim follows from this result.

Full ranks, baseline/model comparisons, conditional 95% query-bootstrap intervals (200 resamples), and available metadata strata are generated in `outputs/smoke/evaluation/metrics.json`, `query_metrics.csv`, `retrievals.csv`, `embeddings.png`, `retrieval_examples.png` and `report.html`. The observed test split contains only the synthetic_water generator label; it cannot establish urban/rural differences. All cloud values are zero and all acquisition gaps are two days. Cloud/time rejection behavior is tested separately; real cloud/season/noise effects have not been measured.

## Automated tests

**112 passed, 1 skipped, 0 failed** in the final full run. The skip is CUDA execution, unavailable on this host. Upstream Rasterio/Affine and NumPy deprecation warnings were present (not suppressed). The suite covers configuration, local raster metadata and real temporary GeoTIFF I/O, nodata/NaN/cloud masks, reprojection, channel/source rules, stable pairs and author-time opt-in, spatial leakage including mixed-CRS rejection, model/loss behavior, checkpoint resume equivalence, provenance mismatches, external evaluation leakage, exact toy retrieval/metrics, download safeguards/redaction, logging and CLI scenarios. No tests use real satellite pixels. Network unit tests use controlled responses; the real bounded acquisition audit is recorded separately.

CPU two-epoch resumed parameters were tested bit-for-bit against an uninterrupted two-epoch run. The interruption scenario observes recovery from a completed-epoch checkpoint; forced mid-write/process-kill durability is untested. An integration regression briefly exposed a missing socket import during concurrent cleanup; it was fixed, and the final suite contains the relevant no-retry regression test.

## Acceptance scenarios

Each row refers to an actually executed test via `python scripts/run_scenarios.py`. The runner writes command/return code, expected behavior, observed assertion status and elapsed time to `outputs/scenarios/results.json`. The table spells out what those assertions observed.

| Scenario / test | Expected behavior | Observed behavior | Outcome |
|---|---|---|---|
| 01: `test_scenario_01_happy_path_cpu` | Prepare, train, evaluate both directions and write a labelled synthetic report | Synthetic prepare/train/evaluate completed; 8 held-out gallery pairs, both model directions and HTML report; CLI returned ranked hits | PASS |
| 02: `test_scenario_02_no_data` | Missing scenes produce an actionable error and CLI status 2 | Missing scene CSV produced contextual message and CLI exit 2 | PASS |
| 03: `test_scenario_03_bad_configuration` | Invalid paths, thresholds, patch sizes and batch sizes are rejected | Invalid path, negative patch size, out-of-range overlap and zero batch size each raised ValueError | PASS |
| 04: `test_scenario_04_cloudy_optical` | All-cloud optical patches rejected with cloud_fraction reason | All-cloud fixtures produced zero accepted patches and cloud_fraction rejection reasons | PASS |
| 05: `test_scenario_05_time_mismatch` | Time gaps beyond seven days rejected with time_gap reason | March optical vs January SAR produced zero accepted patches and time_gap reasons | PASS |
| 06: `test_scenario_06_geographic_leakage` | Cross-split overlapping footprints rejected even with falsified block IDs | Overlapping copy with altered pair/block IDs failed the cross-split geography check | PASS |
| 07: `test_scenario_07_corrupt_raster_and_missing_band` | Corrupt raster and missing bands raise useful errors and write traceback | Corrupt TIFF and band 999 raised descriptive errors; full traceback present in log | PASS |
| 08: `test_scenario_08_cpu_only` | Small encoder forward pass works entirely on CPU | Both encoder outputs had shape [2,32] on CPU | PASS |
| 09: `test_scenario_09_gpu` | Small CUDA forward pass works if CUDA available; otherwise explicitly skipped | CUDA unavailable; no GPU computation observed | SKIP |
| 10: `test_scenario_10_interrupted_training` | One-epoch checkpoint resumes to epoch two with preserved history | Completed epoch 1 resumed to epoch 2; checkpoint epoch/history verified | PASS |
| 11: `test_scenario_11_empty_gallery` | Empty gallery rejected with a clear ValueError | Empty gallery raised a clear ValueError | PASS |
| 12: `test_scenario_12_unexpected_exception` | Unexpected exception writes traceback and yields concise CLI status 1 | Injected RuntimeError produced CLI exit 1, concise error and full traceback in rotating log | PASS |

## Executed notebooks

| Notebook | Executed code cells | Outcome |
|---|---:|---|
| `01_data_exploration.ipynb` | 4 | PASSED |
| `02_pair_generation.ipynb` | 4 | PASSED |
| `03_baseline_retrieval.ipynb` | 3 | PASSED |
| `04_model_training_and_evaluation.ipynb` | 4 | PASSED |

All four executed top-to-bottom using the project interpreter; 15 code cells executed. Computed tables and figures are retained in source notebooks. Console log streams are removed from source notebooks to avoid committing machine paths; full executed copies and timestamps remain in `outputs/notebooks/executed/execution_status.json`. They prepare their own labelled fixture and checkpoint in `outputs/notebook_demo/`. The baseline and model metrics match the independent CLI smoke run.

## Remaining work

Obtain real, geographically separated paired pixels within reviewed source terms, retain timestamp/cloud/processing provenance, run repeated training seeds and region-level uncertainty, and test an available CUDA GPU. The small contiguous India AOI by itself is insufficient for three independent geographic holdouts. See [limitations](limitations.md) for source differences, computational scaling and scientific interpretation. Re-run this record generator only after the tests, scenarios and notebooks have actually completed: `python scripts/record_experiments.py`.
