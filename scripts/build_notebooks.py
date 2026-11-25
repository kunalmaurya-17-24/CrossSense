"""Regenerate the four source notebooks from short, reviewable cell definitions.

This authoring helper does not fabricate execution outputs. Execute the actual
notebooks with scripts/execute_notebooks.py to produce observed result cells.
"""

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def markdown(text: str):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbformat.v4.new_code_cell(text.strip())


def setup(run_id: str):
    return code(f'''
from pathlib import Path
import json, sys, platform
from importlib.metadata import version
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists() and (p / "src/crosssense").exists())
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from crosssense.config import load_config, save_config
from crosssense.logging_config import configure_logging
from crosssense.data.synthetic import create_synthetic_fixture
from crosssense.data.pairing import prepare_pairs
from crosssense.data.dataset import load_manifest, load_pair_arrays, PairedPatchDataset
from crosssense.models.training import set_seed

config = load_config(ROOT / "configs/default.yaml")
config.training.device = "cpu"
set_seed(config.seed)
log_path = configure_logging(ROOT / "outputs/logs", "INFO", "notebook-{run_id}")
WORK = ROOT / "outputs/notebook_demo"
WORK.mkdir(parents=True, exist_ok=True)
save_config(config, WORK / "config.yaml")
scenes_path = WORK / "synthetic/scenes.csv"
if not scenes_path.exists():
    scenes_path = create_synthetic_fixture(WORK / "synthetic", config.data, config.seed)
manifest_path = WORK / "processed/pairs.csv"
if not manifest_path.exists():
    manifest_path = prepare_pairs(scenes_path, WORK / "processed", config.data, config.seed)
frame = load_manifest(manifest_path)
assert set(frame.source) == {{"synthetic"}}, "This default notebook is explicitly a synthetic demonstration"
display(Markdown("**SYNTHETIC PIPELINE TEST — no real satellite observations.**"))
display(pd.DataFrame([{{"python": platform.python_version(), "numpy": version("numpy"), "torch": version("torch"), "rasterio": version("rasterio"), "seed": config.seed, "device": config.training.device}}]))
display(config.to_dict())
''')


def write_notebook(name: str, cells: list) -> None:
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python (CrossSense)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    output = ROOT / "notebooks" / name
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(output)


def main() -> None:
    write_notebook("01_data_exploration.ipynb", [
        markdown('''
# 01 — Inspect sensors, geography and masks

**Question:** what enters the retrieval model, and what metadata supports a positive pair?

The default data are tiny generated GeoTIFFs with known geographic extents,
two SAR channels and thirteen optical channels. Their visual patterns and
land-cover names are synthetic labels. The setup calls reusable package
functions and records a seed, config and versions. It creates local fixtures
only when absent; it performs no network access.

For real imagery, follow `docs/data_sources.md`, build a local scene manifest,
and replace `manifest_path` deliberately while removing the synthetic assertion.
Never call a real-data result synthetic or a synthetic result satellite evidence.
'''),
        setup("01"),
        markdown('''
## Raster inspection and raw scene provenance

GeoTIFF metadata are inspected by `inspect_raster`; malformed files, missing CRS
and invalid band selections raise contextual errors. Raster values are not RGB
photographs: SAR uses calibrated dB here, while optical DN values use the explicit
source scale. Projection and masks matter before any image comparison.
'''),
        code('''
from dataclasses import asdict
from crosssense.data.preprocess import inspect_raster
scenes = pd.read_csv(scenes_path)
display(scenes[["scene_id", "sensor", "timestamp", "source", "band_names", "land_cover"]])
metadata = []
for row in scenes.head(2).itertuples():
    path = Path(row.path)
    path = path if path.is_absolute() else scenes_path.parent / path
    metadata.append({"sensor": row.sensor, **asdict(inspect_raster(path))})
display(pd.DataFrame(metadata))
'''),
        markdown('''
## Paired normalized patches and common validity

This plot shows the tensors actually loaded for training: VV/VH after dB clipping,
B4/B3/B2 as an RGB diagnostic, and the common mask. Optical display brightness
is multiplied by 2.5 only for visualization. Invalid model pixels are zero-filled;
the mask preserves their distinction from valid low values.
'''),
        code('''
sar, optical, mask = load_pair_arrays(frame.iloc[0])
fig, axes = plt.subplots(1, 4, figsize=(13, 3))
axes[0].imshow(np.ma.array(sar[0], mask=~mask), cmap="gray", vmin=0, vmax=1)
axes[1].imshow(np.ma.array(sar[1], mask=~mask), cmap="gray", vmin=0, vmax=1)
rgb = np.moveaxis(np.clip(optical[[2,1,0]] * 2.5, 0, 1), 0, -1)
rgb[~mask] = 0
axes[2].imshow(rgb)
axes[3].imshow(mask, cmap="gray", vmin=0, vmax=1)
for ax, title in zip(axes, ["Synthetic VV", "Synthetic VH", "Synthetic optical B4/B3/B2", "Common valid mask"]):
    ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
display(frame[["pair_id", "source", "crs", "x", "y", "time_gap_days", "cloud_fraction", "valid_fraction", "split"]].head())
'''),
        code('''
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
frame.valid_fraction.hist(ax=axes[0]); axes[0].set_title("Accepted valid fractions")
frame.cloud_fraction.hist(ax=axes[1]); axes[1].set_title("Accepted cloud fractions")
frame.groupby(["split", "land_cover"]).size().unstack(fill_value=0).plot.bar(ax=axes[2])
axes[2].set_title("Synthetic labels by split"); plt.tight_layout(); plt.show()
display(Markdown(f"**Observed in this execution:** {len(frame)} accepted synthetic pairs, SAR shape `{sar.shape}`, optical shape `{optical.shape}`, mask shape `{mask.shape}`. Logs: `{log_path}`. These observations validate the fixture and data path; they do not measure real-sensor retrieval quality."))
'''),
    ])
    write_notebook("02_pair_generation.ipynb", [
        markdown('''
# 02 — Reproduce pairing and spatial holdouts

**Question:** which scenes/patches satisfy the configured overlap, time, cloud
and validity rules, and can nearby geography leak between splits?

The same geographic grid defines a positive pair. This does not mean the sensor
images are pixel-identical. A default seven-day window, 90% smaller-scene overlap,
20% cloud cap and 80% common-valid threshold are configured outside the code.
The synthetic fixture has separated regions so all three spatial holdouts exist.
'''),
        setup("02"),
        code('''
from crosssense.evaluation.splits import validate_splits
manifest_path = prepare_pairs(scenes_path, WORK / "processed", config.data, config.seed)
frame = load_manifest(manifest_path)
validate_splits(frame, buffer_m=config.data.split_buffer_m)
rejections = pd.read_csv(WORK / "processed/rejections.csv")
display(frame[["pair_id", "block_id", "split", "sar_time", "optical_time", "time_gap_days", "cloud_fraction", "valid_fraction"]].head(12))
display(frame.groupby("split").agg(pairs=("pair_id", "size"), blocks=("block_id", "nunique")))
display(rejections.groupby("reason").size().rename("rejections").to_frame())
'''),
        markdown('''
## Geographic assignment

All acquisitions in a spatial component stay together. Footprint overlap or
proximity can merge blocks before assignment, including neighbors across a block
boundary. Random adjacent-patch splits would expose nearly identical ground in
training and test and can inflate retrieval scores. The validation below checks
blocks and footprints independently of how the assignments were generated.
'''),
        code('''
from matplotlib.patches import Rectangle
colors = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:green"}
fig, ax = plt.subplots(figsize=(9, 5))
for row in frame.itertuples():
    ax.add_patch(Rectangle((row.minx, row.miny), row.maxx-row.minx, row.maxy-row.miny, color=colors[row.split], alpha=.4))
for split, rows in frame.groupby("split"):
    ax.scatter(rows.x, rows.y, label=split, color=colors[split], s=12)
ax.autoscale(); ax.set(xlabel="Projected easting (m)", ylabel="Projected northing (m)", title="Synthetic spatial holdouts, EPSG:32643")
ax.legend(); plt.show()
'''),
        markdown('''
## Deliberate time mismatch and leakage failures

The following experiments intentionally violate a rule. Expected errors are
caught and displayed; an unexpected success raises an assertion. Rejection CSVs
preserve why candidates were dropped. These are actual computed outcomes on
fixtures, not statements that every source product was validated.
'''),
        code('''
import copy
strict = copy.deepcopy(config.data)
strict.max_time_gap_days = 0
strict_path = prepare_pairs(scenes_path, WORK / "strict_time", strict, config.seed)
strict_pairs = pd.read_csv(strict_path)
strict_rejections = pd.read_csv(WORK / "strict_time/rejections.csv")
assert len(strict_pairs) == 0
assert "time_gap" in set(strict_rejections.reason)
display(strict_rejections.groupby("reason").size().to_frame("count"))

leaky = frame.copy()
first_block = leaky.block_id.iloc[0]
members = leaky.index[leaky.block_id == first_block]
assert len(members) >= 2
leaky.loc[members[0], "split"] = "test" if leaky.loc[members[1], "split"] != "test" else "train"
try:
    validate_splits(leaky, config.data.split_buffer_m)
except ValueError as exc:
    display(Markdown(f"**Observed expected leakage rejection:** `{exc}`"))
else:
    raise AssertionError("Deliberately shared spatial block was not rejected")
display(Markdown(f"**Observed:** {len(frame)} default pairs; zero with a zero-day time window. All default split checks passed. Inspect `{WORK / 'processed/rejections.csv'}` for rejected candidates."))
'''),
    ])
    write_notebook("03_baseline_retrieval.ipynb", [
        markdown('''
# 03 — Measure an untrained statistical baseline

**Question:** how much retrieval signal comes from a small shared descriptor
without fitting any parameters? Mean, spread, quantiles and edge statistics are
computed from normalized per-modality channel averages. Equal feature dimension
does not establish physical SAR/optical alignment; this is a deliberately weak,
measurable baseline. It uses neither geography nor pair IDs as input features.

Evaluation uses only the geographic test split. Relevance is the exact manifest
pair ID. Nearby unlabelled positives may count as false matches under this rule.
'''),
        setup("03"),
        code('''
from crosssense.retrieval.search import statistical_features
from crosssense.retrieval.index import CosineIndex
from crosssense.evaluation.metrics import paired_ranks, metrics_from_ranks
from crosssense.evaluation.splits import validate_splits
validate_splits(frame, config.data.split_buffer_m)
dataset = PairedPatchDataset(frame, "test")
test_frame = dataset.frame
ids = test_frame.pair_id.tolist()
sar_vectors = np.stack([statistical_features(dataset[i]["sar"].numpy())[0] for i in range(len(dataset))])
optical_vectors = np.stack([statistical_features(dataset[i]["optical"].numpy())[0] for i in range(len(dataset))])
metadata = json.loads(test_frame.to_json(orient="records"))
rows = []
for direction, query, gallery in [("optical_to_sar", optical_vectors, sar_vectors), ("sar_to_optical", sar_vectors, optical_vectors)]:
    index = CosineIndex(gallery, ids, metadata)
    ranks = paired_ranks(index.similarities(query), ids, ids)
    rows.append({"direction": direction, "queries": len(ids), **metrics_from_ranks(ranks), "index_bytes": index.size_bytes})
display(pd.DataFrame(rows))
'''),
        markdown('''
## Ranked IDs, geographic metadata and failure examples

Cosine scores describe embedding similarity and are not calibrated match
probabilities. Every returned item includes its stable ID and manifest metadata.
The chosen example below is an observed top-1 failure if one exists; otherwise
it is explicitly identified as a success.
'''),
        code('''
index = CosineIndex(sar_vectors, ids, metadata)
ranks = paired_ranks(index.similarities(optical_vectors), ids, ids)
failed = np.flatnonzero(ranks > 1)
query_i = int(failed[0]) if len(failed) else 0
hits = index.search(optical_vectors[query_i], 5)[0]
display(pd.DataFrame([{ "rank": rank, "id": hit["id"], "cosine": hit["score"], "relevant": hit["id"] == ids[query_i], "block": hit["metadata"]["block_id"], "land_cover": hit["metadata"]["land_cover"] } for rank, hit in enumerate(hits, 1)]))
lookup = {pair_id: i for i, pair_id in enumerate(ids)}
_, query_optical, mask = load_pair_arrays(test_frame.iloc[query_i])
fig, axes = plt.subplots(1, len(hits)+1, figsize=(3*(len(hits)+1), 3))
rgb = np.moveaxis(np.clip(query_optical[[2,1,0]] * 2.5, 0, 1), 0, -1)
rgb[~mask] = 0
axes[0].imshow(rgb); axes[0].set_title("Synthetic optical query")
for ax, hit in zip(axes[1:], hits):
    sar, _, valid = load_pair_arrays(test_frame.iloc[lookup[hit["id"]]])
    ax.imshow(np.ma.array(sar[0], mask=~valid), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"cosine={hit['score']:.3f}\\n{'positive' if hit['id'] == ids[query_i] else 'non-relevant ID'}")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.show()
display(Markdown(f"**Observed:** {len(failed)}/{len(ids)} optical-to-SAR queries missed their exact positive at rank 1. Shown query positive rank: **{ranks[query_i]}**. These are synthetic fixture results only."))
'''),
    ])
    write_notebook("04_model_training_and_evaluation.ipynb", [
        markdown('''
# 04 — Train and compare a two-branch retrieval model

**Question:** does symmetric paired-data contrastive training change retrieval
on geographically held-out fixture patches relative to the statistical baseline?

Two compact CNN branches handle 2 SAR and 4 selected optical channels, return
32-dimensional L2-normalized vectors, and train with symmetric InfoNCE. Nearby
off-diagonal pairs are excluded from negatives using projected coordinates.
Model selection uses validation recall; the test split is used only here for
final comparison. A fixed four-epoch CPU schedule makes this runnable quickly.
There is no reconstruction, translation, LLM, or evaluated semantic task.
'''),
        setup("04"),
        code('''
from crosssense.models.training import train_model, build_model, load_checkpoint
from crosssense.evaluation.reports import evaluate_experiment
config.training.epochs = 4
config.validate()
checkpoint = WORK / "train/best.pt"
if not checkpoint.exists():
    checkpoint = train_model(config, manifest_path, WORK / "train")
else:
    display(Markdown(f"Reusing existing local checkpoint `{checkpoint}`. Remove `outputs/notebook_demo/train` deliberately to rerun training from the same seed."))
payload = load_checkpoint(checkpoint, build_model(config))
display({"checkpoint_format": payload["format_version"], "selected_epoch": payload["epoch"], "training_config": payload["config"], "manifest_sha256": payload["manifest_sha256"]})
history = pd.read_json(WORK / "train/history.json")
display(history)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
history.plot(x="epoch", y="loss", ax=axes[0], marker="o")
history.plot(x="epoch", y="validation_recall@1", ax=axes[1], marker="o")
axes[0].set_title("Observed training objective"); axes[1].set_title("Geographic validation recall")
plt.tight_layout(); plt.show()
'''),
        markdown('''
## Measured comparison in both retrieval directions

`evaluate_experiment` writes ranked retrieval CSVs, query ranks, JSON metrics,
HTML, embedding PCA and observed examples. Recall@K uses the exact positive ID;
MRR uses its full-gallery rank. Query-bootstrap intervals are conditional on this
gallery and do not represent spatial independence or repeated training seeds.
Latency excludes image loading/encoding and includes exact search/metadata return.
Small-gallery Recall@10 can be trivially 1 when the gallery has at most 10 items.
'''),
        code('''
report_dir = WORK / "evaluation"
report = evaluate_experiment(config, manifest_path, checkpoint, report_dir)
assert report["data_label"] == "SYNTHETIC PIPELINE TEST"
display(pd.read_csv(report_dir / "summary.csv"))
display({"label": report["data_label"], "pairs": report["gallery_pairs"], "selected_epoch": report["selected_epoch"], "uncertainty_definition": report["uncertainty"]})
for method, directions in report["methods"].items():
    for direction, metrics in directions.items():
        display({"method": method, "direction": direction, "95% query-bootstrap": metrics["confidence_intervals_95"], "strata": metrics["strata"]})
'''),
        code('''
display(Image(filename=str(report_dir / "embeddings.png")))
display(Image(filename=str(report_dir / "retrieval_examples.png")))
queries = pd.read_csv(report_dir / "query_metrics.csv")
failures = queries[queries.positive_rank > 1].sort_values("positive_rank", ascending=False)
display(failures.head(12))
display(Markdown(f"**Observed in this execution:** {report['queries_per_direction']} synthetic test queries per sensor direction, {len(failures)} method/direction query rows with positive rank >1. Inspect `{report_dir / 'report.html'}` and `retrievals.csv` for visual and numeric evidence."))
'''),
        markdown('''
## What this experiment can establish

This run verifies that data preparation, geographic splits, training, checkpoints,
and bidirectional evaluation are connected. Synthetic urban/rural labels and
constructed cloud masks only test diagnostic machinery. Real urban/rural sensor
behavior, seasonal change, noise and acquisition-gap robustness remain unmeasured.
Inspect each geography/land-cover stratum and failure row before making claims.
Repeat with separated real regions and repeated seeds before drawing scientific
conclusions. Official SEN12MS access stopped at the configured archive-size cap;
the India extension contains catalogue metadata only.
'''),
    ])


if __name__ == "__main__":
    main()
