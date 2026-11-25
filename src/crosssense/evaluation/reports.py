"""Bidirectional baseline/model measurements with inspectable failure reports."""

import html
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from crosssense.data.dataset import PairedPatchDataset, load_manifest, load_pair_arrays
from crosssense.models.training import (
    build_model,
    embed_dataset,
    load_checkpoint,
    manifest_fingerprint,
    resolve_device,
    set_seed,
    validate_checkpoint_holdout,
    validate_manifest_preprocessing,
    validate_preprocessing,
)
from crosssense.retrieval.index import CosineIndex
from crosssense.retrieval.search import statistical_features

from .metrics import bootstrap_intervals, metrics_from_ranks, paired_ranks
from .splits import validate_splits

LOGGER = logging.getLogger(__name__)


def _stratify(frame: pd.DataFrame, ranks: np.ndarray) -> dict[str, Any]:
    groups = {"geographic_block": frame.block_id.astype(str),
              "land_cover": frame.land_cover.astype(str),
              "cloud_fraction": pd.cut(frame.cloud_fraction, [-0.001, .05, .2, 1],
                                       labels=["0-5%", "5-20%", "20-100%"], include_lowest=True).astype("string").fillna("unknown"),
              "time_gap_days": pd.cut(frame.time_gap_days, [-.001, 1, 3, 7, np.inf],
                                      labels=["0-1", "1-3", "3-7", ">7"], include_lowest=True).astype("string").fillna("unknown")}
    result: dict[str, Any] = {}
    for name, values in groups.items():
        result[name] = {}
        for label in values.dropna().unique():
            selection = np.asarray(values == label)
            result[name][str(label)] = {"queries": int(selection.sum()),
                                        **metrics_from_ranks(ranks[selection])}
    return result


def _report_visuals(frame: pd.DataFrame, vectors: dict[str, tuple[np.ndarray, np.ndarray]],
                    query_rows: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A shared PCA fitted to both modalities, purely a diagnostic projection.
    fig, axes = plt.subplots(1, len(vectors), figsize=(6 * len(vectors), 5), squeeze=False)
    for ax, (name, (sar, optical)) in zip(axes[0], vectors.items()):
        joined = np.concatenate([sar, optical])
        joined = joined - joined.mean(0)
        _, _, vh = np.linalg.svd(joined, full_matrices=False)
        projected = joined @ vh[:2].T
        if projected.shape[1] < 2:
            projected = np.pad(projected, ((0, 0), (0, 2 - projected.shape[1])))
        n = len(sar)
        ax.scatter(*projected[:n].T, marker="^", label="SAR", alpha=.8)
        ax.scatter(*projected[n:].T, marker="o", label="Optical", alpha=.8)
        for i in range(min(n, 40)):
            ax.plot(projected[[i, i + n], 0], projected[[i, i + n], 1], color="gray", alpha=.2)
        ax.set(title=f"{name}: joint PCA (diagnostic)", xlabel="PC1", ylabel="PC2")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "embeddings.png", dpi=130)
    plt.close(fig)

    candidates = query_rows[query_rows.positive_rank > 1].sort_values("positive_rank", ascending=False)
    actual_failures = len(candidates) > 0
    if not actual_failures:
        candidates = query_rows.head(3)
    candidates = candidates.head(6)
    fig, axes = plt.subplots(len(candidates), 3, figsize=(10, 3 * len(candidates)), squeeze=False)
    lookup = {row.pair_id: row for _, row in frame.iterrows()}

    def show(ax: Any, row: pd.Series, sensor: str, title: str) -> None:
        sar, optical, mask = load_pair_arrays(row)
        if sensor == "sar":
            ax.imshow(np.ma.array(sar[0], mask=~mask), cmap="gray", vmin=0, vmax=1)
        else:
            names = json.loads(row.optical_band_names)
            if all(name in names for name in ["B4", "B3", "B2"]):
                rgb = optical[[names.index(name) for name in ["B4", "B3", "B2"]]]
                rgb = np.moveaxis(np.clip(rgb * 2.5, 0, 1), 0, -1)
                rgb[~mask] = 0
                ax.imshow(rgb)
            else:
                ax.imshow(np.ma.array(optical[0], mask=~mask), cmap="gray", vmin=0, vmax=1)
                title += f"\nselected band {names[0]} (grayscale)"
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    for axes_row, (_, query) in zip(axes, candidates.iterrows()):
        sensor = "optical" if query.direction == "optical_to_sar" else "sar"
        target = "sar" if sensor == "optical" else "optical"
        show(axes_row[0], lookup[query.pair_id], sensor,
             f"{query.method} {query.direction}\nquery {query.pair_id}")
        show(axes_row[1], lookup[query.top1_id], target,
             f"Top-1 {query.top1_id}\ncosine={query.top1_score:.3f}, positive rank={query.positive_rank}")
        show(axes_row[2], lookup[query.pair_id], target,
             f"Manifest positive\nland={lookup[query.pair_id].land_cover}, gap={lookup[query.pair_id].time_gap_days:.1f}d")
    fig.suptitle("Observed retrieval failures" if actual_failures else "No observed top-1 failures; example successes")
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(output / "retrieval_examples.png", dpi=130)
    plt.close(fig)


def evaluate_experiment(config: Any, manifest: str | Path, checkpoint: str | Path,
                        output_dir: str | Path, split: str = "test") -> dict[str, Any]:
    """Measure baseline and learned retrieval in both directions on a held-out split."""
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation requires validation or test geographic holdouts")
    set_seed(config.seed)
    complete = load_manifest(manifest)
    band_schema = validate_manifest_preprocessing(config, complete)
    validate_splits(complete, config.data.split_buffer_m)
    dataset = PairedPatchDataset(complete, split)
    frame = dataset.frame
    device = resolve_device(config.training.device)
    model = build_model(config).to(device)
    payload = load_checkpoint(checkpoint, model)
    validate_preprocessing(config, payload["config"])
    if payload.get("band_schema") != band_schema:
        raise ValueError("Evaluation semantic band order differs from checkpoint or checkpoint metadata is missing; "
                         "use matching bands or re-train with current manifests")
    fingerprint = manifest_fingerprint(complete)
    validate_checkpoint_holdout(payload, frame, split, fingerprint, config.data.split_buffer_m)
    if payload["config"]["model"] != {
            key: getattr(config.model, key) for key in payload["config"]["model"]}:
        raise ValueError("Evaluation model settings differ from the trained checkpoint")
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    learned = embed_dataset(model, dataset, device, config.training.batch_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    learned_encoding_s = time.perf_counter() - started
    started = time.perf_counter()
    baseline_sar, baseline_optical = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        baseline_sar.append(statistical_features(sample["sar"].numpy())[0])
        baseline_optical.append(statistical_features(sample["optical"].numpy())[0])
    baseline = (np.stack(baseline_sar), np.stack(baseline_optical))
    baseline_encoding_s = time.perf_counter() - started
    vectors = {"statistical_baseline": baseline, "dual_encoder": learned}
    ids = frame.pair_id.astype(str).tolist()
    metadata = json.loads(frame.to_json(orient="records"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    synthetic_count = int(frame.source.str.startswith("synthetic").sum())
    data_label = ("SYNTHETIC PIPELINE TEST" if synthetic_count == len(frame) else
                  "MIXED REAL/SYNTHETIC DATA" if synthetic_count else "REAL DATA")
    report: dict[str, Any] = {
        "data_label": data_label,
        "source_counts": frame.source.value_counts().to_dict(), "split": split,
        "semantic_band_schema": band_schema,
        "gallery_pairs": len(frame), "queries_per_direction": len(frame), "seed": config.seed,
        "checkpoint": str(Path(checkpoint).resolve()), "selected_epoch": payload["epoch"],
        "evaluation_manifest_sha256": fingerprint,
        "training_manifest_sha256": payload.get("manifest_sha256"),
        "relevance": "Exactly the manifest pair_id; nearby alternative patches may be unlabelled positives.",
        "scores": "Exact cosine similarity, not calibrated probabilities; ties use stable manifest order.",
        "uncertainty": "95% query-bootstrap percentile intervals, 200 resamples; conditional on this gallery, not spatial or training-seed confidence.",
        "latency": "Mean wall time of one exact search including sorting/metadata return after one warmup; excludes image loading and encoding.",
        "index_size": "Float32 embedding bytes plus UTF-8 IDs/metadata; excludes Python object overhead.",
        "encoding_seconds_per_pair": {"dual_encoder": learned_encoding_s / len(frame),
                                      "statistical_baseline": baseline_encoding_s / len(frame)},
        "methods": {}, "skipped_queries": 0,
    }
    all_queries, predictions = [], []
    for method, (sar, optical) in vectors.items():
        report["methods"][method] = {}
        for direction, query, gallery in [("optical_to_sar", optical, sar),
                                          ("sar_to_optical", sar, optical)]:
            index = CosineIndex(gallery, ids, metadata)
            index.search(query[0], 10)
            measured, hits = [], []
            for vector in query:
                started = time.perf_counter()
                hits.append(index.search(vector, 10)[0])
                measured.append(time.perf_counter() - started)
            scores = index.similarities(query)
            ranks = paired_ranks(scores, ids, ids)
            summary = {**metrics_from_ranks(ranks), "queries": len(ids),
                       "gallery_size": len(ids), "latency_ms_per_query": float(np.mean(measured) * 1000),
                       "index_size_bytes": index.size_bytes, "vector_bytes": index.embeddings.nbytes,
                       "confidence_intervals_95": bootstrap_intervals(ranks, config.seed),
                       "strata": _stratify(frame, ranks)}
            report["methods"][method][direction] = summary
            LOGGER.info("Evaluation method=%s direction=%s metrics=%s", method, direction,
                        {k: v for k, v in summary.items() if k not in {"strata", "confidence_intervals_95"}})
            for i, pair_id in enumerate(ids):
                all_queries.append({"method": method, "direction": direction, "pair_id": pair_id,
                                    "positive_rank": int(ranks[i]), "top1_id": hits[i][0]["id"],
                                    "top1_score": hits[i][0]["score"], "block_id": frame.iloc[i].block_id,
                                    "land_cover": frame.iloc[i].land_cover})
                for rank, hit in enumerate(hits[i], 1):
                    predictions.append({"method": method, "direction": direction, "query_id": pair_id,
                                        "rank": rank, "gallery_id": hit["id"], "cosine_score": hit["score"],
                                        "relevant": hit["id"] == pair_id,
                                        "gallery_metadata": json.dumps(hit["metadata"])})
    queries = pd.DataFrame(all_queries)
    queries.to_csv(output / "query_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "retrievals.csv", index=False)
    _report_visuals(frame, vectors, queries, output)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    table = [{"method": method, "direction": direction,
              **{k: values[k] for k in ["recall@1", "recall@5", "recall@10", "mrr", "latency_ms_per_query", "index_size_bytes"]}}
             for method, directions in report["methods"].items() for direction, values in directions.items()]
    summary_frame = pd.DataFrame(table)
    summary_frame.to_csv(output / "summary.csv", index=False)
    page = ("<!doctype html><html><head><meta charset='utf-8'><title>CrossSense measured evaluation</title>"
            "<style>body{max-width:1100px;margin:2em auto;font-family:system-ui}td,th{padding:.5em}img{max-width:100%}</style>"
            f"</head><body><h1>CrossSense: {html.escape(report['data_label'])}</h1>"
            f"<p>Measured {len(frame)} {html.escape(split)} pairs. Selected epoch {payload['epoch']}. "
            "No sensor identity, reconstruction or semantic understanding claim is implied.</p>"
            + summary_frame.to_html(index=False, float_format=lambda v: f"{v:.5f}")
            + f"<p>{html.escape(report['relevance'])}</p><p>{html.escape(report['uncertainty'])}</p>"
            "<h2>Embedding diagnostic</h2><img src='embeddings.png' alt='Joint PCA of modality embeddings'>"
            "<h2>Observed examples</h2><img src='retrieval_examples.png' alt='Query, retrieval and manifest positive'>"
            "<p>Full strata, provenance and uncertainty: metrics.json. Ranked IDs, scores and metadata: retrievals.csv.</p>"
            "</body></html>")
    (output / "report.html").write_text(page, encoding="utf-8")
    LOGGER.info("Evaluation complete pairs=%d skipped=0 report=%s", len(frame), output / "report.html")
    return report
