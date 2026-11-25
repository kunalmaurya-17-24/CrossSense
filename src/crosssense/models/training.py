"""Reproducible CPU/GPU training with resumable local checkpoints."""

import dataclasses
import hashlib
import json
import logging
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from crosssense.data.dataset import PairedPatchDataset, load_manifest
from crosssense.evaluation.metrics import retrieval_metrics
from crosssense.evaluation.splits import validate_splits

from .dual_encoder import DualEncoder
from .losses import symmetric_info_nce

LOGGER = logging.getLogger(__name__)


def configuration_dict(config: Any) -> dict[str, Any]:
    """Serialize validated dataclass configuration with portable path strings."""
    raw = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else vars(config)
    return json.loads(json.dumps(raw, default=lambda x: vars(x) if hasattr(x, "__dict__") else str(x)))


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch and request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    # Tiny fixtures otherwise spend most time starting large CPU thread pools.
    torch.set_num_threads(1)


def resolve_device(requested: str) -> torch.device:
    """Select a requested device and explain unavailable CUDA explicitly."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("Training device must be cpu, cuda, or auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; set training.device: cpu")
    if requested == "cuda":
        free, total = torch.cuda.mem_get_info()
        LOGGER.info("CUDA memory free_bytes=%d total_bytes=%d", free, total)
    return torch.device(requested)


def build_model(config: Any) -> DualEncoder:
    """Construct the configured two-branch compact CNN."""
    m = config.model
    return DualEncoder(m.sar_channels, m.optical_channels, m.embedding_dim,
                       m.width, m.freeze_backbones)


PREPROCESSING_FIELDS = ("patch_size", "resolution_m", "target_crs", "sar_bands", "optical_bands",
                        "sar_units", "sar_clip_db", "optical_scale", "optical_clip")
PREPARATION_FIELDS = (*PREPROCESSING_FIELDS, "stride", "min_valid_fraction", "max_cloud_fraction",
                      "max_time_gap_days", "min_overlap", "block_size_m", "split_buffer_m",
                      "split_fractions", "allow_author_pairs_without_time", "require_cloud_mask")


def validate_manifest_preprocessing(config: Any, frame: Any) -> dict[str, list[str]]:
    """Check prepared-array provenance, shapes and semantic band order before use.

    Source scale/offset may differ across scenes: those source-specific values
    are applied during preparation and recorded separately in the manifest.
    Canonical preparation settings are compared without local input paths.
    """
    snapshot = frame.attrs.get("preprocessing_config")
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1 or not isinstance(snapshot.get("data"), dict):
        raise ValueError("Manifest preprocessing provenance missing/unsupported; rebuild pairs to create "
                         "preprocessing_config.json alongside pairs.csv")
    current = configuration_dict(config)["data"]
    prepared = snapshot["data"]
    for field in PREPARATION_FIELDS:
        if field in current and (field not in prepared or current[field] != prepared[field]):
            raise ValueError(f"Prepared manifest preprocessing mismatch for data.{field}; "
                             "use the preparation config or rebuild pairs")
    schema: dict[str, list[str]] = {}
    for sensor, expected_channels in [("sar", config.model.sar_channels),
                                      ("optical", config.model.optical_channels)]:
        name_column = f"{sensor}_band_names"
        if name_column not in frame:
            raise ValueError(f"Manifest lacks {name_column} semantic channel metadata; rebuild pairs")
        for _, row in frame.iterrows():
            try:
                names = json.loads(row[name_column])
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid {name_column} for pair {row.pair_id}") from exc
            if (not isinstance(names, list) or len(names) != expected_channels or
                    any(not isinstance(name, str) or not name for name in names) or
                    len(set(names)) != len(names)):
                raise ValueError(f"Invalid {name_column} channel count/order for pair {row.pair_id}")
            if sensor in schema and schema[sensor] != names:
                raise ValueError(f"Inconsistent {name_column} semantic channel order across manifest")
            schema[sensor] = names
            array = np.load(row[f"{sensor}_path"], mmap_mode="r", allow_pickle=False)
            expected_shape = (expected_channels, config.data.patch_size, config.data.patch_size)
            if array.shape != expected_shape:
                raise ValueError(f"Prepared {sensor} array shape {array.shape} differs from configured "
                                 f"{expected_shape} for pair {row.pair_id}; rebuild pairs")
    LOGGER.info("Prepared manifest provenance verified semantic_band_schema=%s", schema)
    return schema


def validate_preprocessing(config: Any, checkpoint_config: dict[str, Any]) -> None:
    """Prevent evaluating model weights with silently changed normalization/bands."""
    current = configuration_dict(config)["data"]
    for field in PREPROCESSING_FIELDS:
        if current[field] != checkpoint_config["data"][field]:
            raise ValueError(f"Checkpoint preprocessing mismatch for data.{field}; use the training config")


def validate_checkpoint_holdout(payload: dict[str, Any], frame: Any,
                                split: str, fingerprint: str, buffer_m: float) -> None:
    """Check new evaluation manifests against locations used for fitting/selection."""
    if payload.get("manifest_sha256") == fingerprint:
        return  # Complete original manifest was independently checked by validate_splits.
    import pandas as pd
    records = payload.get("fit_geography")
    if not records:
        raise ValueError("Checkpoint lacks training geography; re-train before evaluating a different manifest")
    fitted = pd.DataFrame(records)
    # A new test set must also be separate from regions used for checkpoint selection.
    if split == "validation":
        fitted = fitted.loc[fitted.split == "train"].copy()
    fitted["split"] = "train"
    columns = ["pair_id", "block_id", "split", "crs", "minx", "miny", "maxx", "maxy"]
    combined = pd.concat([fitted[columns], frame[columns]], ignore_index=True)
    original_buffer = payload.get("config", {}).get("data", {}).get("split_buffer_m", buffer_m)
    try:
        validate_splits(combined, buffer_m=max(buffer_m, original_buffer))
    except ValueError as exc:
        raise ValueError(f"External evaluation overlaps fitted/selection geography: {exc}") from exc


def manifest_fingerprint(frame: Any) -> str:
    """Hash pair assignments, metadata and array content for resume provenance."""
    digest = hashlib.sha256(frame.sort_values("pair_id").to_csv(index=False).encode())
    for row in frame.sort_values("pair_id").itertuples():
        for field in ("sar_path", "optical_path", "mask_path"):
            path = Path(getattr(row, field))
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def embed_dataset(model: DualEncoder, dataset: PairedPatchDataset,
                  device: torch.device, batch_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Encode a complete dataset in manifest order without gradients."""
    if len(dataset) == 0:
        raise ValueError("Cannot encode an empty dataset; check manifest split assignments")
    model.eval()
    sar_vectors, optical_vectors = [], []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            sar, optical = model(batch["sar"].to(device), batch["optical"].to(device))
            sar_vectors.append(sar.cpu().numpy())
            optical_vectors.append(optical.cpu().numpy())
    return np.concatenate(sar_vectors), np.concatenate(optical_vectors)


def _validation_recall(model: DualEncoder, dataset: PairedPatchDataset,
                       device: torch.device, batch_size: int) -> float:
    sar, optical = embed_dataset(model, dataset, device, batch_size)
    ids = dataset.frame.pair_id.tolist()
    scores = optical @ sar.T
    return (retrieval_metrics(scores, ids, ids)["recall@1"] +
            retrieval_metrics(scores.T, ids, ids)["recall@1"]) / 2


def save_checkpoint(path: Path, model: DualEncoder, optimizer: torch.optim.Optimizer,
                    epoch: int, config: Any, **state: Any) -> None:
    """Atomically save model, optimizer, random state and experiment provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 1, "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "epoch": epoch,
               "config": configuration_dict(config), "python_rng": random.getstate(),
               "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               **state}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    LOGGER.info("Checkpoint saved epoch=%d path=%s", epoch, path)


def load_checkpoint(path: str | Path, model: DualEncoder,
                    optimizer: torch.optim.Optimizer | None = None,
                    restore_rng: bool = False) -> dict[str, Any]:
    """Load a trusted local CrossSense checkpoint; never load untrusted pickle files."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Checkpoint missing: {path}; run crosssense train first")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if torch.cuda.is_available() and payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload


def train_model(config: Any, manifest: str | Path, output_dir: str | Path,
                resume: str | Path | None = None) -> Path:
    """Train on geographic training blocks and select best validation Recall@1.

    ``epochs`` is the total target schedule on resume, not additional epochs.
    Only local checkpoints produced by this project should be resumed.
    """
    set_seed(config.seed)
    device = resolve_device(config.training.device)
    frame = load_manifest(Path(manifest))
    band_schema = validate_manifest_preprocessing(config, frame)
    validate_splits(frame, config.data.split_buffer_m)
    if frame.crs.nunique() != 1:
        raise ValueError("Training requires a shared metric projected CRS for geographic negative masking")
    train_data, val_data = PairedPatchDataset(frame, "train"), PairedPatchDataset(frame, "validation")
    if len(train_data) < 2 or len(val_data) < 2:
        raise ValueError("Training requires at least two train and two validation pairs in geographic holdouts")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_dict = configuration_dict(config)
    (output / "config.json").write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    versions = {"python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "device": str(device), "seed": config.seed,
                "deterministic_algorithms": "warn_only", "cpu_threads": 1}
    (output / "versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=config.training.lr)
    generator = torch.Generator().manual_seed(config.seed)
    fingerprint = manifest_fingerprint(frame)
    start_epoch, best_score, stale = 0, -1.0, 0
    history: list[dict[str, Any]] = []
    if resume is not None:
        previous_best = Path(resume).parent / "best.pt"
        if not previous_best.is_file():
            raise FileNotFoundError("Resume requires best.pt alongside the selected checkpoint to preserve "
                                    "the prior best model; restore the original checkpoint directory")
        payload = load_checkpoint(resume, model, optimizer, restore_rng=True)
        if payload["config"]["model"] != config_dict["model"]:
            raise ValueError("Resume model configuration differs from checkpoint")
        if payload.get("manifest_sha256") != fingerprint:
            raise ValueError("Resume manifest/data fingerprint differs from checkpoint; use the original data")
        # A changed objective would make a resumed experiment ambiguous.
        if payload["config"]["seed"] != config.seed:
            raise ValueError("Resume seed differs from checkpoint")
        for field, value in config_dict["data"].items():
            if field != "scene_manifest" and payload["config"]["data"][field] != value:
                raise ValueError(f"Resume data.{field} differs from checkpoint")
        previous = payload["config"]["training"]
        for field in ("batch_size", "lr", "temperature", "num_workers", "patience"):
            if previous[field] != config_dict["training"][field]:
                raise ValueError(f"Resume training.{field} differs from checkpoint")
        start_epoch, best_score = payload["epoch"], payload["best_score"]
        stale, history = payload["stale_epochs"], payload["history"]
        generator.set_state(payload["loader_rng"])
        LOGGER.info("Resumed checkpoint=%s next_epoch=%d", resume, start_epoch + 1)
    if config.training.epochs <= start_epoch:
        raise ValueError(f"Requested epochs={config.training.epochs} must exceed resumed epoch={start_epoch}")
    batch_size = min(config.training.batch_size, len(train_data))
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                        num_workers=config.training.num_workers, generator=generator)
    LOGGER.info("Training architecture=%s parameters=%d trainable=%d device=%s batch_size=%d seed=%d "
                "train_pairs=%d validation_pairs=%d versions=%s", type(model).__name__,
                sum(p.numel() for p in model.parameters()),
                sum(p.numel() for p in model.parameters() if p.requires_grad), device, batch_size,
                config.seed, len(train_data), len(val_data), versions)
    LOGGER.info("Preprocessing=%s", config_dict["data"])
    started = time.perf_counter()
    best_path = output / "best.pt"
    # Preserve the prior best when resuming into a different output directory.
    if resume is not None:
        prior_best = Path(resume).parent / "best.pt"
        if prior_best.is_file() and prior_best.resolve() != best_path.resolve():
            import shutil
            shutil.copyfile(prior_best, best_path)
    for epoch in range(start_epoch + 1, config.training.epochs + 1):
        model.train()
        total_loss, seen, skipped = 0.0, 0, 0
        for batch in loader:
            if len(batch["sar"]) < 2:
                skipped += len(batch["sar"])
                continue
            optimizer.zero_grad(set_to_none=True)
            sar, optical = model(batch["sar"].to(device), batch["optical"].to(device))
            loss = symmetric_info_nce(sar, optical, config.training.temperature,
                                      batch["coords"].to(device), config.data.geographic_exclusion_m,
                                      config.data.patch_size * config.data.resolution_m)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}; inspect normalized patches")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(sar)
            seen += len(sar)
        if not seen:
            raise ValueError("No valid training batches; provide at least two pairs and batch_size >= 2")
        score = _validation_recall(model, val_data, device, batch_size)
        improved = score > best_score
        best_score, stale = (score, 0) if improved else (best_score, stale + 1)
        row = {"epoch": epoch, "loss": total_loss / seen, "validation_recall@1": score,
               "lr": optimizer.param_groups[0]["lr"], "elapsed_s": time.perf_counter() - started,
               "trained_pairs": seen, "skipped_singletons": skipped}
        history.append(row)
        LOGGER.info("Epoch %s", row)
        state = {"best_score": best_score, "stale_epochs": stale, "history": history,
                 "loader_rng": generator.get_state(), "manifest_sha256": fingerprint,
                 "band_schema": band_schema,
                 "fit_geography": frame.loc[frame.split.isin(["train", "validation"]),
                     ["pair_id", "block_id", "split", "crs", "minx", "miny", "maxx", "maxy"]].to_dict("records")}
        save_checkpoint(output / "last.pt", model, optimizer, epoch, config, **state)
        if improved:
            save_checkpoint(best_path, model, optimizer, epoch, config, **state)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stale >= config.training.patience:
            LOGGER.info("Early stopping epoch=%d patience=%d", epoch, config.training.patience)
            break
    return best_path
