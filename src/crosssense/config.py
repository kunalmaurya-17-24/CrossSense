"""Validated YAML settings shared by scripts, notebooks and CLI."""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml
from pyproj import CRS


@dataclass
class DataConfig:
    patch_size: int = 32
    stride: int = 32
    resolution_m: float = 10.0
    target_crs: str = "EPSG:32643"
    sar_bands: list[int] = field(default_factory=lambda: [1, 2])
    optical_bands: list[int] = field(default_factory=lambda: [2, 3, 4, 8])
    sar_units: str = "db"
    sar_clip_db: list[float] = field(default_factory=lambda: [-30.0, 5.0])
    optical_scale: float = 10000.0
    optical_clip: list[float] = field(default_factory=lambda: [0.0, 1.0])
    min_valid_fraction: float = 0.8
    max_cloud_fraction: float = 0.2
    max_time_gap_days: float = 7.0
    min_overlap: float = 0.9
    geographic_exclusion_m: float = 500.0
    block_size_m: float = 10000.0
    split_buffer_m: float = 500.0
    split_fractions: list[float] = field(default_factory=lambda: [0.6, 0.2, 0.2])
    scene_manifest: str | None = None
    require_cloud_mask: bool = False
    allow_author_pairs_without_time: bool = False


@dataclass
class ModelConfig:
    sar_channels: int = 2
    optical_channels: int = 4
    embedding_dim: int = 32
    width: int = 16
    freeze_backbones: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 4
    lr: float = 0.001
    temperature: float = 0.1
    device: str = "cpu"
    patience: int = 5
    num_workers: int = 0


@dataclass
class LoggingConfig:
    level: str = "INFO"
    output_dir: str = "outputs/logs"


@dataclass
class AppConfig:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML/JSON-serializable configuration snapshot."""
        return asdict(self)

    def validate(self) -> "AppConfig":
        """Fail before computation when settings are inconsistent."""
        _integer("seed", self.seed, minimum=0)
        for name in ("patch_size", "stride"):
            _integer(f"data.{name}", getattr(self.data, name), minimum=1)
        if self.data.patch_size < 8:
            raise ValueError("data.patch_size must be at least 8 for the compact CNN")
        for name in ("resolution_m", "optical_scale", "block_size_m"):
            _number(f"data.{name}", getattr(self.data, name), minimum=0, strict=True)
        for name in ("max_time_gap_days", "geographic_exclusion_m", "split_buffer_m"):
            _number(f"data.{name}", getattr(self.data, name), minimum=0)
        for name in ("min_valid_fraction", "max_cloud_fraction", "min_overlap"):
            _number(f"data.{name}", getattr(self.data, name), minimum=0, maximum=1)
        for name in ("sar_bands", "optical_bands"):
            values = getattr(self.data, name)
            if not isinstance(values, list) or not values or any(type(v) is not int for v in values) or len(set(values)) != len(values):
                raise ValueError(f"data.{name} must be a nonempty list of unique 1-based band indices")
            for value in values:
                _integer(f"data.{name}", value, minimum=1)
        for name in ("sar_clip_db", "optical_clip"):
            values = getattr(self.data, name)
            if not isinstance(values, list) or len(values) != 2:
                raise ValueError(f"data.{name} requires [minimum, maximum]")
            for value in values:
                _number(f"data.{name}", value)
            if values[0] >= values[1]:
                raise ValueError(f"data.{name} requires minimum < maximum")
        for name in ('require_cloud_mask', 'allow_author_pairs_without_time'):
            if not isinstance(getattr(self.data, name), bool):
                raise ValueError(f'data.{name} must be boolean')
        if self.data.sar_units not in ("db", "linear"):
            raise ValueError("data.sar_units must be db or linear calibrated backscatter")
        try:
            crs = CRS.from_user_input(self.data.target_crs)
        except Exception as exc:
            raise ValueError("data.target_crs is not a valid CRS") from exc
        if not crs.is_projected or any(axis.unit_name.lower() not in ("metre", "meter") for axis in crs.axis_info):
            raise ValueError("data.target_crs must be projected with metre units")
        fractions = self.data.split_fractions
        if not isinstance(fractions, list) or len(fractions) != 3:
            raise ValueError("data.split_fractions requires train/val/test fractions")
        for value in fractions:
            _number("data.split_fractions", value, minimum=0, maximum=1, strict=True)
        if not math.isclose(sum(fractions), 1.0, abs_tol=1e-6):
            raise ValueError("data.split_fractions must sum to one")
        if self.data.scene_manifest is not None and not Path(self.data.scene_manifest).is_file():
            raise ValueError(f"data.scene_manifest does not exist: {self.data.scene_manifest}. Prepare data first.")
        for name in ("sar_channels", "optical_channels", "embedding_dim", "width"):
            _integer(f"model.{name}", getattr(self.model, name), minimum=1)
        if self.model.sar_channels != len(self.data.sar_bands) or self.model.optical_channels != len(self.data.optical_bands):
            raise ValueError("model channel counts must equal the selected data band counts")
        if not isinstance(self.model.freeze_backbones, bool):
            raise ValueError("model.freeze_backbones must be boolean")
        for name in ("epochs", "patience"):
            _integer(f"training.{name}", getattr(self.training, name), minimum=1)
        _integer("training.batch_size", self.training.batch_size, minimum=2)
        _integer("training.num_workers", self.training.num_workers, minimum=0)
        for name in ("lr", "temperature"):
            _number(f"training.{name}", getattr(self.training, name), minimum=0, strict=True)
        if self.training.device not in ("cpu", "cuda", "auto"):
            raise ValueError("training.device must be cpu, cuda or auto")
        if self.logging.level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError("logging.level must be DEBUG/INFO/WARNING/ERROR/CRITICAL")
        if not isinstance(self.logging.output_dir, str) or not self.logging.output_dir.strip():
            raise ValueError("logging.output_dir must be a nonempty path")
        if Path(self.logging.output_dir).is_file():
            raise ValueError("logging.output_dir points to a file; choose a directory")
        return self


def _integer(name: str, value: Any, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(name: str, value: Any, minimum: float | None = None, maximum: float | None = None, strict: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and (value <= minimum if strict else value < minimum):
        raise ValueError(f"{name} must be {'>' if strict else '>='} {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


def load_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    """Load YAML, reject unknown keys and validate all settings."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration missing: {path}. Use configs/default.yaml.")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Configuration must be a YAML mapping")
    allowed = {f.name for f in fields(AppConfig)}
    if set(payload) - allowed:
        raise ValueError(f"Unknown configuration keys: {sorted(set(payload) - allowed)}")
    nested = {}
    for name, cls in (("data", DataConfig), ("model", ModelConfig), ("training", TrainingConfig), ("logging", LoggingConfig)):
        values = payload.get(name, {})
        if not isinstance(values, dict):
            raise ValueError(f"{name} must be a mapping")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {name} configuration keys: {sorted(unknown)}")
        nested[name] = cls(**values)
    config = AppConfig(seed=payload.get("seed", 42), **nested).validate()
    logging.getLogger(__name__).info("Configuration loaded and validated: %s", path)
    return config


def save_config(config: AppConfig, path: str | Path) -> None:
    """Write a complete resolved configuration for a reproducible run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
