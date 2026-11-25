"""Console and rotating JSON logs, with per-run context and secret redaction."""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class RunFormatter(logging.Formatter):
    """Serialize events and full exception traces with a stable run identifier."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "module": record.name,
            "run_id": self.run_id,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return redact(json.dumps(payload, ensure_ascii=False))


def redact(text: str) -> str:
    """Remove credentials in URLs and common authorization/token assignments."""
    text = re.sub(r"(https?|ftp)://[^\s/@]+(?::[^\s/@]*)?@", r"\1://[REDACTED]@", text)
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]+=*", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)((?:access_token|refresh_token|password|client_secret|api_key)[\s\"']*[:=][\s\"']*)[^\s\"'&,}]+", r"\1[REDACTED]", text)
    return text


def configure_logging(output_dir: str | Path = "outputs/logs", level: str = "INFO", run_id: str = "crosssense") -> Path:
    """Attach a console handler and rotating JSON file; return the log path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{run_id}.log"
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if getattr(handler, "_crosssense", False):
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)
    formatter = RunFormatter(run_id)
    for handler in (logging.StreamHandler(sys.stderr), RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")):
        handler.setFormatter(formatter)
        handler.setLevel(level)
        handler._crosssense = True
        root.addHandler(handler)
    logging.getLogger(__name__).info("Logging initialized: level=%s file=%s", level, log_path)
    return log_path
