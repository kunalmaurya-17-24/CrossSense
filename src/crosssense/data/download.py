"""Bounded, audited downloads from documented dataset services.

No transfer is started without a known size and sufficient disk space. A failed
SEN12MS attempt is sticky: rerunning returns its existing audit instead of
repeatedly probing a failed server. Delete that audit deliberately to retry.
"""

from __future__ import annotations

import errno
import ftplib
import hashlib
import json
import logging
import math
import os
import shutil
import socket
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

LOG = logging.getLogger(__name__)
SEN12MS_RECORD = "https://mediatum.ub.tum.de/1474000"
SEN12MS_HOST = "dataserv.ub.tum.de"
SEN12MS_FILES = ("ROIs2017_winter_s1.tar.gz", "ROIs2017_winter_s2.tar.gz")
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024


class DownloadStopped(RuntimeError):
    """A permanent access error, explicit safeguard, or exhausted budget."""


def utc_now() -> str:
    """Return an unambiguous audit timestamp."""
    return datetime.now(UTC).isoformat()


def safe_url(url: str) -> str:
    """Remove URL credentials and query values, which may contain tokens."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host += f":{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise DownloadStopped("Total download time limit exceeded")
    return remaining


def _space_check(directory: Path, expected: int, max_bytes: int) -> int:
    if expected <= 0:
        raise DownloadStopped("Expected download size is unknown or empty; no body downloaded")
    if expected > max_bytes:
        raise DownloadStopped(f"Expected size {expected} bytes exceeds maximum {max_bytes} bytes; no body downloaded")
    free = shutil.disk_usage(directory).free
    # Reserve room for verification and ordinary system use.
    if free < expected + 10 * 1024 * 1024:
        raise DownloadStopped(f"Insufficient disk space: expected={expected}, free={free} bytes")
    return free


def _write_audit(path: Path, audit: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def bounded_download(
    url: str,
    destination: str | Path,
    *,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
    time_limit_s: float = 90,
    expected_bytes: int | None = None,
    headers: dict[str, str] | None = None,
    expected_sha256: str | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Download one HTTP(S) file with a known size, byte/time caps and audit.

    ``expected_bytes`` may come from a catalogue. Otherwise a HEAD request must
    supply Content-Length. Only temporary network failures and HTTP 408/429/5xx
    are retried, at most twice (delays 1s, 2s). Auth/permission/missing-file/TLS
    failures are permanent. Authorization headers are never included in logs.
    """
    if not math.isfinite(max_bytes) or not math.isfinite(time_limit_s) or max_bytes <= 0 or time_limit_s <= 0:
        raise ValueError("Download byte and time limits must be positive")
    if urlsplit(url).scheme not in {"http", "https"}:
        raise ValueError("bounded_download supports only HTTP(S); SEN12MS uses its dedicated FTP routine")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    audit_path = destination.with_suffix(destination.suffix + ".audit.json")
    if audit_path.exists():
        LOG.info("Returning existing download audit: %s", audit_path)
        return json.loads(audit_path.read_text(encoding="utf-8"))
    if destination.exists():
        raise FileExistsError(f"Destination already exists; refusing to overwrite {destination}")
    audit: dict[str, Any] = {
        "url": safe_url(url), "attempted_file": destination.name,
        "start_time": utc_now(), "end_time": None, "max_bytes": max_bytes,
        "time_limit_s": time_limit_s, "expected_bytes": expected_bytes,
        "bytes_received": 0, "retry_count": 0, "status": "started", "error": None,
    }
    client = session or requests.Session()
    own_session = session is None
    deadline = time.monotonic() + time_limit_s
    partial = destination.with_suffix(destination.suffix + ".part")
    LOG.info("Download start source=%s file=%s cap_bytes=%d time_limit_s=%s", safe_url(url), destination.name, max_bytes, time_limit_s)
    try:
        for attempt in range(3):
            try:
                timeout = max(0.001, min(15, _remaining(deadline)))
                if expected_bytes is None:
                    with client.head(url, headers=headers, allow_redirects=True, timeout=timeout) as response:
                        response.raise_for_status()
                        length = response.headers.get("Content-Length")
                        expected_bytes = int(length) if length else None
                expected = expected_bytes or 0
                audit["expected_bytes"] = expected_bytes
                audit["free_disk_bytes"] = _space_check(destination.parent, expected, max_bytes)
                checksum = hashlib.sha256()
                received = 0
                with client.get(url, headers=headers, stream=True, timeout=max(0.001, min(15, _remaining(deadline)))) as response:
                    response.raise_for_status()
                    if "text/html" in response.headers.get("Content-Type", ""):
                        raise DownloadStopped("Server returned an HTML page instead of the requested data file")
                    response_size = response.headers.get("Content-Length")
                    if response_size and int(response_size) != expected:
                        raise DownloadStopped("GET size differs from preflight/catalogue size")
                    with partial.open("wb") as stream:
                        for chunk in response.iter_content(chunk_size=64 * 1024):
                            _remaining(deadline)
                            if not chunk:
                                continue
                            audit["bytes_received"] += len(chunk)
                            received += len(chunk)
                            if audit["bytes_received"] > max_bytes or received > expected:
                                raise DownloadStopped("Download byte cap or advertised size exceeded")
                            stream.write(chunk)
                            checksum.update(chunk)
                if received != expected:
                    raise DownloadStopped(f"Incomplete body: expected {expected}, received {received} bytes")
                audit["sha256"] = checksum.hexdigest()
                if expected_sha256 and audit["sha256"].lower() != expected_sha256.lower():
                    raise DownloadStopped("SHA256 checksum mismatch")
                partial.replace(destination)
                audit["status"] = "downloaded"
                audit["local_file"] = str(destination.resolve())
                break
            except requests.exceptions.SSLError:
                raise DownloadStopped("TLS certificate validation failed; verification was not bypassed") from None
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                transient = status in {408, 429} or (status is not None and 500 <= status < 600)
                if not transient or attempt == 2:
                    raise DownloadStopped(f"HTTP {status}; {'temporary error retries exhausted' if transient else 'access/missing-file error; no retry'}") from None
                delay = attempt + 1
                if _remaining(deadline) <= delay:
                    raise DownloadStopped("Time budget cannot accommodate retry") from None
                audit["retry_count"] += 1
                LOG.warning("Temporary HTTP %s for %s; retry=%d delay_s=%d", status, safe_url(url), audit["retry_count"], delay)
                time.sleep(delay)
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
                if attempt == 2:
                    raise DownloadStopped("Temporary network failure; two retries exhausted") from None
                delay = attempt + 1
                if _remaining(deadline) <= delay:
                    raise DownloadStopped("Time budget cannot accommodate retry") from None
                audit["retry_count"] += 1
                LOG.warning("Temporary network failure for %s; retry=%d delay_s=%d", safe_url(url), audit["retry_count"], delay)
                time.sleep(delay)
    except (DownloadStopped, ValueError, OSError, requests.RequestException) as exc:
        audit["status"] = "stopped"
        # Do not stringify arbitrary requests exceptions: they can contain URLs or headers.
        audit["error"] = str(exc) if isinstance(exc, DownloadStopped) else f"Invalid response/filesystem/network error ({type(exc).__name__}); no retry"
        LOG.error("Download stopped file=%s reason=%s", destination.name, audit["error"])
    finally:
        partial.unlink(missing_ok=True)
        audit["end_time"] = utc_now()
        _write_audit(audit_path, audit)
        if own_session:
            client.close()
    return audit


def verify_sen12ms_directory(root: str | Path) -> dict[str, Any]:
    """Check readable, georeferenced, matching SAR/optical files after download.

    The routine verifies band counts and filenames without inventing acquisition
    timestamps, which are not reliably encoded in the distributed filenames.
    """
    import rasterio

    root = Path(root)
    files = sorted(root.rglob("*.tif"))
    sar: dict[str, Path] = {}
    optical: dict[str, Path] = {}
    records = []
    for path in files:
        if "_s1_" not in path.name and "_s2_" not in path.name:
            continue
        sensor = "sar" if "_s1_" in path.name else "optical"
        expected = 2 if sensor == "sar" else 13
        try:
            with rasterio.open(path) as src:
                if src.count != expected or src.crs is None or src.width <= 0 or src.height <= 0:
                    raise ValueError(f"Invalid {sensor} metadata in {path.name}: expected {expected} bands and CRS")
                src.read(1, window=((0, min(8, src.height)), (0, min(8, src.width))))
                records.append({"path": str(path.resolve()), "bytes": path.stat().st_size,
                                "sensor": sensor, "crs": str(src.crs),
                                "bounds": list(src.bounds), "resolution": list(src.res),
                                "width": src.width, "height": src.height, "bands": src.count})
        except rasterio.errors.RasterioError as exc:
            raise ValueError(f"Unreadable SEN12MS raster {path.name}") from exc
        key = path.name.replace("_s1_", "_sensor_").replace("_s2_", "_sensor_")
        (sar if sensor == "sar" else optical)[key] = path
    matched = sorted(sar.keys() & optical.keys())
    if not matched:
        raise ValueError("No readable matched 2-band S1 / 13-band S2 GeoTIFF pairs found")
    return {"source": "SEN12MS", "pair_count": len(matched), "files": records,
            "total_bytes": sum(row["bytes"] for row in records),
            "paired_ids": matched, "acquisition_times": "not inferred; supply provenance sidecar"}


def _extract_checked(archive: Path, destination: Path, max_unpacked_bytes: int) -> None:
    """Reject path traversal, links, special files and decompression bombs."""
    destination.mkdir(parents=True, exist_ok=True)
    target_root = destination.resolve()
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        if sum(member.size for member in members) > max_unpacked_bytes:
            raise DownloadStopped("Unpacked archive exceeds configured size cap")
        for member in members:
            target = (target_root / member.name).resolve()
            if not target.is_relative_to(target_root) or not (member.isfile() or member.isdir()):
                raise DownloadStopped("Unsafe archive member rejected")
        stream.extractall(destination, members=members, filter="data")


def attempt_sen12ms(
    output_dir: str | Path,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
    time_limit_s: float = 90,
) -> dict[str, Any]:
    """Attempt the documented official winter archive pair once, safely.

    FTP SIZE is checked for both archives before receiving either body. Only
    temporary transport/FTP 4xx failures retry (max twice, 1s/2s delays). An
    existing audit is returned unchanged so a stopped download is not repeated.
    """
    if not math.isfinite(max_bytes) or not math.isfinite(time_limit_s) or max_bytes <= 0 or time_limit_s <= 0:
        raise ValueError("Download byte and time limits must be positive")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    audit_path = root / "sen12ms_download_audit.json"
    if audit_path.exists():
        LOG.info("SEN12MS already attempted; returning %s", audit_path)
        return json.loads(audit_path.read_text(encoding="utf-8"))
    audit: dict[str, Any] = {
        "source": "SEN12MS", "documented_source": SEN12MS_RECORD,
        "access_documentation": "https://github.com/schmitt-muc/SEN12MS/issues/4",
        "url": f"ftp://{SEN12MS_HOST}/", "attempted_files": list(SEN12MS_FILES),
        "start_time": utc_now(), "end_time": None, "max_bytes": max_bytes,
        "time_limit_s": time_limit_s, "free_disk_bytes": shutil.disk_usage(root).free,
        "expected_sizes": {}, "bytes_received": 0, "retry_count": 0,
        "status": "started", "error": None, "files": [],
    }
    deadline = time.monotonic() + time_limit_s
    LOG.info("SEN12MS preflight source=%s subset=winter_s1+s2 destination=%s cap_bytes=%d time_limit_s=%s free_bytes=%d", SEN12MS_RECORD, root, max_bytes, time_limit_s, audit["free_disk_bytes"])
    ftp: ftplib.FTP | None = None
    partials = []
    try:
        for attempt in range(3):
            try:
                ftp = ftplib.FTP(timeout=max(0.001, min(12, _remaining(deadline))))
                ftp.connect(SEN12MS_HOST)
                # Public dataset-share credentials, published by its authors.
                # They are intentionally omitted from all URLs, exceptions and logs.
                ftp.login("m1474000", "m1474000")
                ftp.voidcmd("TYPE I")
                total_expected = 0
                for name in SEN12MS_FILES:
                    if ftp.sock:
                        ftp.sock.settimeout(max(0.001, min(12, _remaining(deadline))))
                    size = ftp.size(name)
                    audit["expected_sizes"][name] = size
                    if size is None:
                        raise DownloadStopped(f"FTP SIZE unavailable for {name}; no body downloaded")
                    total_expected += size
                    _space_check(root, total_expected, max_bytes)
                for name in SEN12MS_FILES:
                    partial = root / (name + ".part")
                    destination = root / name
                    if destination.exists():
                        raise DownloadStopped(f"Destination {name} already exists; refusing overwrite")
                    partials.append(partial)
                    digest = hashlib.sha256()
                    file_bytes = 0
                    with partial.open("wb") as target:
                        def receive(chunk: bytes, file_name: str = name, file_digest=digest) -> None:
                            nonlocal file_bytes
                            _remaining(deadline)
                            audit["bytes_received"] += len(chunk)
                            file_bytes += len(chunk)
                            if audit["bytes_received"] > max_bytes or file_bytes > audit["expected_sizes"][file_name]:
                                raise DownloadStopped("Transfer byte cap or advertised size exceeded")
                            target.write(chunk)
                            file_digest.update(chunk)
                        ftp.retrbinary(f"RETR {name}", receive, blocksize=64 * 1024)
                    if file_bytes != audit["expected_sizes"][name]:
                        raise DownloadStopped(f"Incomplete archive {name}: expected {audit['expected_sizes'][name]}, got {file_bytes}")
                    partial.replace(destination)
                    audit["files"].append({"file": name, "bytes": file_bytes, "sha256": digest.hexdigest()})
                extracted = root / "extracted"
                for name in SEN12MS_FILES:
                    _remaining(deadline)
                    _extract_checked(root / name, extracted, max_unpacked_bytes=max_bytes * 4)
                verified = verify_sen12ms_directory(extracted)
                _write_audit(root / "verified_files.json", verified)
                audit["verification"] = verified
                audit["status"] = "verified"
                break
            except ftplib.error_perm as exc:
                # FTP text can contain account details; record only status code.
                raise DownloadStopped(f"FTP {str(exc)[:3]} authentication/permission/missing-file error; no retry") from None
            except (ftplib.error_temp, TimeoutError, ConnectionError, OSError) as exc:
                error_kind = type(exc).__name__
                # Local filesystem errors (including disk full/permission) and
                # an invalid hostname are not temporary transfer failures.
                network_errnos = {errno.ECONNABORTED, errno.ECONNREFUSED, errno.ECONNRESET,
                                  errno.ENETDOWN, errno.ENETUNREACH, errno.EHOSTUNREACH,
                                  errno.ETIMEDOUT, 10050, 10051, 10053, 10054, 10060, 10061, 10065}
                transient = (isinstance(exc, (ftplib.error_temp, TimeoutError, ConnectionError))
                             or (isinstance(exc, socket.gaierror) and exc.errno == socket.EAI_AGAIN)
                             or getattr(exc, "errno", None) in network_errnos)
                if not transient:
                    raise DownloadStopped(f"Non-temporary FTP/filesystem/network failure ({error_kind}, errno={getattr(exc, 'errno', None)}); no retry") from None
                if attempt == 2:
                    raise DownloadStopped(f"Temporary FTP/network failure ({error_kind}); two retries exhausted") from None
                delay = attempt + 1
                if _remaining(deadline) <= delay:
                    raise DownloadStopped(f"Temporary FTP/network failure ({error_kind}); time budget cannot accommodate retry") from None
                audit["retry_count"] += 1
                LOG.warning("SEN12MS temporary network error type=%s retry=%d delay_s=%d", error_kind, audit["retry_count"], delay)
                time.sleep(delay)
            finally:
                if ftp is not None:
                    ftp.close()
                    ftp = None
    except (DownloadStopped, ValueError, tarfile.TarError, ftplib.Error) as exc:
        audit["status"] = "stopped"
        audit["error"] = str(exc) if isinstance(exc, (DownloadStopped, ValueError)) else f"Archive/FTP verification error ({type(exc).__name__})"
        LOG.error("SEN12MS stopped: %s", audit["error"])
    finally:
        for partial in partials:
            partial.unlink(missing_ok=True)
        audit["end_time"] = utc_now()
        audit["manual_next_steps"] = (
            "Open the official mediaTUM record, review its terms, obtain a small matched "
            "S1/S2 subset through its browser or author-documented FTP/rsync route, "
            "then run local verification and provide acquisition-time/cloud provenance. "
            "No full seasonal archive is authorized above the configured cap. "
            "Continue pipeline tests with source=synthetic."
        )
        _write_audit(audit_path, audit)
    return audit


def download_cdse_product(
    product_id: str,
    destination: str | Path,
    *,
    expected_bytes: int,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
    time_limit_s: float = 90,
) -> dict[str, Any]:
    """Download a known CDSE product using an externally supplied OAuth token.

    Register via the official CDSE browser and obtain the token using their
    documentation. Tokens stay in the environment and request header only.
    """
    from uuid import UUID

    product_id = str(UUID(product_id))
    token = os.environ.get("CDSE_ACCESS_TOKEN")
    if not token:
        raise DownloadStopped("CDSE_ACCESS_TOKEN is missing. Register/login at https://dataspace.copernicus.eu/ and obtain an OAuth token using the official OData documentation.")
    return bounded_download(
        f"https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value",
        destination, expected_bytes=expected_bytes, max_bytes=max_bytes,
        time_limit_s=time_limit_s, headers={"Authorization": f"Bearer {token}"},
    )
