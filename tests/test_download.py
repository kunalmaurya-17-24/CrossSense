"""Safeguard tests use fakes; they never contact a data service."""

import io
import json
import tarfile

import pytest
import requests

from crosssense.data import download


class Response:
    def __init__(self, content=b"abc", size=3, status=200, content_type="application/octet-stream"):
        self.content = content
        self.status_code = status
        self.headers = {"Content-Type": content_type}
        if size is not None:
            self.headers["Content-Length"] = str(size)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def iter_content(self, chunk_size):
        yield self.content


class Session:
    def __init__(self, head=None, get=None):
        self.head_response = head or Response()
        self.get_response = get or Response()
        self.head_calls = 0
        self.get_calls = 0

    def head(self, *_args, **_kwargs):
        self.head_calls += 1
        return self.head_response

    def get(self, *_args, **_kwargs):
        self.get_calls += 1
        return self.get_response


def test_bounded_download_and_sticky_audit(tmp_path):
    client = Session()
    result = download.bounded_download("https://user:secret@example.org/file?token=secret", tmp_path / "file.bin", session=client)
    assert result["status"] == "downloaded"
    assert result["bytes_received"] == 3
    assert (tmp_path / "file.bin").read_bytes() == b"abc"
    assert "secret" not in json.dumps(result)
    again = download.bounded_download("https://example.org/file", tmp_path / "file.bin", session=client)
    assert again == result
    assert client.get_calls == 1


@pytest.mark.parametrize("size", [None, 100])
def test_size_preflight_prevents_body(tmp_path, size):
    client = Session(head=Response(size=size))
    result = download.bounded_download("https://example.org/data", tmp_path / "file", max_bytes=50, session=client)
    assert result["status"] == "stopped"
    assert result["bytes_received"] == 0
    assert client.get_calls == 0


@pytest.mark.parametrize("status", [401, 403, 404])
def test_permanent_http_errors_never_retry(tmp_path, status):
    client = Session(head=Response(status=status))
    result = download.bounded_download("https://example.org/data", tmp_path / "file", session=client)
    assert result["retry_count"] == 0
    assert f"HTTP {status}" in result["error"]


def test_temporary_http_errors_retry_at_most_twice(tmp_path, monkeypatch):
    delays = []
    monkeypatch.setattr(download.time, "sleep", delays.append)
    client = Session(head=Response(status=503))
    result = download.bounded_download("https://example.org/data", tmp_path / "file", session=client)
    assert result["status"] == "stopped"
    assert result["retry_count"] == 2
    assert client.head_calls == 3
    assert delays == [1, 2]


def test_bad_body_and_checksum_are_rejected(tmp_path):
    client = Session(get=Response(content=b"toolong", size=3))
    result = download.bounded_download("https://example.org/file", tmp_path / "file", session=client)
    assert result["status"] == "stopped"
    assert not (tmp_path / "file").exists()
    assert not (tmp_path / "file.part").exists()
    result = download.bounded_download("https://example.org/file", tmp_path / "checksum", expected_sha256="wrong", session=Session())
    assert result["error"] == "SHA256 checksum mismatch"


def test_sen12ms_preflight_size_and_redaction(tmp_path, monkeypatch):
    class FTP:
        sock = None
        calls = 0

        def __init__(self, **_kwargs):
            FTP.calls += 1

        def connect(self, *_args):
            pass

        def login(self, *_args):
            pass

        def voidcmd(self, *_args):
            pass

        def size(self, _name):
            return 100 * 1024 * 1024

        def retrbinary(self, *_args, **_kwargs):
            raise AssertionError("Oversized archive must never be downloaded")

        def close(self):
            pass

    monkeypatch.setattr(download.ftplib, "FTP", FTP)
    result = download.attempt_sen12ms(tmp_path)
    assert result["status"] == "stopped"
    assert "exceeds maximum" in result["error"]
    assert result["bytes_received"] == 0
    assert result["retry_count"] == 0
    assert "m1474000" not in json.dumps(result)
    assert download.attempt_sen12ms(tmp_path) == result
    assert FTP.calls == 1


def test_sen12ms_ftp_permission_no_retry(tmp_path, monkeypatch):
    class FTP:
        def __init__(self, **_kwargs):
            pass

        def connect(self, *_args):
            raise download.ftplib.error_perm("530 secret login failed")

        def close(self):
            pass

    monkeypatch.setattr(download.ftplib, "FTP", FTP)
    result = download.attempt_sen12ms(tmp_path)
    assert "FTP 530" in result["error"]
    assert "secret" not in json.dumps(result)
    assert result["retry_count"] == 0


def test_archive_traversal_and_expansion_limit(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        info = tarfile.TarInfo("../outside.txt")
        info.size = 3
        stream.addfile(info, io.BytesIO(b"bad"))
    with pytest.raises(download.DownloadStopped, match="Unsafe"):
        download._extract_checked(archive, tmp_path / "target", 100)
    with pytest.raises(download.DownloadStopped, match="Unpacked"):
        download._extract_checked(archive, tmp_path / "target", 1)


def test_cdse_missing_token_is_actionable(tmp_path, monkeypatch):
    monkeypatch.delenv("CDSE_ACCESS_TOKEN", raising=False)
    with pytest.raises(download.DownloadStopped, match="Register/login"):
        download.download_cdse_product("00000000-0000-0000-0000-000000000001", tmp_path / "product.zip", expected_bytes=100)


def test_download_config_limits(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        download.attempt_sen12ms(tmp_path, max_bytes=0)
    with pytest.raises(ValueError, match="positive"):
        download.bounded_download("https://example.org/file", tmp_path / "file", time_limit_s=-1)
    with pytest.raises(ValueError, match="positive"):
        download.attempt_sen12ms(tmp_path, time_limit_s=float("nan"))


@pytest.mark.parametrize("error", [requests.exceptions.TooManyRedirects, requests.exceptions.ContentDecodingError])
def test_malformed_http_response_is_recorded_without_leaking_details(tmp_path, error):
    class Client(Session):
        def get(self, *_args, **_kwargs):
            self.get_calls += 1
            raise error("token=secret")

    client = Client()
    result = download.bounded_download("https://example.org/file", tmp_path / "file", session=client)
    assert result["status"] == "stopped"
    assert result["retry_count"] == 0
    assert client.get_calls == 1
    assert "secret" not in json.dumps(result)


def test_ftp_local_permission_error_is_not_retried(tmp_path, monkeypatch):
    class FTP:
        def __init__(self, **_kwargs):
            pass

        def connect(self, *_args):
            raise PermissionError(13, "private local detail")

        def close(self):
            pass

    monkeypatch.setattr(download.ftplib, "FTP", FTP)
    result = download.attempt_sen12ms(tmp_path)
    assert result["retry_count"] == 0
    assert "Non-temporary" in result["error"]
    assert "private local detail" not in json.dumps(result)


def test_downloaded_raster_verification_requires_paired_expected_bands(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    for sensor, count in [("s1", 2), ("s2", 13)]:
        path = tmp_path / f"ROIs2017_winter_{sensor}_1_p1.tif"
        with rasterio.open(path, "w", driver="GTiff", width=8, height=8, count=count,
                           dtype="float32", crs="EPSG:32643", transform=from_origin(400000, 2000000, 10, 10)) as dataset:
            dataset.write(np.ones((count, 8, 8), dtype="float32"))
    result = download.verify_sen12ms_directory(tmp_path)
    assert result["pair_count"] == 1
    assert result["total_bytes"] == sum(path.stat().st_size for path in tmp_path.glob("*.tif"))
    (tmp_path / "ROIs2017_winter_s2_1_p1.tif").unlink()
    with pytest.raises(ValueError, match="No readable matched"):
        download.verify_sen12ms_directory(tmp_path)
