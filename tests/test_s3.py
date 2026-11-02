"""Output straight to S3 (-o s3://bucket/prefix), against a local moto S3 server."""

from __future__ import annotations

import json
import os
import uuid

import pytest

from conftest import T0
from wrf_fixture import hourly, write_wrfout
from modelatlasify.cli import main

moto_server = pytest.importorskip("moto.server")
boto3 = pytest.importorskip("boto3")


@pytest.fixture(scope="session")
def s3_endpoint():
    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


def _clear_s3fs_cache():
    import s3fs

    s3fs.S3FileSystem.clear_instance_cache()  # instances remember old credentials


@pytest.fixture
def s3(s3_endpoint, monkeypatch):
    """A fresh bucket on the mock server, with dummy credentials in the environment."""
    for key in list(os.environ):
        if key.startswith("AWS_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("AWS_ENDPOINT_URL", s3_endpoint)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    _clear_s3fs_cache()
    bucket = f"test-{uuid.uuid4().hex[:12]}"
    client = boto3.client("s3", endpoint_url=s3_endpoint)
    client.create_bucket(Bucket=bucket)
    yield client, bucket
    _clear_s3fs_cache()


def _keys(client, bucket, prefix=""):
    pages = client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
    return {o["Key"] for page in pages for o in page.get("Contents", [])}


def _run(tmp_path, name="run01", n=3):
    run = tmp_path / "inputs" / name
    write_wrfout(run, hourly(T0, n))
    return run


def test_cli_writes_everything_to_s3(s3, tmp_path, monkeypatch, capsys):
    from rasterio.io import MemoryFile

    client, bucket = s3
    run = _run(tmp_path)
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)  # nothing may be written here (or next to the inputs)
    before = {p for p in tmp_path.rglob("*")}

    url = f"s3://{bucket}/atlas"
    assert main([str(run), "-o", url, "--cut-dirs", "99", "--no-validate", "-np", "2"]) == 0
    out, err = capsys.readouterr()
    assert out.strip() == f"{url}/catalog.json"
    assert f"-> {url}" in err and "all asset links resolve" not in err  # --no-validate

    assert {p for p in tmp_path.rglob("*")} == before
    keys = _keys(client, bucket, "atlas/")
    assert {"atlas/catalog.json", "atlas/run01/collection.json",
            "atlas/run01/model-images.html",
            "atlas/run01/model-images/full/surface_wind_speed.html"} <= keys
    assert any(k.startswith("atlas/run01/command-used.") for k in keys)
    tifs = sorted(k for k in keys if k.endswith(".tif"))
    assert len(tifs) == 3 * 3 + 3 + 2  # wind speed + u10 + v10, precipitation, interval
    assert any(k.startswith("atlas/run01/model-images/full/precipitation/") and k.endswith(".png")
               for k in keys)
    assert not any(".modelatlasify-write-check" in k for k in keys)  # probe cleaned up

    catalog = json.loads(client.get_object(Bucket=bucket, Key="atlas/catalog.json")["Body"].read())
    assert all(not ln["href"].startswith("s3://") for ln in catalog["links"]
               if ln["rel"] != "self")  # self-contained: relative links
    body = client.get_object(Bucket=bucket, Key=tifs[0])["Body"].read()
    with MemoryFile(body) as mem, mem.open() as src:
        assert src.crs is not None and src.count == 1

    def ctype(key):
        return client.head_object(Bucket=bucket, Key=key)["ContentType"]

    assert ctype("atlas/catalog.json") == "application/json"
    assert ctype("atlas/run01/model-images.html").startswith("text/html")
    assert ctype(next(k for k in keys if k.endswith(".png"))) == "image/png"
    assert ctype(tifs[0]).startswith("image/tiff")


def test_s3_catalog_validates_and_accumulates_runs(s3, tmp_path, capsys):
    client, bucket = s3
    url = f"s3://{bucket}/atlas/"  # a trailing slash is fine
    for name in ("runA", "runB"):
        run = _run(tmp_path, name, n=2)
        assert main([str(run), "-o", url, "--cut-dirs", "99", "--no-plots",
                     "--no-validate"]) == 0
    from modelatlasify.core.validation import validate_catalog

    report = validate_catalog(f"s3://{bucket}/atlas/catalog.json", schemas=False)
    assert report.ok and report.n_objects == 1 + 2 + 4
    catalog = json.loads(client.get_object(Bucket=bucket, Key="atlas/catalog.json")["Body"].read())
    children = [ln["href"] for ln in catalog["links"] if ln["rel"] == "child"]
    assert children == ["./runA/collection.json", "./runB/collection.json"]

    # A link check notices a missing asset.
    tif = next(k for k in _keys(client, bucket, "atlas/runB/") if k.endswith(".tif"))
    client.delete_object(Bucket=bucket, Key=tif)
    _clear_s3fs_cache()
    report = validate_catalog(f"s3://{bucket}/atlas/catalog.json", schemas=False)
    assert len(report.missing_hrefs) == 1 and tif in report.missing_hrefs[0]


def test_missing_credentials_is_a_clear_error(tmp_path, monkeypatch, capsys):
    for key in list(os.environ):
        if key.startswith("AWS_"):
            monkeypatch.delenv(key)
    empty = tmp_path / "empty"
    empty.write_text("")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(empty))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(empty))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")  # no instance-role lookup
    _clear_s3fs_cache()
    run = _run(tmp_path, n=1)
    assert main([str(run), "-o", "s3://some-bucket/atlas"]) == 1
    err = capsys.readouterr().err
    assert "no AWS credentials found" in err
    assert "aws configure" in err and "AWS_ACCESS_KEY_ID" in err
    assert "Traceback" not in err and "Found WRF run" not in err  # failed before any work
    _clear_s3fs_cache()


def test_missing_bucket_is_a_clear_error(s3, tmp_path, capsys):
    run = _run(tmp_path, n=1)
    assert main([str(run), "-o", "s3://no-such-bucket-modelatlasify/atlas"]) == 1
    err = capsys.readouterr().err
    assert "the bucket 'no-such-bucket-modelatlasify' does not exist" in err


def test_access_denied_is_a_clear_error(s3, tmp_path, monkeypatch, capsys):
    import s3fs

    def denied(self, path, data, **kwargs):
        raise PermissionError("Access Denied")

    monkeypatch.setattr(s3fs.S3FileSystem, "pipe_file", denied)
    run = _run(tmp_path, n=1)
    _, bucket = s3
    assert main([str(run), "-o", f"s3://{bucket}/atlas"]) == 1
    err = capsys.readouterr().err
    assert "access denied" in err and "not allowed to write" in err and "aws sso login" in err


def test_no_credential_options_on_the_command_line():
    from modelatlasify.cli import build_parser

    options = {o for a in build_parser()._actions for o in a.option_strings}
    assert not {o for o in options if "key" in o or "secret" in o or "token" in o}


def test_bucket_name_is_required(capsys, tmp_path):
    assert main([str(tmp_path), "-o", "s3://"]) == 1
    assert "has no bucket name" in capsys.readouterr().err
