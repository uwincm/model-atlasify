"""Where the products go: a local directory or an S3 prefix.

Output paths are ``UPath`` objects (universal_pathlib): a plain local path for
a directory, or an s3fs-backed path for ``s3://bucket/prefix``. Both support
the ``pathlib`` API, so the pipeline joins and lists paths the same way. Every
file is written through :func:`write_bytes` / :func:`write_text`, which upload
straight to S3 (nothing is staged on the local disk) with a Content-Type
guessed from the file name, so pages and images display properly in a browser.

S3 credentials come only from the standard AWS chain, never from the command
line: ``aws configure`` / ``aws sso login`` (``~/.aws``, optionally with
``AWS_PROFILE``), or ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``
(+ ``AWS_SESSION_TOKEN``) in the environment.
"""

from __future__ import annotations

import mimetypes
import os
import posixpath
import uuid
from pathlib import Path

import pystac
from upath import UPath

from modelatlasify.core.errors import ModelAtlasifyError

S3_PREFIX = "s3://"

#: Content types for the files ModelAtlasify writes (mimetypes is
#: platform-dependent and does not know some of them).
CONTENT_TYPES = {
    ".tif": "image/tiff; application=geotiff; profile=cloud-optimized",
    ".png": "image/png",
    ".json": "application/json",
    ".html": "text/html; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}

AUTH_HELP = (
    "Authenticate first, in one of these ways:\n"
    "  - AWS CLI: run 'aws configure' (access keys) or 'aws sso login' (SSO), and set "
    "AWS_PROFILE if you use a non-default profile;\n"
    "  - environment: export AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (and "
    "AWS_SESSION_TOKEN for temporary credentials).\n"
    "Access keys are not accepted on the command line."
)


class OutputAccessError(ModelAtlasifyError):
    """The output location cannot be written (e.g. no AWS credentials)."""


def is_remote(path: str | os.PathLike) -> bool:
    return str(path).startswith(S3_PREFIX)


def output_root(text: str | os.PathLike) -> UPath | Path:
    """The output root for ``-o``: a UPath for ``s3://bucket[/prefix]``, or a
    plain absolute ``Path`` for a local directory."""
    text = str(text)
    if is_remote(text):
        bucket = text[len(S3_PREFIX):].split("/", 1)[0]
        if not bucket:
            raise OutputAccessError(f"'{text}' has no bucket name; use s3://bucket/prefix.")
        return UPath(text.rstrip("/"))
    return Path(os.path.abspath(text))


def content_type(path: str | os.PathLike) -> str:
    ext = posixpath.splitext(str(path))[1].lower()
    return CONTENT_TYPES.get(ext) or mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def write_bytes(path: UPath | Path, data: bytes) -> None:
    """Write a file: locally (creating its folder) or straight to S3."""
    if is_remote(path):
        path.fs.pipe_file(path.path, data, ContentType=content_type(path))
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def write_text(path: UPath | Path, text: str) -> None:
    write_bytes(path, text.encode("utf-8"))


def read_bytes(path: UPath | Path) -> bytes:
    return path.read_bytes()


def read_text(path: UPath | Path) -> str:
    return read_bytes(path).decode("utf-8")


def refresh(path: UPath | Path) -> None:
    """Forget cached S3 listings under ``path`` (other processes may have
    written there since), so ``glob``/``iterdir`` see every file."""
    if is_remote(path):
        path.fs.invalidate_cache(path.path)


def as_path(href: str) -> UPath | Path:
    """The path for a STAC href (an s3:// URL or a local file path)."""
    return UPath(href) if is_remote(href) else Path(href)


def check_writable(root: UPath) -> None:
    """Fail early, with advice, if ``root`` on S3 cannot be written: write and
    delete a small probe object. Local roots are not checked."""
    if not is_remote(root):
        return
    from botocore.exceptions import BotoCoreError, NoCredentialsError, PartialCredentialsError

    probe = root / f".modelatlasify-write-check-{uuid.uuid4().hex}"
    where = str(root)
    bucket = root.path.split("/", 1)[0]
    try:
        root.fs.pipe_file(probe.path, b"ok", ContentType="text/plain")
        root.fs.rm_file(probe.path)
    except (NoCredentialsError, PartialCredentialsError) as exc:
        raise OutputAccessError(
            f"Cannot write to {where}: no AWS credentials found ({exc}).\n{AUTH_HELP}") from None
    except FileNotFoundError:
        raise OutputAccessError(
            f"Cannot write to {where}: the bucket '{bucket}' does not exist (or is not "
            "visible to your AWS account).") from None
    except PermissionError as exc:
        raise OutputAccessError(
            f"Cannot write to {where}: access denied ({exc}). The AWS credentials found "
            f"(profile '{os.environ.get('AWS_PROFILE', 'default')}' or AWS_* environment "
            "variables) are not allowed to write there, or have expired.\n" + AUTH_HELP) from None
    except (BotoCoreError, OSError) as exc:
        raise OutputAccessError(f"Cannot write to {where}: {exc}\n{AUTH_HELP}") from None


class FsspecStacIO(pystac.StacIO):
    """pystac I/O that reads and writes local files or S3 objects."""

    def read_text(self, source, *args, **kwargs) -> str:
        href = pystac.utils.make_posix_style(source) if not isinstance(source, str) else source
        if is_remote(href) or os.path.exists(href):
            return read_text(as_path(href))
        return pystac.StacIO.default().read_text(source, *args, **kwargs)  # e.g. http(s)

    def write_text(self, dest, txt: str, *args, **kwargs) -> None:
        href = dest if isinstance(dest, str) else str(dest)
        write_text(as_path(href), txt)
