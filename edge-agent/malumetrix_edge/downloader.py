"""
Model downloads that survive bad networks.

The file is written to <name>.part and continued with an HTTP Range request
after a cut. It only takes its real name once its size and SHA-256 match what
the platform said, so a truncated or tampered file is never loaded.
"""
import hashlib
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

CHUNK = 1 << 20


class LinkExpired(Exception):
    """The signed link stopped working; ask the platform for a fresh one."""


class ChecksumMismatch(Exception):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, sha256: str, size: Optional[int] = None, *, timeout: int = 60,
             ca_bundle: Optional[str] = None, progress: Optional[Callable[[int, Optional[int]], None]] = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and _sha256(dest) == sha256:
        return dest
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if size is not None and have > size:
        part.unlink()
        have = 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    ctx = ssl.create_default_context(cafile=ca_bundle) if ca_bundle else None
    try:
        resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout, context=ctx)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 410):
            raise LinkExpired(f"download link refused ({e.code})") from None
        if e.code == 416 and size is not None and have == size:
            resp = None  # already complete
        else:
            raise
    if resp is not None:
        with resp:
            resumed = resp.status == 206
            mode = "ab" if resumed else "wb"  # the server ignored Range: start over
            done = have if resumed else 0
            with open(part, mode) as f:
                for chunk in iter(lambda: resp.read(CHUNK), b""):
                    f.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, size)
                f.flush()
                os.fsync(f.fileno())

    if size is not None and part.stat().st_size != size:
        raise ChecksumMismatch(f"got {part.stat().st_size} bytes, expected {size}")
    actual = _sha256(part)
    if actual != sha256:
        part.unlink()  # corrupt: don't resume from it
        raise ChecksumMismatch(f"SHA-256 {actual[:12]}… does not match {sha256[:12]}…")
    os.replace(part, dest)
    return dest
