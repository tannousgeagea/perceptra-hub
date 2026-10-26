"""
Image header probe — learn an image's size without downloading it.

When a registration does not say how big an image is (a person registering
files from the console does not know), the hub reads only the first bytes of
the blob with a ranged request and lets Pillow parse the header. Nothing is
stored; the probe needs only read permission (a sp=r SAS is enough).

Most formats carry their size in the first few KB; JPEGs with a large EXIF
block (embedded thumbnails) can push the frame header further, so a second,
larger read is tried before giving up.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

READ_STEPS = (64 * 1024, 1024 * 1024)
PROBE_WORKERS = 8


class ProbeError(Exception):
    """The image could not be sized from its header."""


@dataclass
class ProbeResult:
    width: int
    height: int
    file_format: Optional[str]
    file_size: Optional[int]


def _parse_size(data: bytes) -> Optional[Tuple[int, int, Optional[str]]]:
    from PIL import ImageFile

    parser = ImageFile.Parser()
    try:
        parser.feed(data)
    except Exception:
        return None
    image = parser.image
    if image is None or not image.size or 0 in image.size:
        return None
    return image.size[0], image.size[1], (image.format or "").lower() or None


def probe_image(adapter, key: str) -> ProbeResult:
    """Width/height (and size in bytes when the backend reports it) of `key`."""
    parsed = None
    for length in READ_STEPS:
        try:
            data = adapter.download_range(key, 0, length)
        except Exception as e:
            raise ProbeError(f"cannot read '{key}': {e}") from None
        parsed = _parse_size(data)
        if parsed or len(data) < length:  # sized, or the whole file was read
            break
    if not parsed:
        raise ProbeError(f"'{key}' is not a readable image (could not parse its header)")
    file_size = None
    try:
        file_size = adapter.get_file_metadata(key).size
    except Exception:
        pass  # size is optional metadata
    width, height, file_format = parsed
    return ProbeResult(width=width, height=height, file_format=file_format, file_size=file_size)


def probe_many(jobs: Iterable[Tuple[object, str]]) -> Dict[Tuple[int, str], object]:
    """Probe (adapter, key) pairs in parallel; returns {(id(adapter), key): ProbeResult | ProbeError}."""
    jobs = list(jobs)
    results: Dict[Tuple[int, str], object] = {}

    def run(job):
        adapter, key = job
        try:
            return job, probe_image(adapter, key)
        except ProbeError as e:
            return job, e

    with ThreadPoolExecutor(max_workers=PROBE_WORKERS) as pool:
        for (adapter, key), outcome in pool.map(run, jobs):
            results[(id(adapter), key)] = outcome
    return results
