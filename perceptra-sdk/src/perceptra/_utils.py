"""Internal utilities."""
from __future__ import annotations

from typing import Any


def strip_none(d: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *d* with all None-valued keys removed."""
    return {k: v for k, v in d.items() if v is not None}
