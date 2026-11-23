"""
Device credentials.

An enrollment code is short and typed by a person ("K7QM-4XPA-9TWE"), valid
for a day and usable once. The device trades it for a device key
("mtxd_<prefix>_<secret>"), which it keeps and sends on every call. Only
SHA-256 hashes of codes and secrets are stored; the prefix finds the device.
"""
import hashlib
import hmac
import secrets
from datetime import timedelta
from typing import Optional, Tuple

from django.utils import timezone

# No 0/O, 1/I/L: codes are read off a screen and typed on a device.
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
CODE_TTL = timedelta(hours=24)
KEY_SCHEME = "mtxd"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def normalize_code(code: str) -> str:
    return "".join(ch for ch in (code or "").upper() if ch.isalnum())


def new_enrollment_code() -> Tuple[str, str, "timezone.datetime"]:
    """(code to show once, hash to store, expiry)."""
    raw = "".join(secrets.choice(CODE_ALPHABET) for _ in range(12))
    code = "-".join(raw[i:i + 4] for i in range(0, 12, 4))
    return code, _sha(raw), timezone.now() + CODE_TTL


def code_hash(code: str) -> str:
    return _sha(normalize_code(code))


def new_device_key() -> Tuple[str, str, str]:
    """(key to hand to the device once, prefix, hash of the secret)."""
    prefix = secrets.token_hex(4)
    secret = secrets.token_urlsafe(32)
    return f"{KEY_SCHEME}_{prefix}_{secret}", prefix, _sha(secret)


def split_key(key: str) -> Optional[Tuple[str, str]]:
    parts = (key or "").strip().split("_", 2)
    if len(parts) != 3 or parts[0] != KEY_SCHEME or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


def secret_matches(secret: str, stored_hash: str) -> bool:
    return bool(stored_hash) and hmac.compare_digest(_sha(secret), stored_hash)
