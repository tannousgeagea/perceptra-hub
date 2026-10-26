"""
Short-lived signed URLs for files Perceptra serves itself.

Browsers load images with plain <img src>, which cannot send auth headers.
When a storage backend cannot mint its own short-lived URL (an Azure SAS-only
credential, local storage), Perceptra streams the file through its own media
route instead, and authorizes the request with an HMAC signature over the
path and an expiry carried in the query string.

The key is MEDIA_URL_SIGNING_KEY (falls back to SECRET_KEY): rotating it
invalidates every outstanding URL.
"""
import base64
import hashlib
import hmac
import time
from typing import Optional
from urllib.parse import urlencode

from django.conf import settings

DEFAULT_EXPIRATION_S = 3600


def _key() -> bytes:
    key = getattr(settings, "MEDIA_URL_SIGNING_KEY", None) or settings.SECRET_KEY
    return key.encode() if isinstance(key, str) else key


def sign(path: str, expires_at: int) -> str:
    digest = hmac.new(_key(), f"{path}|{int(expires_at)}".encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def signed_query(path: str, expiration: int = DEFAULT_EXPIRATION_S) -> str:
    """
    'exp=…&sig=…' authorizing GET `path` for at least `expiration` seconds.

    The expiry is rounded up to the end of the next window of `expiration`
    seconds (so it lasts up to twice as long): every request in the same
    window gets the same URL, and browsers can serve the file from cache.
    """
    window = max(60, int(expiration))
    expires_at = (int(time.time()) // window + 2) * window
    return urlencode({"exp": expires_at, "sig": sign(path, expires_at)})


def verify(path: str, exp: Optional[str], sig: Optional[str]) -> bool:
    """True when `sig` signs `path` and `exp` has not passed."""
    if not exp or not sig:
        return False
    try:
        expires_at = int(exp)
    except (TypeError, ValueError):
        return False
    if expires_at < time.time():
        return False
    return hmac.compare_digest(sign(path, expires_at), sig)
