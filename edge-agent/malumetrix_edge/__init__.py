"""
Malumetrix edge agent: keeps a device's models in step with the platform.

Three ways to use it (see README.md):
  1. the sidecar:   `malumetrix-edge run` next to your own inference service
  2. the SDK:       `EdgeClient(...)` inside your own Python application
  3. neither:       implement the three-call HTTP protocol yourself
"""
import os as _os

# A published image carries the hub release it was built with; a pip install
# reports the package version.
__version__ = _os.environ.get("MALUMETRIX_EDGE_VERSION") or "0.1.0"
PROTOCOL = 1

from .sdk import EdgeClient, ModelUpdate  # noqa: E402,F401
