"""
Malumetrix edge agent: keeps a device's models in step with the platform.

Three ways to use it (see README.md):
  1. the sidecar:   `malumetrix-edge run` next to your own inference service
  2. the SDK:       `EdgeClient(...)` inside your own Python application
  3. neither:       implement the three-call HTTP protocol yourself
"""
__version__ = "0.1.0"
PROTOCOL = 1

from .sdk import EdgeClient, ModelUpdate  # noqa: E402,F401
