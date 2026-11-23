"""
malumetrix-edge enroll --url https://… --code K7QM-4XPA-9TWE
malumetrix-edge run    [--config /etc/malumetrix/edge.json]
malumetrix-edge status

In Docker, `run` enrolls by itself first when MALUMETRIX_ENROLL_CODE is set
and the data volume has no credentials yet.
"""
import argparse
import json
import logging
import os
import signal
import sys
import urllib.request

from . import __version__
from .config import load
from .protocol import ApiError, NetworkError
from .sdk import Agent, NotEnrolled, enroll
from .store import Store


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="malumetrix-edge", description="Keeps this device's models in step with Malumetrix.")
    p.add_argument("--config", help="JSON settings file (or MALUMETRIX_CONFIG)")
    p.add_argument("--url", help="Platform URL (or MALUMETRIX_URL)")
    p.add_argument("--data-dir", help="Where models and credentials live (default /var/lib/malumetrix)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("enroll", help="Trade a one-time code from Fleet for this device's key")
    e.add_argument("--code", required=True)
    sub.add_parser("run", help="Run the agent (heartbeats, updates, local API)")
    sub.add_parser("status", help="Ask the running agent how it's doing (local API)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = load(args.config, url=args.url, data_dir=args.data_dir)

    try:
        if args.command == "enroll":
            answer = enroll(cfg, args.code)
            print(f"Enrolled as {answer.get('name')} ({answer['device_id']}). Start it with: malumetrix-edge run")
            return 0

        if args.command == "status":
            if not cfg.local_api or cfg.local_api == "off":
                print("The local API is off; nothing to ask.", file=sys.stderr)
                return 1
            with urllib.request.urlopen(f"http://{cfg.local_api}/v1/status", timeout=5) as resp:
                print(json.dumps(json.load(resp), indent=2))
            return 0

        # run
        code = os.getenv("MALUMETRIX_ENROLL_CODE")
        if code and not Store(cfg.data_path).credentials():
            enroll(cfg, code)
        agent = Agent(cfg)
        signal.signal(signal.SIGTERM, lambda *_: (agent.stop(), sys.exit(0)))
        agent.run_forever()
        return 0
    except NotEnrolled as e:
        print(e, file=sys.stderr)
        return 2
    except ApiError as e:
        print(f"The platform refused: {e.detail}", file=sys.stderr)
        return 1
    except (NetworkError, OSError) as e:
        print(f"Couldn't reach it: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
