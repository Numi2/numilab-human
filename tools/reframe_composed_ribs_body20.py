"""Reframe the source003 composed rib rows into the existing body20 frame."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from numilab_human.composed_bone_body_frame import reframe_composed_rib_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--actual-capture-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = (
        args.source_receipt, args.registration, args.partition,
        args.actual_capture_pack,
    )
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        parser.error("missing pinned input(s): " + ", ".join(missing))
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    result = reframe_composed_rib_payload(
        args.source_receipt,
        args.registration,
        args.partition,
        args.actual_capture_pack,
        args.output,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
