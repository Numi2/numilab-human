"""Independently verify the retained BodyParts3D organ geometry inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from numilab_human.organ_geometry import SCHEMA, inventory  # noqa: E402
from numilab_human.physiology import canonical  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify(path: Path) -> dict:
    recorded = json.loads(path.read_text(encoding="utf-8"))
    actual = inventory()
    if recorded.get("schema") != SCHEMA:
        raise ValueError("unsupported inventory schema")
    if canonical(recorded) != canonical(actual):
        raise ValueError("inventory does not match the current pinned source inputs")
    counts = actual["counts"]
    if counts != {
        "region_count": 18,
        "member_count": 378,
        "declared_membership_count": 386,
        "closed_quotient_member_count": 371,
        "open_or_defective_quotient_member_count": 7,
        "region_with_open_or_defective_member_count": 3,
    }:
        raise ValueError(f"unexpected inventory counts: {counts}")
    if actual["qualification"]["physical_stepping"]:
        raise ValueError("source inventory cannot claim physical stepping")
    return {"schema": "HumanPack.organ-geometry-verification.v1", "status": "pass",
            "inventory_sha256": sha256(canonical(actual) + b"\n"), "counts": counts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory", type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(args.inventory.resolve()), sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"organ geometry verification: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
