"""Independently verify the retained source-frame organ moment receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from numilab_human.organ_geometry_moments import SCHEMA, compile_moments  # noqa: E402
from numilab_human.physiology import canonical  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify(path: Path) -> dict:
    recorded = json.loads(path.read_text(encoding="utf-8"))
    actual = compile_moments()
    if recorded.get("schema") != SCHEMA:
        raise ValueError("unsupported organ moment schema")
    if canonical(recorded) != canonical(actual):
        raise ValueError("organ moment receipt does not match the current pinned source inputs")
    if actual["counts"] != {
        "region_count": 18,
        "member_count": 378,
        "declared_membership_count": 386,
        "closed_quotient_member_count": 371,
        "open_or_defective_quotient_member_count": 7,
        "region_with_open_or_defective_member_count": 3,
        "moment_computed_member_count": 357,
        "not_single_closed_component_member_count": 14,
        "source_topology_defective_member_count": 7,
    }:
        raise ValueError(f"unexpected organ moment counts: {actual['counts']}")
    if any(row["physical_volume_m3"] is not None or row["mechanical_mass_kg"] is not None
           for row in actual["members"]):
        raise ValueError("source moments cannot claim physical volume or mass")
    if actual["qualification"]["physical_stepping"]:
        raise ValueError("source moments cannot claim physical stepping")
    return {"schema": "HumanPack.organ-geometry-moment-verification.v1",
            "status": "pass", "moment_receipt_sha256": sha256(canonical(actual) + b"\n"),
            "counts": actual["counts"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("moments", type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(args.moments.resolve()), sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"organ geometry moment verification: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
