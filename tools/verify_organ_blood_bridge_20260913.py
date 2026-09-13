"""Independently verify the source-to-hydraulic organ blood bridge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from numilab_human.organ_blood_bridge import SCHEMA, compile_bridge  # noqa: E402
from numilab_human.physiology import canonical  # noqa: E402


def verify(path: Path) -> dict:
    recorded = json.loads(path.read_text(encoding="utf-8"))
    actual = compile_bridge()
    if recorded.get("schema") != SCHEMA:
        raise ValueError("unsupported organ blood bridge schema")
    if canonical(recorded) != canonical(actual):
        raise ValueError("organ blood bridge does not match current pinned source inputs")
    if [row["source_index"] for row in actual["bindings"]] != [15, 16, 19, 20]:
        raise ValueError("unexpected cavity source indices")
    if any(row["physical_volume_owner"] is not None or row["mechanical_mass_owner"] is not None
           or row["density_kg_per_m3"] is not None for row in actual["bindings"]):
        raise ValueError("bridge cannot assign physical mass or density")
    if actual["qualification"]["physical_volume_authority_assigned"]:
        raise ValueError("bridge cannot admit a physical volume authority")
    return {"schema": "HumanPack.organ-blood-cavity-bridge-verification.v1",
            "status": "pass", "bridge_sha256": actual["identity_sha256"],
            "cavity_bindings": len(actual["bindings"]),
            "intersecting_triangle_pairs": actual["cvsim_cavity_reference"]["intersecting_triangle_pairs"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bridge", type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(args.bridge.resolve()), sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"organ blood bridge verification: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
