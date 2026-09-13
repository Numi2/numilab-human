"""Independently verify the pinned cardiac loading/support contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from numilab_human.cardiac_loading_contract import SCHEMA, compile_contract  # noqa: E402
from numilab_human.physiology import canonical  # noqa: E402


def verify(path: Path) -> dict:
    recorded = json.loads(path.read_text(encoding="utf-8"))
    actual = compile_contract()
    if recorded.get("schema") != SCHEMA:
        raise ValueError("unsupported cardiac loading contract schema")
    if canonical(recorded) != canonical(actual):
        raise ValueError("cardiac loading contract does not match current source inputs")
    if len(actual["unresolved_gates"]) != 6 or any(gate["admitted"] for gate in actual["unresolved_gates"]):
        raise ValueError("unresolved loading/support gates were promoted")
    if actual["qualification"]["native_anatomical_wall_admitted"]:
        raise ValueError("source contract cannot admit anatomical wall stepping")
    return {"schema": "HumanPack.cardiac-loading-contract-verification.v1", "status": "pass",
            "contract_identity_sha256": actual["identity_sha256"],
            "unresolved_gate_count": len(actual["unresolved_gates"]),
            "closure_label_count": len(actual["closure_labels"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(args.contract.resolve()), sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"cardiac loading contract verification: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
