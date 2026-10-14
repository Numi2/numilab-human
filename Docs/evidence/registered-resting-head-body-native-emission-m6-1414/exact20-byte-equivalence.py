#!/usr/bin/env python3
"""Metadata-only exact array comparison for the explicitly scoped 20 typed surfaces."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

D = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
OLD = D / "head006-intercostal-native-emission-diagnostic-005/native-run/accepted-geometry/step-0.mrvpack"
NEW = D / "registered-resting-head-body-native-emission-diagnostic-007/native-run/accepted-geometry/step-0.mrvpack"
OLD_RECEIPT = OLD.with_suffix(".receipt.json")
NEW_RECEIPT = NEW.with_suffix(".receipt.json")
VERIFIER_ROOT = Path("/Users/n/numi-human-head-eye-placement-gate-20261014")
VERIFIER = VERIFIER_ROOT / "src/numilab_human/resting_anatomy_verify.py"
EXPECTED_VERIFIER_SHA256 = "a72878c0380279947f57e50c6c651e1b448554f224bc72f4cce84733b7761a76"
OUT = D / "head007-exact20-byte-equivalence-001"
FIELDS = ("positions", "faces", "normals", "global_gids")
KEYS = tuple(
    [(51005, stable_id) for stable_id in range(153, 159)]
    + [(51007, 1)]
    + [(51010, stable_id) for stable_id in range(311, 318)]
    + [(51023, stable_id) for stable_id in range(305, 310)]
    + [(51024, 310)]
)
EXPECTED_OLD_PACK_SHA256 = "d3c28f5dc6d5f06711f7e48f86513df71c41c9ef4ed9a6988aebf0852c6f8c0c"
EXPECTED_NEW_PACK_SHA256 = "fffd226819b8fe59e1b4ca0d6273a035347851c399b952bb3dae810bef504f89"
EXPECTED_OLD_RECEIPT_SHA256 = "93596adc6806e754142fb320e93c247496695685c26e61af048fd39a4c5c31d5"
EXPECTED_NEW_RECEIPT_SHA256 = "74113429ce6141258665ad7652cf4873954e66a97503e57db5ebec0a4ceb822c"
ALL875_REPORT = D / "head007-step0-byte-equivalence-001/surface-byte-equivalence.json"
ALL875_REPORT_SHA256 = "6a5991d34e11447f57f7b2777c9f48e4300dca463cdbbf2f0b59f2698fe7ff4e"
OLD_PROGRESS = D / "head006-intercostal-native-emission-diagnostic-005/head006-actual-all-nearby-005-output/progress.json"
OLD_RUNNER = D / "head006-intercostal-native-emission-diagnostic-005/head006-actual-all-nearby-005.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_digest(row: tuple) -> tuple[str, dict]:
    per_field = {}
    aggregate = hashlib.sha256()
    for field, value in zip(FIELDS, row, strict=True):
        if value is None:
            per_field[field] = {"present": False, "sha256": None}
            aggregate.update(field.encode("ascii") + b"\0NONE")
            continue
        array = np.ascontiguousarray(value)
        raw = array.tobytes()
        per_field[field] = {
            "present": True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        aggregate.update(field.encode("ascii") + b"\0")
        aggregate.update(json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)},
                                    sort_keys=True, separators=(",", ":")).encode("ascii"))
        aggregate.update(b"\0" + raw)
    return aggregate.hexdigest(), per_field


def main() -> None:
    if len(KEYS) != 20 or len(set(KEYS)) != 20:
        raise RuntimeError("the explicitly scoped typed-key set must contain exactly 20 unique keys")
    if OUT.exists():
        raise FileExistsError(OUT)
    if sha(OLD) != EXPECTED_OLD_PACK_SHA256 or sha(NEW) != EXPECTED_NEW_PACK_SHA256:
        raise ValueError("native capture pack pin mismatch")
    if sha(OLD_RECEIPT) != EXPECTED_OLD_RECEIPT_SHA256 or sha(NEW_RECEIPT) != EXPECTED_NEW_RECEIPT_SHA256:
        raise ValueError("native capture receipt pin mismatch")
    verifier_sha = sha(VERIFIER)
    if verifier_sha != EXPECTED_VERIFIER_SHA256:
        raise ValueError("shared capture decoder source pin mismatch")
    sys.path.insert(0, str(VERIFIER_ROOT / "src"))
    from numilab_human import resting_anatomy_verify as av

    if Path(av.__file__).resolve() != VERIFIER.resolve():
        raise ValueError("imported capture decoder differs from frozen source path")
    old_surfaces, old_identity = av.load_capture(OLD, OLD_RECEIPT)
    new_surfaces, new_identity = av.load_capture(NEW, NEW_RECEIPT)
    rows = []
    missing = []
    for key in KEYS:
        old = old_surfaces.get(key)
        new = new_surfaces.get(key)
        if old is None or new is None:
            missing.append(list(key))
            continue
        old_digest, old_fields = row_digest(old)
        new_digest, new_fields = row_digest(new)
        rows.append({
            "typed_key": list(key),
            "byte_identical": old_digest == new_digest,
            "old_surface_sha256": old_digest,
            "new_surface_sha256": new_digest,
            "fields": {
                name: {
                    "byte_identical": old_fields[name] == new_fields[name],
                    "old": old_fields[name],
                    "new": new_fields[name],
                }
                for name in FIELDS
            },
        })
    progress = json.loads(OLD_PROGRESS.read_text())
    passed = [stage["target"] for stage in progress.get("stages", [])
              if stage.get("stage") == "target_passed"]
    report = {
        "schema": "numi.human.native-step0-exact20-surface-byte-equivalence.v1",
        "status": "all_20_typed_surfaces_byte_identical" if not missing and all(r["byte_identical"] for r in rows) else "surface_byte_mismatch",
        "qualification": "metadata-only exact array identity; no geometry predicates were run",
        "scope": "20 typed surfaces requested for native007 publication comparison",
        "typed_keys": [list(key) for key in KEYS],
        "decoder": {"path": str(VERIFIER), "sha256": verifier_sha},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "old_capture": {
            "pack_path": str(OLD), "pack_sha256": sha(OLD),
            "receipt_path": str(OLD_RECEIPT), "receipt_sha256": sha(OLD_RECEIPT),
            "accepted_step": old_identity.get("accepted_step"),
        },
        "new_capture": {
            "pack_path": str(NEW), "pack_sha256": sha(NEW),
            "receipt_path": str(NEW_RECEIPT), "receipt_sha256": sha(NEW_RECEIPT),
            "accepted_step": new_identity.get("accepted_step"),
        },
        "prior_gate_reuse_basis": {
            "old_runner_path": str(OLD_RUNNER), "old_runner_sha256": sha(OLD_RUNNER),
            "old_progress_path": str(OLD_PROGRESS), "old_progress_sha256": sha(OLD_PROGRESS),
            "previously_passed_target_count": len(passed),
            "previously_passed_targets": passed,
            "full_875_surface_inventory_path": str(ALL875_REPORT),
            "full_875_surface_inventory_sha256": ALL875_REPORT_SHA256,
            "interpretation": "Identity comparison supports reuse only where exact typed target arrays and predicate implementation/input pins match; it is not a fresh predicate result.",
        },
        "comparison": {
            "field_order": list(FIELDS),
            "typed_surface_count": len(rows),
            "missing_typed_keys": missing,
            "all_identical": not missing and all(r["byte_identical"] for r in rows),
            "rows": rows,
        },
    }
    OUT.mkdir(parents=True, exist_ok=False)
    out = OUT / "exact20-byte-equivalence.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "script_sha256": sha(Path(__file__)),
                      "report_path": str(out), "report_sha256": sha(out),
                      "typed_surface_count": len(rows), "missing": missing}, sort_keys=True))


if __name__ == "__main__":
    main()
