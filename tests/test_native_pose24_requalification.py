from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORCE = ROOT / "Docs/media/native-force-audit-pose24-20260915"
STAND = ROOT / "Docs/media/native-passive-stand-pose24-20260915"


def test_pose24_force_audit_closes_static_generalized_balance() -> None:
    receipt = json.loads((FORCE / "receipt-v1.json").read_text(encoding="utf-8"))
    assert receipt["source"]["commit"] == "7625ec565e086faf0dcd349846dadc2d22d65e86"
    assert receipt["command"]["passive_pose_sweeps"] == 24
    assert receipt["results"]["coordinate_identity_rows"] == 128
    assert receipt["results"]["internal_balanced"]
    assert receipt["results"]["maximum_absolute_force_residual"] < 1.0e-3
    assert receipt["results"]["maximum_closure_ratio"] < 0.05
    assert receipt["qualification"]["generalized_force_closed"]
    assert not receipt["qualification"]["force_convergence"]
    assert not receipt["qualification"]["sustained_standing"]


def test_pose24_artifacts_and_stand_release_are_hash_bound() -> None:
    receipt = json.loads((FORCE / "receipt-v1.json").read_text(encoding="utf-8"))
    for key in ("coordinate_map", "force_snapshot", "force_ledger", "native_stdout", "native_stderr"):
        artifact = receipt["artifacts"][key]
        assert hashlib.sha256((FORCE / artifact).read_bytes()).hexdigest() == receipt["artifacts"][f"{key}_sha256"]
    stand = json.loads((STAND / "receipt-v1.json").read_text(encoding="utf-8"))
    assert stand["source"]["commit"] == receipt["source"]["commit"]
    assert stand["results"]["persistent_completed_steps"] == 512
    assert stand["results"]["compiled_stand_balanced"]
    assert stand["results"]["persistent_max_acceleration_mps2"] < 5.0
    assert stand["qualification"]["bounded_exact_clock_release"]
    assert not stand["qualification"]["sustained_standing"]
