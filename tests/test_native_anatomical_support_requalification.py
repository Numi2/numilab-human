from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "Docs/media/anatomical-support-candidate-20260915"


def test_pose_search_requalification_is_hash_bound_and_fail_closed() -> None:
    receipt = json.loads((EVIDENCE / "native-static-support-receipt-v2.json").read_text())
    assert receipt["schema"] == "HumanPack.native-anatomical-support-wrench.v2"
    assert receipt["status"] == "partial"
    assert receipt["native"]["commit"] == "413d08aa6a4253fc09628afa9816200a0f870969"
    assert receipt["run"]["activation_sweeps"] == 4096
    assert receipt["run"]["pose_sweeps"] == 12
    assert receipt["run"]["internal_normalized_residual_rms"] == 0.71909649283
    assert receipt["run"]["internal_balanced"] is False
    assert receipt["qualification"]["static_unilateral_support_wrench_closed"] is True
    assert receipt["qualification"]["internal_generalized_equilibrium"] is False
    assert receipt["qualification"]["standing"] is False
    assert receipt["qualification"]["walking"] is False
    for key in ("build_log", "native_run_log", "force_snapshot", "force_ledger"):
        artifact = receipt["artifacts"][key]
        expected = receipt["artifacts"][f"{key}_sha256"]
        assert hashlib.sha256((EVIDENCE / artifact).read_bytes()).hexdigest() == expected

    ledger = json.loads((EVIDENCE / receipt["artifacts"]["force_ledger"]).read_text())
    assert ledger["status"] == "partial"
    assert ledger["coverage"]["nv"] == 128
    assert ledger["coverage"]["source_contributions_per_dof"] == 6
    assert ledger["qualification"]["per_dof_source_audit"] is True
    assert ledger["qualification"]["force_convergence"] is False
    assert ledger["residual"]["maximum_closure_ratio"] == 1.0
