#!/usr/bin/env python3
"""Verify stored artifacts and recompute the offline reaction audit."""
import hashlib
import json
from pathlib import Path

from verify_reactions import HERE, MANIFEST, MANIFEST_SHA, audit, parse


def main() -> None:
    if not __debug__:
        raise RuntimeError("receipt verification requires Python assertion checks")
    receipt = json.loads((HERE / "receipt.json").read_text())
    assert receipt["schema"] == "numi.human.offline-limit-reaction-receipt.v1"
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == MANIFEST_SHA
    for entry in receipt["artifacts"]:
        path = (HERE / entry["path"]).resolve()
        assert path.is_relative_to(HERE) and path.is_file()
        data = path.read_bytes()
        assert len(data) == entry["bytes"], entry["path"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]
    qualification = json.loads((HERE / "qualification.json").read_text())
    assert qualification["native_commit"] == receipt["native_commit"]
    assert qualification["working_tree_clean"] and qualification["working_tree_clean_after"]
    manifest = json.loads(MANIFEST.read_text())
    for run in qualification["runs"]:
        assert run["returncode"] == 0
        sweeps = run["sweeps"]
        log = (HERE / f"qualified-{sweeps}-stdout.log").read_bytes()
        digest = hashlib.sha256(log).hexdigest()
        assert digest == run["stdout"]["sha256"]
        q, reactions, summary = parse(log.decode())
        result = audit(q, reactions, manifest)
        assert result["status"] == "offline_reactions_passed"
        stored = json.loads((HERE / f"reaction-audit-{sweeps}.json").read_text())
        for key, value in result.items():
            assert stored[key] == value, key
        assert stored["log_sha256"] == digest
        assert summary["internal_balanced"] == "false"
        assert float(summary["internal_normalized_residual_rms"]) > receipt["balance_threshold"]
        source = json.loads((HERE / f"source-oracle-{sweeps}.json").read_text())
        assert source["certificate_sha256"] == digest
        assert source["primitive_geometry_passed"] and source["lowering_passed"] and source["wrench_passed"]
    for key in ("internal_equilibrium", "dynamic_limits", "dynamic_contact", "v5_tissue_acceptance",
                "standing", "walking", "costal_requalification", "calibration", "performance", "full_release"):
        assert receipt["qualification"][key] is False
    print(json.dumps({"status": "bounded_offline_reactions_passed", "artifacts": len(receipt["artifacts"]),
                      "internal_equilibrium": False, "dynamic_limits": False, "full_release": False}))


if __name__ == "__main__":
    main()
