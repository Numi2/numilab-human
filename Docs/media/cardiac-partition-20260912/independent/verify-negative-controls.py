#!/usr/bin/env python3
"""Exercise captured-evidence rejection without modifying canonical evidence."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
from verify_cardiac_partition import verify, EVIDENCE


def main():
    record = json.loads((EVIDENCE / "execution.json").read_text())
    artifact = json.loads((EVIDENCE / "partition.json").read_text())
    checks = []
    cases = [
        ("missing_owner", lambda e,p: e["inputs_before"].pop("src/numilab_human/cardiac_face_arrangement.py"), "owner or source hash drift"),
        ("failed_regression", lambda e,p: e["runs"][1].update(exit_code=1), "failed or unbound run"),
        ("wrong_test_command", lambda e,p: e["runs"][1]["command"].pop(), "captured regression command differs"),
        ("promoted_biology", lambda e,p: p["qualification"].update(anatomical_valve_interface_selected=True), "qualification scope differs"),
        ("integer_boolean_claim", lambda e,p: p["qualification"].update(both_emitted_candidates_have_disjoint_interiors=1), "qualification scope differs"),
        ("added_mass", lambda e,p: p.update(added_mechanical_mass_kg=0.001), "unsupported physical promotion"),
        ("missing_candidate", lambda e,p: p["candidates"].pop("right_atrium_priority"), "candidate coverage differs"),
        ("changed_mesh_identity", lambda e,p: p["candidates"]["right_atrium_priority"].update(geometry_sha256="0"*64), "candidate geometry hash drift"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory)
        for filename in ("compile.log", "tests.log"):
            shutil.copyfile(EVIDENCE / filename, target / filename)
        for name, mutate, expected in cases:
            execution, payload = copy.deepcopy(record), copy.deepcopy(artifact)
            mutate(execution, payload)
            encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"))+"\n").encode()
            (target / "partition.json").write_bytes(encoded)
            # Rebind the manipulated JSON so semantic rejection, not only a
            # trivial stale output digest, is exercised by these controls.
            execution["outputs"]["partition.json"] = hashlib.sha256(encoded).hexdigest()
            (target / "execution.json").write_text(json.dumps(execution)+"\n")
            try:
                verify(target)
            except ValueError as error:
                if expected not in str(error):
                    raise AssertionError((name, str(error))) from error
                checks.append({"case": name, "rejected": True, "reason": str(error)})
            else:
                raise AssertionError("unsupported evidence admitted: "+name)
    result = {"schema": "HumanPack.cardiac-partition-evidence-negative-controls.v1", "status": "pass",
        "checks": checks, "canonical_evidence_modified": False, "physical_stepping": False,
        "execution_sha256": hashlib.sha256((EVIDENCE / "execution.json").read_bytes()).hexdigest(),
        "verifier_sha256": hashlib.sha256((ROOT / "tools/verify_cardiac_partition.py").read_bytes()).hexdigest()}
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
