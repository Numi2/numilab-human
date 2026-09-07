from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys
import time

from numilab_human import model
from numilab_human.target_coverage import canonical_bytes, materialize, validate_manifest, validate_transition

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "Build/target-coverage-20260908"
PREVIOUS = BASE / "HumanPack.target-coverage.v1.json"
MANIFEST = BASE / "HumanPack.target-coverage.with-nonmuscle-tendons.v1.json"
IR = BASE / "myosim-export-with-nonmuscle-tendons.json"
REFERENCE = ROOT / "Build/completion-20260908/constraint-oracle"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_record(path):
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(data), "sha256": sha(data)}


previous = json.loads(PREVIOUS.read_text())
current = json.loads(MANIFEST.read_text())
exported = json.loads(IR.read_text())
started = time.monotonic()
repeated = materialize(
    sources=ROOT / "Sources", source_lock=ROOT / "sources.lock.json",
    myosim_ir=IR, myosim_ir_sha256=sha(IR.read_bytes()), previous=previous,
)
repeat_seconds = time.monotonic() - started
assert canonical_bytes(repeated) + b"\n" == MANIFEST.read_bytes()
validate_manifest(current)
validate_transition(previous, current)
old_leaves = {entry["leaf_sha256"]: entry for entry in previous["leaves"]}
new_leaves = {entry["leaf_sha256"]: entry for entry in current["leaves"]}
assert len(old_leaves) == 27737
assert all(new_leaves[key] == value for key, value in old_leaves.items())
assert len(new_leaves.keys() - old_leaves.keys()) == 64
assert all(entry["evidence_status"] == "unknown" for entry in current["leaves"])
assert current["integrated_qualification"] == "unknown"

# Execute the actual lowerer unchanged through rigid/equality serialization,
# then stop before site/muscle processing. No fitting stub or synthetic payload.
function = model.myosim_fullbody_reference_artifacts
lines, first_line = inspect.getsourcelines(function)
stop_line = first_line + next(index for index, line in enumerate(lines)
                              if line.startswith("    site_by_id ="))
captured = {}


class PrefixComplete(BaseException):
    pass


def trace(frame, event, arg):
    if frame.f_code is function.__code__ and event == "line" and frame.f_lineno == stop_line:
        captured.update(frame.f_locals)
        raise PrefixComplete()
    return trace


started = time.monotonic()
try:
    sys.settrace(trace)
    function(exported)
    raise AssertionError("lowerer reached muscle fitting unexpectedly")
except PrefixComplete:
    pass
finally:
    sys.settrace(None)
prefix_seconds = time.monotonic() - started
identity = {}
for key, filename in (
    ("rigid_payload", "myosim-fullbody-core-reference.nhrigid"),
    ("equality_payload", "myosim-fullbody-joint-equalities.nheq"),
    ("equality_compliance_payload", "myosim-fullbody-joint-equalities-source-compliance.nheq"),
):
    payload = captured[key]
    reference = REFERENCE / filename
    assert payload == reference.read_bytes(), key
    identity[key] = {"bytes": len(payload), "sha256": sha(payload),
                     "exact_reference_match": True, "reference": file_record(reference)}

sources = {entry["record_sha256"]: entry["source_id"] for entry in current["source_records"]}
unresolved = [
    {"source_id": sources[entry["source_record_sha256"]], "path": entry["path"],
     "status": entry["status"]}
    for entry in current["registers"] if entry["status"] != "materialized"
]
assert len(unresolved) == 3
files = [
    ROOT / path for path in (
        "src/numilab_human/myosim_export.py", "src/numilab_human/target_coverage.py",
        "src/numilab_human/model.py", "tests/test_target_coverage.py",
        "tests/test_myosim_source_inventory.py", "schemas/humanpack-target-coverage.v1.schema.json",
        "sources.lock.json",
    )
] + [IR, Path(__file__).resolve()]
receipt = {
    "schema": "numi.human.target-coverage-materialization-receipt.v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "host": platform.platform(), "python": platform.python_version(),
    "scope_status": current["scope_status"],
    "integrated_qualification": "unknown", "every_leaf_evidence_status": "unknown",
    "manifest": {**file_record(MANIFEST), "manifest_sha256": current["manifest_sha256"],
                 "canonical_json_sha256": sha(canonical_bytes(current))},
    "previous_manifest": {**file_record(PREVIOUS), "manifest_sha256": previous["manifest_sha256"],
                          "canonical_json_sha256": sha(canonical_bytes(previous))},
    "counts": {key: current["counts"][key] for key in ("leaves", "mandatory_leaves", "unresolved_current_registers")},
    "retention": {"previous_leaves": 27737, "exact_previous_leaf_records_retained": 27737,
                  "new_leaves": 64, "removed_leaves": 0},
    "new_source_inventory": {"nonmuscle_tendons": 8, "route_elements": 24,
                             "sites": 24, "wrap_geometries": 8, "native_mechanics_status": "not_lowered"},
    "repeat_byte_identical": True, "repeat_seconds": repeat_seconds,
    "unresolved_registers": unresolved,
    "files": [file_record(path) for path in files],
    "focused_tests": {"runner": "unittest", "coverage_methods_passed": 13,
                      "nonmuscle_tamper_cases_passed": 8, "pinned_source_integration_methods_passed": 1,
                      "source_interpreter": ".venv-myosim/bin/python", "mujoco_version": "3.12.0",
                      "integration_test_path": "tests/test_myosim_source_inventory.py"},
    "fresh_lowering_identity": {
        "scope": "actual lowerer through rigid and equality prefix; no muscle fitting or GPU work",
        "seconds": prefix_seconds, "source_equality_count": len(exported["joint_equalities"]),
        "artifacts": identity,
    },
}
output = BASE / "materialization-with-nonmuscle-receipt.json"
output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps({"receipt": str(output.relative_to(ROOT)), "bytes": output.stat().st_size,
                  "sha256": sha(output.read_bytes()), "repeat_seconds": repeat_seconds,
                  "prefix_seconds": prefix_seconds, "identity": identity}, indent=2))
