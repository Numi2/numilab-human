#!/usr/bin/env python3
"""Import, pin, and no-work control for the 772->782/791 continuation derivative."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import tempfile

H = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
DRIVER = H / "continue_adaptive_fixed_boundary_cones_772_v782_791_root.py"
EXPECTED_DRIVER_SHA = "3af7794ce5c5e98adfa12d88f7e9249a551b419332fdb3494a5854ed56729301"
OUTER = H / "fixture-772-791-root"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

assert sha(DRIVER) == EXPECTED_DRIVER_SHA
spec = importlib.util.spec_from_file_location("continuation_772_791_test", DRIVER)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

pins = module.validate_dependencies()
assert pins["promoter"]["sha256"] == module.EXPECTED_PROMOTER_SHA256
assert pins["source_verifier"]["sha256"] == module.EXPECTED_SOURCE_VERIFIER_SHA256
assert pins["producer"]["sha256"] == module.EXPECTED_PRODUCER_SHA256

try:
    module.check_pin(module.PROMOTER, "0" * 64)
except ValueError as exc:
    tamper = {"status": "PASS_REJECTED", "error": str(exc)}
else:
    raise AssertionError("tampered continuation dependency hash was accepted")

OUTER.mkdir(exist_ok=False)
state = OUTER / "synthetic-zero-quality-state.json"
state.write_text(json.dumps({
    "Skin_current": {"native_quality_failures": 0},
    "Fat_current": {"native_quality_failures": 0},
    "physical_admission": False,
    "native_steps": 0,
}, indent=2) + "\n")
output = OUTER / "zero-round-control"
old_argv = sys.argv
try:
    sys.argv = [str(DRIVER), "--state", str(state), "--output", str(output), "--rounds", "0"]
    module.main()
finally:
    sys.argv = old_argv
completion_path = output / "completion.json"
completion = json.loads(completion_path.read_text())
assert completion["status"] == "ZERO_NATIVE_QUALITY_FAILURES"
assert completion["physical_admission"] is False and completion["native_steps"] == 0
assert not list(output.glob("round*")), "zero-round fixture unexpectedly launched a geometry round"
result = {
    "schema": "numi.h706.continuation-772-782-791-controls.v1",
    "status": "PASS_IMPORT_PINS_TAMPER_REJECTION_AND_ZERO_ROUND_FIXTURE",
    "driver": {"path": str(DRIVER), "sha256": sha(DRIVER)},
    "dependency_pins": pins,
    "tamper_control": tamper,
    "zero_round_fixture": {
        "synthetic_state": str(state),
        "completion": {"path": str(completion_path), "sha256": sha(completion_path), "status": completion["status"]},
        "geometry_rounds": 0,
        "physical_admission": False,
        "native_steps": 0,
    },
}
receipt = OUTER / "receipt.json"
receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
print("receipt_sha256", sha(receipt), "path", receipt)
