"""Verify the retained support failure and repaired same-stack physical cohort."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "source_trace_audit", BASE.parent / "source-compliant-equilibrium-20260910/verify_receipt.py")
TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    raw = path.read_bytes()
    return (gzip.decompress(raw) if path.suffix == ".gz" else raw).decode()


def audit_failure(text):
    trace = TRACE.audit_trace(text, 64, require_equivalence=False)
    require(trace["replay"] == "bitwise" and not trace["dropout_matches_zero"], "retained failure disappeared")
    require(trace["first_dropout_difference"]["root"] == 1, "first failed root")
    buffers = {}
    samples = []
    for line in text.splitlines():
        if line.startswith("mrnx_initial_buffer="):
            row = json.loads(line.split("=", 1)[1])
            buffers.setdefault(row["name"], []).append(row["values"])
        elif line.startswith("mrnx_initial_support="):
            samples.append(json.loads(line.split("=", 1)[1]))
    require(len(buffers) == 6 and all(len(v) == 4 for v in buffers.values()), "first-root snapshots incomplete")
    for name in ["muscle_force", "bias", "free_velocity", "initial_q"]:
        require(all(v == buffers[name][0] for v in buffers[name]), "source input diverged: " + name)
    for name in ["coupled_force", "coupled_velocity"]:
        require(buffers[name][2] != buffers[name][3], "retained coupled mismatch disappeared")
    require(len(samples) == 4 * 17 * 18, "support assembly trace incomplete")
    groups = [samples[i * 306:(i + 1) * 306] for i in range(4)]
    require(groups[0] == groups[1] == groups[2], "earlier scenarios differ")
    for group in groups:
        require([(r["assembly"], r["row"]) for r in group] ==
                [(a, r) for a in range(17) for r in range(18)], "support assembly order")
    for good, bad in zip(groups[2][:18], groups[3][:18]):
        require({k: v for k, v in good.items() if k != "normal_constraint"} ==
                {k: v for k, v in bad.items() if k != "normal_constraint"},
                "initial contact geometry or Jacobian diverged")
    good, bad = groups[2][1], groups[3][1]
    require(good["normal_constraint"] == 17.9370499 and bad["normal_constraint"] == 395.966522,
            "retained recovery mismatch")
    return trace


def verify(base=BASE):
    receipt = json.loads((base / "receipt.json").read_text())
    require(receipt["schema"] == "numi.human.coupled-initial-poses.v1", "receipt schema")
    actual = {str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()
              and p.name != "receipt.json" and "__pycache__" not in p.parts}
    require(actual == set(receipt["artifacts"]), "artifact inventory")
    for name, digest in receipt["artifacts"].items():
        require(hashlib.sha256((base / name).read_bytes()).hexdigest() == digest, "artifact hash: " + name)
    require(hashlib.sha256(Path(SPEC.origin).read_bytes()).hexdigest() == receipt["trace_auditor_sha256"],
            "trace auditor drift")
    failed = audit_failure(read(base / "horizon-support-trace.log.gz"))
    launch = json.loads((base / "horizon-initial-bodies-launch.json").read_text())
    require(launch["returncode"] == 0 and not launch["competing_workloads"], "physical execution failed or contended")
    for owner in ["native", "brain"]:
        state = launch["source_state"][owner]
        require(state["revision"] == receipt["revisions"][owner] and state["status"] == "",
                "physical source revision or cleanliness: " + owner)
    repaired_text = read(base / "horizon-initial-bodies.log.gz")
    passed = TRACE.audit_trace(repaired_text, 64)
    buffers = {}
    for line in repaired_text.splitlines():
        if line.startswith("mrnx_initial_buffer="):
            row = json.loads(line.split("=", 1)[1])
            buffers.setdefault(row["name"], []).append(row["values"])
    require(len(buffers) == 6 and all(len(v) == 4 and all(x == v[0] for x in v)
                                   for v in buffers.values()), "repaired first-root physical buffers diverge")
    tests = read(base / "initial-bodies-regressions-qualified.log")
    require("100% tests passed out of 10" in tests, "native regressions")
    require("support_initial_bodies_missing_and_short_rejected=1" in tests, "initial-pose admission checks")
    require("command_buffer_address_reuse=not_observed" in tests, "allocator coverage must remain partial")
    require(receipt["qualification"] == {
        "support_initial_pose_ownership": True, "six_point_four_ms_replay_and_dropout": True,
        "allocator_address_reuse_observed": False, "standing": False, "walking": False,
        "registered_anatomical_tissue": False, "experimental_calibration": False}, "qualification boundary")
    return {"retained_failure": failed, "repaired": passed, "native_tests_passed": 10,
            "allocator_address_reuse": "not_observed"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
