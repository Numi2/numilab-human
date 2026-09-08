"""Verify bounded prepared recruitment evidence; never promote behavior."""
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct

BASE = Path(__file__).resolve().parent
SCENARIOS = {"recruited", "replay", "zero", "unavailable"}
COUNTS = {"qv": 257, "motor": 416, "activation": 416, "tendon_force": 416}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def audit_trace(text):
    records = {}
    for line in text.splitlines():
        if not line.startswith("prepared_recruitment={"):
            continue
        row = json.loads(line.split("=", 1)[1])
        scenario, root, kind = row["scenario"], row["root"], row["kind"]
        require(row["schema"] == "numi.human.prepared-recruitment-trace.v1", "trace schema")
        require(scenario in SCENARIOS and type(root) is int and root in range(1, 5)
                and kind in COUNTS and row["elapsed_microseconds"] == root * 100, "trace identity or clock")
        key = (scenario, root, kind)
        require(key not in records, "duplicate trace")
        raw = base64.b64decode(row["fp32_le_base64"], validate=True)
        require(len(raw) == COUNTS[kind] * 4, "trace extent")
        values = struct.unpack(f"<{COUNTS[kind]}f", raw)
        require(all(math.isfinite(v) for v in values), "nonfinite trace")
        if kind in {"motor", "activation"}:
            require(all(0 <= v <= 1 for v in values), "excitation or activation bounds")
        if kind == "qv":
            require(abs(sum(v*v for v in values[3:7]) - 1) <= 16 * 2**-23, "root quaternion")
        records[key] = (raw, values)
    require(len(records) == 64, "missing scenario, root or observable")
    for root in range(1, 5):
        for kind in COUNTS:
            require(records["recruited", root, kind][0] == records["replay", root, kind][0], "replay drift")
            require(records["zero", root, kind][0] == records["unavailable", root, kind][0], "dropout changed physical outcome")
        for scenario in SCENARIOS:
            motors = records[scenario, root, "motor"][1]
            if root == 1 or scenario in {"zero", "unavailable"}:
                require(all(v == 0 for v in motors), "unavailable command must be zero")
            else:
                require(max(motors) > 0, "recruitment absent")
    def final(scenario, kind):
        return records[scenario, 4, kind][1]
    differences = {}
    for kind in {"activation", "tendon_force"}:
        difference = max(abs(a-b) for a, b in zip(final("recruited", kind), final("zero", kind)))
        require(difference > 0, "native muscle response absent")
        differences[kind] = difference
    metrics = {
        "scenarios": 4, "accepted_roots_per_scenario": 4, "seconds_per_scenario": 0.0004,
        "full_state_command_activation_force_replay": "bitwise",
        "dropout_matches_zero_command_physics": True,
        "maximum_delivered_excitation": max(final("recruited", "motor")),
        "maximum_terminal_activation_difference": differences["activation"],
        "maximum_terminal_tendon_force_difference_n": differences["tendon_force"],
        "maximum_terminal_q_difference": max(abs(a-b) for a,b in zip(final("recruited", "qv")[:129], final("zero", "qv")[:129])),
    }
    for scenario in ["recruited", "zero"]:
        values = final(scenario, "qv")
        metrics[scenario] = {
            "right_ankle_velocity_rad_s": values[129+108],
            "root_linear_speed_m_s": math.sqrt(sum(v*v for v in values[129:132])),
        }
    return metrics


def verify(base=BASE):
    receipt = json.loads((base / "receipt.json").read_text())
    require(receipt["schema"] == "numi.human.prepared-recruitment-receipt.v1", "receipt schema")
    require(set(receipt["qualification"]) == {"bounded_prepared_recruitment", "loaded_equilibrium", "registered_anatomical_tissue", "standing", "walking", "calibration", "costal_timeout_resolved", "performance", "full_release"}, "qualification coverage")
    for name, value in receipt["qualification"].items():
        require(type(value) is bool and value == (name == "bounded_prepared_recruitment"), "unsupported qualification")
    for artifact in receipt["artifacts"]:
        raw = (base / artifact["path"]).read_bytes()
        require(len(raw) == artifact["bytes"] and hashlib.sha256(raw).hexdigest() == artifact["sha256"], "artifact drift")
    for label in ["describe-clean", "recruited-clean", "legacy-clean"]:
        launch = json.loads((base / (label + "-launch.json")).read_text())
        require(launch["returncode"] == 0, "execution failure")
        for owner in ["native", "brain"]:
            source = launch["source_state"][owner]
            require(source["revision"] == receipt[owner + "_commit"] and source["status"] == "", "source drift")
    legacy = json.loads((base / "legacy-clean-launch.json").read_text())
    require(len(legacy["brain_test_binary_sha256"]) == 64 and legacy["brain_test_binary_bytes"] > 0, "binary identity missing")
    for label, tests in [("recruited-clean", 1), ("legacy-clean", 4)]:
        text = gzip.decompress((base / (label + ".log.gz")).read_bytes()).decode()
        require(f"Executed {tests} tests, with 0 failures" in text or
                f"Executed {tests} test, with 0 failures" in text, "test closure missing")
    metrics = audit_trace(gzip.decompress((base / "recruited-clean.log.gz").read_bytes()).decode())
    require(metrics == receipt["measurements"], "measurement drift")
    body = json.loads((base / "body.json").read_text())
    describe = gzip.decompress((base / "describe-clean.log.gz").read_bytes()).decode()
    require(json.JSONDecoder().raw_decode(describe[describe.index('{'):])[0] == body, "native description drift")
    # Re-author from the admitted bytes, never from the recorded metric summary.
    root = next(p for p in base.parents if (p / "src/numilab_human").is_dir())
    manifest = root / "Docs/media/support-stance-20260908/source-manifest.json"
    require(hashlib.sha256(manifest.read_bytes()).hexdigest() == receipt["source_manifest_sha256"], "anatomy identity drift")
    ankle = next(j for j in json.loads(manifest.read_text())["core_tree"]["source_joint_map"] if j["core_v_index"] == 108)
    require(ankle["source_name"] == "ankle_angle_r" and ankle["source_type"] == 3, "angular observable binding drift")
    import sys
    sys.path.insert(0, str(root / "src"))
    from numilab_human.locomotor import compile_program
    prior = root / "Docs/media/prepared-state-20260908"
    require(hashlib.sha256((prior / "receipt.json").read_bytes()).hexdigest() == receipt["prior_prepared_receipt_sha256"], "prepared admission evidence drift")
    payload = gzip.decompress((base / "muscle.nhmyo.gz").read_bytes())
    program = compile_program(body, payload, prepared_state=(prior / "fixture/prepared.nhinit").read_bytes(),
                              length_gain=0, velocity_gain=0, maximum=1)
    require(program == json.loads((base / "prepared-program.json").read_text()), "authoring replay drift")
    return {"status": "bounded_prepared_recruitment_verified", "measurements": metrics, "standing": False, "walking": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
