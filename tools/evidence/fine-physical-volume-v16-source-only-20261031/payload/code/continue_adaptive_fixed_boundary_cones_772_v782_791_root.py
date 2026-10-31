#!/usr/bin/env python3
"""Versioned 772 continuation using the 782 promoter and shared 791 source gate."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, subprocess, sys

H = Path(__file__).resolve().parent
PRODUCER = H / "quality-partial-cavity-684-root/batch_interior_steiner_kernel_cone_quality_767_root.py"
PROMOTER = H / "quality-partial-cavity-684-root/independent_promoter_782_791_root.py"
SOURCE_VERIFIER = H / "verify_current_source_evidence_791_root.py"
EXPECTED_PRODUCER_SHA256 = "c9d11fbcc2403755f283a9773053dd8e91c4d81dde8f3594c3f29be208a7da75"
EXPECTED_PROMOTER_SHA256 = "8904c4d9832abdde9a717d7457c8e067870aaedda365e0ec85d8667742cf3658"
EXPECTED_SOURCE_VERIFIER_SHA256 = "b4da6ac3509767e11f3df6bd712a9908adf6b9bd0c1efc2122b16ec15df04ed9"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def pin(path):
    return {"path": str(Path(path)), "sha256": sha(path)}

def check_pin(path, expected):
    actual = pin(path)
    if actual["sha256"] != expected:
        raise ValueError("pinned continuation dependency SHA mismatch: " + str(path))
    return actual

def validate_dependencies():
    pins = {
        "producer": check_pin(PRODUCER, EXPECTED_PRODUCER_SHA256),
        "promoter": check_pin(PROMOTER, EXPECTED_PROMOTER_SHA256),
        "source_verifier": check_pin(SOURCE_VERIFIER, EXPECTED_SOURCE_VERIFIER_SHA256),
    }
    spec = importlib.util.spec_from_file_location("promoter_782_791_for_continuation_772", PROMOTER)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import pinned 782/791 promoter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "verify_and_promote", None)):
        raise ValueError("pinned 782/791 promoter has no verify_and_promote entry point")
    return pins

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")

def run(command, log_path):
    with Path(log_path).open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(str(command) + ": " + Path(log_path).read_text()[-2000:])

def main():
    validate_dependencies()
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    state = args.state
    progress = []
    for index in range(args.rounds):
        current = read(state)
        before = sum(current[role + "_current"]["native_quality_failures"] for role in ("Skin", "Fat"))
        if not before:
            break
        output = args.output / ("round%03d" % (index + 1))
        output.mkdir()
        proposal = output / "proposal"
        run([
            sys.executable, str(PRODUCER), "--state", str(state), "--output", str(proposal),
            "--seconds", "90", "--maximum-attempts", "10000", "--maximum-operations", "1000"
        ], output / "producer.log")
        report = read(proposal / "report.json")
        if report["status"] == "NO_VALID_BOUNDED_PROGRESS":
            break
        if not any(row.get("status") == "PASS_INTERIOR_STEINER_KERNEL_CONE_BATCH"
                   for row in report["role_results"]):
            break
        verified = output / "verified-state.json"
        run([
            sys.executable, str(PROMOTER), "--state", str(state), "--journal", str(proposal / "report.json"),
            "--output", str(verified), "--receipt", str(output / "independent-replay.json")
        ], output / "verifier.log")
        current = read(verified)
        counts = {role: current[role + "_current"]["native_quality_failures"] for role in ("Skin", "Fat")}
        if sum(counts.values()) >= before:
            raise ValueError("verified adaptive cone round did not strictly reduce total native failures")
        state = verified
        row = {"round": index + 1, "state": pin(state), **counts}
        progress.append(row)
        write(args.output / "latest-verified.json", {"current": row, "progress": progress})
        print("VERIFIED_ADAPTIVE_CONE", json.dumps(row), flush=True)
    current = read(state)
    remaining = sum(current[role + "_current"]["native_quality_failures"] for role in ("Skin", "Fat"))
    write(args.output / "completion.json", {
        "status": "ZERO_NATIVE_QUALITY_FAILURES" if remaining == 0 else "BOUNDED_METHOD_COMPLETE_NOT_FINISHED",
        "state": pin(state), "progress": progress, "physical_admission": False, "native_steps": 0
    })

if __name__ == "__main__":
    main()
