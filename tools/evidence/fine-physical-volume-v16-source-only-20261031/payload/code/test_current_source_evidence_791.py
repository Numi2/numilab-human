#!/usr/bin/env python3
"""Corpus and negative-control tests for the versioned 791 source verifier."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np

H = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
MAIN = H / "verify_current_source_evidence_791_root.py"
BASE = H / "verify_current_source_evidence_791_validation_root.py"
EXPECTED_MAIN_SHA = "b4da6ac3509767e11f3df6bd712a9908adf6b9bd0c1efc2122b16ec15df04ed9"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

assert sha(MAIN) == EXPECTED_MAIN_SHA
spec = importlib.util.spec_from_file_location("source_evidence_791_tested", MAIN)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

original_np_load = module.V.np.load
npz_cache = {}
load_calls = 0

class CachedNPZ:
    def __init__(self, archive):
        self.archive = archive
    def __getitem__(self, key):
        return self.archive[key]
    def __contains__(self, key):
        return key in self.archive
    def close(self):
        pass

def cached_np_load(file, *args, **kwargs):
    global load_calls
    load_calls += 1
    key = str(Path(file).resolve())
    if key not in npz_cache:
        npz_cache[key] = original_np_load(file, *args, **kwargs)
    return CachedNPZ(npz_cache[key])

module.V.np.load = cached_np_load
states = {
    "757": H / "all-quality-757-root/round003/trial02/verified-state.json",
    "763": H / "all-quality-763-root/state.json",
    "767": H / "all-quality-767-root/state.json",
    "772": H / "all-quality-772-root/round001/verified-state.json",
    "773": H / "all-quality-773-root/round001/trial01/verified-state.json",
    "776": H / "all-quality-774-resume-root/verified-state-776.json",
    "777": H / "all-quality-777-root/round-02/verified-state.json",
    "780": H / "all-quality-780-root/round003/trial01/verified-state.json",
    "783": H / "all-quality-783-root/round001/verified-state.json",
    "latest789": H / "all-quality-789-root/round-02/verified-state.json",
}
results = {}
started = time.monotonic()
try:
    for label, path in states.items():
        if not path.is_file():
            results[label] = {"status": "MISSING", "path": str(path)}
            continue
        try:
            proof = module.verify_current_source_evidence(path)
            results[label] = {
                "status": proof.get("status"),
                "mode": proof.get("mode"),
                "state_sha256": sha(path),
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if label == "776" and "bounded movement lacks its recorded 775 audit invocation" in error:
                results[label] = {
                    "status": "EXPECTED_LEGACY_REJECTION",
                    "reason": error,
                    "state_sha256": sha(path),
                }
            else:
                results[label] = {
                    "status": "FAIL",
                    "error": error,
                    "state_sha256": sha(path),
                }

    # Exercise the exact fallback rule using measured movement from actual
    # Float32 coordinate arrays: missing count means zero only when unchanged
    # is declared and independently measured.
    baseline = np.array([[0., 0., 0.], [1., 0., 0.]], dtype=np.float32)
    unchanged = baseline.copy()
    changed = baseline.copy()
    changed[1, 0] = np.nextafter(changed[1, 0], np.float32(2.0))
    moved_unchanged = int(np.count_nonzero(
        np.any(baseline.view(np.uint32) != unchanged.view(np.uint32), axis=1)))
    moved_changed = int(np.count_nonzero(
        np.any(baseline.view(np.uint32) != changed.view(np.uint32), axis=1)))
    try:
        actual = module.V._reported_moved_point_count(
            {"f32_coordinate_bytes_unchanged": True}, moved_unchanged, "fixture-zero")
        results["missing-count-zero-measured"] = {"status": "PASS", "reported": actual}
    except Exception as exc:
        results["missing-count-zero-measured"] = {"status": "FAIL", "error": repr(exc)}
    try:
        module.V._reported_moved_point_count(
            {"f32_coordinate_bytes_unchanged": True}, moved_changed, "fixture-changed")
        results["missing-count-changed-measured"] = {"status": "FAIL_ACCEPTED"}
    except ValueError as exc:
        results["missing-count-changed-measured"] = {
            "status": "PASS_REJECTED", "measured_moved": moved_changed, "error": str(exc)}
finally:
    module.V.np.load = original_np_load
    for archive in npz_cache.values():
        try:
            archive.close()
        except Exception:
            pass

expected_legacy = results.get("776", {}).get("status") == "EXPECTED_LEGACY_REJECTION"
all_available_accounted = all(
    r["status"].startswith("PASS") or r["status"] == "EXPECTED_LEGACY_REJECTION"
    for r in results.values()
)
missing = [k for k, v in results.items() if v["status"] == "MISSING"]
out = {
    "schema": "numi.h706.current-source-evidence-791-corpus-controls.v1",
    "status": "PASS_9_STATES_AND_EXPECTED_LEGACY_776_REJECTION"
              if all_available_accounted and expected_legacy and not missing else
              "PASS_WITH_MISSING_EXPECTED_CORPUS_STATE"
              if all_available_accounted and missing else
              "FAIL",
    "helper": {"path": str(MAIN), "sha256": sha(MAIN)},
    "base": {"path": str(BASE), "sha256": sha(BASE)},
    "states": results,
    "npz_unique_paths_loaded": len(npz_cache),
    "npz_load_calls": load_calls,
    "elapsed_s": time.monotonic() - started,
    "physical_admission": False,
    "native_steps": 0,
}
receipt = H / "current-source-evidence-791-root-corpus-tests.json"
receipt.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
print("receipt_sha256", sha(receipt), "receipt_path", receipt)
if not all_available_accounted:
    raise SystemExit(1)
