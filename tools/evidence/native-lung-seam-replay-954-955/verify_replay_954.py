#!/usr/bin/env python3
"""Validate the parameterized replay against retained 952 and identity failures."""
import copy
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

D = pathlib.Path(__file__).resolve().parent
E = pathlib.Path("/Users/n/numi-human-resting-evidence-20261005")
RUNNER = D / "replay_lobe_seam_witnesses.py"
CASE = D / "known-952-witnesses.json"
OUTPUT = D / "replay-952-parameterized.json"
OLD = E / "native-lung-pleura-self-seam-diagnosis-952/report.json"
OLD_ENVELOPE = E / "native-lung-pleura-self-seam-diagnosis-952/captured-displacement-envelope.json"
RESULT = D / "verification-results.json"
RETAINED = D / "retained-artifacts.json"


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def verify_external_pins():
    retained = json.loads(RETAINED.read_text())
    if retained.get("schema") != "numi.human.retained-artifacts.v1":
        raise AssertionError("external retained-input index has an unexpected schema")
    for pin in retained.get("external_inputs", []):
        path = pathlib.Path(pin["path"])
        if not path.is_file():
            raise AssertionError(f"missing pinned external input: {path}")
        if path.stat().st_size != pin["bytes"] or sha(path) != pin["sha256"]:
            raise AssertionError(f"external input path/hash/size mismatch: {path}")
    return {"checked_external_input_count": len(retained.get("external_inputs", [])),
            "retained_index_sha256": sha(RETAINED)}

def run_rejected(case_path, output_path, expected_text):
    if output_path.exists():
        raise RuntimeError(f"test output already exists: {output_path}")
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--case", str(case_path), "--out", str(output_path)],
        text=True, capture_output=True, check=False)
    combined = proc.stdout + proc.stderr
    if proc.returncode != 2 or expected_text not in combined or output_path.exists():
        raise AssertionError(f"expected rejection {expected_text!r}; rc={proc.returncode}, output={combined!r}")
    return {"returncode": proc.returncode, "expected_rejection": expected_text, "output": combined.strip()}


external_pin_result = verify_external_pins()
case = json.loads(CASE.read_text())
actual = json.loads(OUTPUT.read_text())
old_report = json.loads(OLD.read_text())
old_envelope = json.loads(OLD_ENVELOPE.read_text())
if actual["case_sha256"] != sha(CASE):
    raise AssertionError("replay output does not bind the explicit witness case")
if actual["summary"]["witness_count"] != 25:
    raise AssertionError("known selected witness count changed")
if actual["summary"]["captured_native_exact_hit_count"] != 25:
    raise AssertionError("known exact native hit count changed")
if not actual["summary"]["captured_native_exact_hits_remain_unallowed"]:
    raise AssertionError("native intersections must remain unallowed")
if actual["summary"]["binary64_map_relation_counts"] != {"disjoint": 17, "single_point_contact": 8}:
    raise AssertionError("binary64 replay relation classes differ from 952")
if actual["summary"]["source_relation_counts"] != {"disjoint": 17, "single_point_contact": 8}:
    raise AssertionError("source relation classes differ from 952")

old_events = {(tuple(x["parent_owner_pair"]), tuple(x["parent_face_ids"])): x for x in old_report["events"]}
old_pairs = {(tuple(x["owners"]), tuple(x["source_face_ids"])): x for x in old_envelope["pairs"]}
new_events = {(tuple(x["owners"]), tuple(x["face_ids"])): x for x in actual["witnesses"]}
if old_events.keys() != new_events.keys() or old_pairs.keys() != new_events.keys():
    raise AssertionError("explicit pair/face witness inventory differs from retained 952")
max_euclidean_delta = 0.0
for key, event in old_events.items():
    new = new_events[key]
    if (event["source_relation"]["hit"] != new["source_relation"]["hit"]
            or event["source_relation"]["points"] != new["source_relation"]["points"]):
        raise AssertionError(f"source relation mismatch for {key}")
    if (event["high_precision_map_relation"]["hit"] != new["binary64_map_relation"]["hit"]
            or event["high_precision_map_relation"]["points"] != new["binary64_map_relation"]["points"]):
        raise AssertionError(f"binary64 relation mismatch for {key}")
    if (event["captured_native_relation"]["hit"] != new["captured_native_relation"]["hit"]
            or event["captured_native_relation"]["points"] != new["captured_native_relation"]["points"]):
        raise AssertionError(f"native relation mismatch for {key}")
    for before, after in zip(event["vertices"], new["vertices"]):
        diff = abs(before["capture_minus_map64_m"] - after["capture_to_replay_euclidean_diagnostic_m"])
        max_euclidean_delta = max(max_euclidean_delta, diff)
        if diff != 0.0:
            raise AssertionError(f"established replay evaluator mismatch for {key}: {diff}")
        if after["capture_to_replay_l1_upper_m"] < after["capture_to_replay_euclidean_diagnostic_m"]:
            raise AssertionError("upward L1 envelope is below the Euclidean diagnostic")
    if new["pairwise_capture_to_replay_l1_upper_m"] < old_pairs[key]["pairwise_hausdorff_displacement_bound_m"]:
        raise AssertionError(f"L1 envelope fell below established Euclidean envelope for {key}")
    if not new["ideal_gap_within_l1_envelope_diagnostic"]:
        raise AssertionError(f"selected ideal gap exceeds the L1 envelope for {key}")
    if new["kind"] == "pleura_parent_pair" and not all(
            x["native_parent_coordinates_exact_match"] for x in new["pleura_parent_coordinate_matches"]):
        raise AssertionError(f"native pleura parent coordinate lineage mismatch for {key}")

bad_dir = D / "identity-tests"
bad_dir.mkdir(exist_ok=True)
classification = json.loads(pathlib.Path(case["inputs"]["classification_report"]["path"]).read_text())
# Rename source/scan/lineage input labels without changing path/hash evidence.
# The replay binds these records by exact path and digest, not historical labels.
generic_classification = copy.deepcopy(classification)
for old_name, new_name in (("NHA924", "source_payload"),
                           ("938_full_scan_report", "complete_scan"),
                           ("row310_to_parent_face_lineage", "pleura_parent_map")):
    if old_name not in generic_classification["inputs"]:
        raise AssertionError(f"expected pinned record label absent from baseline fixture: {old_name}")
    generic_classification["inputs"][new_name] = generic_classification["inputs"].pop(old_name)
generic_classification_path = bad_dir / "classification-generic-input-labels.json"
generic_classification_path.write_text(json.dumps(generic_classification, sort_keys=True, indent=2) + "\n")
generic_case = copy.deepcopy(case)
generic_case["inputs"]["classification_report"] = {"path": str(generic_classification_path), "sha256": sha(generic_classification_path)}
generic_case_path = bad_dir / "known-952-generic-classification-labels.json"
generic_case_path.write_text(json.dumps(generic_case, sort_keys=True, indent=2) + "\n")
with tempfile.TemporaryDirectory(prefix="generic-classification-labels-", dir=bad_dir) as td:
    generic_output = pathlib.Path(td) / "generic-classification-labels-report.json"
    generic_proc = subprocess.run([sys.executable, str(RUNNER), "--case", str(generic_case_path), "--out", str(generic_output)],
                                  text=True, capture_output=True, check=False)
    if generic_proc.returncode != 0 or not generic_output.is_file():
        raise AssertionError(f"path/hash-bound classification labels failed: {generic_proc.stdout} {generic_proc.stderr}")
    generic_report = json.loads(generic_output.read_text())
    if generic_report["summary"]["witness_count"] != 25:
        raise AssertionError("renamed classification records changed selected witness count")
    generic_label_result = {"case_sha256": sha(generic_case_path), "classification_sha256": sha(generic_classification_path),
                            "report_sha256": sha(generic_output), "witness_count": generic_report["summary"]["witness_count"]}
direct_event = next(x for x in classification["cross"]["events"]
                    if len(x.get("owners", [])) == 2 and all(int(owner) in {305, 306, 307, 308, 309} for owner in x["owners"]))
direct_case = copy.deepcopy(case)
direct_case["witnesses"].append({"kind": "lobe_pair", "owners": direct_event["owners"], "face_ids": direct_event["face_ids"]})
direct_case_path = bad_dir / "known-952-plus-lobe-cross.json"
direct_case_path.write_text(json.dumps(direct_case, sort_keys=True, indent=2) + "\n")
with tempfile.TemporaryDirectory(prefix="direct-lobe-pair-", dir=bad_dir) as td:
    direct_output = pathlib.Path(td) / "direct-lobe-pair-report.json"
    direct_proc = subprocess.run([sys.executable, str(RUNNER), "--case", str(direct_case_path), "--out", str(direct_output)],
                                 text=True, capture_output=True, check=False)
    if direct_proc.returncode != 0 or not direct_output.is_file():
        raise AssertionError(f"direct lobe-pair witness failed: {direct_proc.stdout} {direct_proc.stderr}")
    direct_report = json.loads(direct_output.read_text())
    direct_result = {"case_sha256": sha(direct_case_path), "report_sha256": sha(direct_output),
                     "witness_count": direct_report["summary"]["witness_count"],
                     "direct_lobe_pair": next(x for x in direct_report["witnesses"] if x["kind"] == "lobe_pair")}
    if direct_result["witness_count"] != 26 or not direct_result["direct_lobe_pair"]["captured_native_relation"]["hit"]:
        raise AssertionError("direct lobe-pair path did not retain the selected exact native event")
bad_source = copy.deepcopy(case)
bad_source["inputs"]["source_nha"]["sha256"] = "0" * 64
bad_source_path = bad_dir / "wrong-source-hash.json"
bad_source_path.write_text(json.dumps(bad_source, sort_keys=True, indent=2) + "\n")
source_result = run_rejected(bad_source_path, bad_dir / "wrong-source-output.json", "source_nha SHA-256 mismatch")

wrong_step = 6111
wrong_receipt_path = E / f"native-terminal-cycle-931/accepted-geometry/step-{wrong_step}.receipt.json"
wrong_receipt = json.loads(wrong_receipt_path.read_text())
wrong_case = copy.deepcopy(case)
wrong_case["accepted_step"] = wrong_step
wrong_case["inputs"]["accepted_receipt"] = {"path": str(wrong_receipt_path), "sha256": sha(wrong_receipt_path)}
wrong_case["inputs"]["native_pack"] = {"path": wrong_receipt["accepted_pack_path"], "sha256": wrong_receipt["pack_file_sha256"]}
param = wrong_receipt["skin_source_mapping"]["anatomy_parameters"]
wrong_case["inputs"]["anatomy_parameters"] = {"path": param["path"], "sha256": param["sha256"]}
wrong_case_path = bad_dir / "wrong-capture-step.json"
wrong_case_path.write_text(json.dumps(wrong_case, sort_keys=True, indent=2) + "\n")
capture_result = run_rejected(wrong_case_path, bad_dir / "wrong-capture-output.json", "full scan pack/receipt identity differs")

result = {
    "schema": "numi.human.parameterized-lobe-seam-replay-verification.v1",
    "status": "passed",
    "runner_sha256": sha(RUNNER),
    "case_sha256": sha(CASE),
    "successful_replay_sha256": sha(OUTPUT),
    "successful_replay_comparison": {
        "witnesses": 25,
        "source_relations": actual["summary"]["source_relation_counts"],
        "binary64_relations": actual["summary"]["binary64_map_relation_counts"],
        "captured_exact_hits": actual["summary"]["captured_native_exact_hit_count"],
        "max_euclidean_diagnostic_delta_vs_952_m": max_euclidean_delta,
        "all_l1_bounds_dominate_euclidean_diagnostics": True,
        "all_pleura_parent_lineage_coordinates_exact": True,
    },
    "direct_lobe_pair_witness": direct_result,
    "classification_input_labels_are_path_hash_bound": generic_label_result,
    "external_retained_inputs": external_pin_result,
    "wrong_identity_rejections": {
        "source": source_result,
        "capture": capture_result,
    },
    "limits": ["Tests use the known 952 terminal capture and witnesses only.", "This offline replay does not qualify other captures or change exact-intersection acceptance."],
}
RESULT.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
print(json.dumps(result, sort_keys=True, indent=2))
print("verification_sha256", sha(RESULT))
