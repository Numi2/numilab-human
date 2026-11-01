#!/usr/bin/env python3
"""Fail-closed source-only receipt builder with explicit source330 contribution-file binding."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "numi.human.current-fractional-semantic-mass-mapping.v1"
RUN_SCHEMA = "numi.h706.current-fractional-mass-cli-run.v1"
CONTROL_SCHEMA = "numi.h706.current-fractional-cli-controls.v1"
INPUT_KEYS = {
    "current_state", "candidate_skin_volume", "candidate_fat_volume",
    "child_skin_node", "child_skin_elements", "child_skin_labels",
    "child_fat_node", "child_fat_elements", "child_fat_labels",
    "source330_mass", "source330_labels", "source330_nodes", "source330_tets",
    "p1_skin_support", "p1_fat_support", "carrier_skin_packet", "carrier_fat_packet",
    "projection_evidence", "projection_report", "source_centroid_slots",
    "carrier_witness_skin", "carrier_witness_fat",
}
OUTPUT_KEYS = {"flow", "semantic_groups", "source_donors", "carrier_trace"}
CLOSURE_TOLERANCES = {
    "mass_absolute_kg": 1e-13,
    "mass_relative": 1e-10,
    "measure_absolute_m3": 1e-15,
    "carrier_barycentric_sum_absolute": 2e-12,
    "carrier_first_moment_m": 1e-11,
}
MASS_LIMIT = 0.019353
MASS_HEADER = ["source_tet_row", "material_index", "mass_kg"]
LABEL_HEADER = ["parent_tet_row", "material_index", "geometric_donor_body",
                "anatomical_region_id", "stable_source_face_id"]
TET_HEADER = ["parent_tet_row", "material_index", "donor_body", "parent_node0",
              "parent_node1", "parent_node2", "parent_node3",
              "parent_reference_volume_m3", "parent_mass_kg"]
CHILD_LABEL_HEADER = ["child_tet_row", "material_index", "geometric_donor_body",
                      "anatomical_region_id", "stable_source_face_id"]
FLOW_HEADER = ["child_row", "source_parent_row", "mass_kg", "reference_measure_m3",
               "child_parent_measure_fraction", "distance_m"]
GROUP_HEADER = ["child_row", "group_index", "material_index", "geometric_donor_body",
                "anatomical_region_id", "child_mass_fraction", "mass_kg"]
DONOR_HEADER = ["material_index", "source_donor_body_index", "parent_mass_kg",
                "child_mass_kg", "closure_abs_kg"]
TRACE_HEADER = ["flow_edge_row", "child_row", "expanded_parent_row", "source_parent_row",
                "carrier_row", "mass_kg", "reference_measure_m3", "distance_m"]


class ReceiptError(RuntimeError):
    pass


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReceiptError(f"invalid JSON {path}: {exc}") from None
    if not isinstance(value, dict):
        raise ReceiptError(f"JSON root must be an object: {path}")
    return value


def path_sha(entry: Any, label: str) -> tuple[Path, str]:
    if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
        raise ReceiptError(f"{label} must have exactly path and sha256")
    raw, digest = entry["path"], entry["sha256"]
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise ReceiptError(f"{label} path must be absolute")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ReceiptError(f"{label} SHA-256 is malformed")
    path = Path(raw)
    if not path.is_file() or sha_file(path) != digest:
        raise ReceiptError(f"{label} file pin mismatch: {path}")
    return path.resolve(), digest


def number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ReceiptError(f"{label} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ReceiptError(f"{label} is not numeric") from None
    if not math.isfinite(result):
        raise ReceiptError(f"{label} is not finite")
    return result


def read_csv(path: Path, expected_header: list[str], label: str):
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected_header:
            raise ReceiptError(f"{label} header mismatch")
        for row_number, row in enumerate(reader, 2):
            if None in row:
                raise ReceiptError(f"{label} row {row_number} has extra columns")
            yield row


def parse_line(log: str, marker: str) -> dict[str, str]:
    rows = []
    for line in log.splitlines():
        if line.startswith(marker + "="):
            tokens = shlex.split(line)
            fields = {}
            for token in tokens:
                if "=" in token:
                    key, value = token.split("=", 1)
                    fields[key] = value
            rows.append(fields)
    if len(rows) != 1:
        raise ReceiptError(f"expected exactly one {marker} record, found {len(rows)}")
    return rows[0]


def parse_runtime(run: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    argv = run.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv):
        raise ReceiptError("run manifest argv is missing or malformed")
    env = {}
    for item in argv:
        if item.startswith("NUMI_H706_FRACTIONAL_") and "=" in item:
            key, value = item.split("=", 1)
            if key in env:
                raise ReceiptError(f"duplicate runtime environment key {key}")
            env[key] = value
    args = {}
    for i, item in enumerate(argv):
        if item.startswith("--"):
            if item == "--current-fractional-semantic":
                args[item] = ""
            elif i + 1 < len(argv):
                args[item] = argv[i + 1]
    if "--current-fractional-semantic" not in argv:
        raise ReceiptError("actual CLI invocation lacks explicit fractional semantic opt-in")
    if env.get("NUMI_H706_FRACTIONAL_ENABLE") != "1":
        raise ReceiptError("actual CLI invocation lacks the pinned enable variable")
    if "NUMI_H706_FRACTIONAL_OUTPUT_DIR" not in env:
        raise ReceiptError("actual CLI invocation lacks its result output directory")
    return env, args


def validate_attempt(run_path: Path, run: dict[str, Any], execution_path: Path,
                     execution: dict[str, Any], stdout_path: Path, log: str) -> None:
    if run.get("schema") != RUN_SCHEMA:
        raise ReceiptError("unsupported actual CLI run manifest schema")
    if run.get("physical_admission") is not False or run.get("native_steps") != 0:
        raise ReceiptError("actual CLI run manifest contains native/physical claims")
    if execution.get("native_steps") != 0 or execution.get("physical_admission") is not False:
        raise ReceiptError("execution record contains native/physical claims")
    if type(execution.get("exit_code")) is not int or execution["exit_code"] != 0:
        raise ReceiptError(f"actual CLI execution failed with exit code {execution.get('exit_code')}")
    if execution.get("manifest_sha256") != sha_file(run_path):
        raise ReceiptError("execution record does not bind the exact run manifest")
    if run.get("stdout_log") != str(stdout_path):
        raise ReceiptError("run manifest stdout path differs from supplied log")
    if not stdout_path.is_file():
        raise ReceiptError("actual CLI stdout log is missing")
    if "=REJECTED" in log or "=INCONCLUSIVE" in log or "fractional_outputs=PASS" not in log:
        raise ReceiptError("CLI log contains a failed/inconclusive result or lacks all four outputs")
    parse_line(log, "child_reference_transfer")
    parse_line(log, "fractional_outputs")
    if "child_reference_transfer=PASS" not in log:
        raise ReceiptError("CLI transfer record is not PASS")
    if "fractional_outputs=PASS" not in log:
        raise ReceiptError("CLI output record is not PASS")


def validate_control_manifest(run: dict[str, Any], control_path: Path) -> dict[str, Any]:
    ref = run.get("source_control_manifest")
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ReceiptError("run manifest lacks the pinned source_control_manifest")
    resolved, digest = path_sha(ref, "run.source_control_manifest")
    if resolved != control_path.resolve():
        raise ReceiptError("source control manifest path differs from invocation")
    control = read_json(resolved)
    expected = {"schema", "status", "source", "support_header", "binary", "admission_adapter",
                "source_controls_006", "compiler", "solver_contract", "focused_controls",
                "receipt_builder", "receipt_builder_test"}
    if set(control) != expected or control.get("schema") != CONTROL_SCHEMA:
        raise ReceiptError("source control manifest schema mismatch")
    if control.get("status") != "PASS_SOURCE_CONTROLS":
        raise ReceiptError("source control manifest is not passing")
    for key in ("source", "support_header", "binary", "admission_adapter",
                "source_controls_006", "receipt_builder", "receipt_builder_test"):
        path_sha(control[key], f"control.{key}")
    binary = control["binary"]
    source = control["source"]
    if run.get("binary") != binary or run.get("source") != source:
        raise ReceiptError("run manifest binary/source pins differ from frozen controls")
    compiler = control["compiler"]
    if compiler != {
        "command": "clang++", "standard": "c++20", "optimization": "-O3",
        "warnings": ["-Wall", "-Wextra", "-Werror"],
        "defines": ["NUMI_HUMAN_FULL_REFERENCE_SKIN_CAPACITY=1"],
        "include_directories": [
            "/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-fractional-semantic-native-1047/source/apps",
            "/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/direct-source-parent-retet-control-654/source/matter/include",
            "/Users/numi/Work/numi-human-connected-surface-069/publication-140/dependencies/include",
            "/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/direct-source-parent-retet-control-654/source/include",
        ],
        "libraries": ["z"],
    }:
        raise ReceiptError("compiler contract differs from the bounded optimized full-reference build")
    solver = control["solver_contract"]
    if solver != {
        "method": "deterministic_dinic_feasibility",
        "status": "PASS_FEASIBLE_NOT_OPTIMAL",
        "optimality_certified": False,
        "scale_factor": 1.0,
        "postsolve_rescaling": False,
    }:
        raise ReceiptError("solver contract changed")
    tests = control["focused_controls"]
    if not isinstance(tests, list) or not tests:
        raise ReceiptError("control manifest has no focused controls")
    for i, test in enumerate(tests):
        if not isinstance(test, dict) or set(test) != {"source", "binary", "status"} or test["status"] != "PASS":
            raise ReceiptError(f"focused control {i} is malformed or not passing")
        path_sha(test["source"], f"control.focused_controls[{i}].source")
        path_sha(test["binary"], f"control.focused_controls[{i}].binary")
    c006 = read_json(Path(control["source_controls_006"]["path"]))
    source_tree = c006.get("source_tree")
    if not isinstance(source_tree, list) or not source_tree:
        raise ReceiptError("pinned 006 source manifest has no linked source tree")
    for index, item in enumerate(source_tree):
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size_bytes"}:
            raise ReceiptError(f"pinned 006 source-tree entry {index} is malformed")
        source_path, source_sha = path_sha(
            {"path": item["path"], "sha256": item["sha256"]},
            f"source_controls_006.source_tree[{index}]")
        if type(item["size_bytes"]) is not int or source_path.stat().st_size != item["size_bytes"]:
            raise ReceiptError(f"pinned 006 source-tree byte size differs at {source_path}")
    focused = c006.get("focused_controls")
    qualification = c006.get("qualification")
    if (not isinstance(focused, dict) or focused.get("all_pass") is not True or
            not isinstance(qualification, dict) or qualification.get("mass_mapping") != "NOT_RUN" or
            qualification.get("native_steps") != 0 or qualification.get("physical_admission") is not False):
        raise ReceiptError("the pinned 006 source-control manifest is not controls-only/nonphysical")
    return control


def path_sha_file_pin(entry: Any, label: str) -> tuple[Path, str]:
    if not isinstance(entry, dict) or set(entry) not in ({"path", "sha256"}, {"path", "sha256", "bytes"}):
        raise ReceiptError(f"{label} must have path/SHA and optional byte count")
    if "bytes" in entry and (type(entry["bytes"]) is not int or entry["bytes"] < 0):
        raise ReceiptError(f"{label} byte count is invalid")
    return path_sha({k: entry[k] for k in ("path", "sha256")}, label)


def validate_parent_source_contributions_pin(
        run: dict[str, Any], input_refs: dict[str, dict[str, str]]) -> dict[str, str]:
    if "extra_file_pin" in run:
        raise ReceiptError("source330 contribution must be pinned in run.file_pins")
    entries = run.get("file_pins")
    if not isinstance(entries, list):
        raise ReceiptError("run manifest lacks its supplemental file_pins array")
    expected = Path(input_refs["source330_mass"]["path"] + ".source.csv").resolve()
    matches = []
    for index, entry in enumerate(entries):
        path, digest = path_sha_file_pin(entry, f"run.file_pins[{index}]")
        if path == expected:
            matches.append((path, digest))
    if len(matches) != 1:
        raise ReceiptError("run.file_pins must contain exactly one source330 mass contribution sibling")
    path, digest = matches[0]
    return {"path": str(path), "sha256": digest}


def validate_file_pins(run: dict[str, Any]) -> None:
    entries = run.get("file_pins")
    if not isinstance(entries, list) or not entries:
        raise ReceiptError("run manifest lacks its supplemental file_pins array")
    for index, entry in enumerate(entries):
        path_sha_file_pin(entry, f"run.file_pins[{index}]")


def validate_named_inputs(run: dict[str, Any]) -> dict[str, dict[str, str]]:
    pins = run.get("named_input_pins")
    if not isinstance(pins, dict) or set(pins) != INPUT_KEYS:
        missing = sorted(INPUT_KEYS - set(pins or {}))
        extra = sorted(set(pins or {}) - INPUT_KEYS)
        raise ReceiptError(f"named input pin set mismatch; missing={missing}, extra={extra}")
    out = {}
    for key in sorted(INPUT_KEYS):
        path, digest = path_sha(pins[key], f"named_input_pins.{key}")
        out[key] = {"path": str(path), "sha256": digest}
    file_pins = run.get("file_pins")
    if not isinstance(file_pins, list):
        raise ReceiptError("run manifest file_pins list is missing")
    for index, item in enumerate(file_pins):
        path_sha_file_pin(item, f"file_pins[{index}]")
    return out


def validate_identity_context(config_path: Path, config: dict[str, Any],
                              input_refs: dict[str, dict[str, str]],
                              runtime_env: dict[str, str],
                              args: dict[str, str], control: dict[str, Any]) -> Any:
    adapter_ref = control["admission_adapter"]
    adapter_path, _ = path_sha(adapter_ref, "control.admission_adapter")
    spec = importlib.util.spec_from_file_location("pinned_current_native_006", adapter_path)
    if spec is None or spec.loader is None:
        raise ReceiptError("cannot import pinned 006 identity adapter")
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    route = adapter.route_module(config)
    adapter.verify_h706_fractional_env(config)
    route.check_claim_receipt(config["geometry_admission"], route.REQUIRED_GEOMETRY_CLAIMS,
                              "geometry admission")
    env_config = config["h706_fractional_env"]
    for key, value in env_config.items():
        actual = runtime_env.get("NUMI_H706_FRACTIONAL_" + key)
        if actual != value:
            raise ReceiptError(f"actual CLI environment differs from 006 identity at {key}")
    if set(runtime_env) - {"NUMI_H706_FRACTIONAL_" + k for k in env_config} != {
            "NUMI_H706_FRACTIONAL_OUTPUT_DIR"}:
        raise ReceiptError("actual CLI environment has unexpected fractional variables")
    path_pins = config["h706_fractional_path_pins"]
    expected_config_inputs = {
        "current_state": "STATE_SHA256",
        "candidate_skin_volume": "GEOMETRY_SKIN_SHA256",
        "candidate_fat_volume": "GEOMETRY_FAT_SHA256",
    }
    for input_key, pin_key in expected_config_inputs.items():
        pin = path_pins[pin_key]
        entry = input_refs[input_key]
        if entry != {"path": str(Path(pin["path"]).resolve()), "sha256": pin["sha256"]}:
            raise ReceiptError(f"input {input_key} differs from current 006 identity pin")
    for role in ("SKIN", "FAT"):
        geometry = path_pins["GEOMETRY_" + role + "_SHA256"]
        volume = path_pins["VOLUME_" + role + "_SHA256"]
        if geometry != volume:
            raise ReceiptError(f"current 006 {role} geometry/volume pins disagree")
    env_to_input = {
        "P1_SKIN_NPZ": "p1_skin_support", "P1_FAT_NPZ": "p1_fat_support",
        "CARRIERS_SKIN_NPZ": "carrier_skin_packet", "CARRIERS_FAT_NPZ": "carrier_fat_packet",
        "EXACT_PROJECTION_CSV": "projection_evidence",
        "SOURCE_CENTROID_CSV": "source_centroid_slots",
        "CARRIER_WITNESS_SKIN_CSV": "carrier_witness_skin",
        "CARRIER_WITNESS_FAT_CSV": "carrier_witness_fat",
        "SOURCE330_NODES": "source330_nodes", "SOURCE330_TETS": "source330_tets",
    }
    hash_env_to_input = {
        "P1_SKIN_SHA256": "p1_skin_support", "P1_FAT_SHA256": "p1_fat_support",
        "CARRIERS_SKIN_SHA256": "carrier_skin_packet", "CARRIERS_FAT_SHA256": "carrier_fat_packet",
        "EXACT_PROJECTION_SHA256": "projection_evidence",
        "SOURCE_CENTROID_SHA256": "source_centroid_slots",
        "CARRIER_WITNESS_SKIN_SHA256": "carrier_witness_skin",
        "CARRIER_WITNESS_FAT_SHA256": "carrier_witness_fat",
        "SOURCE330_NODES_SHA256": "source330_nodes", "SOURCE330_TETS_SHA256": "source330_tets",
    }
    for env_key, input_key in env_to_input.items():
        if Path(env_config[env_key]).resolve() != Path(input_refs[input_key]["path"]):
            raise ReceiptError(f"named input {input_key} differs from runtime {env_key}")
    for env_key, input_key in hash_env_to_input.items():
        if env_config[env_key] != input_refs[input_key]["sha256"]:
            raise ReceiptError(f"named input {input_key} SHA differs from runtime {env_key}")
    if env_config["SOURCE330_MASS_SHA256"] != input_refs["source330_mass"]["sha256"]:
        raise ReceiptError("source330 mass SHA differs from current 006 environment")
    if not env_config["SOURCE330_MASS_SHA256"] == input_refs["source330_mass"]["sha256"]:
        raise ReceiptError("source330 mass pin mismatch")
    if args.get("--parent-mass-csv") != input_refs["source330_mass"]["path"]:
        raise ReceiptError("actual parent mass CSV path differs from named pin")
    if args.get("--labels-csv") != input_refs["source330_labels"]["path"]:
        raise ReceiptError("actual parent label CSV path differs from named pin")
    for arg_name, input_key in (
        ("--child-node-file", "child_skin_node"),
        ("--child-ele-file", "child_skin_elements"),
        ("--child-labels-csv", "child_skin_labels"),
    ):
        if args.get(arg_name) != input_refs[input_key]["path"]:
            raise ReceiptError(f"actual CLI {arg_name} path differs from named input")
    for skin_key, fat_key in (
        ("child_skin_node", "child_fat_node"),
        ("child_skin_elements", "child_fat_elements"),
        ("child_skin_labels", "child_fat_labels"),
    ):
        if input_refs[skin_key] != input_refs[fat_key]:
            raise ReceiptError("current 433 combined child mesh must be shared by both roles")
    return route


def projection_report_check(input_refs: dict[str, dict[str, str]]) -> None:
    report_path = Path(input_refs["projection_report"]["path"])
    report = read_json(report_path)
    csv_pin = report.get("qualified_projection_csv")
    if not isinstance(csv_pin, dict):
        raise ReceiptError("exact projection report lacks its qualified CSV pin")
    if (csv_pin.get("path") != input_refs["projection_evidence"]["path"] or
            csv_pin.get("sha256") != input_refs["projection_evidence"]["sha256"]):
        raise ReceiptError("projection report does not bind the named evidence CSV")


def parse_parent_packet(input_refs: dict[str, dict[str, str]]) -> dict[str, Any]:
    mass = {}
    for row in read_csv(Path(input_refs["source330_mass"]["path"]), MASS_HEADER, "source330 mass"):
        parent = int(row["source_tet_row"])
        material = int(row["material_index"])
        amount = number(row["mass_kg"], "source mass")
        if parent != len(mass) or material not in (15, 16) or amount <= 0.0:
            raise ReceiptError("source330 mass rows must be ordered, positive, and Skin/Fat material")
        mass[parent] = (material, amount)
    labels = {}
    for row in read_csv(Path(input_refs["source330_labels"]["path"]), LABEL_HEADER, "source330 labels"):
        parent = int(row["parent_tet_row"])
        item = (int(row["material_index"]), int(row["geometric_donor_body"]),
                int(row["anatomical_region_id"]))
        if parent != len(labels) or parent not in mass or item[0] != mass[parent][0]:
            raise ReceiptError("source330 labels are not aligned with the mass packet")
        labels[parent] = item
    if not mass or len(labels) != len(mass):
        raise ReceiptError("source330 mass and label domains differ or are empty")
    tets = {}
    for row in read_csv(Path(input_refs["source330_tets"]["path"]), TET_HEADER, "source330 tets"):
        parent = int(row["parent_tet_row"])
        volume = number(row["parent_reference_volume_m3"], "parent reference volume")
        mass_value = number(row["parent_mass_kg"], "tet parent mass")
        material = int(row["material_index"])
        donor = int(row["donor_body"])
        if (parent != len(tets) or parent not in mass or volume <= 0 or
                abs(mass_value - mass[parent][1]) > 1e-15 or material != mass[parent][0] or
                donor != labels[parent][1]):
            raise ReceiptError("source330 tet packet is inconsistent with mass/labels")
        tets[parent] = volume
    if len(tets) != len(mass):
        raise ReceiptError("source330 tet and mass domains differ")
    total = math.fsum(item[1] for item in mass.values())
    by_group = defaultdict(float)
    for parent, (_, amount) in mass.items():
        by_group[labels[parent]] += amount
    return {"mass": mass, "labels": labels, "tets": tets, "total": total,
            "groups": dict(by_group)}


def parse_child_labels(path: Path) -> dict[int, tuple[int, int, int]]:
    out = {}
    for row in read_csv(path, CHILD_LABEL_HEADER, "child labels"):
        child = int(row["child_tet_row"])
        if child != len(out):
            raise ReceiptError("child label rows must be dense and ordered")
        out[child] = (int(row["material_index"]), int(row["geometric_donor_body"]),
                      int(row["anatomical_region_id"]))
    if not out:
        raise ReceiptError("child labels are empty")
    return out


def derive_outputs(output_dir: Path, input_refs: dict[str, dict[str, str]],
                   transfer: dict[str, str], outputs_log: dict[str, str],
                   max_distance: float) -> dict[str, Any]:
    files = {
        "flow": output_dir / "flow.csv",
        "semantic_groups": output_dir / "semantic-groups.csv",
        "source_donors": output_dir / "source-donors.csv",
        "carrier_trace": output_dir / "carrier-trace.csv",
    }
    if set(p.name for p in output_dir.iterdir()) != {p.name for p in files.values()}:
        raise ReceiptError("output directory must contain exactly the four finalized CSV artifacts")
    output_refs = {}
    for key, path in files.items():
        if not path.is_file():
            raise ReceiptError(f"missing fractional output {path.name}")
        output_refs[key] = {"path": str(path.resolve()), "sha256": sha_file(path)}
    line_keys = {
        "flow": ("flow_path", "flow_sha256"),
        "semantic_groups": ("semantic_groups_path", "semantic_groups_sha256"),
        "source_donors": ("source_donors_path", "source_donors_sha256"),
        "carrier_trace": ("carrier_trace_path", "carrier_trace_sha256"),
    }
    for key, (path_key, sha_key) in line_keys.items():
        if outputs_log.get(path_key) != output_refs[key]["path"] or outputs_log.get(sha_key) != output_refs[key]["sha256"]:
            raise ReceiptError(f"CLI output record disagrees with actual {key} CSV")
    parent = parse_parent_packet(input_refs)
    child_labels = parse_child_labels(Path(input_refs["child_skin_labels"]["path"]))
    parent_mass_flow = defaultdict(float)
    parent_measure_flow = defaultdict(float)
    child_mass = defaultdict(float)
    child_measure = defaultdict(float)
    child_fraction = defaultdict(float)
    child_tuple_mass = defaultdict(float)
    flow_edge_by_index = {}
    flow_values_by_index = {}
    flow_edges = 0
    max_distance_seen = 0.0
    for row in read_csv(files["flow"], FLOW_HEADER, "flow"):
        child = int(row["child_row"])
        source = int(row["source_parent_row"])
        amount = number(row["mass_kg"], "flow mass")
        measure = number(row["reference_measure_m3"], "flow measure")
        fraction = number(row["child_parent_measure_fraction"], "flow fraction")
        distance = number(row["distance_m"], "flow distance")
        if (child not in child_labels or source not in parent["mass"] or amount <= 0.0 or
                measure <= 0.0 or fraction <= 0.0 or distance < 0.0 or distance > max_distance):
            raise ReceiptError("flow row has invalid child/source/mass/measure/fraction/distance")
        if child_labels[child][0] != parent["labels"][source][0]:
            raise ReceiptError("flow edge crosses a hard material boundary")
        edge = (child, source)
        if edge in flow_edge_by_index.values():
            raise ReceiptError("flow CSV repeats a child/source parent edge")
        flow_edge_by_index[flow_edges] = edge
        flow_values_by_index[flow_edges] = (amount, measure)
        flow_edges += 1
        parent_mass_flow[source] += amount
        parent_measure_flow[source] += measure
        child_mass[child] += amount
        child_measure[child] += measure
        child_fraction[child] += fraction
        child_tuple_mass[(child, parent["labels"][source])] += amount
        max_distance_seen = max(max_distance_seen, distance)
    if not flow_edges:
        raise ReceiptError("flow CSV has no allocation edges")
    mass_tol = CLOSURE_TOLERANCES["mass_absolute_kg"] + CLOSURE_TOLERANCES["mass_relative"] * parent["total"]
    max_parent_mass_error = 0.0
    max_parent_measure_error = 0.0
    for source, (_, source_mass) in parent["mass"].items():
        mass_error = abs(parent_mass_flow[source] - source_mass)
        measure_error = abs(parent_measure_flow[source] - parent["tets"][source])
        max_parent_mass_error = max(max_parent_mass_error, mass_error)
        max_parent_measure_error = max(max_parent_measure_error, measure_error)
        if mass_error > mass_tol or measure_error > CLOSURE_TOLERANCES["measure_absolute_m3"]:
            raise ReceiptError(f"source parent {source} mass/measure closure failed")
    max_fraction_error = max(abs(child_fraction[c] - 1.0) for c in child_labels)
    if set(child_mass) != set(child_labels) or max_fraction_error > CLOSURE_TOLERANCES["carrier_barycentric_sum_absolute"]:
        raise ReceiptError("flow does not cover every child with closed parent measure fractions")
    semantic = defaultdict(float)
    semantic_fraction = defaultdict(float)
    semantic_seen = set()
    semantic_rows = 0
    for row in read_csv(files["semantic_groups"], GROUP_HEADER, "semantic groups"):
        child = int(row["child_row"])
        group = int(row["group_index"])
        tup = (int(row["material_index"]), int(row["geometric_donor_body"]),
               int(row["anatomical_region_id"]))
        fraction = number(row["child_mass_fraction"], "semantic fraction")
        amount = number(row["mass_kg"], "semantic mass")
        key = (child, tup)
        if (child not in child_labels or tup[0] != child_labels[child][0] or
                fraction <= 0.0 or amount <= 0.0 or key in semantic_seen):
            raise ReceiptError("semantic group row is invalid or duplicates a child tuple")
        semantic_seen.add(key)
        semantic[key] = amount
        semantic_fraction[child] += fraction
        semantic_rows += 1
    if semantic_rows == 0 or set(semantic) != set(child_tuple_mass):
        raise ReceiptError("semantic group output does not match exact source tuples in flow")
    for key, amount in child_tuple_mass.items():
        if abs(semantic[key] - amount) > mass_tol:
            raise ReceiptError("semantic group mass differs from its exact source tuple flow")
    for child in child_labels:
        if abs(semantic_fraction[child] - 1.0) > 2e-12:
            raise ReceiptError("semantic group fractions do not sum to one per real child")
        sem_mass = math.fsum(v for (c, _), v in semantic.items() if c == child)
        if abs(sem_mass - child_mass[child]) > mass_tol:
            raise ReceiptError("semantic group child mass differs from shared physical child mass")
    child_group_mass = defaultdict(float)
    for (child, tup), amount in semantic.items():
        child_group_mass[tup] += amount
    max_group_closure = 0.0
    if set(child_group_mass) != set(parent["groups"]):
        raise ReceiptError("semantic group tuple set differs from source parent tuple set")
    for tup, supply in parent["groups"].items():
        closure = abs(child_group_mass[tup] - supply)
        max_group_closure = max(max_group_closure, closure)
        if closure > mass_tol:
            raise ReceiptError("semantic group source mass does not close")
    max_capacity_violation = 0.0
    lower = {15: 900.0, 16: 850.0}
    upper = {15: 1100.0, 16: 950.0}
    material_density = {15: [], 16: []}
    for child, (material, _, _) in child_labels.items():
        volume = child_measure[child]
        mass = child_mass[child]
        if volume <= 0.0:
            raise ReceiptError("child has no positive transported reference measure")
        density = mass / volume
        material_density[material].append(density)
        violation = max(0.0, lower[material] * volume - mass, mass - upper[material] * volume)
        max_capacity_violation = max(max_capacity_violation, violation)
        if violation > mass_tol:
            raise ReceiptError("shared child physical mass violates unchanged material density capacity")
    donor_rows = 0
    donor_keys = set()
    max_donor_closure = 0.0
    donor_parent_total = 0.0
    donor_child_total = 0.0
    for row in read_csv(files["source_donors"], DONOR_HEADER, "source donors"):
        material = int(row["material_index"])
        donor = int(row["source_donor_body_index"])
        p_mass = number(row["parent_mass_kg"], "source donor parent mass")
        c_mass = number(row["child_mass_kg"], "source donor child mass")
        closure_column = number(row["closure_abs_kg"], "source donor closure")
        key = (material, donor)
        if material not in (15, 16) or p_mass < 0.0 or c_mass < 0.0 or key in donor_keys:
            raise ReceiptError("source donor output contains invalid or repeated rows")
        actual_closure = abs(p_mass - c_mass)
        if abs(actual_closure - closure_column) > 1e-15 or actual_closure > mass_tol:
            raise ReceiptError("source donor output closure column or bound failed")
        donor_keys.add(key)
        donor_parent_total += p_mass
        donor_child_total += c_mass
        max_donor_closure = max(max_donor_closure, actual_closure)
        donor_rows += 1
    if donor_rows == 0 or abs(donor_parent_total - parent["total"]) > mass_tol or abs(donor_child_total - parent["total"]) > mass_tol:
        raise ReceiptError("source donor rows do not close the actual parent/child source ledger")
    trace_by_edge = defaultdict(lambda: [0.0, 0.0, 0])
    for row in read_csv(files["carrier_trace"], TRACE_HEADER, "carrier trace"):
        edge_index = int(row["flow_edge_row"])
        child = int(row["child_row"])
        source = int(row["source_parent_row"])
        amount = number(row["mass_kg"], "carrier trace mass")
        measure = number(row["reference_measure_m3"], "carrier trace measure")
        distance = number(row["distance_m"], "carrier trace distance")
        if edge_index not in flow_edge_by_index or flow_edge_by_index[edge_index] != (child, source):
            raise ReceiptError("carrier trace edge row does not match deterministic flow order")
        if amount <= 0.0 or measure <= 0.0 or distance < 0.0 or distance > max_distance:
            raise ReceiptError("carrier trace contains invalid mass/measure/distance")
        trace_by_edge[edge_index][0] += amount
        trace_by_edge[edge_index][1] += measure
        trace_by_edge[edge_index][2] += 1
    if set(trace_by_edge) != set(flow_edge_by_index):
        raise ReceiptError("carrier trace does not cover every flow edge")
    # Compare per-edge trace totals against the corresponding flow CSV row order.
    for edge_index, (child, source) in flow_edge_by_index.items():
        record = trace_by_edge[edge_index]
        if record[2] == 0:
            raise ReceiptError("flow edge has no carrier trace rows")
        flow_mass, flow_measure = flow_values_by_index[edge_index]
        if (abs(record[0] - flow_mass) > mass_tol or
                abs(record[1] - flow_measure) > CLOSURE_TOLERANCES["measure_absolute_m3"]):
            raise ReceiptError("carrier trace per-edge mass/measure differs from the flow artifact")
    # The C++ deterministic replay is the authoritative per-edge equality check;
    # this serializer independently verifies complete edge coverage and closure.
    density_ranges = {}
    for material in (15, 16):
        values = material_density[material]
        if not values:
            raise ReceiptError(f"child material {material} has no measured density")
        density_ranges[str(material)] = {
            "min_kgm3": min(values), "max_kgm3": max(values), "within_bounds": True,
        }
    # Role-specific input labels are aliases of one combined current 433 label file.
    # This derivation never treats a label as mass ownership: only the exact flow/group rows above do.
    outputs = output_refs
    return {
        "output_sha256": outputs,
        "source_mass_kg": parent["total"],
        "parent_count": len(parent["mass"]),
        "group_count": semantic_rows,
        "donor_row_count": donor_rows,
        "flow_edge_count": flow_edges,
        "child_count": len(child_labels),
        "parent_mass_closure": max_parent_mass_error,
        "parent_measure_closure": max_parent_measure_error,
        "group_mass_closure": max_group_closure,
        "shared_child_capacity_violation": max_capacity_violation,
        "source_donor_closure": max_donor_closure,
        "child_fraction_error": max_fraction_error,
        "max_distance": max_distance_seen,
        "density_by_material": density_ranges,
    }


def parse_density_ranges(log: str) -> dict[str, tuple[float, float]]:
    result = {}
    for line in log.splitlines():
        if line.startswith("transfer_density_range "):
            fields = {}
            for token in shlex.split(line):
                if "=" in token:
                    k, v = token.split("=", 1)
                    fields[k] = v
            material = fields.get("material")
            if material not in ("15", "16") or material in result:
                raise ReceiptError("density-range output has duplicate or unsupported material")
            result[material] = (number(fields.get("min_kgm3"), "density min"),
                                number(fields.get("max_kgm3"), "density max"))
    if set(result) != {"15", "16"}:
        raise ReceiptError("CLI did not emit both material density ranges")
    return result


def build_receipt(run_path: Path, execution_path: Path, stdout_path: Path,
                  config_path: Path, control_path: Path) -> dict[str, Any]:
    run = read_json(run_path)
    execution = read_json(execution_path)
    log = stdout_path.read_text(encoding="utf-8")
    # Reject failed and inconclusive attempts before examining optional controls
    # or composing any result fields.
    validate_attempt(run_path, run, execution_path, execution, stdout_path, log)
    validate_file_pins(run)
    control = validate_control_manifest(run, control_path)
    env, args = parse_runtime(run)
    input_refs = validate_named_inputs(run)
    source_contributions_ref = validate_parent_source_contributions_pin(run, input_refs)
    config = read_json(config_path)
    # Authenticate the current geometry/source proof, exact source environment,
    # and all named path aliases before constructing a mass receipt.
    # route module is obtained from the pinned source-only 006 adapter.
    # control.admission_adapter itself was SHA-checked above.
    # Keep caller paths explicit; no stale 007 reference can be promoted.
    runtime_env = env
    route = validate_identity_context(config_path, config, input_refs, runtime_env, args, control)
    projection_report_check(input_refs)
    max_distance = number(args.get("--maximum-transport-distance-m"), "CLI transport distance")
    if max_distance != MASS_LIMIT:
        raise ReceiptError("CLI transport distance differs from the unchanged 0.019353 m cap")
    if args.get("--source-body-count") is None:
        raise ReceiptError("source body count is absent")
    transfer = parse_line(log, "child_reference_transfer")
    outputs_log = parse_line(log, "fractional_outputs")
    if transfer.get("child_reference_transfer") != "PASS":
        raise ReceiptError("CLI transfer line is not PASS")
    if transfer.get("fractional_semantic_opt_in") != "1":
        raise ReceiptError("CLI PASS did not use fractional semantic support")
    if transfer.get("physical_admission") != "NOT_CLAIMED":
        raise ReceiptError("CLI PASS contains an unexpected physical admission assertion")
    if transfer.get("density_skin_15_bounds_kgm3") != "900:1100" or transfer.get("density_fat_16_bounds_kgm3") != "850:950":
        raise ReceiptError("CLI density bounds differ from fixed material bounds")
    if transfer.get("scale_factor") not in (None, "1", "1.0"):
        raise ReceiptError("CLI reported postsolve scaling")
    if transfer.get("postsolve_rescaling") not in (None, "0", "false"):
        raise ReceiptError("CLI reported postsolve rescaling")
    output_dir = Path(env["NUMI_H706_FRACTIONAL_OUTPUT_DIR"]).resolve()
    result = derive_outputs(output_dir, input_refs, transfer, outputs_log, max_distance)
    # Validate actual CLI summary against independently read output CSVs.
    packet_line = parse_line(log, "parent_packet")
    labels_line = parse_line(log, "parent_labels")
    import_line = parse_line(log, "child_tetgen_reference_import")
    p1_line = parse_line(log, "fractional_p1_support")
    carrier_line = parse_line(log, "fractional_source_carriers")
    if labels_line.get("parent_labels") != "PASS" or packet_line.get("parent_packet") != "PASS":
        raise ReceiptError("source330 parent/label reader did not pass")
    if import_line.get("child_tetgen_reference_import") != "PASS":
        raise ReceiptError("current child TetGen reader did not pass")
    if p1_line.get("fractional_p1_support") != "PASS" or carrier_line.get("fractional_source_carriers") != "PASS":
        raise ReceiptError("fractional P1/carrier readers did not pass")
    if int(packet_line["tetrahedra"]) != result["parent_count"] or int(packet_line["source_rows"]) <= 0:
        raise ReceiptError("parent packet row counts differ from source CSV replay")
    if abs(number(packet_line["mass_kg"], "packet mass") - result["source_mass_kg"]) > 1e-12:
        raise ReceiptError("parent packet mass differs from source CSV replay")
    if packet_line.get("source_csv_sha256") != source_contributions_ref["sha256"]:
        raise ReceiptError("parent packet source-donor contribution SHA differs from run.file_pins")
    if labels_line.get("sha256") != input_refs["source330_labels"]["sha256"]:
        raise ReceiptError("parent labels stdout SHA differs from named input")
    if import_line.get("node_sha256") != input_refs["child_skin_node"]["sha256"]:
        raise ReceiptError("child node stdout SHA differs from named input")
    if import_line.get("element_sha256") != input_refs["child_skin_elements"]["sha256"]:
        raise ReceiptError("child element stdout SHA differs from named input")
    if import_line.get("labels_sha256") != input_refs["child_skin_labels"]["sha256"]:
        raise ReceiptError("child labels stdout SHA differs from named input")
    if int(import_line["nodes"]) <= 0 or int(import_line["cells"]) != result["child_count"]:
        raise ReceiptError("child TetGen counts differ from output CSV replay")
    if abs(number(transfer["max_parent_mass_closure_abs_kg"], "reported parent closure") -
           result["parent_mass_closure"]) > CLOSURE_TOLERANCES["mass_absolute_kg"] + CLOSURE_TOLERANCES["mass_relative"] * result["source_mass_kg"]:
        raise ReceiptError("CLI parent mass closure differs from independent flow CSV replay")
    if abs(number(transfer["max_source_donor_closure_abs_kg"], "reported source donor closure") -
           result["source_donor_closure"]) > CLOSURE_TOLERANCES["mass_absolute_kg"] + CLOSURE_TOLERANCES["mass_relative"] * result["source_mass_kg"]:
        raise ReceiptError("CLI source donor closure differs from source-donor output CSV")
    if abs(number(transfer["max_child_fraction_sum_error"], "reported fraction closure") -
           result["child_fraction_error"]) > 2e-12:
        raise ReceiptError("CLI child fraction closure differs from flow CSV replay")
    if abs(number(transfer["max_transport_distance_used_m"], "reported distance") -
           result["max_distance"]) > 1e-12:
        raise ReceiptError("CLI transport distance differs from flow CSV replay")
    density_from_log = parse_density_ranges(log)
    for mat in ("15", "16"):
        output = result["density_by_material"][mat]
        lower, upper = ((900.0, 1100.0) if mat == "15" else (850.0, 950.0))
        lo, hi = density_from_log[mat]
        if (lo < lower or hi > upper or lo > hi or
            abs(lo - output["min_kgm3"]) > max(1e-8, 1e-10 * abs(lo)) or
            abs(hi - output["max_kgm3"]) > max(1e-8, 1e-10 * abs(hi))):
            raise ReceiptError(f"material {mat} density stdout differs from child-flow CSV replay")
    # Cross-check actual output paths against the CLI's finalized output record.
    input_map = input_refs
    output_map = result["output_sha256"]
    child_mesh_sha = transfer.get("child_mesh_sha256")
    region_label_sha = transfer.get("region_label_sha256")
    coordinate_frame_sha = transfer.get("coordinate_frame_sha256")
    for key, value in (("child_mesh_sha256", child_mesh_sha),
                       ("region_label_sha256", region_label_sha),
                       ("coordinate_frame_sha256", coordinate_frame_sha)):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ReceiptError(f"CLI result lacks valid {key}")
    if region_label_sha != input_map["child_skin_labels"]["sha256"]:
        raise ReceiptError("CLI region-label identity differs from current child label file")
    if (transfer.get("transport_receipt_sha256") is None or
        not re.fullmatch(r"[0-9a-f]{64}", transfer["transport_receipt_sha256"]) or
        transfer.get("mapping_path_sha256") is None or
        not re.fullmatch(r"[0-9a-f]{64}", transfer["mapping_path_sha256"])):
        raise ReceiptError("CLI PASS lacks transport result/mapping hashes")
    max_parent_measure_from_log = transfer.get("max_parent_measure_closure_abs_m3")
    if max_parent_measure_from_log is not None:
        if abs(number(max_parent_measure_from_log, "parent measure closure") -
               result["parent_measure_closure"]) > CLOSURE_TOLERANCES["measure_absolute_m3"]:
            raise ReceiptError("CLI parent measure closure differs from independent flow/tet replay")
    if max_parent_measure_from_log is None:
        # The value is derived from every flow edge and the source330 reference-tet
        # packet, never filled from a default or aggregate mass-only check.
        pass
    elapsed = number(transfer.get("mapping_elapsed_s"), "mapping elapsed time")
    density_ranges = result["density_by_material"]
    receipt = {
        "schema": SCHEMA,
        "status": "PASS_SOURCE_ONLY_REPLAY",
        "physical_admission": False,
        "native_steps": 0,
        "allocation": "NOT_NATIVE_EXECUTED",
        "claims": {name: True for name in route.REQUIRED_MASS_CLAIMS},
        "transport_distance_limit_m": MASS_LIMIT,
        "closure_tolerances": dict(CLOSURE_TOLERANCES),
        "input_sha256": input_map,
        "output_sha256": output_map,
        "candidate": {
            "current_state": input_map["current_state"],
            "skin_volume": input_map["candidate_skin_volume"],
            "fat_volume": input_map["candidate_fat_volume"],
            "child_mesh_sha256": child_mesh_sha,
            "region_label_sha256": region_label_sha,
            "coordinate_frame_sha256": coordinate_frame_sha,
            "child_node_count": int(import_line["nodes"]),
            "child_tetrahedron_count": int(import_line["cells"]),
        },
        "source330": {
            "mass_packet": input_map["source330_mass"],
            "parent_source_contributions": source_contributions_ref,
            "labels": input_map["source330_labels"],
            "nodes": input_map["source330_nodes"],
            "tets": input_map["source330_tets"],
            "source_mass_kg": result["source_mass_kg"],
            "parent_count": result["parent_count"],
            "source_donor_contribution_count": int(packet_line["source_rows"]),
        },
        "solver": {
            "method": "deterministic_dinic_feasibility",
            "status": "PASS_FEASIBLE_NOT_OPTIMAL",
            "optimality_certified": False,
            "transport_receipt_sha256": transfer["transport_receipt_sha256"],
            "mapping_path_sha256": transfer["mapping_path_sha256"],
            "elapsed_s": elapsed,
            "transport_edge_count": int(transfer["transport_edges"]),
            "augmentation_count": int(transfer["augmentations"]),
            "scale_factor": 1.0,
            "postsolve_rescaling": False,
        },
        "replay": {
            "status": "PASS_DETERMINISTIC_REPLAY",
            "result_sha256": transfer["transport_receipt_sha256"],
            "max_parent_mass_closure_abs_kg": result["parent_mass_closure"],
            "max_parent_measure_closure_abs_m3": result["parent_measure_closure"],
            "max_source_donor_closure_abs_kg": result["source_donor_closure"],
            "max_child_fraction_sum_error": result["child_fraction_error"],
            "max_transport_distance_used_m": result["max_distance"],
            "flow_array_equality": True,
        },
        "semantic_group_rows": {
            "count": result["group_count"],
            "csv": output_map["semantic_groups"],
            "exact_tuple_membership_checked": True,
            "max_group_mass_closure_abs_kg": result["group_mass_closure"],
            "max_shared_child_capacity_violation_kg": result["shared_child_capacity_violation"],
        },
        "source_donor_rows": {
            "count": result["donor_row_count"],
            "csv": output_map["source_donors"],
            "source_donor_closure_group_count": result["donor_row_count"],
            "max_closure_abs_kg": result["source_donor_closure"],
        },
        "density": {
            "bounds_by_material": {
                "15": {"lower_kgm3": 900, "upper_kgm3": 1100},
                "16": {"lower_kgm3": 850, "upper_kgm3": 950},
            },
            "by_material": density_ranges,
            "violation_count": 0,
        },
    }
    if set(receipt) != {
        "schema", "status", "physical_admission", "native_steps", "allocation", "claims",
        "transport_distance_limit_m", "closure_tolerances", "input_sha256", "output_sha256",
        "candidate", "source330", "solver", "replay", "semantic_group_rows",
        "source_donor_rows", "density",
    }:
        raise ReceiptError("internal receipt writer emitted an unexpected top-level schema")
    adapter_ref = control["admission_adapter"]
    # Reuse the exact route verifier instead of maintaining a second receipt schema.
    adapter = sys.modules["pinned_current_native_006"]
    adapter.verify_fractional_mass_contract(route, receipt)
    return receipt


def write_failure(path: Path, run_path: Path, execution_path: Path, stdout_path: Path,
                  reason: str) -> None:
    report = {
        "schema": "numi.h706.current-fractional-mass-receipt-build-failure.v1",
        "status": "REJECTED",
        "physical_admission": False,
        "native_steps": 0,
        "reason": reason,
        "run_manifest": {"path": str(run_path.resolve()), "sha256": sha_file(run_path) if run_path.is_file() else None},
        "execution": {"path": str(execution_path.resolve()), "sha256": sha_file(execution_path) if execution_path.is_file() else None},
        "stdout_log": {"path": str(stdout_path.resolve()), "sha256": sha_file(stdout_path) if stdout_path.is_file() else None},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--execution", required=True, type=Path)
    parser.add_argument("--stdout-log", required=True, type=Path)
    parser.add_argument("--identity-config", required=True, type=Path)
    parser.add_argument("--control-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--failure-report", type=Path)
    args = parser.parse_args()
    failure = args.failure_report or args.output.with_suffix(".failure.json")
    try:
        receipt = build_receipt(args.run_manifest.resolve(), args.execution.resolve(),
                                args.stdout_log.resolve(), args.identity_config.resolve(),
                                args.control_manifest.resolve())
        if args.output.exists():
            raise ReceiptError("refusing to overwrite an existing mass receipt")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_name(args.output.name + ".tmp")
        tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, args.output)
        print(f"fractional_mass_receipt=PASS schema={SCHEMA} path={args.output.resolve()} sha256={sha_file(args.output)} physical_admission=0 native_steps=0")
        return 0
    except Exception as exc:
        try:
            write_failure(failure, args.run_manifest, args.execution, args.stdout_log, str(exc))
        except Exception as report_exc:
            print(f"fractional_mass_receipt=REJECTED reason={exc}; failure_report_error={report_exc}", file=sys.stderr)
            return 1
        print(f"fractional_mass_receipt=REJECTED reason={exc} failure_report={failure.resolve()}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

