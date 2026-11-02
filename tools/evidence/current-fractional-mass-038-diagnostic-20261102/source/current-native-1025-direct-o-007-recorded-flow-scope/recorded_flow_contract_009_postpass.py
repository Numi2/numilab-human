from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = "numi.human.current-native-recorded-flow-bundle.v1"
ADAPTER_VERSION = "009-p029-source-controls-recorded-flow"
BUILDER_SHA256 = "e9f7166deb4101fe4e3d532dcbc0e2bbeb8b1937133de53bab682dcb26bb88db"
RECEIPT_BUILDER_COMPAT_PATH = Path(__file__).with_name(
    "build_current_fractional_mass_receipt_013_optional_bytes_compat_009.py"
)
RECEIPT_BUILDER_COMPAT_SHA256 = "eed5a7e577c6311f6334fd5c31bc1609d2c66595f478cd2656ab7ea176b56eae"
RECEIPT_BUILDER_COMPAT_TEST_PATH = Path(__file__).with_name(
    "test_build_current_fractional_mass_receipt_013_optional_bytes_compat_009.py"
)
RECEIPT_BUILDER_COMPAT_TEST_SHA256 = "2d82c6144c23fb9e96ebe616dadfd2054838b5f923c5d39140eb079519e1e258"
RECEIPT_BUILDER_NESTED_COMPAT_PATH = Path(__file__).with_name(
    "build_current_fractional_mass_receipt_013_nested_controls_compat_009.py"
)
RECEIPT_BUILDER_NESTED_COMPAT_SHA256 = "c31ba26110f4b9da3606eb0811bf8fc22f1e39be6e6731c80d0c94b9ee6031e7"
RECEIPT_BUILDER_NESTED_COMPAT_TEST_PATH = Path(__file__).with_name(
    "test_postpass_009_entrypoints.py"
)
RECEIPT_BUILDER_NESTED_COMPAT_TEST_SHA256 = "a54d52a3262dfcc5cb1f558d7312c4c22da242624380fd21d543cd34d2ed5e67"
POSTPASS_TOOL_FILES = {
    "contract": "recorded_flow_contract_009_postpass.py",
    "identity_helper": "materialize_current_fractional_013_identity_009.py",
    "mass_receipt_launcher": "build_current_fractional_mass_receipt_013_nested_controls_compat_009.py",
    "mass_receipt_compat_launcher": "build_current_fractional_mass_receipt_013_optional_bytes_compat_009.py",
    "mass_receipt_compat_test": "test_build_current_fractional_mass_receipt_013_optional_bytes_compat_009.py",
    "materializer": "materialize_current_native_1025_direct_o_009.py",
    "recorded_flow_materializer": "materialize_current_native_1025_direct_o_009_recorded_flow_dir.py",
    "prepare_adapter": "prepare_current_native_1025_direct_o_009.py",
    "postpass_driver": "run_current_native_1025_winning_1047_p029_postpass_009.sh",
    "focused_test": "test_postpass_009_entrypoints.py",
    "converter_source": "/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-fractional-recorded-flow-1050/solver-free-converter-001/convert_current_fractional_combined_flow.py",
    "converter_binary": "/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-fractional-recorded-flow-1050/solver-free-converter-001/convert-current-fractional-combined-flow",
    "motion_verifier": "/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-motion-1026-verifier/verify_current_motion_geometry_1026_v14.py",
    "accepted_surface_exporter": "/Users/numi/Work/skin-fat-native-integration-209/build-native/numi-human-accepted-skin-export",
}


def postpass_tool_refs() -> dict[str, dict[str, str]]:
    refs = {}
    for key, filename in POSTPASS_TOOL_FILES.items():
        candidate = Path(filename)
        path = candidate.resolve() if candidate.is_absolute() else Path(__file__).resolve().with_name(filename)
        if not path.is_file():
            raise ValueError(f"009 post-pass tool is missing: {path}")
        refs[key] = {"path": str(path), "sha256": sha_file(path)}
    return refs


def verify_postpass_tool_pins(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    declared = config.get("fractional_mass_postpass_009")
    if not isinstance(declared, dict) or set(declared) != set(POSTPASS_TOOL_FILES):
        raise ValueError("identity config lacks the closed 009 post-pass tool pin set")
    actual = postpass_tool_refs()
    for key, expected in actual.items():
        got = declared.get(key)
        if not isinstance(got, dict) or set(got) != {"path", "sha256"} or got != expected:
            raise ValueError(f"009 post-pass tool pin changed: {key}")
    return actual


SCHEMA_MASS = "numi.human.current-fractional-semantic-mass-mapping.v1"
FILE_SPECS = {
    "CHILD_MASS_CSV": ("child-mass.csv", ("child_row", "reference_volume_m3", "mass_kg", "density_kg_m3")),
    "TRANSPORT_FLOW_CSV": ("transport-flow.csv", ("child_row", "parent_row", "mass_kg", "source_measure_m3", "distance_m")),
    "SOURCE_CARRIER_FLOW_CSV": ("source-carrier-flow.csv", ("child_row", "parent_row", "expanded_parent_row", "carrier_row", "mass_kg", "source_measure_m3", "distance_m")),
    "CHILD_SOURCE_DONOR_CSV": ("child-source-donor.csv", ("child_row", "source_donor_body", "mass_kg")),
    "SOURCE_PARENT_CSV": ("source-parent.csv", ("parent_row", "global_source330_row", "material", "geometric_donor", "region", "stable_source_face", "mass_kg", "reference_volume_m3", "source_centroid_x_m", "source_centroid_y_m", "source_centroid_z_m")),
    "SOURCE_PARENT_DONOR_CSV": ("source-parent-donor.csv", ("parent_row", "source_donor_body", "mass_kg")),
}
ROLE_KEYS = {
    "status", "recorded_flow_feasibility", "deterministic_optimizer_replay",
    "complete_allocation", "physical_admission", "native_steps", "solver_metrics_scope",
    "paired_solver_run_sha256", "source_carrier_packet_sha256", "source_carrier_array_sha256",
    "source_carrier_flow_trace_sha256", "expected_input_sha256", "transport_receipt_sha256",
    "transport_edges", "candidate_edges", "augmentations", "flow_csv",
}
PFX = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_"
H706_PFX = "NUMI_H706_FRACTIONAL_RECORDED_FLOW_"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _ref(value: Any, label: str, *, size: bool = False) -> tuple[Path, str]:
    keys = {"path", "sha256", "size_bytes"} if size else {"path", "sha256"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} must have exactly {sorted(keys)}")
    raw, digest = value["path"], _hex(value["sha256"], f"{label}.sha256")
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise ValueError(f"{label} path must be absolute")
    path = Path(raw)
    if not path.is_file() or sha_file(path) != digest:
        raise ValueError(f"{label} file hash mismatch: {path}")
    if size and (type(value["size_bytes"]) is not int or value["size_bytes"] != path.stat().st_size):
        raise ValueError(f"{label} byte count mismatch")
    return path.resolve(), digest


def validate_run_source_control_pin(
    run: dict[str, Any], expected_ref: dict[str, Any]
) -> dict[str, str]:
    run_control = run.get("source_control_manifest")
    allowed_keys = ({"path", "sha256"}, {"path", "sha256", "bytes"})
    if not isinstance(expected_ref, dict) or set(expected_ref) != {"path", "sha256"}:
        raise ValueError("007 source-control config pin schema mismatch")
    if not isinstance(run_control, dict) or set(run_control) not in allowed_keys:
        raise ValueError("1047 run lacks exact source control")
    raw_path = expected_ref["path"]
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError("configured source-control path must be absolute")
    control_path = Path(raw_path)
    if str(control_path.resolve()) != raw_path or run_control["path"] != raw_path:
        raise ValueError("1047 run source-control path differs from canonical configured path")
    digest = _hex(expected_ref["sha256"], "configured source-control SHA")
    if run_control["sha256"] != digest or sha_file(control_path) != digest:
        raise ValueError("1047 run does not bind current configured source control")
    if "bytes" in run_control and (
        type(run_control["bytes"]) is not int or run_control["bytes"] != control_path.stat().st_size
    ):
        raise ValueError("1047 run source-control byte count mismatch")
    return {"path": raw_path, "sha256": digest}


def verify_nested_artifact_refs(value: Any, context: str) -> None:
    """Rehash every path/SHA leaf, including versioned nested focus records."""
    if isinstance(value, list):
        for index, item in enumerate(value):
            verify_nested_artifact_refs(item, f"{context}[{index}]")
        return
    if not isinstance(value, dict):
        return
    if "status" in value:
        status = value["status"]
        if not isinstance(status, str) or not status.startswith("PASS"):
            raise ValueError(f"{context} has a non-passing status")
    if "path" in value or "sha256" in value:
        keys = set(value)
        exact_ref_shape = keys in ({"path", "sha256"}, {"path", "sha256", "bytes"})
        compiler_stderr_shape = (
            context.endswith(".compiler_include_trace.stderr") and
            keys == {"path", "sha256", "bytes", "status"} and
            value.get("status") == "PASS"
        )
        if not exact_ref_shape and not compiler_stderr_shape:
            raise ValueError(f"{context} artifact reference has unexpected fields")
        raw, digest = value["path"], _hex(value["sha256"], f"{context}.sha256")
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            raise ValueError(f"{context} artifact path must be absolute")
        path = Path(raw)
        if str(path.resolve()) != raw or not path.is_file() or sha_file(path) != digest:
            raise ValueError(f"{context} artifact path/SHA changed: {raw}")
        if "bytes" in value and (type(value["bytes"]) is not int or value["bytes"] < 0 or
                                  value["bytes"] != path.stat().st_size):
            raise ValueError(f"{context} artifact byte count changed: {raw}")
        return
    for key, item in value.items():
        verify_nested_artifact_refs(item, f"{context}.{key}")


def _strip_verified_bytes(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_verified_bytes(item) for item in value]
    if isinstance(value, dict):
        if "path" in value and "sha256" in value and set(value) in (
                {"path", "sha256"}, {"path", "sha256", "bytes"}):
            return {"path": value["path"], "sha256": value["sha256"]}
        return {key: _strip_verified_bytes(item) for key, item in value.items()}
    return value


def verify_nested_control_refs(control: dict[str, Any]) -> None:
    """Rehash refs; preserve explicit P028 failure and P029 NOT_RUN status."""
    validation_control = control
    if control.get("variant") == "028-implicit-complete-residual-dinic-path-only-materialization":
        validation_control = json.loads(json.dumps(control))
        focused = validation_control.get("focused_controls")
        if not isinstance(focused, list):
            raise ValueError("P028 focused controls are missing")
        diagnostic_rows = [row for row in focused if isinstance(row, dict) and
                           row.get("status") == "FAIL_NUMERICAL_PARITY_DIAGNOSTIC"]
        if len(diagnostic_rows) != 1:
            raise ValueError("P028 nested-reference check requires one explicit failed diagnostic")
        _validate_numeric_parity_diagnostic(diagnostic_rows[0], focused.index(diagnostic_rows[0]))
        solver_contract = validation_control.get("solver_contract")
        expected_not_run_solver = {
            "method": "implicit_dinic_feasibility_on_exact_residual_graph",
            "status": "NOT_RUN_FULL_ALLOCATOR",
            "optimality_certified": False,
            "scale_factor": 1.0,
            "postsolve_rescaling": False,
        }
        if solver_contract != expected_not_run_solver:
            raise ValueError("P028 raw solver contract must retain its exact NOT_RUN status")
        solver_contract.pop("status")
        validation_control["status"] = "PASS_SOURCE_CONTROLS_DIAGNOSTIC_RETAINED"
        diagnostic_rows[0].pop("status")
    elif control.get("variant") == "029-canonical-source-tet-sites":
        if control.get("status") != "PASS_SOURCE_CONTROLS_FULL_ALLOCATOR_NOT_RUN":
            raise ValueError("P029 source controls must retain full allocator NOT_RUN status")
        expected_solver = {
            "method": "implicit_dinic_feasibility_on_exact_residual_graph",
            "status": "NOT_RUN_FULL_ALLOCATOR",
            "optimality_certified": False,
            "scale_factor": 1.0,
            "postsolve_rescaling": False,
        }
        if control.get("solver_contract") != expected_solver:
            raise ValueError("P029 solver contract must retain its exact NOT_RUN declaration")
        rows = control.get("focused_controls")
        pass_rows = [r for r in rows if isinstance(r, dict) and r.get("status") == "PASS"] if isinstance(rows, list) else []
        compile_rows = [r for r in rows if isinstance(r, dict) and r.get("status") == "PASS_COMPILE_ONLY"] if isinstance(rows, list) else []
        if (not isinstance(rows, list) or len(rows) != 2 or len(pass_rows) != 1 or
            len(compile_rows) != 1 or
            pass_rows[0].get("name") != "canonical-source-tet-quadrature-reader-controls" or
            compile_rows[0].get("name") != "production-cli-o3-compile-only"):
            raise ValueError("P029 must retain one reader PASS and one compile-only PASS")
        validation_control = json.loads(json.dumps(control))
        validation_control["solver_contract"].pop("status")
    elif (control.get("status") == "PASS_SOURCE_CONTROLS_FULL_ALLOCATOR_NOT_RUN" and
          isinstance(control.get("variant"), str) and
          control["variant"].startswith("027-")):
        validation_control = json.loads(json.dumps(control))
        solver_contract = validation_control.get("solver_contract")
        expected_not_run_solver = {
            "method": "deterministic_dinic_feasibility",
            "status": "NOT_RUN",
            "optimality_certified": False,
            "scale_factor": 1.0,
            "postsolve_rescaling": False,
        }
        if solver_contract != expected_not_run_solver:
            raise ValueError("P027 raw solver contract must retain its exact NOT_RUN status")
        solver_contract.pop("status")
    verify_nested_artifact_refs(validation_control, "source control")

def _json_close(left: Any, right: Any, *, abs_tol: float = 1e-15) -> bool:
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and
                set(left) == set(right) and
                all(_json_close(left[key], right[key], abs_tol=abs_tol) for key in left))
    if isinstance(left, list) or isinstance(right, list):
        return (isinstance(left, list) and isinstance(right, list) and
                len(left) == len(right) and
                all(_json_close(a, b, abs_tol=abs_tol) for a, b in zip(left, right)))
    if isinstance(left, (float, int)) and not isinstance(left, bool):
        if isinstance(right, bool) or not isinstance(right, (float, int)):
            return False
        return abs(float(left) - float(right)) <= abs_tol
    return type(left) is type(right) and left == right


def _variant_generation(variant: Any, context: str) -> str:
    if not isinstance(variant, str):
        raise ValueError(f"{context} variant is missing")
    match = re.fullmatch(r"([0-9]{3})-[a-z0-9]+(?:-[a-z0-9]+)*", variant)
    if match is None:
        raise ValueError(f"{context} variant is not a versioned lower-case slug")
    return match.group(1)


def _numeric_parity_generations(name: Any, context: str) -> tuple[str, str]:
    match = re.fullmatch(
        r"full-allocator-flow-numerical-parity-([0-9]{3})-vs-([0-9]{3})",
        name if isinstance(name, str) else "")
    if match is None:
        raise ValueError(f"{context} numerical-parity label is malformed")
    return match.group(1), match.group(2)


def _validate_numeric_parity_focus(
    item: Any, index: int, *, expected_generation: str | None = None
) -> dict[str, Any]:
    """Independently verify a versioned numeric-parity report before 013 projection."""
    expected_keys = {
        "name", "status", "source", "binary", "stdout", "stderr",
        "comparison_report", "comparison", "comparator_source", "execution",
    }
    if not isinstance(item, dict) or set(item) != expected_keys:
        raise ValueError(f"focused control {index} numerical-parity row schema mismatch")
    name = item.get("name")
    baseline_generation, candidate_generation = _numeric_parity_generations(
        name, f"focused control {index}")
    if expected_generation is None:
        expected_generation = candidate_generation
    if (item.get("status") != "PASS_NUMERICAL_PARITY" or
        candidate_generation != expected_generation):
        raise ValueError(f"focused control {index} numerical-parity status/name mismatch")
    for key in ("source", "binary", "stdout", "stderr", "comparator_source"):
        _ref_with_optional_bytes(item[key], f"focused control {index} {key}")
    report_path, _ = _ref_with_optional_bytes(
        item["comparison_report"], f"focused control {index} numerical comparison report")
    report = read_json(report_path)
    expected_report_keys = {
        "candidate_counts_and_augmentations", "comparison", "conservation_checks",
        "conservation_counters", "conservation_errors", "maximum_absolute_error",
        "maximum_error_record", "maximum_relative_error", "maximum_relative_error_record",
        "numerical_gate_errors", "schema", "status", "structural_counters",
        "structural_mismatches", "tolerance",
    }
    if set(report) != expected_report_keys:
        raise ValueError("P027 numerical report top-level schema mismatch")
    verify_nested_artifact_refs(report, "P027 numerical report")
    if (report.get("schema") != "numi.h706.flow-numerical-parity.v2" or
        report.get("status") != "PASS_NUMERICAL_PARITY"):
        raise ValueError("P027 numerical report is not a v2 numerical-parity PASS")
    tolerance = report.get("tolerance")
    if tolerance != {"absolute_kg_or_native_value": 1e-12, "relative": 1e-12}:
        raise ValueError("P027 numerical parity tolerance differs from the approved 1e-12 policy")
    comparison = report.get("comparison")
    comparison_keys = {
        "comparison_basis", "raw_stdout_identical", "baseline_stdout", "candidate_stdout",
        "baseline_stderr", "candidate_stderr", "stdout_diff", "baseline_binary",
        "candidate_binary", "baseline_source", "candidate_source",
        "baseline_mass_transport_header", "candidate_mass_transport_header",
        "baseline_fractional_support_header", "candidate_fractional_support_header",
        "comparator_source", "comparator_execution",
    }
    if not isinstance(comparison, dict) or set(comparison) != comparison_keys:
        raise ValueError("numerical comparison schema mismatch")
    if comparison.get("comparison_basis") != "normalized_numeric_rows":
        raise ValueError("P027 numerical comparison basis is unsupported")
    fixture_ids = []
    for key in ("baseline_source", "candidate_source"):
        fixture_path, _ = _ref_with_optional_bytes(comparison[key], f"numeric comparison {key}")
        fixture_match = re.fullmatch(r"FlowParityNumericFixture_([0-9]{3})\.cpp", fixture_path.name)
        if fixture_match is None:
            raise ValueError(f"numeric comparison {key} is not a versioned fixture source")
        fixture_ids.append(fixture_match.group(1))
    if tuple(fixture_ids) != (baseline_generation, candidate_generation):
        raise ValueError("numerical-parity label differs from baseline/candidate source refs")
    if type(comparison.get("raw_stdout_identical")) is not bool:
        raise ValueError("P027 raw stdout equality marker is malformed")
    execution = item.get("execution")
    if not isinstance(execution, dict) or set(execution) != {"command", "exit_code", "stdout", "stderr"}:
        raise ValueError("P027 comparator execution receipt schema mismatch")
    command = execution.get("command")
    if (not isinstance(command, list) or not command or
        any(not isinstance(part, str) or not part for part in command) or
        type(execution.get("exit_code")) is not int or execution["exit_code"] != 0):
        raise ValueError("P027 comparator execution did not exit successfully")
    execution_stdout, _ = _ref_with_optional_bytes(
        execution["stdout"], f"focused control {index} comparator stdout")
    execution_stderr, _ = _ref_with_optional_bytes(
        execution["stderr"], f"focused control {index} comparator stderr")
    if execution_stderr.read_bytes() != b"":
        raise ValueError("P027 comparator emitted stderr")
    comparator_ref = comparison["comparator_source"]
    expected_command = [
        "python3", comparator_ref["path"],
        "--baseline-stdout", comparison["baseline_stdout"]["path"],
        "--candidate-stdout", comparison["candidate_stdout"]["path"],
        "--baseline-stderr", comparison["baseline_stderr"]["path"],
        "--candidate-stderr", comparison["candidate_stderr"]["path"],
        "--baseline-binary", comparison["baseline_binary"]["path"],
        "--candidate-binary", comparison["candidate_binary"]["path"],
        "--baseline-source", comparison["baseline_source"]["path"],
        "--candidate-source", comparison["candidate_source"]["path"],
        "--baseline-mass-header", comparison["baseline_mass_transport_header"]["path"],
        "--candidate-mass-header", comparison["candidate_mass_transport_header"]["path"],
        "--baseline-support-header", comparison["baseline_fractional_support_header"]["path"],
        "--candidate-support-header", comparison["candidate_fractional_support_header"]["path"],
        "--output-json", str(report_path),
        "--output-diff", comparison["stdout_diff"]["path"],
    ]
    if command != expected_command:
        raise ValueError("P027 comparator command differs from its pinned inputs/outputs")
    if _strip_verified_bytes(item["comparison"]) != _strip_verified_bytes(comparison):
        raise ValueError("focused-control numeric comparison differs from its pinned report")
    if _strip_verified_bytes(item["comparator_source"]) != _strip_verified_bytes(comparison["comparator_source"]):
        raise ValueError("focused-control comparator source differs from its pinned report")
    for item_key, report_key in (
        ("source", "candidate_source"), ("binary", "candidate_binary"),
        ("stdout", "candidate_stdout"), ("stderr", "candidate_stderr"),
    ):
        if _strip_verified_bytes(item[item_key]) != _strip_verified_bytes(comparison[report_key]):
            raise ValueError(f"focused-control candidate {item_key} differs from the report")
    comparator_source, comparator_sha = _ref_with_optional_bytes(
        comparison["comparator_source"], "P027 comparator implementation")
    spec = importlib.util.spec_from_file_location(
        f"_p027_numeric_comparator_{comparator_sha[:16]}", comparator_source)
    if spec is None or spec.loader is None:
        raise ValueError("P027 comparator implementation cannot be loaded")
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)

    baseline_stdout, _ = _ref_with_optional_bytes(comparison["baseline_stdout"], "P027 baseline stdout")
    candidate_stdout, _ = _ref_with_optional_bytes(comparison["candidate_stdout"], "P027 candidate stdout")
    _ref_with_optional_bytes(comparison["baseline_stderr"], "P027 baseline stderr")
    _ref_with_optional_bytes(comparison["candidate_stderr"], "P027 candidate stderr")
    baseline_bytes, candidate_bytes = baseline_stdout.read_bytes(), candidate_stdout.read_bytes()
    if comparison["raw_stdout_identical"] != (baseline_bytes == candidate_bytes):
        raise ValueError("P027 raw stdout identity marker is false")
    diff_path, _ = _ref_with_optional_bytes(comparison["stdout_diff"], "P027 raw stdout diff")
    import difflib
    expected_diff = "".join(difflib.unified_diff(
        baseline_stdout.read_text(encoding="utf-8").splitlines(keepends=True),
        candidate_stdout.read_text(encoding="utf-8").splitlines(keepends=True),
        fromfile="baseline-raw-stdout", tofile="candidate-raw-stdout")).encode("utf-8")
    if diff_path.read_bytes() != expected_diff:
        raise ValueError("P027 raw stdout diff does not match its pinned streams")

    b_rows, b_summaries, b_counts = comparator.parse_output(baseline_stdout)
    c_rows, c_summaries, c_counts = comparator.parse_output(candidate_stdout)
    if set(b_rows) != set(c_rows) or b_counts != c_counts or set(b_summaries) != set(c_summaries):
        raise ValueError("P027 numeric parity has structural row, domain, or phase differences")
    phases = {"minimum", "maximum"}
    if (set(b_counts) != phases or set(c_counts) != phases or
        b_counts != {"maximum": {"parents": 2, "children": 5},
                     "minimum": {"parents": 2, "children": 6}}):
        raise ValueError("P027 numeric parity does not cover the pinned min/max domains")
    numeric_samples = []
    numeric_errors = []
    structural_fields = 0
    for key in sorted(b_rows, key=str):
        b_row, c_row = b_rows[key], c_rows[key]
        if b_row["struct"] != c_row["struct"]:
            raise ValueError(f"P027 numeric parity identity mismatch at {key}")
        if len(b_row["numbers"]) != len(c_row["numbers"]):
            raise ValueError(f"P027 numeric parity column mismatch at {key}")
        structural_fields += len(b_row["struct"])
        for column, (base_value, candidate_value) in enumerate(zip(
                b_row["numbers"], c_row["numbers"])):
            if not math.isfinite(base_value) or not math.isfinite(candidate_value):
                raise ValueError(f"P027 numeric parity contains nonfinite values at {key}")
            delta = abs(base_value - candidate_value)
            relative = delta / max(abs(base_value), abs(candidate_value), 1e-300)
            allowed = max(1e-12, 1e-12 * max(abs(base_value), abs(candidate_value)))
            numeric_samples.append((key, column, b_row["fields"][column], base_value, candidate_value))
            if delta > allowed:
                numeric_errors.append((key, column, delta, relative, allowed))
    if not numeric_samples:
        raise ValueError("P027 numerical parity contains no numeric rows")
    if (len(b_rows) != 47 or len(c_rows) != 47 or len(numeric_samples) != 103 or
        structural_fields != 49):
        raise ValueError("P027 parity row/identity coverage differs from the frozen fixture")
    max_abs_sample = max(numeric_samples, key=lambda sample: abs(sample[3] - sample[4]))
    max_rel_sample = max(numeric_samples, key=lambda sample:
                         abs(sample[3] - sample[4]) / max(abs(sample[3]), abs(sample[4]), 1e-300))
    max_abs = abs(max_abs_sample[3] - max_abs_sample[4])
    max_rel = max(abs(base - candidate) / max(abs(base), abs(candidate), 1e-300)
                  for _, _, _, base, candidate in numeric_samples)
    def error_record(sample: tuple[Any, int, str, float, float]) -> dict[str, Any]:
        key, column, field, base, candidate = sample
        return {"key": list(key), "column": column, "field": field,
                "baseline": base, "candidate": candidate}
    if numeric_errors:
        raise ValueError("P027 numerical rows exceed the declared absolute/relative tolerance")
    b_checks, b_errors = comparator.conservation(b_rows, b_counts)
    c_checks, c_errors = comparator.conservation(c_rows, c_counts)
    checks = report.get("conservation_checks")
    if (not isinstance(checks, dict) or set(checks) != {"baseline", "candidate"} or
        len(b_checks) != 2 or len(c_checks) != 2 or
        [row.get("phase") for row in b_checks] != ["maximum", "minimum"] or
        [row.get("phase") for row in c_checks] != ["maximum", "minimum"]):
        raise ValueError("P027 conservation report lacks both actual phases")
    if b_errors or c_errors or not _json_close(checks["baseline"], b_checks) or not _json_close(
            checks["candidate"], c_checks):
        raise ValueError("P027 conservation checks do not match complete stdout recomputation")
    for rows in (b_checks, c_checks):
        for row in rows:
            if (row.get("checks_evaluated") != 9 or
                row.get("parent_input_rows_checked") != row.get("parent_count") or
                row.get("parent_out_rows_checked") != row.get("parent_count") or
                row.get("parent_source_rows_checked") != row.get("parent_count") or
                row.get("child_rows_checked") != row.get("child_count") or
                row.get("child_source_rows_checked") != row.get("child_count")):
                raise ValueError("P027 conservation phase lacks full parent/child row coverage")
    if report.get("conservation_errors") != [] or report.get("numerical_gate_errors") != [] or \
            report.get("structural_mismatches") != []:
        raise ValueError("P027 report contains a hidden structural, numerical, or conservation error")
    b_numeric_rows, c_numeric_rows = len(b_rows), len(c_rows)
    counters = report.get("structural_counters")
    expected_counters = {
        "baseline_numeric_rows": b_numeric_rows, "candidate_numeric_rows": c_numeric_rows,
        "numeric_values_compared": len(numeric_samples),
        "integer_or_identity_values_compared": structural_fields, "mismatch_count": 0,
    }
    if counters != expected_counters:
        raise ValueError("P027 structural/numeric counters differ from stdout recomputation")
    comparator_execution = comparison.get("comparator_execution")
    expected_exec = {
        "numeric_row_count": len(numeric_samples),
        "structural_field_count": structural_fields, "structural_mismatch_count": 0,
        "numeric_gate_error_count": 0, "conservation_error_count": 0,
    }
    if comparator_execution != expected_exec:
        raise ValueError("P027 comparator execution counters differ from stdout recomputation")
    conservation_counters = {
        "checks_per_phase": 9, "baseline_phases": 2, "candidate_phases": 2,
        "baseline_checks_evaluated": 18, "candidate_checks_evaluated": 18, "error_count": 0,
    }
    if report.get("conservation_counters") != conservation_counters:
        raise ValueError("P027 conservation counters are empty or inconsistent")
    if not math.isclose(report.get("maximum_absolute_error", float("nan")), max_abs,
                        rel_tol=0.0, abs_tol=1e-18):
        raise ValueError("P027 maximum absolute error does not match stdout recomputation")
    if not math.isclose(report.get("maximum_relative_error", float("nan")), max_rel,
                        rel_tol=0.0, abs_tol=1e-18):
        raise ValueError("P027 maximum relative error does not match stdout recomputation")
    max_rel_sample = max(numeric_samples, key=lambda sample:
                         abs(sample[3] - sample[4]) / max(abs(sample[3]), abs(sample[4]), 1e-300))
    if (report.get("maximum_error_record") != error_record(max_abs_sample) or
        report.get("maximum_relative_error_record") != error_record(max_rel_sample)):
        raise ValueError("P027 error witness records differ from stdout recomputation")
    try:
        execution_summary = json.loads(execution_stdout.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("P027 comparator stdout is not a JSON execution summary") from exc
    expected_execution_summary = {
        "status": "PASS_NUMERICAL_PARITY", "max_abs": max_abs, "max_rel": max_rel,
        "structural_mismatches": 0, "numeric_errors": 0,
        "conservation_errors": 0, "rows": len(numeric_samples),
    }
    if not _json_close(execution_summary, expected_execution_summary, abs_tol=1e-18):
        raise ValueError("P027 comparator execution summary differs from recomputed results")
    if max_abs > 1e-12 or max_rel > 1e-12:
        raise ValueError("P027 numeric maxima exceed the declared tolerance")
    telemetry = report.get("candidate_counts_and_augmentations")
    if not isinstance(telemetry, dict) or set(telemetry) != phases:
        raise ValueError("P027 candidate telemetry does not cover both phases")
    for phase in phases:
        row = telemetry[phase]
        if not isinstance(row, dict) or set(row) != {
            "baseline", "candidate", "candidate_edge_count_equal",
            "augmentation_count_equal", "receipt_equal", "telemetry_only",
        } or row["telemetry_only"] is not True:
            raise ValueError("P027 candidate telemetry schema is invalid")
        if row["baseline"] != b_summaries[phase] or row["candidate"] != c_summaries[phase]:
            raise ValueError("P027 candidate telemetry differs from stdout")
        if (row["candidate_edge_count_equal"] !=
                (b_summaries[phase]["candidates"] == c_summaries[phase]["candidates"]) or
            row["augmentation_count_equal"] !=
                (b_summaries[phase]["augmentations"] == c_summaries[phase]["augmentations"]) or
            row["receipt_equal"] !=
                (b_summaries[phase]["receipt"] == c_summaries[phase]["receipt"])):
            raise ValueError("P027 candidate telemetry flags differ from stdout")
    return {"source": _strip_verified_bytes(item["source"]),
            "binary": _strip_verified_bytes(item["binary"]), "status": "PASS"}


def _validate_numeric_parity_diagnostic(item: Any, index: int) -> dict[str, Any]:
    """Recompute P028's immutable failed comparator evidence; never call it PASS."""
    expected_keys = {
        "name", "status", "source", "binary", "stdout", "stderr",
        "comparison_report", "comparison", "comparator_source", "execution",
    }
    if not isinstance(item, dict) or set(item) != expected_keys:
        raise ValueError(f"focused control {index} P028 diagnostic schema mismatch")
    match = re.fullmatch(
        r"full-allocation-numeric-comparison-([0-9]{3})-vs-([0-9]{3})-diagnostic",
        item.get("name") if isinstance(item.get("name"), str) else "")
    if match is None or match.groups() != ("024", "028"):
        raise ValueError("P028 diagnostic label does not bind the pinned 024/028 comparison")
    if item.get("status") != "FAIL_NUMERICAL_PARITY_DIAGNOSTIC":
        raise ValueError("P028 numeric comparison diagnostic must retain its FAIL status")
    for key in ("source", "binary", "stdout", "stderr", "comparator_source"):
        _ref_with_optional_bytes(item[key], f"focused control {index} diagnostic {key}")
    report_path, _ = _ref_with_optional_bytes(
        item["comparison_report"], f"focused control {index} diagnostic report")
    report = read_json(report_path)
    expected_report_keys = {
        "candidate_counts_and_augmentations", "comparison", "conservation_checks",
        "conservation_counters", "conservation_errors", "maximum_absolute_error",
        "maximum_error_record", "maximum_relative_error", "maximum_relative_error_record",
        "numerical_gate_errors", "schema", "status", "structural_counters",
        "structural_mismatches", "tolerance",
    }
    if set(report) != expected_report_keys:
        raise ValueError("P028 diagnostic report schema mismatch")
    if (report.get("schema") != "numi.h706.flow-numerical-parity.v2" or
        report.get("status") != "FAIL" or
        report.get("tolerance") != {"absolute_kg_or_native_value": 1e-12, "relative": 1e-12}):
        raise ValueError("P028 report is not the declared v2 FAIL diagnostic")
    comparison = report.get("comparison")
    comparison_keys = {
        "comparison_basis", "raw_stdout_identical", "baseline_stdout", "candidate_stdout",
        "baseline_stderr", "candidate_stderr", "stdout_diff", "baseline_binary",
        "candidate_binary", "baseline_source", "candidate_source",
        "baseline_mass_transport_header", "candidate_mass_transport_header",
        "baseline_fractional_support_header", "candidate_fractional_support_header",
        "comparator_source", "comparator_execution",
    }
    if not isinstance(comparison, dict) or set(comparison) != comparison_keys:
        raise ValueError("P028 comparison schema mismatch")
    if (comparison.get("comparison_basis") != "normalized_numeric_rows" or
        type(comparison.get("raw_stdout_identical")) is not bool or comparison["raw_stdout_identical"]):
        raise ValueError("P028 diagnostic must retain distinct raw allocator outputs")
    for key in comparison_keys - {"comparison_basis", "raw_stdout_identical", "comparator_execution"}:
        _ref_with_optional_bytes(comparison[key], f"P028 comparison {key}")
    if (_strip_verified_bytes(item["comparison"]) != _strip_verified_bytes(comparison) or
        _strip_verified_bytes(item["comparator_source"]) !=
        _strip_verified_bytes(comparison["comparator_source"])):
        raise ValueError("P028 focused diagnostic differs from its pinned report")
    fixture_ids = []
    for key in ("baseline_source", "candidate_source"):
        path, _ = _ref_with_optional_bytes(comparison[key], f"P028 {key}")
        fixture_match = re.fullmatch(r"FlowParityNumericFixture_([0-9]{3})\.cpp", path.name)
        if fixture_match is None:
            raise ValueError(f"P028 {key} is not a versioned fixture")
        fixture_ids.append(fixture_match.group(1))
    if fixture_ids != ["024", "028"]:
        raise ValueError("P028 comparison refs do not bind 024 and 028")
    for item_key, report_key in (("source", "candidate_source"), ("binary", "candidate_binary"),
                                 ("stdout", "candidate_stdout"), ("stderr", "candidate_stderr")):
        if _strip_verified_bytes(item[item_key]) != _strip_verified_bytes(comparison[report_key]):
            raise ValueError(f"P028 candidate {item_key} differs from its report ref")
    execution = item.get("execution")
    execution_shapes = ({"command", "exit_code"}, {"command", "exit_code", "stdout", "stderr"})
    if (not isinstance(execution, dict) or set(execution) not in execution_shapes or
        type(execution.get("exit_code")) is not int or execution["exit_code"] != 2):
        raise ValueError("P028 diagnostic comparator must retain its failing exit code")
    expected_command = [
        "python3", comparison["comparator_source"]["path"],
        "--baseline-stdout", comparison["baseline_stdout"]["path"],
        "--candidate-stdout", comparison["candidate_stdout"]["path"],
        "--baseline-stderr", comparison["baseline_stderr"]["path"],
        "--candidate-stderr", comparison["candidate_stderr"]["path"],
        "--baseline-binary", comparison["baseline_binary"]["path"],
        "--candidate-binary", comparison["candidate_binary"]["path"],
        "--baseline-source", comparison["baseline_source"]["path"],
        "--candidate-source", comparison["candidate_source"]["path"],
        "--baseline-mass-header", comparison["baseline_mass_transport_header"]["path"],
        "--candidate-mass-header", comparison["candidate_mass_transport_header"]["path"],
        "--baseline-support-header", comparison["baseline_fractional_support_header"]["path"],
        "--candidate-support-header", comparison["candidate_fractional_support_header"]["path"],
        "--output-json", str(report_path), "--output-diff", comparison["stdout_diff"]["path"],
    ]
    if execution.get("command") != expected_command:
        raise ValueError("P028 diagnostic command differs from pinned comparator inputs")
    execution_stdout = execution_stderr = None
    if set(execution) == {"command", "exit_code", "stdout", "stderr"}:
        execution_stdout, _ = _ref_with_optional_bytes(execution["stdout"], "P028 comparator stdout")
        execution_stderr, _ = _ref_with_optional_bytes(execution["stderr"], "P028 comparator stderr")
        if execution_stderr.read_bytes():
            raise ValueError("P028 comparator stderr is not empty")
    import difflib
    baseline_stdout, _ = _ref_with_optional_bytes(comparison["baseline_stdout"], "P028 baseline stdout")
    candidate_stdout, _ = _ref_with_optional_bytes(comparison["candidate_stdout"], "P028 candidate stdout")
    diff_path, _ = _ref_with_optional_bytes(comparison["stdout_diff"], "P028 raw stdout diff")
    expected_diff = "".join(difflib.unified_diff(
        baseline_stdout.read_text(encoding="utf-8").splitlines(keepends=True),
        candidate_stdout.read_text(encoding="utf-8").splitlines(keepends=True),
        fromfile="baseline-raw-stdout", tofile="candidate-raw-stdout")).encode("utf-8")
    if diff_path.read_bytes() != expected_diff:
        raise ValueError("P028 stdout diff does not match pinned streams")
    comparator_path, comparator_sha = _ref_with_optional_bytes(
        comparison["comparator_source"], "P028 comparator source")
    spec = importlib.util.spec_from_file_location(f"_p028_numeric_comparator_{comparator_sha[:16]}", comparator_path)
    if spec is None or spec.loader is None:
        raise ValueError("P028 comparator implementation cannot be loaded")
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    b_rows, b_summaries, b_counts = comparator.parse_output(baseline_stdout)
    c_rows, c_summaries, c_counts = comparator.parse_output(candidate_stdout)
    structural = []
    if set(b_rows) != set(c_rows):
        for key in sorted(set(b_rows) - set(c_rows), key=str):
            structural.append({"kind": "candidate_missing_row", "key": list(key)})
        for key in sorted(set(c_rows) - set(b_rows), key=str):
            structural.append({"kind": "unexpected_candidate_row", "key": list(key)})
    if b_counts != c_counts:
        structural.append({"kind": "parent_child_counts_differ", "baseline": b_counts, "candidate": c_counts})
    if set(b_summaries) != set(c_summaries):
        structural.append({"kind": "phase_sets_differ", "baseline": sorted(b_summaries), "candidate": sorted(c_summaries)})
    numeric_errors, numeric_samples = [], []
    for key in sorted(set(b_rows) & set(c_rows), key=str):
        b, c = b_rows[key], c_rows[key]
        if b["struct"] != c["struct"]:
            structural.append({"kind": "integer_or_identity_fields_differ", "key": list(key),
                               "baseline": b["struct"], "candidate": c["struct"]})
        if len(b["numbers"]) != len(c["numbers"]):
            raise ValueError("P028 diagnostic numeric row width mismatch")
        for column, (bv, cv) in enumerate(zip(b["numbers"], c["numbers"])):
            delta = abs(bv - cv)
            relative = delta / max(abs(bv), abs(cv), 1e-300)
            allowed = max(1e-12, 1e-12 * max(abs(bv), abs(cv)))
            numeric_samples.append((key, column, b["fields"][column], bv, cv))
            if delta > allowed:
                numeric_errors.append({"key": list(key), "field": b["fields"][column],
                    "baseline": bv, "candidate": cv, "absolute_error": delta,
                    "relative_error": relative, "allowed_error": allowed})
    b_cons, b_cons_errors = comparator.conservation(b_rows, b_counts)
    c_cons, c_cons_errors = comparator.conservation(c_rows, c_counts)
    conservation_errors = ([{"run": "baseline", **row} for row in b_cons_errors] +
                           [{"run": "candidate", **row} for row in c_cons_errors])
    max_abs, max_rel, max_record, max_rel_record = comparator.numeric_extrema(numeric_samples)
    counters = {
        "baseline_numeric_rows": len(b_rows), "candidate_numeric_rows": len(c_rows),
        "numeric_values_compared": len(numeric_samples),
        "integer_or_identity_values_compared": sum(len(row["struct"]) for row in b_rows.values()),
        "mismatch_count": len(structural),
    }
    telemetry = {}
    for phase in sorted(set(b_summaries) | set(c_summaries)):
        b, c = b_summaries.get(phase, {}), c_summaries.get(phase, {})
        telemetry[phase] = {"baseline": b, "candidate": c,
            "candidate_edge_count_equal": b.get("candidates") == c.get("candidates"),
            "augmentation_count_equal": b.get("augmentations") == c.get("augmentations"),
            "receipt_equal": b.get("receipt") == c.get("receipt"), "telemetry_only": True}
    comparison_exec = {
        "numeric_row_count": len(numeric_samples),
        "structural_field_count": sum(len(row["struct"]) for row in b_rows.values()),
        "structural_mismatch_count": len(structural),
        "numeric_gate_error_count": len(numeric_errors),
        "conservation_error_count": len(conservation_errors),
    }
    conservation_counters = {
        "checks_per_phase": 9, "baseline_phases": len(b_cons), "candidate_phases": len(c_cons),
        "baseline_checks_evaluated": sum(row["checks_evaluated"] for row in b_cons),
        "candidate_checks_evaluated": sum(row["checks_evaluated"] for row in c_cons),
        "error_count": len(conservation_errors),
    }
    if (not structural or not numeric_errors or conservation_errors or
        report["structural_mismatches"] != structural or
        report["numerical_gate_errors"] != numeric_errors or report["conservation_errors"] != [] or
        report["conservation_checks"] != {"baseline": b_cons, "candidate": c_cons} or
        report["structural_counters"] != counters or
        report["conservation_counters"] != conservation_counters or
        comparison["comparator_execution"] != comparison_exec or
        report["candidate_counts_and_augmentations"] != telemetry or
        report["maximum_error_record"] != max_record or
        report["maximum_relative_error_record"] != max_rel_record or
        not math.isclose(report["maximum_absolute_error"], max_abs, rel_tol=0.0, abs_tol=1e-18) or
        not math.isclose(report["maximum_relative_error"], max_rel, rel_tol=0.0, abs_tol=1e-18)):
        raise ValueError("P028 diagnostic report differs from recomputed failed comparison")
    if execution_stdout is not None:
        try:
            summary = json.loads(execution_stdout.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("P028 diagnostic comparator stdout is not JSON") from exc
        expected_summary = {"status": "FAIL", "max_abs": max_abs, "max_rel": max_rel,
            "structural_mismatches": len(structural), "numeric_errors": len(numeric_errors),
            "conservation_errors": 0, "rows": len(numeric_samples)}
        if not _json_close(summary, expected_summary, abs_tol=1e-18):
            raise ValueError("P028 diagnostic execution summary differs from recomputed failure")
    return {"name": item["name"], "status": "FAIL_NUMERICAL_PARITY_DIAGNOSTIC",
            "comparison_report": _strip_verified_bytes(item["comparison_report"]),
            "structural_mismatches": len(structural), "numeric_gate_errors": len(numeric_errors),
            "conservation_errors": 0}


def _focus_for_frozen_013(
    item: Any, index: int, *, variant: str | None = None
) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"focused control {index} must be an object")
    status = item.get("status")
    if status == "PASS_NUMERICAL_PARITY":
        generation = _variant_generation(variant, f"focused control {index}")
        return _validate_numeric_parity_focus(
            item, index, expected_generation=generation)
    if status == "PASS" and set(item) in (
            {"source", "binary", "status"},
            {"source", "binary", "status", "stderr", "stdout"},
            {"binary", "compile_command", "name", "source", "status", "stderr", "stdout"},
            {"binary", "execution", "name", "source", "status", "stderr", "stdout"}):
        if "name" in item and (not isinstance(item["name"], str) or not item["name"]):
            raise ValueError(f"focused control {index} has an invalid named-control label")
        if "compile_command" in item and (
                not isinstance(item["compile_command"], str) or not item["compile_command"]):
            raise ValueError(f"focused control {index} has an invalid compile command")
        if "execution" in item:
            execution = item["execution"]
            source_path, _ = _ref_with_optional_bytes(item["source"], f"focused control {index} source")
            _ref_with_optional_bytes(item["stdout"], f"focused control {index} stdout")
            _ref_with_optional_bytes(item["stderr"], f"focused control {index} stderr")
            if item["binary"] is None:
                expected_command = ["python3", str(source_path)] if item.get("name") == (
                    "numeric-comparator-negative-controls") else None
            else:
                binary_path, _ = _ref_with_optional_bytes(item["binary"], f"focused control {index} binary")
                expected_command = [str(binary_path)]
            if (not isinstance(execution, dict) or set(execution) != {"command", "exit_code"} or
                expected_command is None or execution.get("command") != expected_command or
                type(execution.get("exit_code")) is not int or execution["exit_code"] != 0):
                raise ValueError(f"focused control {index} execution is not its pinned test run")
        return {"source": _strip_verified_bytes(item["source"]),
                "binary": _strip_verified_bytes(item["binary"]), "status": "PASS"}
    compile_only_shapes = (
        {"binary", "compile_command", "name", "source", "status"},
        {"binary", "compile_command", "execution_scope", "name", "source", "status"},
    )
    if status == "PASS_COMPILE_ONLY" and set(item) in compile_only_shapes:
        expected_compile_name = ("production-cli-o3-compile-only" if "execution_scope" in item
                                 else "production-cli-o3-compile")
        if item.get("name") != expected_compile_name:
            raise ValueError(f"focused control {index} compile-only label is unsupported")
        if not isinstance(item.get("compile_command"), str) or not item["compile_command"]:
            raise ValueError(f"focused control {index} compile-only command is missing")
        if "execution_scope" in item and item["execution_scope"] != (
                "Pinned existing O3 CLI build only; no allocator or current-input invocation."):
            raise ValueError(f"focused control {index} compile-only scope is unsupported")
        _ref_with_optional_bytes(item["source"], f"focused control {index} compile-only source")
        _ref_with_optional_bytes(item["binary"], f"focused control {index} compile-only binary")
        return {"source": _strip_verified_bytes(item["source"]),
                "binary": _strip_verified_bytes(item["binary"]),
                "status": "PASS_COMPILE_ONLY"}
    if status == "PASS_EXACT_OUTPUT_PARITY" and set(item) in ({
            "baseline", "bytewise_compare", "candidate", "compared", "source", "status"},
            {"baseline", "bytewise_compare", "candidate", "compared", "name", "source", "status"}):
        if "name" in item and (not isinstance(item["name"], str) or not item["name"]):
            raise ValueError(f"focused control {index} has an invalid named-control label")
        baseline, candidate = item["baseline"], item["candidate"]
        stream_keys = {"binary", "stdout", "variant"}
        stream_keys_with_stderr = stream_keys | {"stderr"}
        baseline_shapes = (stream_keys, stream_keys_with_stderr)
        candidate_shapes = baseline_shapes + (
            stream_keys | {"compile_command"},
            stream_keys_with_stderr | {"compile_command"},
        )
        if (not isinstance(baseline, dict) or set(baseline) not in baseline_shapes or
            not isinstance(candidate, dict) or set(candidate) not in candidate_shapes or
            ("stderr" in baseline) != ("stderr" in candidate) or
            not isinstance(baseline.get("variant"), str) or not baseline["variant"] or
            not isinstance(candidate.get("variant"), str) or not candidate["variant"] or
            baseline["variant"] == candidate["variant"] or
            not isinstance(item.get("compared"), str) or not item["compared"]):
            raise ValueError(f"focused control {index} parity record has an unsupported shape")
        if "compile_command" in candidate and (
                not isinstance(candidate["compile_command"], str) or
                not candidate["compile_command"]):
            raise ValueError(f"focused control {index} parity compile command is invalid")
        baseline_out = _ref_with_optional_bytes(baseline["stdout"], f"focused control {index} baseline stdout")
        candidate_out = _ref_with_optional_bytes(candidate["stdout"], f"focused control {index} candidate stdout")
        if "stderr" in baseline:
            _ref_with_optional_bytes(baseline["stderr"], f"focused control {index} baseline stderr")
            _ref_with_optional_bytes(candidate["stderr"], f"focused control {index} candidate stderr")
        if baseline_out[1] != candidate_out[1] or baseline_out[0].read_bytes() != candidate_out[0].read_bytes():
            raise ValueError(f"focused control {index} exact-output parity is not byte-identical")
        compare_path, _ = _ref_with_optional_bytes(item["bytewise_compare"],
                                                  f"focused control {index} bytewise comparison")
        comparison_bytes = compare_path.read_bytes()
        if comparison_bytes:
            try:
                comparison_text = comparison_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"focused control {index} comparison record is not UTF-8"
                ) from exc
            if not comparison_text.startswith("PASS:"):
                raise ValueError(f"focused control {index} comparison record does not report PASS")
        # Some standard diff tools emit an empty file when the compared output
        # streams are identical. This is accepted only in the explicit parity
        # branch above, after both stdout path/SHA pins are revalidated and the
        # actual bytes are compared for exact equality.
        return {"source": _strip_verified_bytes(item["source"]),
                "binary": _strip_verified_bytes(candidate["binary"]), "status": "PASS"}
    raise ValueError(f"focused control {index} is not a supported passing record")


def _ref_with_optional_bytes(value: Any, context: str) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) not in (
            {"path", "sha256"}, {"path", "sha256", "bytes"}):
        raise ValueError(f"{context} has an unsupported reference shape")
    raw, digest = value["path"], _hex(value["sha256"], f"{context}.sha256")
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise ValueError(f"{context} path must be absolute")
    path = Path(raw)
    if str(path.resolve()) != raw or not path.is_file() or sha_file(path) != digest:
        raise ValueError(f"{context} path/SHA changed")
    if "bytes" in value and (type(value["bytes"]) is not int or value["bytes"] < 0 or
                              value["bytes"] != path.stat().st_size):
        raise ValueError(f"{context} byte count changed")
    return path.resolve(), digest


def _normalized_focus_list(focus: Any, context: str) -> list[Any]:
    if not isinstance(focus, list) or not focus:
        raise ValueError(f"{context} focused controls must be a nonempty list")
    return [_strip_verified_bytes(item) for item in focus]


def _verify_local_trial_include_pins(source_root: Path, source_rows: list[Any]) -> None:
    """Require every quoted include resolving under the trial root to be closure-pinned."""
    root = source_root.resolve(strict=True)
    pinned: set[Path] = set()
    queue: list[Path] = []
    for index, item in enumerate(source_rows):
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
            raise ValueError(f"trial source pin {index} has the wrong schema")
        path = Path(item["path"]).resolve(strict=True)
        if path.is_relative_to(root):
            pinned.add(path)
            queue.append(path)
    seen: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r'^\s*#\s*include\s*"([^"]+)"', line)
            if match is None:
                continue
            relative = match.group(1)
            for candidate in (path.parent / relative, root / relative):
                try:
                    included = candidate.resolve(strict=True)
                except FileNotFoundError:
                    continue
                if not included.is_file() or not included.is_relative_to(root):
                    continue
                if included not in pinned:
                    raise ValueError(f"local source include is not closure-pinned: {included}")
                queue.append(included)
                break


def _control_projection_for_013(control: dict[str, Any], *,
                                rejected_parent_exact_parity: bool = False,
                                mass_receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    expected = {"schema", "status", "source", "support_header", "binary", "admission_adapter",
                "source_controls_006", "compiler", "solver_contract", "focused_controls",
                "receipt_builder", "receipt_builder_test"}
    extended_023 = expected | {
        "compile_command", "mass_transport_header", "parent_manifests",
        "threshold_policy", "variant",
    }
    if (set(control) not in (expected, extended_023) or
        control.get("schema") != "numi.h706.current-fractional-cli-controls.v1"):
        raise ValueError("current source-control manifest schema is unsupported")
    is_extended_023 = set(control) == extended_023
    is_p028_diagnostic = control.get("variant") == "028-implicit-complete-residual-dinic-path-only-materialization"
    is_p029_source_controls = control.get("variant") == "029-canonical-source-tet-sites"
    allowed_statuses = ({"PASS_SOURCE_CONTROLS", "PASS_SOURCE_CONTROLS_FULL_ALLOCATOR_NOT_RUN"}
                        if is_extended_023 else {"PASS_SOURCE_CONTROLS"})
    if is_p028_diagnostic:
        allowed_statuses = {"SOURCE_CONTROLS_PASS_NUMERICAL_PARITY_DIAGNOSTIC_FAIL_FULL_ALLOCATOR_NOT_RUN"}
    if control.get("status") not in allowed_statuses:
        raise ValueError("current source-control status is unsupported")
    if is_extended_023:
        if not isinstance(control.get("variant"), str) or not control["variant"]:
            raise ValueError("extended source-control variant is missing")
        if not isinstance(control.get("compile_command"), str) or not control["compile_command"]:
            raise ValueError("extended source-control compile command is missing")
        if not isinstance(control.get("parent_manifests"), list) or not control["parent_manifests"]:
            raise ValueError("extended source-control parent manifest list is missing")
        threshold = control.get("threshold_policy")
        base_threshold_keys = {
            "bounded_scaled_levels", "complete_positive_fallback", "infeasibility_certificate",
            "maximum_phase", "physical_flow", "productive_delta_cache",
            "residual_capacity_semantics", "search",
        }
        parallel_batch_keys = {
            "candidate_pool_cap", "candidate_source", "edge_budget", "physical_flow",
            "prefix_policy", "selected_cross_edge_cap",
        }
        threshold_keys = set(threshold) if isinstance(threshold, dict) else set()
        terminal_policy_keys = {
            "fallback", "maximum_phase", "minimum_phase", "physical_graph",
        }
        base_and_batch = base_threshold_keys | {"parallel_cross_edge_batch"}
        base_batch_terminal = base_and_batch | {"positive_demand_child_terminal_policy"}
        p027_threshold_keys = {
            "compaction", "cross_edge_selection", "edge_cap", "infeasibility_certificate",
            "maximum_phase", "physical_flow", "search",
        }
        p028_threshold_keys = {
            "compaction", "implicit_materialization", "implicit_search", "live_candidate_cap",
            "physical_flow", "provider_errors", "telemetry",
        }
        supported_shapes = {
            "023-capacity-threshold-residual-forest": frozenset(base_threshold_keys),
            "024-parallel-reached-parent-cross-path-residual-forest": frozenset(base_and_batch),
            "025-terminal-positive-demand-child-residual-forest": frozenset(base_batch_terminal),
            "026-full-positive-residual-per-target-frontier": frozenset(base_batch_terminal),
        }
        variant = control.get("variant")
        live_compaction_shape = frozenset(threshold_keys) == frozenset(p027_threshold_keys)
        implicit_dinic_shape = frozenset(threshold_keys) == frozenset(p028_threshold_keys)
        if live_compaction_shape:
            _variant_generation(variant, "live-compaction source-control")
            if any(not isinstance(threshold[key], str) or not threshold[key]
                   for key in p027_threshold_keys):
                raise ValueError("live-compaction threshold policy schema is invalid")
        elif implicit_dinic_shape:
            if (not (is_p028_diagnostic or is_p029_source_controls) or
                any(not isinstance(threshold[key], str) or not threshold[key]
                    for key in p028_threshold_keys)):
                raise ValueError("implicit-Dinic threshold/variant schema is invalid")
        elif (not isinstance(threshold, dict) or variant not in supported_shapes or
              frozenset(threshold_keys) != supported_shapes[variant] or
              type(threshold.get("bounded_scaled_levels")) is not int or
              (threshold["bounded_scaled_levels"] != 0 if variant ==
               "026-full-positive-residual-per-target-frontier" else threshold["bounded_scaled_levels"] <= 0) or
              any(not isinstance(threshold[key], str) or not threshold[key]
                  for key in base_threshold_keys - {"bounded_scaled_levels"})):
            raise ValueError("extended source-control threshold policy schema is invalid")
        if "parallel_cross_edge_batch" in threshold:
            batch = threshold["parallel_cross_edge_batch"]
            if not isinstance(batch, dict) or set(batch) != parallel_batch_keys:
                raise ValueError("parallel cross-edge batch policy schema is invalid")
            if variant == "026-full-positive-residual-per-target-frontier":
                if any(not isinstance(batch[key], str) or not batch[key] for key in parallel_batch_keys):
                    raise ValueError("026 parallel cross-edge batch policy schema is invalid")
            elif (type(batch.get("candidate_pool_cap")) is not int or batch["candidate_pool_cap"] <= 0 or
                  type(batch.get("selected_cross_edge_cap")) is not int or
                  batch["selected_cross_edge_cap"] <= 0 or
                  any(not isinstance(batch[key], str) or not batch[key]
                      for key in parallel_batch_keys - {"candidate_pool_cap", "selected_cross_edge_cap"})):
                raise ValueError("parallel cross-edge batch policy schema is invalid")
        if "positive_demand_child_terminal_policy" in threshold:
            terminal = threshold["positive_demand_child_terminal_policy"]
            if (not isinstance(terminal, dict) or set(terminal) != terminal_policy_keys or
                any(not isinstance(terminal[key], str) or not terminal[key]
                    for key in terminal_policy_keys)):
                raise ValueError("positive-demand child terminal policy schema is invalid")
        if live_compaction_shape:
            generation = _variant_generation(variant, "live-compaction source-control")
            rows = control["focused_controls"]
            compile_rows = [item for item in rows if isinstance(item, dict) and
                            item.get("status") == "PASS_COMPILE_ONLY"]
            numeric_rows = [item for item in rows if isinstance(item, dict) and
                            item.get("status") == "PASS_NUMERICAL_PARITY"]
            names = [item.get("name") for item in rows if isinstance(item, dict)]
            allowed_statuses = {"PASS", "PASS_NUMERICAL_PARITY", "PASS_COMPILE_ONLY"}
            if (len(rows) != 7 or len(names) != 7 or
                any(not isinstance(name, str) or not name for name in names) or
                len(set(names)) != 7 or len(compile_rows) != 1 or
                len(numeric_rows) != 1 or
                sum(item.get("status") == "PASS" for item in rows) != 5 or
                any(item.get("status") not in allowed_statuses for item in rows)):
                raise ValueError("live-compaction focused-control roster/status shape is invalid")
            item = compile_rows[0]
            if (item.get("name") != "production-cli-o3-compile-only" or
                _strip_verified_bytes(item.get("source")) != _strip_verified_bytes(control["source"]) or
                _strip_verified_bytes(item.get("binary")) != _strip_verified_bytes(control["binary"]) or
                item.get("compile_command") != control.get("compile_command") or
                item.get("execution_scope") !=
                "Pinned existing O3 CLI build only; no allocator or current-input invocation."):
                raise ValueError("027 compile-only record is not the pinned existing CLI build")
            _, numeric_candidate_generation = _numeric_parity_generations(
                numeric_rows[0].get("name"), "live-compaction focused control")
            if numeric_candidate_generation != generation:
                raise ValueError("live-compaction numerical-parity row does not bind its variant")
        if is_p028_diagnostic:
            rows = control["focused_controls"]
            diagnostic_rows = [item for item in rows if isinstance(item, dict) and
                               item.get("status") == "FAIL_NUMERICAL_PARITY_DIAGNOSTIC"]
            compile_rows = [item for item in rows if isinstance(item, dict) and
                            item.get("status") == "PASS_COMPILE_ONLY"]
            pass_rows = [item for item in rows if isinstance(item, dict) and
                         item.get("status") == "PASS"]
            if (len(rows) != 8 or len(diagnostic_rows) != 1 or len(compile_rows) != 1 or
                len(pass_rows) != 6 or
                any(not isinstance(item, dict) or item.get("status") not in {
                    "PASS", "PASS_COMPILE_ONLY", "FAIL_NUMERICAL_PARITY_DIAGNOSTIC"
                } for item in rows)):
                raise ValueError("P028 focused-control roster/status shape is invalid")
            compile_item = compile_rows[0]
            if (set(compile_item) != {"binary", "compile_command", "execution_scope", "name",
                                      "source", "status", "stderr", "stdout"} or
                compile_item.get("name") != "production-cli-o3-compile-only" or
                compile_item.get("execution_scope") != "compile_only_no_full_allocator_run" or
                compile_item.get("compile_command") != control.get("compile_command") or
                _strip_verified_bytes(compile_item.get("source")) !=
                    _strip_verified_bytes(control.get("source")) or
                _strip_verified_bytes(compile_item.get("binary")) !=
                    _strip_verified_bytes(control.get("binary"))):
                raise ValueError("P028 compile-only record differs from pinned production CLI build")
            for stream in ("stdout", "stderr"):
                stream_path, _ = _ref_with_optional_bytes(
                    compile_item[stream], f"P028 compile-only {stream}")
                if stream_path.read_bytes():
                    raise ValueError(f"P028 compile-only {stream} must be empty")
        if variant == "026-full-positive-residual-per-target-frontier":
            compile_only = [item for item in control["focused_controls"]
                            if isinstance(item, dict) and item.get("status") == "PASS_COMPILE_ONLY"]
            if (len(control["focused_controls"]) != 5 or len(compile_only) != 1 or
                sum(item.get("status") == "PASS" for item in control["focused_controls"]) != 3 or
                sum(item.get("status") == "PASS_EXACT_OUTPUT_PARITY" for item in control["focused_controls"]) != 1):
                raise ValueError("026 focused controls must retain three passes, exact parity, and one compile-only record")
            item = compile_only[0]
            if (item.get("name") != "production-cli-o3-compile" or
                _strip_verified_bytes(item.get("source")) != _strip_verified_bytes(control["source"]) or
                _strip_verified_bytes(item.get("binary")) != _strip_verified_bytes(control["binary"]) or
                item.get("compile_command") != control.get("compile_command")):
                raise ValueError("026 compile-only record is not the pinned production CLI compile")
        mass_transport = control.get("mass_transport_header")
        if not isinstance(mass_transport, dict) or set(mass_transport) not in (
                {"path", "sha256"}, {"path", "sha256", "bytes"}):
            raise ValueError("extended source-control mass transport header pin is invalid")
    focused = control.get("focused_controls")
    if not isinstance(focused, list) or not focused:
        raise ValueError("current source-control manifest lacks focused controls")
    projection = {key: _strip_verified_bytes(value) for key, value in control.items()
                  if key in expected and key != "focused_controls"}
    if is_extended_023:
        # The 023 compile has one extra include root: its own immutable trial
        # source directory. Keep the raw control/closure untouched and fully
        # verified, but prove that this is the only compiler-contract delta
        # before adapting to frozen 013's historical shared include contract.
        parent_ref = control["parent_manifests"][0]
        parent_path, _ = _ref_with_optional_bytes(
            parent_ref, "extended source-control immediate parent")
        parent_control = read_json(parent_path)
        current_focus = control.get("focused_controls")
        parent_focused = parent_control.get("focused_controls")
        current_has_numeric_parity = (
            isinstance(current_focus, list) and
            any(isinstance(item, dict) and item.get("status") == "PASS_NUMERICAL_PARITY"
                for item in current_focus))
        parent_has_rejected_exact_parity = (
            current_has_numeric_parity and isinstance(parent_focused, list) and
            any(isinstance(item, dict) and item.get("status") == "PASS_EXACT_OUTPUT_PARITY"
                for item in parent_focused))
        parent_projection = _control_projection_for_013(
            parent_control,
            rejected_parent_exact_parity=parent_has_rejected_exact_parity,
        )
        current_compiler = control["compiler"]
        parent_compiler = parent_projection.get("compiler")
        if not isinstance(current_compiler, dict) or not isinstance(parent_compiler, dict):
            raise ValueError("extended source-control compiler contract is malformed")
        compiler_keys = {"command", "standard", "optimization", "warnings", "defines",
                         "include_directories", "libraries"}
        if set(current_compiler) != compiler_keys or set(parent_compiler) != compiler_keys:
            raise ValueError("extended source-control compiler fields are unsupported")
        compare_current_compiler = dict(current_compiler)
        compare_parent_compiler = dict(parent_compiler)
        if compare_current_compiler.get("command") != compare_parent_compiler.get("command"):
            if not is_p029_source_controls:
                raise ValueError("extended source-control compiler command changed")
            current_command = compare_current_compiler["command"]
            parent_command = compare_parent_compiler["command"]
            current_path = (Path(current_command) if "/" in current_command else
                            Path(shutil.which(current_command) or ""))
            parent_path = (Path(parent_command) if "/" in parent_command else
                           Path(shutil.which(parent_command) or ""))
            if (not current_path.is_file() or not parent_path.is_file() or
                current_path.resolve() != parent_path.resolve() or
                sha_file(current_path) != sha_file(parent_path)):
                raise ValueError("P029 compiler command aliases do not resolve to the same pinned executable")
            compare_current_compiler["command"] = compare_parent_compiler["command"]
        current_dirs = current_compiler.get("include_directories")
        baseline_dirs = parent_compiler.get("include_directories")
        source_ref = control.get("source")
        source_path, _ = _ref_with_optional_bytes(source_ref, "extended source-control source")
        trial_root = str(source_path.parent)
        if (not isinstance(current_dirs, list) or not isinstance(baseline_dirs, list) or
            any(not isinstance(item, str) or not item for item in [*current_dirs, *baseline_dirs])):
            raise ValueError("extended source-control include roots are malformed")
        current_dirs_resolved = [str(Path(item).resolve(strict=True)) for item in current_dirs]
        baseline_dirs_resolved = [str(Path(item).resolve(strict=True)) for item in baseline_dirs]
        trial_root_resolved = str(Path(trial_root).resolve(strict=True))
        if (current_dirs_resolved != [trial_root_resolved, *baseline_dirs_resolved] or
            trial_root_resolved in baseline_dirs_resolved or
            {key: value for key, value in compare_current_compiler.items() if key != "include_directories"} !=
            {key: value for key, value in compare_parent_compiler.items() if key != "include_directories"}):
            raise ValueError("extended source-control compiler is not the baseline contract plus its source root")
        command_tokens = shlex.split(control["compile_command"])
        binary_ref = control.get("binary")
        if not isinstance(binary_ref, dict):
            raise ValueError("extended source-control binary pin is malformed")
        expected_tokens = [current_compiler["command"],
                           f"-std={current_compiler['standard']}",
                           current_compiler["optimization"],
                           *current_compiler["warnings"],
                           *(f"-D{define}" for define in current_compiler["defines"]),
                           *(f"-I{directory}" for directory in current_dirs),
                           source_ref["path"],
                           *(f"-l{library}" for library in current_compiler["libraries"]),
                           "-o", binary_ref.get("path")]
        if command_tokens != expected_tokens:
            raise ValueError("extended source-control compile command differs from its exact compiler/source/binary pins")
        projection["compiler"] = parent_projection["compiler"]
        projection["status"] = "PASS_SOURCE_CONTROLS"
    if rejected_parent_exact_parity:
        # Validate any retained parent exact-output claim against its pinned streams;
        # omit only a demonstrably false claim from this compiler-only projection.
        parity_rows = [item for item in focused if isinstance(item, dict) and
                       item.get("status") == "PASS_EXACT_OUTPUT_PARITY"]
        if len(parity_rows) != 1:
            raise ValueError("parent projection requires exactly one historical exact-output claim")
        parity = parity_rows[0]
        base_ref = parity.get("baseline", {}).get("stdout")
        candidate_ref = parity.get("candidate", {}).get("stdout")
        base_path, base_sha = _ref_with_optional_bytes(base_ref, "historical P026 baseline stdout")
        candidate_path, candidate_sha = _ref_with_optional_bytes(
            candidate_ref, "historical P026 candidate stdout")
        if base_sha == candidate_sha or base_path.read_bytes() == candidate_path.read_bytes():
            raise ValueError("parent exact-output claim is not the preserved bytewise mismatch")
        compare_path, _ = _ref_with_optional_bytes(
            parity.get("bytewise_compare"), "historical P026 exact-parity comparison")
        compare_bytes = compare_path.read_bytes()
        if not compare_bytes:
            raise ValueError("parent exact-output rejection lacks its pinned nonempty diff")
        projected_controls = []
        for index, item in enumerate(focused):
            if item is parity:
                continue
            projected_controls.append(_focus_for_frozen_013(
                item, index, variant=control.get("variant")))
        if len(projected_controls) != 4:
            raise ValueError("parent projection lost a non-parity control")
        projection["focused_controls"] = projected_controls
        projection["status"] = "PASS_COMPILER_CONTRACT_WITH_REJECTED_PARENT_PARITY"
    elif is_p028_diagnostic:
        diagnostic_rows = [item for item in focused if isinstance(item, dict) and
                           item.get("status") == "FAIL_NUMERICAL_PARITY_DIAGNOSTIC"]
        if len(diagnostic_rows) != 1:
            raise ValueError("P028 projection requires one explicit failed diagnostic")
        _validate_numeric_parity_diagnostic(diagnostic_rows[0], focused.index(diagnostic_rows[0]))
        compile_rows = [item for item in focused if isinstance(item, dict) and
                        item.get("status") == "PASS_COMPILE_ONLY"]
        if len(compile_rows) != 1:
            raise ValueError("P028 projection requires exactly one compile-only control")
        compile_item = compile_rows[0]
        projected_controls = []
        for index, item in enumerate(focused):
            if item is diagnostic_rows[0]:
                continue
            if item is compile_item:
                projected_controls.append({
                    "source": _strip_verified_bytes(item["source"]),
                    "binary": _strip_verified_bytes(item["binary"]),
                    "status": "PASS_COMPILE_ONLY",
                })
            else:
                projected_controls.append(
                    _focus_for_frozen_013(item, index, variant=control.get("variant")))
        projection["focused_controls"] = projected_controls
        if (len(projection["focused_controls"]) != 7 or
            sum(row.get("status") == "PASS" for row in projection["focused_controls"]) != 6 or
            sum(row.get("status") == "PASS_COMPILE_ONLY" for row in projection["focused_controls"]) != 1):
            raise ValueError("P028 feasibility-only projection has an unexpected PASS-only roster")
        projection["status"] = "PASS_SOURCE_CONTROLS"
        if mass_receipt is not None:
            raw_solver = control["solver_contract"]
            expected_raw_solver = {
                "method": "implicit_dinic_feasibility_on_exact_residual_graph",
                "status": "NOT_RUN_FULL_ALLOCATOR",
                "optimality_certified": False,
                "scale_factor": 1.0,
                "postsolve_rescaling": False,
            }
            if raw_solver != expected_raw_solver:
                raise ValueError("P028 raw solver contract differs from its retained NOT_RUN declaration")
            if (not isinstance(mass_receipt, dict) or
                mass_receipt.get("schema") != SCHEMA_MASS or
                mass_receipt.get("status") != "PASS_SOURCE_ONLY_REPLAY" or
                mass_receipt.get("physical_admission") is not False or
                mass_receipt.get("native_steps") != 0 or
                mass_receipt.get("allocation") != "NOT_NATIVE_EXECUTED"):
                raise ValueError("P028 013 projection requires a passing source-only mass receipt")
            claims = mass_receipt.get("claims")
            if (not isinstance(claims, dict) or not claims or
                any(value is not True for value in claims.values())):
                raise ValueError("P028 013 projection requires all mass claims to pass")
            solver = mass_receipt.get("solver")
            replay = mass_receipt.get("replay")
            if (not isinstance(solver, dict) or
                solver.get("method") != "deterministic_dinic_feasibility" or
                solver.get("status") != "PASS_FEASIBLE_NOT_OPTIMAL" or
                solver.get("optimality_certified") is not False or
                solver.get("scale_factor") != 1.0 or
                solver.get("postsolve_rescaling") is not False or
                not isinstance(replay, dict) or
                replay.get("status") != "PASS_DETERMINISTIC_REPLAY" or
                replay.get("flow_array_equality") is not True):
                raise ValueError("P028 receipt lacks deterministic feasibility/replay evidence")
            projection["solver_contract"] = {
                "method": "deterministic_dinic_feasibility",
                "status": "PASS_FEASIBLE_NOT_OPTIMAL",
                "optimality_certified": False,
                "scale_factor": 1.0,
                "postsolve_rescaling": False,
            }
    elif is_p029_source_controls:
        rows = focused
        pass_rows = [item for item in rows if isinstance(item, dict) and item.get("status") == "PASS"]
        compile_rows = [item for item in rows if isinstance(item, dict) and item.get("status") == "PASS_COMPILE_ONLY"]
        if (len(rows) != 2 or len(pass_rows) != 1 or len(compile_rows) != 1 or
            any(not isinstance(item, dict) or item.get("status") not in {"PASS", "PASS_COMPILE_ONLY"}
                for item in rows)):
            raise ValueError("P029 projection requires exactly its reader PASS and CLI compile-only PASS")
        reader = pass_rows[0]
        if (reader.get("name") != "canonical-source-tet-quadrature-reader-controls" or
            set(reader) != {"binary", "execution", "name", "source", "status", "stderr", "stdout"}):
            raise ValueError("P029 reader control record has an unsupported shape")
        source_path, _ = _ref_with_optional_bytes(reader["source"], "P029 reader source")
        reader_binary, _ = _ref_with_optional_bytes(reader["binary"], "P029 reader binary")
        stdout_path, _ = _ref_with_optional_bytes(reader["stdout"], "P029 reader stdout")
        stderr_path, _ = _ref_with_optional_bytes(reader["stderr"], "P029 reader stderr")
        expected_reader_output = "PASS 029 canonical source-tet quadrature: 11 positive degree-one sites, owner/packet bary replay, exact child map, tuple group, one-hop and unchanged R; malformed provenance rejected\n"
        if (reader["execution"] != {"command": [str(reader_binary)], "exit_code": 0} or
            stdout_path.read_text(encoding="utf-8") != expected_reader_output or stderr_path.read_bytes() or
            source_path.name != "CanonicalSourceTetQuadratureCheck_029.cpp"):
            raise ValueError("P029 reader output/execution is not the pinned PASS control")
        compile_item = compile_rows[0]
        if (compile_item.get("name") != "production-cli-o3-compile-only" or
            set(compile_item) != {"binary", "compile_command", "execution", "execution_scope",
                                  "name", "source", "status", "stderr", "stdout"} or
            compile_item.get("compile_command") != control.get("compile_command") or
            compile_item.get("execution_scope") != "compile_only_no_full_allocator_run" or
            _strip_verified_bytes(compile_item.get("source")) != _strip_verified_bytes(control.get("source")) or
            _strip_verified_bytes(compile_item.get("binary")) != _strip_verified_bytes(control.get("binary")) or
            compile_item.get("execution") != {"command": shlex.split(control["compile_command"]), "exit_code": 0}):
            raise ValueError("P029 production CLI compile-only record differs from the pinned build")
        for stream in ("stdout", "stderr"):
            stream_path, _ = _ref_with_optional_bytes(compile_item[stream], f"P029 compile {stream}")
            if stream_path.read_bytes():
                raise ValueError(f"P029 compile {stream} must be empty")
        # The original P029 control and closure retain and verify the distinct
        # compile-only row. Frozen 013 only accepts PASS rows, so its ephemeral
        # legacy view contains the functional reader control only; it never
        # promotes or rewrites the raw compile-only evidence.
        projection["focused_controls"] = [
            {"source": _strip_verified_bytes(reader["source"]),
             "binary": _strip_verified_bytes(reader["binary"]), "status": "PASS"},
        ]
        projection["status"] = "PASS_SOURCE_CONTROLS"
        raw_solver = control.get("solver_contract")
        expected_raw_solver = {
            "method": "implicit_dinic_feasibility_on_exact_residual_graph",
            "status": "NOT_RUN_FULL_ALLOCATOR",
            "optimality_certified": False,
            "scale_factor": 1.0,
            "postsolve_rescaling": False,
        }
        if raw_solver != expected_raw_solver:
            raise ValueError("P029 solver contract differs from its retained NOT_RUN declaration")
        if mass_receipt is not None:
            if (not isinstance(mass_receipt, dict) or
                mass_receipt.get("schema") != SCHEMA_MASS or
                mass_receipt.get("status") != "PASS_SOURCE_ONLY_REPLAY" or
                mass_receipt.get("physical_admission") is not False or
                mass_receipt.get("native_steps") != 0 or
                mass_receipt.get("allocation") != "NOT_NATIVE_EXECUTED"):
                raise ValueError("P029 compatibility projection requires a passing source-only mass receipt")
            claims = mass_receipt.get("claims")
            if (not isinstance(claims, dict) or not claims or
                any(value is not True for value in claims.values())):
                raise ValueError("P029 source-only receipt has a failed/missing claim")
            solver = mass_receipt.get("solver")
            replay = mass_receipt.get("replay")
            if (not isinstance(solver, dict) or solver.get("method") != "deterministic_dinic_feasibility" or
                solver.get("status") != "PASS_FEASIBLE_NOT_OPTIMAL" or
                solver.get("optimality_certified") is not False or solver.get("scale_factor") != 1.0 or
                solver.get("postsolve_rescaling") is not False or
                not isinstance(replay, dict) or replay.get("status") != "PASS_DETERMINISTIC_REPLAY" or
                replay.get("flow_array_equality") is not True):
                raise ValueError("P029 mass receipt lacks deterministic feasibility/replay evidence")
            projection["solver_contract"] = {
                "method": "deterministic_dinic_feasibility",
                "status": "PASS_FEASIBLE_NOT_OPTIMAL",
                "optimality_certified": False,
                "scale_factor": 1.0,
                "postsolve_rescaling": False,
            }
    else:
        projection["focused_controls"] = [
            _focus_for_frozen_013(item, index, variant=control.get("variant"))
            for index, item in enumerate(focused)
        ]
    return projection


def _verify_allocator_source_closure(
    closure: dict[str, Any], control_path: Path, control_sha: str,
    control: dict[str, Any]
) -> None:
    expected_keys = {
        "binaries", "compaction_api_parent", "compaction_api_test", "comparison_report",
        "compile_command", "compile_control", "compiler", "compiler_identity",
        "full_allocator_run", "parent_source_closure", "parent_source_control_manifest",
        "schema", "scope", "source_control_manifest", "sources", "status", "variant",
    }
    if set(closure) != expected_keys:
        raise ValueError("027 source closure schema mismatch")
    is_p028_diagnostic = closure.get("variant") == "028-implicit-complete-residual-dinic-path-only-materialization"
    is_p029_source_controls = closure.get("variant") == "029-canonical-source-tet-sites"
    expected_closure_status = ("SOURCE_CLOSED_FOCUSED_CONTROLS_PASS_NUMERICAL_PARITY_DIAGNOSTIC_FAIL_FULL_ALLOCATOR_NOT_RUN"
                               if is_p028_diagnostic else
                               "SOURCE_CLOSED_FOCUSED_CONTROLS_PASS_FULL_ALLOCATOR_NOT_RUN")
    expected_control_statuses = ({"SOURCE_CONTROLS_PASS_NUMERICAL_PARITY_DIAGNOSTIC_FAIL_FULL_ALLOCATOR_NOT_RUN"}
                                 if is_p028_diagnostic else
                                 {"PASS_SOURCE_CONTROLS_FULL_ALLOCATOR_NOT_RUN", "PASS_SOURCE_CONTROLS"})
    expected_scope = ("Source and focused synthetic feasibility/conservation/replay controls only. Numeric allocation identity comparison is retained as FAIL diagnostic. This does not establish full current-input completion, performance, admission, or physical qualification."
                      if is_p028_diagnostic else
                      "P029 canonical source-tet reader controls and production CLI compile only. The inherited P028 allocation-parity diagnostic remains FAIL; no full allocator feasibility, mass mapping, native execution, or physical qualification is claimed."
                      if is_p029_source_controls else
                      "Source/control closure only. Synthetic numeric parity and focused algorithm/remapping controls do not establish completion, performance, feasibility, admission, or physical qualification on full current inputs.")
    if (closure.get("schema") != "numi.h706.allocator-source-closure.v1" or
        not isinstance(closure.get("variant"), str) or
        closure.get("variant") != control.get("variant") or
        closure.get("status") != expected_closure_status or
        control.get("status") not in expected_control_statuses or
        closure.get("full_allocator_run") != "NOT_RUN" or closure.get("scope") != expected_scope):
        raise ValueError("allocator source closure status/scope is unsupported")
    current = closure.get("source_control_manifest")
    if (not isinstance(current, dict) or set(current) != {"path", "sha256", "bytes"} or
        current.get("path") != str(control_path) or current.get("sha256") != control_sha or
        current.get("bytes") != control_path.stat().st_size):
        raise ValueError("027 source closure does not bind the current source-control manifest")
    parents = control.get("parent_manifests")
    if not isinstance(parents, list) or len(parents) != 4:
        raise ValueError("027 source control must pin its four parent artifacts")
    for closure_key, parent_index in (
        ("parent_source_control_manifest", 0), ("parent_source_closure", 1),
        ("compaction_api_parent", 2), ("compaction_api_test", 3),
    ):
        ref = closure.get(closure_key)
        path, digest = _ref_with_optional_bytes(ref, f"027 source closure {closure_key}")
        parent = parents[parent_index]
        if (_strip_verified_bytes(ref) != _strip_verified_bytes(parent) or
            str(path) != parent["path"] or digest != parent["sha256"]):
            raise ValueError(f"027 source closure {closure_key} differs from source-control lineage")
    if (closure.get("compile_command") != control.get("compile_command") or
        closure.get("compiler") != control.get("compiler")):
        raise ValueError("027 source closure compiler/command differs from source control")
    identity = closure.get("compiler_identity")
    if (not isinstance(identity, dict) or set(identity) != {"target", "version"} or
        any(not isinstance(identity[key], str) or not identity[key] for key in identity)):
        raise ValueError("027 compiler identity is malformed")
    source_rows = closure.get("sources")
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("027 source closure source list is empty")
    present: set[tuple[str, str]] = set()
    paths: set[str] = set()
    for index, ref in enumerate(source_rows):
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256", "bytes"}:
            raise ValueError(f"027 source closure source {index} schema mismatch")
        path, digest = _ref_with_optional_bytes(ref, f"027 source closure source {index}")
        if type(ref["bytes"]) is not int or ref["bytes"] != path.stat().st_size:
            raise ValueError(f"027 source closure source {index} size mismatch")
        if str(path) in paths:
            raise ValueError("027 source closure has duplicate source paths")
        paths.add(str(path))
        present.add((str(path), digest))
    for key in ("source", "support_header", "mass_transport_header"):
        ref = control.get(key)
        if not isinstance(ref, dict) or (ref.get("path"), ref.get("sha256")) not in present:
            raise ValueError(f"027 source closure omits current source-control {key}")
    if is_p028_diagnostic:
        diagnostic_rows = [row for row in control.get("focused_controls", [])
                           if isinstance(row, dict) and row.get("status") == "FAIL_NUMERICAL_PARITY_DIAGNOSTIC"]
        if len(diagnostic_rows) != 1:
            raise ValueError("P028 closure requires exactly one retained failed parity diagnostic")
        diagnostic_result = _validate_numeric_parity_diagnostic(diagnostic_rows[0],
                                                                 control["focused_controls"].index(diagnostic_rows[0]))
        if (_strip_verified_bytes(closure["comparison_report"]) !=
            _strip_verified_bytes(diagnostic_rows[0]["comparison_report"]) or
            diagnostic_result.get("status") != "FAIL_NUMERICAL_PARITY_DIAGNOSTIC"):
            raise ValueError("P028 closure must bind the retained FAIL diagnostic, not a PASS")
        comparator_ref = diagnostic_rows[0]["comparator_source"]
    elif is_p029_source_controls:
        parents = control.get("parent_manifests")
        if not isinstance(parents, list) or len(parents) != 4:
            raise ValueError("P029 must pin the P028 control/closure and compaction owner pair")
        parent_control_path, parent_control_sha = _ref_with_optional_bytes(
            parents[0], "P029 inherited P028 control")
        parent_closure_path, parent_closure_sha = _ref_with_optional_bytes(
            parents[1], "P029 inherited P028 closure")
        parent_control = read_json(parent_control_path)
        if (parent_control.get("variant") != "028-implicit-complete-residual-dinic-path-only-materialization" or
            parent_control.get("status") != "SOURCE_CONTROLS_PASS_NUMERICAL_PARITY_DIAGNOSTIC_FAIL_FULL_ALLOCATOR_NOT_RUN"):
            raise ValueError("P029 must inherit the explicit P028 failed parity diagnostic")
        verify_nested_control_refs(parent_control)
        verify_source_closure(parent_closure_path, parent_closure_sha,
                              parent_control_path, parent_control_sha, parent_control)
        diagnostic_rows = [row for row in parent_control.get("focused_controls", [])
                           if isinstance(row, dict) and row.get("status") == "FAIL_NUMERICAL_PARITY_DIAGNOSTIC"]
        if len(diagnostic_rows) != 1:
            raise ValueError("P029 inherited P028 closure must contain one retained failed parity diagnostic")
        diagnostic = diagnostic_rows[0]
        diagnostic_result = _validate_numeric_parity_diagnostic(
            diagnostic, parent_control["focused_controls"].index(diagnostic))
        if (_strip_verified_bytes(closure["comparison_report"]) !=
            _strip_verified_bytes(diagnostic["comparison_report"]) or
            diagnostic_result.get("status") != "FAIL_NUMERICAL_PARITY_DIAGNOSTIC"):
            raise ValueError("P029 closure must retain the inherited P028 FAIL diagnostic")
        comparator_ref = diagnostic["comparator_source"]
    else:
        numeric_rows = [row for row in control.get("focused_controls", [])
                        if isinstance(row, dict) and row.get("status") == "PASS_NUMERICAL_PARITY"]
        if len(numeric_rows) != 1 or _strip_verified_bytes(closure["comparison_report"]) != \
                _strip_verified_bytes(numeric_rows[0]["comparison_report"]):
            raise ValueError("027 source closure comparison report differs from the numeric focused control")
        comparator_ref = numeric_rows[0]["comparator_source"]
    if (comparator_ref["path"], comparator_ref["sha256"]) not in present:
        raise ValueError("027 source closure omits the pinned numerical comparator source")
    _verify_local_trial_include_pins(Path(control["source"]["path"]).parent, source_rows)
    binary_rows = closure.get("binaries")
    if not isinstance(binary_rows, list) or not binary_rows:
        raise ValueError("compaction source closure binary roster is empty")
    binary_paths: set[str] = set()
    binary_ids: set[tuple[str, str]] = set()
    for index, ref in enumerate(binary_rows):
        path, digest = _ref_with_optional_bytes(ref, f"027 source closure binary {index}")
        if type(ref.get("bytes")) is not int or ref["bytes"] != path.stat().st_size:
            raise ValueError(f"027 source closure binary {index} size mismatch")
        if str(path) in binary_paths:
            raise ValueError("027 source closure has duplicate binaries")
        binary_paths.add(str(path))
        binary_ids.add((str(path), digest))
    top_binary = control.get("binary")
    if (not isinstance(top_binary, dict) or (top_binary.get("path"), top_binary.get("sha256")) not in binary_ids):
        raise ValueError("027 source closure omits production CLI binary")
    for row_index, row in enumerate(control["focused_controls"]):
        binary = row.get("binary") if isinstance(row, dict) else None
        if not isinstance(binary, dict) or (binary.get("path"), binary.get("sha256")) in binary_ids:
            continue
        source_ref = row.get("source")
        execution = row.get("execution")
        if not isinstance(source_ref, dict) or not isinstance(execution, dict):
            raise ValueError(f"027 source closure omits focused-control binary {row_index}")
        source_path, _ = _ref_with_optional_bytes(source_ref, f"focused control {row_index} source")
        binary_path, _ = _ref_with_optional_bytes(binary, f"focused control {row_index} binary")
        python_tool_invocation = (
            source_path.suffix == ".py" and
            execution.get("command") == ["python3", str(source_path)] and
            binary_path.name.startswith("python3.") and
            "Python.framework" in binary_path.parts
        )
        if not python_tool_invocation:
            raise ValueError(f"027 source closure omits focused-control binary {row_index}")
    compile_ref = closure["compile_control"]
    compile_path, _ = _ref_with_optional_bytes(compile_ref, "027 compile control")
    compile_control = read_json(compile_path)
    generation = _variant_generation(closure["variant"], "allocator source closure")
    expected_compile_schema = f"numi.h706.p{generation}-compile-control.v1"
    expected_compile_status = "PASS_FOCUSED_SOURCE_CONTROLS_FULL_ALLOCATOR_NOT_RUN"
    if is_p028_diagnostic:
        # P028's original compile-control predates the corrected path-only
        # variant envelope and independently recomputed FAIL diagnostic.
        # The immutable corrected successor keeps the original compile/source/
        # binary identity and binds the current diagnostic row exactly.
        expected_compile_schema = "numi.h706.compile-control.v1"
        expected_compile_status = (
            "SOURCE_COMPILED_FOCUSED_CONTROLS_COMPLETE_NUMERICAL_PARITY_DIAGNOSTIC_FAIL_FULL_ALLOCATOR_NOT_RUN"
        )
    if (compile_control.get("schema") != expected_compile_schema or
        compile_control.get("status") != expected_compile_status or
        compile_control.get("full_allocator_run") != "NOT_RUN" or
        compile_control.get("variant") != closure["variant"] or
        compile_control.get("compile_command") != control["compile_command"] or
        compile_control.get("compiler") != control["compiler"] or
        _strip_verified_bytes(compile_control.get("source")) != _strip_verified_bytes(control["source"]) or
        _strip_verified_bytes(compile_control.get("binary")) != _strip_verified_bytes(control["binary"])):
        raise ValueError("027 compile control does not match the pinned source-control build")
    if _strip_verified_bytes(compile_control.get("focused_controls")) != \
            _strip_verified_bytes(control.get("focused_controls")):
        raise ValueError("027 compile control focused records differ from source control")


def verify_source_closure(
    closure_path: Path, closure_sha: str, control_path: Path,
    control_sha: str, control: dict[str, Any]
) -> None:
    closure = read_json(closure_path)
    if sha_file(closure_path) != closure_sha:
        raise ValueError("source closure hash changed")
    schema = closure.get("schema")
    legacy_schema = "numi.h706.fractional-allocator-source-closure.v1"
    parity_schema = "numi.h706.allocator-source-closure.v1"
    if schema not in {legacy_schema, parity_schema}:
        raise ValueError("source closure schema is unsupported")
    if (schema == parity_schema and
        isinstance(closure.get("variant"), str) and
        set(closure) == {
            "binaries", "compaction_api_parent", "compaction_api_test", "comparison_report",
            "compile_command", "compile_control", "compiler", "compiler_identity",
            "full_allocator_run", "parent_source_closure", "parent_source_control_manifest",
            "schema", "scope", "source_control_manifest", "sources", "status", "variant",
        }):
        allowed_closure_status = {
            "SOURCE_CLOSED_FOCUSED_CONTROLS_PASS_FULL_ALLOCATOR_NOT_RUN",
            "SOURCE_CLOSED_FOCUSED_CONTROLS_PASS_NUMERICAL_PARITY_DIAGNOSTIC_FAIL_FULL_ALLOCATOR_NOT_RUN",
        }
        if closure.get("status") not in allowed_closure_status:
            raise ValueError("allocator source closure status is not source-only closed")
        closure_without_status = {key: value for key, value in closure.items() if key != "status"}
        verify_nested_artifact_refs(closure_without_status, "027 source closure")
        _verify_allocator_source_closure(closure, control_path, control_sha, control)
        return
    verify_nested_artifact_refs(closure, "source closure")
    current_023_keys = {
        "compile_command", "compiler", "parent_source_control_manifest", "schema",
        "source_control_manifest", "sources", "variant",
    }
    if schema == parity_schema and set(closure) == current_023_keys:
        current_ref = closure["source_control_manifest"]
        if (not isinstance(current_ref, dict) or set(current_ref) not in (
                {"path", "sha256"}, {"path", "sha256", "bytes"}) or
            current_ref.get("path") != str(control_path) or
            current_ref.get("sha256") != control_sha):
            raise ValueError("023 source closure does not pin the current source-control manifest")
        if "bytes" in current_ref and (
                type(current_ref["bytes"]) is not int or
                current_ref["bytes"] != control_path.stat().st_size):
            raise ValueError("023 source closure source-control byte count mismatch")
        parent_ref = closure["parent_source_control_manifest"]
        parents = control.get("parent_manifests")
        if (not isinstance(parent_ref, dict) or set(parent_ref) not in (
                {"path", "sha256"}, {"path", "sha256", "bytes"}) or
            not isinstance(parents, list) or
            not any(isinstance(parent, dict) and
                    parent.get("path") == parent_ref.get("path") and
                    parent.get("sha256") == parent_ref.get("sha256") and
                    ("bytes" not in parent_ref or parent.get("bytes") == parent_ref["bytes"])
                    for parent in parents)):
            raise ValueError("023 source closure parent source-control pin is not declared by current control")
        if closure.get("compiler") != control.get("compiler"):
            raise ValueError("023 source closure compiler contract differs from source control")
        if (closure.get("compile_command") != control.get("compile_command") or
            closure.get("variant") != control.get("variant")):
            raise ValueError("023 source closure compile command/variant differs from source control")
        source_rows = closure.get("sources")
        if not isinstance(source_rows, list) or not source_rows:
            raise ValueError("023 source closure source list is missing")
        present: set[tuple[str, str]] = set()
        paths: set[str] = set()
        for index, item in enumerate(source_rows):
            if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
                raise ValueError(f"023 source closure source {index} has the wrong schema")
            raw, digest = item["path"], _hex(item["sha256"], f"023 source closure source {index} SHA")
            if not isinstance(raw, str) or not Path(raw).is_absolute():
                raise ValueError(f"023 source closure source {index} path must be absolute")
            source_path = Path(raw)
            resolved = str(source_path.resolve())
            if raw != resolved or resolved in paths:
                raise ValueError("023 source closure sources contain a noncanonical or duplicate path")
            paths.add(resolved)
            if (type(item["bytes"]) is not int or not source_path.is_file() or
                source_path.stat().st_size != item["bytes"] or sha_file(source_path) != digest):
                raise ValueError(f"023 source closure source {index} hash/size changed")
            present.add((raw, digest))
        for key in ("source", "support_header", "mass_transport_header"):
            ref = control.get(key)
            if not isinstance(ref, dict) or (ref.get("path"), ref.get("sha256")) not in present:
                raise ValueError(f"023 source closure does not include current source-control {key}")
        _verify_local_trial_include_pins(Path(control["source"]["path"]).parent, source_rows)
        binary = control.get("binary")
        if (not isinstance(binary, dict) or not isinstance(binary.get("path"), str) or
            binary["path"] not in closure["compile_command"]):
            raise ValueError("023 source closure compile command does not name the pinned CLI binary")
        return
    control_ref = closure.get("source_control_manifest")
    allowed = ({"path", "sha256"}, {"path", "sha256", "bytes"})
    if not isinstance(control_ref, dict) or set(control_ref) not in allowed:
        raise ValueError("source closure lacks exact current source-control reference")
    if control_ref.get("path") != str(control_path) or control_ref.get("sha256") != control_sha:
        raise ValueError("source closure references a different source-control manifest")
    if "bytes" in control_ref and (
        type(control_ref["bytes"]) is not int or control_ref["bytes"] != control_path.stat().st_size
    ):
        raise ValueError("source closure source-control byte count mismatch")
    if closure.get("compiler") != control.get("compiler"):
        raise ValueError("source closure compiler contract differs from source control")
    compiler = closure.get("compiled_cli")
    control_binary = control.get("binary")
    if (not isinstance(compiler, dict) or set(compiler) != {"path", "sha256", "bytes"} or
        not isinstance(control_binary, dict) or compiler.get("path") != control_binary.get("path") or
        compiler.get("sha256") != control_binary.get("sha256") or
        type(compiler.get("bytes")) is not int):
        raise ValueError("source closure compiled CLI identity differs from source control binary")
    cli_path = Path(compiler["path"])
    if (not cli_path.is_absolute() or str(cli_path.resolve()) != compiler["path"] or
        not cli_path.is_file() or cli_path.stat().st_size != compiler["bytes"] or
        sha_file(cli_path) != compiler["sha256"]):
        raise ValueError("source closure compiled CLI path/hash/size changed")
    local_files = closure.get("local_files")
    if not isinstance(local_files, list) or not local_files:
        raise ValueError("source closure local file list is missing")
    present: set[tuple[str, str]] = set()
    paths: set[str] = set()
    for index, item in enumerate(local_files):
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
            raise ValueError(f"source closure local file {index} has the wrong schema")
        raw, digest = item["path"], _hex(item["sha256"], f"source closure local file {index} SHA")
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            raise ValueError(f"source closure local file {index} path must be absolute")
        file_path = Path(raw)
        resolved = str(file_path.resolve())
        if raw != resolved or resolved in paths:
            raise ValueError("source closure local files contain a noncanonical or duplicate path")
        paths.add(resolved)
        if type(item["bytes"]) is not int or not file_path.is_file() or file_path.stat().st_size != item["bytes"] or sha_file(file_path) != digest:
            raise ValueError(f"source closure local file {index} hash/size changed")
        present.add((raw, digest))
    present.add((compiler["path"], compiler["sha256"]))
    for key in ("source", "support_header", "binary"):
        ref = control.get(key)
        if (not isinstance(ref, dict) or set(ref) not in allowed or
            (ref["path"], ref["sha256"]) not in present):
            raise ValueError(f"source closure does not include current source-control {key}")
    focused = closure.get("focused_controls")
    declared = control.get("focused_controls")
    if schema == parity_schema:
        if (_normalized_focus_list(focused, "source closure") !=
            _normalized_focus_list(declared, "source control")):
            raise ValueError("source closure focused records differ from the source-control records")
        # Every parity row is checked here too; do not rely on the manifest's status label.
        for index, item in enumerate(focused):
            _focus_for_frozen_013(item, index, variant=closure.get("variant"))
    else:
        if (not isinstance(focused, dict) or not focused or
            any(not isinstance(value, str) or not value.startswith("PASS") for value in focused.values()) or
            not isinstance(declared, list) or not declared or
            any(not isinstance(item, dict) or item.get("status") != "PASS" for item in declared)):
            raise ValueError("source-control closure focused controls are missing or failing")

def verify_receipt_builder_compat_pin(value: Any) -> None:
    expected = {
        "launcher": {"path": str(RECEIPT_BUILDER_COMPAT_PATH.resolve()),
                     "sha256": RECEIPT_BUILDER_COMPAT_SHA256},
        "test": {"path": str(RECEIPT_BUILDER_COMPAT_TEST_PATH.resolve()),
                 "sha256": RECEIPT_BUILDER_COMPAT_TEST_SHA256},
        "nested_controls_launcher": {
            "path": str(RECEIPT_BUILDER_NESTED_COMPAT_PATH.resolve()),
            "sha256": RECEIPT_BUILDER_NESTED_COMPAT_SHA256,
        },
        "nested_controls_test": {
            "path": str(RECEIPT_BUILDER_NESTED_COMPAT_TEST_PATH.resolve()),
            "sha256": RECEIPT_BUILDER_NESTED_COMPAT_TEST_SHA256,
        },
    }
    if not isinstance(value, dict) or set(value) != {
            "launcher", "test", "nested_controls_launcher", "nested_controls_test"}:
        raise ValueError("007 config lacks exact 013 compatibility launcher/test pins")
    for key, ref in expected.items():
        got = value.get(key)
        if not isinstance(got, dict) or set(got) != {"path", "sha256"}:
            raise ValueError(f"013 compatibility {key} pin schema mismatch")
        if got != ref or not Path(ref["path"]).is_file() or sha_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError(f"013 compatibility {key} pin changed")


def verify_source_control(run: dict[str, Any], bundle: dict[str, Any],
                          mass_receipt: dict[str, Any] | None = None) -> None:
    if not isinstance(bundle, dict) or set(bundle) != {
        "schema", "unified_mass_receipt", "source_control_manifest", "source_closure",
        "paired_solver_run_sha256", "roles"
    } or bundle["schema"] != SCHEMA:
        raise ValueError("recorded-flow bundle schema mismatch")
    control_path, control_sha = _ref(bundle["source_control_manifest"], "source control")
    validate_run_source_control_pin(run, bundle["source_control_manifest"])
    closure_path, closure_sha = _ref(bundle["source_closure"], "source closure")
    control = read_json(control_path)
    if sha_file(control_path) != control_sha:
        raise ValueError("configured source control hash changed")
    verify_nested_control_refs(control)
    projection = _control_projection_for_013(control, mass_receipt=mass_receipt)
    for key in ("source", "binary"):
        _ref_with_optional_bytes(run.get(key), f"1047 run {key}")
        if _strip_verified_bytes(run.get(key)) != projection[key]:
            raise ValueError(f"1047 run {key} differs from current source-control")
    builder_ref = control.get("receipt_builder")
    builder_path, builder_sha = _ref_with_optional_bytes(builder_ref, "pinned receipt builder 013")
    if builder_sha != BUILDER_SHA256:
        raise ValueError("source control does not pin the expected receipt builder")
    spec = importlib.util.spec_from_file_location("fractional_mass_receipt_builder_013_pinned", builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned receipt builder 013")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    # Builder 013 is immutable and accepts only its historical compact controls.
    # Verify the newer original manifest recursively first, then pass a temporary
    # compatibility projection through the frozen validator. The actual run and
    # source-control files remain untouched and separately hash-bound.
    with tempfile.TemporaryDirectory(prefix="h706-013-control-view-") as temp:
        projected_path = Path(temp) / "source-control-013-view.json"
        projected_bytes = (json.dumps(projection, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        projected_path.write_bytes(projected_bytes)
        builder_run = dict(run)
        builder_run["source_control_manifest"] = {
            "path": str(projected_path.resolve()),
            "sha256": hashlib.sha256(projected_bytes).hexdigest(),
        }
        builder_run["source"] = projection["source"]
        builder_run["binary"] = projection["binary"]
        if builder.validate_control_manifest(builder_run, projected_path) != projection:
            raise ValueError("builder 013 did not validate the verified compatibility projection")
    verify_source_closure(closure_path, closure_sha, control_path, control_sha, control)


def _expected_inputs(mass: dict[str, Any], role: str) -> list[str]:
    role = role.lower()
    inputs, candidate, source = mass["input_sha256"], mass["candidate"], mass["source330"]
    support = "p1_" + role + "_support"
    carrier = "carrier_" + role + "_packet"
    result = [
        source["mass_packet"]["sha256"], source["parent_source_contributions"]["sha256"],
        candidate["child_mesh_sha256"], candidate["region_label_sha256"],
        candidate["coordinate_frame_sha256"], inputs["child_skin_node"]["sha256"],
        inputs["child_skin_elements"]["sha256"], inputs["child_skin_labels"]["sha256"],
        inputs[support]["sha256"], inputs[carrier]["sha256"],
    ]
    for i, value in enumerate(result):
        _hex(value, f"{role} expected input {i}")
    return result


def _csv_file(ref: Any, label: str, expected_path: Path,
              expected_header: tuple[str, ...], receipt_entry: Any) -> tuple[Path, str]:
    path, digest = _ref(ref, label)
    if path != expected_path.resolve():
        raise ValueError(f"{label} is not the canonical role output path")
    if not isinstance(receipt_entry, dict) or set(receipt_entry) != {"sha256", "size_bytes"}:
        raise ValueError(f"{label} receipt entry has the wrong schema")
    if receipt_entry["sha256"] != digest or type(receipt_entry["size_bytes"]) is not int:
        raise ValueError(f"{label} differs from role receipt")
    if receipt_entry["size_bytes"] != path.stat().st_size or receipt_entry["size_bytes"] <= 0:
        raise ValueError(f"{label} size differs from role receipt")
    with path.open("r", encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream), None)
    if tuple(header or ()) != expected_header:
        raise ValueError(f"{label} header differs from native importer")
    return path, digest


def verify_recorded_flow_bundle(bundle: dict[str, Any], cfg: dict[str, Any],
                                mass: dict[str, Any], mass_path: Path,
                                mass_sha: str) -> dict[str, str]:
    run_ref = cfg.get("h706_fractional_environment_source")
    if not isinstance(run_ref, dict) or set(run_ref) != {"path", "resolved_path", "sha256", "argv_sha256"}:
        raise ValueError("recorded flow lacks exact run-manifest source pin")
    run_path = Path(run_ref["path"])
    if not run_path.is_absolute() or str(run_path.resolve()) != run_ref["resolved_path"] or not run_path.is_file() or sha_file(run_path) != run_ref["sha256"]:
        raise ValueError("recorded flow run-manifest pin changed")
    run = read_json(run_path)
    unified_path, unified_sha = _ref(bundle["unified_mass_receipt"], "unified mass receipt")
    if unified_path != mass_path.resolve() or unified_sha != mass_sha:
        raise ValueError("recorded flow is not bound to configured 1047 receipt")
    if mass.get("schema") != SCHEMA_MASS or mass.get("status") != "PASS_SOURCE_ONLY_REPLAY":
        raise ValueError("recorded flow requires a successful unified 1047 source-only receipt")
    verify_source_control(run, bundle, mass_receipt=mass)
    paired = _hex(bundle["paired_solver_run_sha256"], "paired solver run")
    transport = _hex(mass["solver"]["transport_receipt_sha256"], "unified transport receipt")
    if paired != transport:
        raise ValueError("recorded flow paired run differs from unified receipt")
    roles = bundle["roles"]
    if not isinstance(roles, dict) or set(roles) != {"SKIN", "FAT"}:
        raise ValueError("recorded flow must contain exactly Skin and Fat")
    output_dir = None
    marker = "NUMI_H706_FRACTIONAL_OUTPUT_DIR="
    for arg in run.get("argv", []):
        if isinstance(arg, str) and arg.startswith(marker):
            if output_dir is not None:
                raise ValueError("run manifest has duplicate fractional output directory")
            output_dir = Path(arg[len(marker):])
    if output_dir is None or not output_dir.is_absolute():
        raise ValueError("run manifest lacks absolute fractional output directory")
    raw_output_dir = output_dir
    if str(raw_output_dir) != str(raw_output_dir.resolve()) or raw_output_dir.is_symlink():
        raise ValueError("run output directory is not a canonical non-symlink path")
    output_dir = raw_output_dir.resolve()
    role_dirs = []
    for role in ("SKIN", "FAT"):
        entry = roles.get(role)
        if not isinstance(entry, dict) or not isinstance(entry.get("receipt"), dict):
            raise ValueError(f"{role} role receipt reference is malformed")
        raw_receipt_path = Path(entry["receipt"].get("path", ""))
        if not raw_receipt_path.is_absolute() or raw_receipt_path.is_symlink():
            raise ValueError(f"{role} role receipt path is not absolute/non-symlink")
        receipt_path = raw_receipt_path.resolve(strict=True)
        role_dir = receipt_path.parent
        allowed_child = output_dir / "recorded-role-flow-1050"
        if role_dir not in (output_dir, allowed_child) or str(role_dir) != str(role_dir.resolve()):
            raise ValueError(f"{role} role files are outside the authenticated run output directory")
        if receipt_path.name != f"{role.lower()}-recorded-flow-receipt.json":
            raise ValueError(f"{role} role receipt filename is not canonical")
        role_dirs.append(role_dir)
    if role_dirs[0] != role_dirs[1]:
        raise ValueError("Skin and Fat recorded-flow files do not share one exact output directory")
    output_dir = role_dirs[0]
    expected_parent = mass["source330"]["parent_count"]
    expected_child = mass["candidate"]["child_tetrahedron_count"]
    if type(expected_parent) is not int or type(expected_child) is not int:
        raise ValueError("unified receipt lacks material-domain counts")
    recorded = {
        PFX + "RECORDED_CARRIER_ALLOCATION_CSV": "1",
        H706_PFX + "ADMISSION_RECEIPT_JSON": str(mass_path.resolve()),
        H706_PFX + "ADMISSION_RECEIPT_SHA256": mass_sha,
    }
    parent_total = child_total = 0
    global_candidates = None
    for role in ("SKIN", "FAT"):
        role_lc = role.lower()
        entry = roles[role]
        if not isinstance(entry, dict) or set(entry) != {"receipt", "files"}:
            raise ValueError(f"{role} bundle schema mismatch")
        receipt_path, receipt_sha = _ref(entry["receipt"], f"{role} receipt")
        if receipt_path != (output_dir / f"{role_lc}-recorded-flow-receipt.json").resolve():
            raise ValueError(f"{role} receipt is not at canonical 1050 output path")
        receipt = read_json(receipt_path)
        expected_keys = ROLE_KEYS | {f"parent_{role_lc}_tets", f"child_{role_lc}_tets"}
        if set(receipt) != expected_keys:
            raise ValueError(f"{role} receipt schema mismatch")
        if (receipt["status"] != f"PASS_SOURCE_ONLY_{role}_SOURCE_CARRIER_FEASIBLE_ALLOCATION" or
            receipt["recorded_flow_feasibility"] != "PASS_SOLVER_FREE" or
            receipt["deterministic_optimizer_replay"] != "NOT_RUN" or
            receipt["complete_allocation"] is not True or receipt["physical_admission"] is not False or
            type(receipt["native_steps"]) is not int or receipt["native_steps"] != 0 or
            receipt["solver_metrics_scope"] != "paired"):
            raise ValueError(f"{role} receipt is not complete source-only solver-free allocation")
        if receipt["paired_solver_run_sha256"] != paired or receipt["transport_receipt_sha256"] != transport:
            raise ValueError(f"{role} receipt does not match unified solver run")
        carrier_key = "carrier_" + role_lc + "_packet"
        expected_carrier_sha = mass["input_sha256"][carrier_key]["sha256"]
        if receipt["source_carrier_packet_sha256"] != expected_carrier_sha:
            raise ValueError(f"{role} carrier packet differs from unified input")
        _hex(receipt["source_carrier_array_sha256"], role + " carrier array")
        _hex(receipt["source_carrier_flow_trace_sha256"], role + " carrier trace")
        if receipt["expected_input_sha256"] != _expected_inputs(mass, role):
            raise ValueError(f"{role} expected-input list differs from unified receipt")
        parent_count = receipt[f"parent_{role_lc}_tets"]
        child_count = receipt[f"child_{role_lc}_tets"]
        if type(parent_count) is not int or parent_count <= 0 or type(child_count) is not int or child_count <= 0:
            raise ValueError(f"{role} material-domain counts are invalid")
        for key in ("transport_edges", "candidate_edges"):
            if type(receipt[key]) is not int or receipt[key] <= 0:
                raise ValueError(f"{role} {key} is invalid")
        if receipt["candidate_edges"] < receipt["transport_edges"]:
            raise ValueError(f"{role} candidate edge count is below its recorded transport edge count")
        if type(receipt["augmentations"]) is not int or receipt["augmentations"] < 0:
            raise ValueError(f"{role} augmentation count is invalid")
        if global_candidates is None:
            global_candidates = receipt["candidate_edges"]
        elif global_candidates != receipt["candidate_edges"]:
            raise ValueError("Skin and Fat receipts disagree on paired semantic candidate edges")
        parent_total += parent_count
        child_total += child_count
        files = entry["files"]
        if not isinstance(files, dict) or set(files) != set(FILE_SPECS):
            raise ValueError(f"{role} flow map differs from six-file importer contract")
        receipt_files = receipt["flow_csv"]
        expected_basenames = {f"{role_lc}-{filename}" for filename, _ in FILE_SPECS.values()}
        if not isinstance(receipt_files, dict) or set(receipt_files) != expected_basenames:
            raise ValueError(f"{role} receipt does not bind exactly six role-local files")
        env_values = {
            f"RECORDED_ALLOCATION_{role}_RECEIPT_JSON": str(receipt_path),
            f"RECORDED_ALLOCATION_{role}_RECEIPT_SHA256": receipt_sha,
        }
        for key, (filename, header) in FILE_SPECS.items():
            expected = output_dir / f"{role_lc}-{filename}"
            path, digest = _csv_file(files[key], role + "." + key, expected, header, receipt_files[expected.name])
            if key == "SOURCE_CARRIER_FLOW_CSV" and receipt["source_carrier_flow_trace_sha256"] != digest:
                raise ValueError(f"{role} carrier trace hash differs from source-carrier CSV")
            env_key = f"RECORDED_ALLOCATION_{role}_{key}"
            env_values[env_key] = str(path)
            env_values[env_key + "_SHA256"] = digest
        recorded.update({PFX + key: value for key, value in env_values.items()})
    if parent_total != expected_parent or child_total != expected_child:
        raise ValueError("Skin/Fat role counts do not exactly partition unified parent/child domains")
    return recorded


def verify_recorded_env_in_plan(environment: dict[str, str], expected: dict[str, str] | None) -> None:
    role_prefix = PFX + "RECORDED_ALLOCATION_"
    unified_prefix = H706_PFX
    expected = expected or {}
    observed_role = {k: v for k, v in environment.items() if k.startswith(role_prefix)}
    observed_unified = {k: v for k, v in environment.items() if k.startswith(unified_prefix)}
    expected_role = {k: v for k, v in expected.items() if k.startswith(role_prefix)}
    expected_unified = {k: v for k, v in expected.items() if k.startswith(unified_prefix)}
    if observed_role != expected_role or observed_unified != expected_unified:
        raise ValueError("saved plan recorded-flow environment differs from authenticated receipt bundle")
    enable = environment.get(PFX + "RECORDED_CARRIER_ALLOCATION_CSV")
    if expected and enable != "1":
        raise ValueError("saved plan lacks explicit recorded-allocation opt-in")
    if not expected and enable is not None:
        raise ValueError("saved plan unexpectedly enables recorded allocation")
    if enable not in (None, "1"):
        raise ValueError("saved plan has an invalid recorded-allocation enable")
