"""Compile a source-bound whole-body recruitment candidate.

The native source-compliant preparation already solves pose, support sharing,
muscle recruitment and stationary fibre state together.  This compiler makes
that state independently inspectable: it verifies the frozen native transcript,
the 416-route activation vector, the monotone search history and the complete
force-balance vectors.  It is deliberately a recruitment *candidate* rather
than an activation calibration claim; no measured EMG or held-out load data is
present in the source package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.activation-recruitment-candidate.v1"
PROFILE = ROOT / "config/activation-recruitment-candidate.v1.json"
ROUTES = 416
DOFS = 128
SUPPORTS = 18


class RecruitmentError(HumanImportError):
    """A source recruitment candidate cannot be admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecruitmentError("activation recruitment: " + message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _read_canonical(path: Path, label: str, *, require_canonical: bool = True) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RecruitmentError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    if require_canonical:
        _require(raw == canonical(value) + b"\n", f"{label} is not canonical")
    return value, _sha256(path)


def _load_profile(path: Path) -> tuple[dict[str, Any], str]:
    profile, profile_sha = _read_canonical(path, "recruitment profile")
    required = {
        "schema", "id", "source_receipt", "source_log", "fp32_log",
        "expected_route_count", "expected_dof_count", "expected_support_count",
        "activation_upper_bound", "expected_nonzero_routes",
        "expected_upper_bound_routes", "maximum_acceleration",
        "maximum_force_residual", "maximum_force_residual_abs",
        "minimum_support_gap", "maximum_loaded_support_gap",
        "activation_transport_tolerance", "boundary",
    }
    _require(set(profile) == required, "recruitment profile fields differ")
    _require(profile["schema"] == "numi.human.activation-recruitment-candidate.v1",
             "unsupported recruitment profile schema")
    _require(profile["id"] == "source_compliant_fullbody_recruitment",
             "unsupported recruitment profile")
    for key in ("source_receipt", "source_log", "fp32_log"):
        value = profile[key]
        _require(isinstance(value, str) and value.strip() and not Path(value).is_absolute()
                 and ".." not in Path(value).parts and "\\" not in value,
                 f"{key} path is unsafe")
    for key, expected in (("expected_route_count", ROUTES),
                          ("expected_dof_count", DOFS),
                          ("expected_support_count", SUPPORTS)):
        _require(profile[key] == expected, f"{key} differs from the source contract")
    upper = profile["activation_upper_bound"]
    _require(type(upper) in (int, float) and math.isfinite(float(upper)) and 0.0 < upper <= 1.0,
             "activation upper bound is invalid")
    for key in ("expected_nonzero_routes", "expected_upper_bound_routes"):
        _require(type(profile[key]) is int and 0 <= profile[key] <= ROUTES,
                 f"{key} is invalid")
    for key in ("maximum_acceleration", "maximum_force_residual",
                "maximum_force_residual_abs", "minimum_support_gap",
                "maximum_loaded_support_gap", "activation_transport_tolerance"):
        _require(type(profile[key]) in (int, float) and math.isfinite(float(profile[key])),
                 f"{key} is not finite")
    _require(float(profile["activation_transport_tolerance"]) >= 0.0,
             "activation transport tolerance is negative")
    _require(isinstance(profile["boundary"], str) and profile["boundary"].strip(),
             "recruitment profile boundary is missing")
    return profile, profile_sha


def _native_records(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"source transcript is not a regular file: {path}")
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("source_compliant_equilibrium="):
            _require("equilibrium" not in records, "source transcript has duplicate equilibrium records")
            records["equilibrium"] = json.loads(line.split("=", 1)[1])
        elif line.startswith("compiled_equilibrium_muscles="):
            _require("muscles" not in records, "source transcript has duplicate muscle records")
            records["muscles"] = json.loads(line.split("=", 1)[1])
    _require(set(records) == {"equilibrium", "muscles"},
             "source transcript lacks the equilibrium and muscle records")
    equilibrium, muscles = records["equilibrium"], records["muscles"]
    _require(equilibrium.get("schema") == "numi.human.source-compliant-equilibrium.v1",
             "unsupported source equilibrium schema")
    _require(muscles.get("schema") == "numi.human.offline-muscle-state.v1",
             "unsupported source muscle schema")
    return equilibrium, muscles


def _finite_vector(value: Any, count: int, label: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == count,
             f"{label} must contain {count} values")
    result = []
    for index, item in enumerate(value):
        _require(type(item) in (int, float) and math.isfinite(float(item)),
                 f"{label}[{index}] is not finite")
        result.append(float(item))
    return result


def _artifact_entry(receipt: dict[str, Any], relative: str) -> dict[str, Any]:
    artifacts = receipt.get("artifacts")
    _require(isinstance(artifacts, list), "source receipt has no artifact inventory")
    rows = [row for row in artifacts if isinstance(row, dict) and row.get("path") == relative]
    _require(len(rows) == 1, f"source receipt has no unique artifact entry for {relative}")
    row = rows[0]
    _require(type(row.get("bytes")) is int and type(row.get("sha256")) is str,
             f"source receipt artifact entry is incomplete for {relative}")
    return row


def compile_candidate(*, profile: Path = PROFILE) -> dict[str, Any]:
    profile_doc, profile_sha = _load_profile(Path(profile))
    receipt_path = (ROOT / profile_doc["source_receipt"]).resolve()
    source_path = (ROOT / profile_doc["source_log"]).resolve()
    fp32_path = (ROOT / profile_doc["fp32_log"]).resolve()
    for path, label in ((receipt_path, "source receipt"),
                        (source_path, "source transcript"),
                        (fp32_path, "FP32 transcript")):
        _require(path.is_relative_to(ROOT), f"{label} resolves outside repository")
    receipt, receipt_sha = _read_canonical(
        receipt_path, "source receipt", require_canonical=False
    )
    _require(receipt.get("schema") == "numi.human.source-compliant-preparation-receipt.v1",
             "unsupported source preparation receipt")
    expected_qual = {
        "offline_source_compliant_equilibrium": True,
        "fp32_pose_recruitment_balance": True,
        "bounded_native_transactions": True,
        "six_millisecond_dropout_equivalence": False,
        "registered_anatomical_tissue": False,
        "standing": False,
        "walking": False,
        "experimental_calibration": False,
        "performance": False,
        "full_release": False,
    }
    _require(receipt.get("qualification") == expected_qual,
             "source preparation qualification changed")
    source_rel = _relative(source_path)
    fp32_rel = _relative(fp32_path)
    source_artifact_rel = str(source_path.relative_to(receipt_path.parent))
    fp32_artifact_rel = str(fp32_path.relative_to(receipt_path.parent))
    for path, relative in ((source_path, source_artifact_rel), (fp32_path, fp32_artifact_rel)):
        row = _artifact_entry(receipt, relative)
        _require(row["bytes"] == path.stat().st_size and row["sha256"] == _sha256(path),
                 f"source receipt artifact drift: {relative}")

    equilibrium, muscles = _native_records(source_path)
    fp32_equilibrium, fp32_muscles = _native_records(fp32_path)
    _require(equilibrium.get("balanced") is True and fp32_equilibrium.get("balanced") is True,
             "source equilibrium is not balanced")
    activations = _finite_vector(muscles.get("activation_fp64"), ROUTES, "activation_fp64")
    activation_fp32 = _finite_vector(muscles.get("activation_fp32"), ROUTES, "activation_fp32")
    fp32_activations = _finite_vector(fp32_muscles.get("activation_fp64"), ROUTES, "FP32 activation_fp64")
    upper = float(profile_doc["activation_upper_bound"])
    _require(all(0.0 <= value <= upper for value in activations),
             "recruitment contains an out-of-range activation")
    _require(all(abs(value - round(value, 7)) <= 1.0e-6 for value in activations),
             "activation precision is not bounded to the native source state")
    transport_error = max(
        max(abs(left - right) for left, right in zip(activations, activation_fp32)),
        max(abs(left - right) for left, right in zip(activations, fp32_activations)),
    )
    _require(transport_error <= float(profile_doc["activation_transport_tolerance"]),
             "FP32 transport changed the admitted activation vector beyond its bound")
    nonzero = sum(value > 0.0 for value in activations)
    at_upper = sum(abs(value - upper) <= 1.0e-7 for value in activations)
    _require(nonzero == profile_doc["expected_nonzero_routes"], "nonzero recruitment count drifted")
    _require(at_upper == profile_doc["expected_upper_bound_routes"], "upper-bound recruitment count drifted")
    _require(nonzero < ROUTES and at_upper < ROUTES,
             "candidate is a uniform maximal-activation diagnostic")

    reference_lengths = _finite_vector(muscles.get("reference_fiber_length_m"), ROUTES,
                                        "reference_fiber_length_m")
    forces = _finite_vector(muscles.get("actuator_force_n"), ROUTES, "actuator_force_n")
    _require(all(value > 0.0 for value in reference_lengths),
             "recruited fibre state has a nonpositive reference length")
    acceleration = _finite_vector(equilibrium.get("acceleration"), DOFS, "acceleration")
    residual = _finite_vector(equilibrium.get("force_residual"), DOFS, "force_residual")
    gravity = _finite_vector(equilibrium.get("gravity_target"), DOFS, "gravity_target")
    muscle_force = _finite_vector(equilibrium.get("muscle_force"), DOFS, "muscle_force")
    equality = _finite_vector(equilibrium.get("equality_force"), DOFS, "equality_force")
    limit = _finite_vector(equilibrium.get("limit_force"), DOFS, "limit_force")
    support = _finite_vector(equilibrium.get("support_force"), DOFS, "support_force")
    support_force = _finite_vector(equilibrium.get("support_normal_force"), SUPPORTS,
                                    "support_normal_force")
    support_gap = _finite_vector(equilibrium.get("support_gap"), SUPPORTS, "support_gap")
    _require(all(value >= 0.0 for value in support_force), "support normal force is negative")
    history = _finite_vector(equilibrium.get("objective_history"),
                             len(equilibrium.get("objective_history", [])),
                             "objective_history")
    _require(len(history) >= 2 and all(after < before for before, after in zip(history, history[1:])),
             "recruitment objective history is not strictly decreasing")
    max_acceleration = max(abs(value) for value in acceleration)
    max_residual = max(abs(value) for value in residual)
    _require(max_acceleration <= float(profile_doc["maximum_acceleration"]),
             "source equilibrium acceleration exceeds the profile gate")
    _require(max_residual <= float(profile_doc["maximum_force_residual"]),
             "source equilibrium force residual exceeds the profile gate")
    _require(max(abs(muscle_force[index] + equality[index] + limit[index]
                     + support[index] - gravity[index] - residual[index])
                 for index in range(DOFS)) <= float(profile_doc["maximum_force_residual_abs"]),
             "generalized force decomposition is not closed")
    _require(min(support_gap) >= float(profile_doc["minimum_support_gap"]),
             "support gap is below the profile gate")
    _require(max(value for value in support_gap if value <= 1.0e-8)
             <= float(profile_doc["maximum_loaded_support_gap"]),
             "loaded support gap is above the profile gate")

    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.activation-recruitment-candidate.1",
        "status": "partial",
        "source": {
            "profile": _relative(Path(profile)),
            "profile_sha256": profile_sha,
            "preparation_receipt": _relative(receipt_path),
            "preparation_receipt_sha256": receipt_sha,
            "source_log": source_rel,
            "source_log_sha256": _sha256(source_path),
            "fp32_log": fp32_rel,
            "fp32_log_sha256": _sha256(fp32_path),
            "native_commit": receipt.get("native_commit"),
            "subject": "one adult male source package",
        },
        "counts": {
            "source_routes": ROUTES,
            "generalized_dofs": DOFS,
            "support_witnesses": SUPPORTS,
            "nonzero_routes": nonzero,
            "upper_bound_routes": at_upper,
            "zero_routes": ROUTES - nonzero,
        },
        "recruitment": {
            "activation_fp64": activations,
            "activation_fp32": activation_fp32,
            "reference_fiber_length_m": reference_lengths,
            "actuator_force_n": forces,
            "objective_history": history,
            "iterations": equilibrium.get("iterations"),
            "rejected_candidates": equilibrium.get("rejected"),
            "maximum_fp32_transport_error": transport_error,
            "uniform_maximal_activation": False,
        },
        "equilibrium": {
            "balanced": True,
            "acceleration_rms": equilibrium.get("acceleration_rms"),
            "maximum_acceleration": max_acceleration,
            "maximum_force_residual": max_residual,
            "support_normal_force_n": support_force,
            "support_gap_m": support_gap,
            "support_total_force_n": sum(support_force),
        },
        "qualification": {
            "source_route_activation_complete": True,
            "nonmaximal_recruitment_candidate": True,
            "pose_recruitment_fiber_state_bound": True,
            "generalized_force_decomposition_closed": True,
            "fp32_activation_transport_bounded": True,
            "fiber_tendon_equilibrium_calibrated": False,
            "activation_calibration": False,
            "held_out_force_validation": False,
            "anatomical_supports_loading": False,
            "sustained_standing": False,
            "recovery": False,
            "walking": False,
        },
        "boundary": profile_doc["boundary"],
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    candidate = compile_candidate(profile=arguments.profile)
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        _require(output.read_bytes() == json.dumps(candidate, indent=2, sort_keys=True).encode() + b"\n",
                 "candidate receipt is immutable and differs")
    else:
        output.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "status": candidate["status"],
        "source_routes": ROUTES,
        "nonzero_routes": candidate["counts"]["nonzero_routes"],
        "maximum_acceleration": candidate["equilibrium"]["maximum_acceleration"],
        "maximum_force_residual": candidate["equilibrium"]["maximum_force_residual"],
        "output": str(output),
    }, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except RecruitmentError as error:
        parser.exit(2, f"activation-recruitment: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
