"""Compare source-bound Human standing traces across exact timestep grids.

This reader is deliberately diagnostic. It verifies identities and complete
trace records, then compares same-time state, aggregate constraint impulses,
post-projection residuals, bounded force work, and production constraint-stage
impulse work. The native trace still lacks complete per-row reaction vectors,
target-relative dissipation, and a mass-consistent energy account for the final
exact-coordinate overwrite, so this module cannot issue a force-convergence or
standing qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "numi.human.native-trace-refinement.v3"
CASE_SCHEMA = "numi.human.current-refinement-case.v3"
TRACE_SCHEMA = "numi.human.persistent-stand-trace.v5"
CASE_SPEC = {
    "100us": (100_000, 64),
    "50us": (50_000, 128),
    "25us": (25_000, 256),
    "12p5us": (12_500, 512),
}
COMMON_DURATION_NS = 6_400_000
COMMON_INTERVAL_NS = 100_000
Q_COUNT = 129
V_COUNT = 128

PHYSICAL_MACHINE_FIELDS = (
    "architecture",
    "chip",
    "machine_identity_sha256",
    "machine_model",
    "machine_name",
    "memory",
    "os_build",
    "os_version",
)

REACTION_FIELDS = (
    "normal_impulse",
    "maximum_normal_contact_impulse_ns",
    "maximum_tangential_contact_impulse_ns",
    "maximum_source_limit_impulse_ns_or_nms",
    "total_source_limit_absolute_impulse_ns_or_nms",
    "maximum_equality_impulse",
    "total_equality_impulse",
)
RESIDUAL_FIELDS = (
    "post_projection_normal_contact_target_velocity_residual_m_s",
    "post_projection_source_limit_target_velocity_residual_m_s_or_rad_s",
    "post_projection_equality_target_velocity_residual_m_s_or_rad_s",
)
TENDON_RESIDUAL_FIELDS = (
    "tendon_max_force_residual_n",
    "tendon_max_moment_residual_nm",
)
WORK_FIELDS = (
    "muscle_virtual_work_j",
    "passive_joint_potential_work_j",
    "support_virtual_work_j",
)
IMPULSE_WORK_FIELDS = (
    "contact_normal_impulse_work_j",
    "contact_tangential_impulse_work_j",
    "equality_impulse_work_j",
    "source_limit_impulse_work_j",
)
ABSOLUTE_IMPULSE_WORK_FIELDS = (
    "contact_normal_absolute_impulse_work_j",
    "contact_tangential_absolute_impulse_work_j",
    "equality_absolute_impulse_work_j",
    "source_limit_absolute_impulse_work_j",
)
TRACE_IMPULSE_WORK_TOTALS = dict(
    zip(
        IMPULSE_WORK_FIELDS + ABSOLUTE_IMPULSE_WORK_FIELDS,
        (
            "total_contact_normal_impulse_work_j",
            "total_contact_tangential_impulse_work_j",
            "total_equality_impulse_work_j",
            "total_source_limit_impulse_work_j",
            "total_contact_normal_absolute_impulse_work_j",
            "total_contact_tangential_absolute_impulse_work_j",
            "total_equality_absolute_impulse_work_j",
            "total_source_limit_absolute_impulse_work_j",
        ),
        strict=True,
    )
)
OWNER_FIELDS = (
    "maximum_normal_contact_impulse_index",
    "maximum_tangential_contact_impulse_index",
    "maximum_source_limit_impulse_dof",
    "maximum_equality_impulse_index",
)
VELOCITY_STAGE_FIELDS = (
    "free_force_acceleration",
    "constraint_velocity_delta",
    "pre_projection_velocity_delta",
    "published_velocity_delta",
)


class TraceRefinementError(ValueError):
    """A source, trace, or comparison contract was violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TraceRefinementError(message)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise TraceRefinementError(f"non-finite JSON constant: {value}")


def _strict_json(text: str, context: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_invalid_constant,
        )
    except (json.JSONDecodeError, TypeError) as error:
        raise TraceRefinementError(f"{context} is not strict JSON: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite(value: Any, context: str) -> float:
    _require(type(value) in (int, float), f"{context} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{context} is not finite")
    return result


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    _require(
        type(value) is int and value >= minimum,
        f"{context} is not an integer >= {minimum}",
    )
    return value


def _hash(value: Any, context: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{context} is not a lowercase SHA-256",
    )
    return value


def _commit(value: Any, context: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value),
        f"{context} is not a full lowercase commit",
    )
    return value


def _physical_machine_receipt(value: Any, context: str) -> dict[str, str]:
    _require(isinstance(value, dict), f"{context} is not an object")
    _require(
        set(value) == {*PHYSICAL_MACHINE_FIELDS, "sha256"},
        f"{context} fields differ",
    )
    receipt: dict[str, str] = {}
    for field in PHYSICAL_MACHINE_FIELDS:
        item = value.get(field)
        _require(
            isinstance(item, str) and item.strip() == item and bool(item),
            f"{context} {field} is not a non-empty canonical string",
        )
        receipt[field] = item
    receipt["machine_identity_sha256"] = _hash(
        receipt["machine_identity_sha256"],
        f"{context} machine_identity_sha256",
    )
    reported = _hash(value.get("sha256"), f"{context} sha256")
    _require(
        _canonical_sha256(receipt) == reported,
        f"{context} digest mismatch",
    )
    receipt["sha256"] = reported
    return receipt


def _is_physical_m4_mac_mini(receipt: dict[str, str]) -> bool:
    identity_text = " ".join(receipt[field] for field in PHYSICAL_MACHINE_FIELDS)
    return (
        receipt["architecture"] == "arm64"
        and receipt["machine_name"] == "Mac mini"
        and receipt["chip"].startswith("Apple M4")
        and "paravirtual" not in identity_text.lower()
    )


def _prefixed_json(text: str, prefix: str, context: str) -> dict[str, Any]:
    rows = [line[len(prefix):] for line in text.splitlines() if line.startswith(prefix)]
    _require(len(rows) == 1, f"{context} must contain exactly one {prefix[:-1]} record")
    value = _strict_json(rows[0], context)
    _require(isinstance(value, dict), f"{context} {prefix[:-1]} record is not an object")
    return value


def _vector(value: Any, count: int, context: str) -> list[float]:
    _require(
        isinstance(value, list) and len(value) == count,
        f"{context} must contain {count} values",
    )
    return [_finite(item, f"{context}[{index}]") for index, item in enumerate(value)]


def _sample(sample: Any, index: int, timestep_seconds: float) -> dict[str, Any]:
    _require(isinstance(sample, dict), f"trace sample {index} is not an object")
    _require(
        _integer(sample.get("step"), f"trace sample {index} step") == index,
        f"trace sample {index} is out of order",
    )
    time_seconds = _finite(sample.get("time_seconds"), f"trace sample {index} time")
    _require(
        abs(time_seconds - index * timestep_seconds)
        <= max(1.0e-12, timestep_seconds * 1.0e-8),
        f"trace sample {index} has the wrong exact-clock time",
    )
    q = _vector(sample.get("q"), Q_COUNT, f"trace sample {index} q")
    v = _vector(sample.get("v"), V_COUNT, f"trace sample {index} v")
    quaternion_norm = math.sqrt(sum(value * value for value in q[3:7]))
    _require(
        abs(quaternion_norm - 1.0) <= 2.0e-5,
        f"trace sample {index} root quaternion is not normalized",
    )

    normalized = dict(sample)
    normalized["time_seconds"] = time_seconds
    normalized["q"] = q
    normalized["v"] = v
    for field in REACTION_FIELDS + RESIDUAL_FIELDS + TENDON_RESIDUAL_FIELDS + WORK_FIELDS + (
        *IMPULSE_WORK_FIELDS,
        *ABSOLUTE_IMPULSE_WORK_FIELDS,
        "passive_joint_energy_j",
    ):
        normalized[field] = _finite(sample.get(field), f"trace sample {index} {field}")
    for field in REACTION_FIELDS + RESIDUAL_FIELDS + TENDON_RESIDUAL_FIELDS:
        _require(normalized[field] >= 0.0, f"trace sample {index} {field} is negative")
    for signed, absolute in zip(
        IMPULSE_WORK_FIELDS, ABSOLUTE_IMPULSE_WORK_FIELDS, strict=True
    ):
        _require(
            normalized[absolute] >= 0.0,
            f"trace sample {index} {absolute} is negative",
        )
        _require(
            normalized[absolute] + 1.0e-12 >= abs(normalized[signed]),
            f"trace sample {index} {absolute} hides signed work",
        )
    for field in OWNER_FIELDS:
        normalized[field] = _integer(sample.get(field), f"trace sample {index} {field}")
    stage_presence = [field in sample for field in VELOCITY_STAGE_FIELDS]
    _require(
        not any(stage_presence) or all(stage_presence),
        f"trace sample {index} has a partial velocity-stage record",
    )
    if all(stage_presence):
        for field in VELOCITY_STAGE_FIELDS:
            normalized[field] = _finite(sample[field], f"trace sample {index} {field}")
            _require(normalized[field] >= 0.0, f"trace sample {index} {field} is negative")
    return normalized


def load_case(directory: Path) -> dict[str, Any]:
    directory = Path(directory).resolve()
    summary_path = directory / "case-summary.json"
    _require(
        summary_path.is_file() and not summary_path.is_symlink(),
        f"{directory} has no regular case-summary.json",
    )
    summary = _strict_json(summary_path.read_text(encoding="utf-8"), str(summary_path))
    _require(
        isinstance(summary, dict) and summary.get("schema") == CASE_SCHEMA,
        f"{summary_path} schema mismatch",
    )
    name = summary.get("name")
    _require(name in CASE_SPEC, f"{summary_path} has an unknown grid")
    timestep_ns, steps = CASE_SPEC[name]
    _require(summary.get("timestep_nanoseconds") == timestep_ns, f"{name} timestep differs")
    _require(summary.get("step_count") == steps, f"{name} step count differs")
    _require(summary.get("duration_nanoseconds") == COMMON_DURATION_NS, f"{name} duration differs")
    _require(summary.get("exit_code") == 0, f"{name} native process did not complete")
    _require(summary.get("required_metric_mismatches") == {}, f"{name} execution contract differs")
    _require(
        summary.get("velocity_stage_diagnostics_complete") is True,
        f"{name} lacks velocity-stage diagnostics",
    )
    _require(
        summary.get("constraint_impulse_work_complete") is True,
        f"{name} lacks constraint impulse work",
    )
    _require(summary.get("compiled_static_balance") is True, f"{name} static balance failed")
    _commit(summary.get("native_commit"), f"{name} native commit")
    _commit(summary.get("input_commit"), f"{name} input commit")
    _hash(summary.get("binary_sha256"), f"{name} binary")
    _hash(summary.get("launcher_sha256"), f"{name} launcher")
    summary["physical_machine_receipt"] = _physical_machine_receipt(
        summary.get("physical_machine_receipt"),
        f"{name} physical_machine_receipt",
    )
    payloads = summary.get("payload_sha256")
    _require(isinstance(payloads, dict) and payloads, f"{name} payload identities are missing")
    for key, value in payloads.items():
        _require(isinstance(key, str) and key, f"{name} payload key is invalid")
        _hash(value, f"{name} payload {key}")

    stdout = directory / "stdout.txt"
    stderr = directory / "stderr.txt"
    for path, key in ((stdout, "stdout_sha256"), (stderr, "stderr_sha256")):
        _require(
            path.is_file() and not path.is_symlink(),
            f"{name} {path.name} is missing or redirected",
        )
        _require(
            _sha256(path) == _hash(summary.get(key), f"{name} {key}"),
            f"{name} {path.name} hash mismatch",
        )
    _require(summary.get("stderr_nonbanner_lines") == [], f"{name} has native stderr diagnostics")

    text = stdout.read_text(encoding="utf-8")
    trace = _prefixed_json(text, "persistent_stand_trace=", f"{name} stdout")
    _require(trace.get("schema") == TRACE_SCHEMA, f"{name} trace schema mismatch")
    work_scope = trace.get("work_scope")
    _require(
        isinstance(work_scope, str)
        and "production_constraint_impulse_work_by_family" in work_scope
        and "exact_coordinate_projection_is_an_unowned_overwrite_not_impulse_work"
        in work_scope,
        f"{name} trace work scope is incomplete",
    )
    _require(
        trace.get("endpoint_equivalent") == "bitwise"
        and trace.get("endpoint_max_q_delta") == 0
        and trace.get("endpoint_max_v_delta") == 0,
        f"{name} deterministic trace endpoint differs",
    )
    samples = trace.get("samples")
    _require(
        isinstance(samples, list) and len(samples) == steps + 1,
        f"{name} trace sample count differs",
    )
    timestep_seconds = timestep_ns * 1.0e-9
    normalized_samples = [
        _sample(sample, index, timestep_seconds) for index, sample in enumerate(samples)
    ]
    for field in IMPULSE_WORK_FIELDS + ABSOLUTE_IMPULSE_WORK_FIELDS:
        _require(
            normalized_samples[0][field] == 0.0,
            f"{name} initial sample {field} is not zero",
        )
    for sample_field, total_field in TRACE_IMPULSE_WORK_TOTALS.items():
        reported = _finite(trace.get(total_field), f"{name} trace {total_field}")
        expected = sum(sample[sample_field] for sample in normalized_samples[1:])
        _require(
            abs(reported - expected) <= 1.0e-12 * (1.0 + abs(expected)),
            f"{name} trace {total_field} disagrees with samples",
        )
    _require(
        all(all(field in sample for field in VELOCITY_STAGE_FIELDS) for sample in normalized_samples),
        f"{name} trace does not retain all velocity stages",
    )

    return {
        "name": name,
        "directory": str(directory),
        "summary": summary,
        "trace": trace,
        "samples": normalized_samples,
    }


def _maximum_delta(first: Iterable[float], second: Iterable[float]) -> float:
    pairs = list(zip(first, second, strict=True))
    return max(abs(left - right) for left, right in pairs) if pairs else 0.0


def _quaternion_delta(first: list[float], second: list[float]) -> float:
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    a = [value / first_norm for value in first]
    b = [value / second_norm for value in second]
    sign = 1.0 if sum(left * right for left, right in zip(a, b, strict=True)) >= 0.0 else -1.0
    numerator = math.sqrt(
        sum((left - sign * right) ** 2 for left, right in zip(a, b, strict=True))
    )
    denominator = math.sqrt(
        sum((left + sign * right) ** 2 for left, right in zip(a, b, strict=True))
    )
    return 4.0 * math.atan2(numerator, denominator)


def _interval_sum(
    samples: list[dict[str, Any]], field: str, start: int, stop: int
) -> float:
    return sum(float(sample[field]) for sample in samples[start:stop])


def compare_cases(directories: Iterable[Path]) -> dict[str, Any]:
    cases = [load_case(Path(directory)) for directory in directories]
    _require(len(cases) == 4, "exactly four refinement case directories are required")
    by_name = {case["name"]: case for case in cases}
    _require(
        len(by_name) == 4 and set(by_name) == set(CASE_SPEC),
        "the four canonical timestep grids are required exactly once",
    )

    identity_fields = ("native_commit", "input_commit", "binary_sha256", "launcher_sha256")
    for field in identity_fields:
        _require(
            len({case["summary"][field] for case in cases}) == 1,
            f"refinement cases mix {field}",
        )
    payload_keys = set(next(iter(cases))["summary"]["payload_sha256"])
    _require(
        all(set(case["summary"]["payload_sha256"]) == payload_keys for case in cases),
        "refinement cases have different payload sets",
    )
    for key in payload_keys:
        _require(
            len({case["summary"]["payload_sha256"][key] for case in cases}) == 1,
            f"refinement cases mix payload {key}",
        )
    machine_receipts = [
        case["summary"]["physical_machine_receipt"] for case in cases
    ]
    _require(
        all(receipt == machine_receipts[0] for receipt in machine_receipts[1:]),
        "refinement cases mix physical_machine_receipt",
    )
    physical_machine_receipt = machine_receipts[0]
    physical_m4_validation = _is_physical_m4_mac_mini(
        physical_machine_receipt
    )

    reference = by_name["12p5us"]
    comparisons: list[dict[str, Any]] = []
    for name in ("100us", "50us", "25us"):
        case = by_name[name]
        timestep_ns, _ = CASE_SPEC[name]
        stride = COMMON_INTERVAL_NS // timestep_ns
        reference_stride = COMMON_INTERVAL_NS // CASE_SPEC["12p5us"][0]
        metrics: dict[str, list[float]] = {
            "root_translation": [],
            "root_orientation": [],
            "scalar_configuration": [],
            "velocity": [],
            "normal_reaction": [],
            "maximum_normal_contact_impulse_rate": [],
            "maximum_tangential_contact_impulse_rate": [],
            "total_equality_impulse_rate": [],
            "total_source_limit_absolute_impulse_rate": [],
            "muscle_work": [],
            "passive_potential_work": [],
            "support_work": [],
            "passive_energy": [],
        }
        residual_deltas = {field: [] for field in RESIDUAL_FIELDS}
        tendon_residual_deltas = {
            field: [] for field in TENDON_RESIDUAL_FIELDS
        }
        stage_deltas = {field: [] for field in VELOCITY_STAGE_FIELDS}
        impulse_work_deltas = {field: [] for field in IMPULSE_WORK_FIELDS}
        absolute_impulse_work_deltas = {
            field: [] for field in ABSOLUTE_IMPULSE_WORK_FIELDS
        }
        owner_mismatches = {field: 0 for field in OWNER_FIELDS}

        for common_index in range(COMMON_DURATION_NS // COMMON_INTERVAL_NS + 1):
            sample = case["samples"][common_index * stride]
            finest = reference["samples"][common_index * reference_stride]
            metrics["root_translation"].append(
                _maximum_delta(sample["q"][:3], finest["q"][:3])
            )
            metrics["root_orientation"].append(
                _quaternion_delta(sample["q"][3:7], finest["q"][3:7])
            )
            metrics["scalar_configuration"].append(
                _maximum_delta(sample["q"][7:], finest["q"][7:])
            )
            metrics["velocity"].append(_maximum_delta(sample["v"], finest["v"]))
            metrics["passive_energy"].append(
                abs(sample["passive_joint_energy_j"] - finest["passive_joint_energy_j"])
            )
            for field in RESIDUAL_FIELDS:
                residual_deltas[field].append(abs(sample[field] - finest[field]))
            for field in TENDON_RESIDUAL_FIELDS:
                tendon_residual_deltas[field].append(
                    abs(sample[field] - finest[field])
                )
            for field in VELOCITY_STAGE_FIELDS:
                stage_deltas[field].append(abs(sample[field] - finest[field]))
            for field in OWNER_FIELDS:
                owner_mismatches[field] += int(sample[field] != finest[field])

            if common_index == 0:
                continue
            start = (common_index - 1) * stride + 1
            stop = common_index * stride + 1
            reference_start = (common_index - 1) * reference_stride + 1
            reference_stop = common_index * reference_stride + 1
            duration_seconds = COMMON_INTERVAL_NS * 1.0e-9
            reaction_fields = {
                "normal_reaction": "normal_impulse",
                "maximum_normal_contact_impulse_rate": "maximum_normal_contact_impulse_ns",
                "maximum_tangential_contact_impulse_rate": "maximum_tangential_contact_impulse_ns",
                "total_equality_impulse_rate": "total_equality_impulse",
                "total_source_limit_absolute_impulse_rate": (
                    "total_source_limit_absolute_impulse_ns_or_nms"
                ),
            }
            for output, field in reaction_fields.items():
                coarse_value = (
                    _interval_sum(case["samples"], field, start, stop) / duration_seconds
                )
                fine_value = (
                    _interval_sum(
                        reference["samples"], field, reference_start, reference_stop
                    )
                    / duration_seconds
                )
                metrics[output].append(abs(coarse_value - fine_value))
            for output, field in (
                ("muscle_work", "muscle_virtual_work_j"),
                ("passive_potential_work", "passive_joint_potential_work_j"),
                ("support_work", "support_virtual_work_j"),
            ):
                metrics[output].append(
                    abs(
                        _interval_sum(case["samples"], field, start, stop)
                        - _interval_sum(
                            reference["samples"], field, reference_start, reference_stop
                        )
                    )
                )
            for field in IMPULSE_WORK_FIELDS:
                impulse_work_deltas[field].append(
                    abs(
                        _interval_sum(case["samples"], field, start, stop)
                        - _interval_sum(
                            reference["samples"], field,
                            reference_start, reference_stop
                        )
                    )
                )
            for field in ABSOLUTE_IMPULSE_WORK_FIELDS:
                absolute_impulse_work_deltas[field].append(
                    abs(
                        _interval_sum(case["samples"], field, start, stop)
                        - _interval_sum(
                            reference["samples"], field,
                            reference_start, reference_stop
                        )
                    )
                )

        comparisons.append(
            {
                "case": name,
                "reference": "12p5us",
                "common_timestamp_count": len(metrics["root_translation"]),
                "common_interval_count": len(metrics["normal_reaction"]),
                "maximum_root_translation_delta_m": max(metrics["root_translation"]),
                "maximum_root_orientation_delta_rad": max(metrics["root_orientation"]),
                "maximum_scalar_configuration_delta_mixed_units": max(
                    metrics["scalar_configuration"]
                ),
                "maximum_generalized_velocity_delta_mixed_units": max(
                    metrics["velocity"]
                ),
                "maximum_normal_reaction_delta_n": max(metrics["normal_reaction"]),
                "rms_normal_reaction_delta_n": math.sqrt(
                    sum(value * value for value in metrics["normal_reaction"])
                    / len(metrics["normal_reaction"])
                ),
                "maximum_max_contact_impulse_rate_delta_n": max(
                    metrics["maximum_normal_contact_impulse_rate"]
                ),
                "maximum_max_tangential_impulse_rate_delta_n": max(
                    metrics["maximum_tangential_contact_impulse_rate"]
                ),
                "maximum_total_equality_impulse_rate_delta_mixed_units_per_s": max(
                    metrics["total_equality_impulse_rate"]
                ),
                "maximum_total_limit_absolute_impulse_rate_delta_mixed_units_per_s": max(
                    metrics["total_source_limit_absolute_impulse_rate"]
                ),
                "maximum_post_projection_residual_deltas": {
                    field: max(values) for field, values in residual_deltas.items()
                },
                "maximum_tendon_residual_deltas": {
                    field: max(values)
                    for field, values in tendon_residual_deltas.items()
                },
                "maximum_velocity_stage_deltas": {
                    field: max(values) for field, values in stage_deltas.items()
                },
                "maximum_interval_work_deltas_j": {
                    "muscle": max(metrics["muscle_work"]),
                    "passive_potential": max(metrics["passive_potential_work"]),
                    "support_reference": max(metrics["support_work"]),
                },
                "maximum_interval_constraint_impulse_work_deltas_j": {
                    field: max(values)
                    for field, values in impulse_work_deltas.items()
                },
                "maximum_interval_constraint_absolute_impulse_work_deltas_j": {
                    field: max(values)
                    for field, values in absolute_impulse_work_deltas.items()
                },
                "maximum_passive_energy_delta_j": max(metrics["passive_energy"]),
                "maximum_impulse_owner_mismatch_counts": owner_mismatches,
            }
        )

    case_rows = []
    for name in ("100us", "50us", "25us", "12p5us"):
        case = by_name[name]
        samples = case["samples"]
        timestep_ns, steps = CASE_SPEC[name]
        duration_seconds = COMMON_DURATION_NS * 1.0e-9
        case_rows.append(
            {
                "name": name,
                "timestep_nanoseconds": timestep_ns,
                "step_count": steps,
                "wall_seconds": _finite(
                    case["summary"].get("wall_seconds"), f"{name} wall_seconds"
                ),
                "mean_normal_reaction_n": sum(
                    sample["normal_impulse"] for sample in samples[1:]
                )
                / duration_seconds,
                "maximum_post_projection_residuals": {
                    field: max(sample[field] for sample in samples[1:])
                    for field in RESIDUAL_FIELDS
                },
                "maximum_tendon_residuals": {
                    field: max(sample[field] for sample in samples[1:])
                    for field in TENDON_RESIDUAL_FIELDS
                },
                "maximum_velocity_stages": {
                    field: max(sample[field] for sample in samples[1:])
                    for field in VELOCITY_STAGE_FIELDS
                },
                "terminal_passive_joint_energy_j": samples[-1][
                    "passive_joint_energy_j"
                ],
                "total_trace_work_j": {
                    field: sum(sample[field] for sample in samples[1:])
                    for field in WORK_FIELDS
                },
                "total_constraint_impulse_work_j": {
                    field: sum(sample[field] for sample in samples[1:])
                    for field in IMPULSE_WORK_FIELDS
                },
                "total_constraint_absolute_impulse_work_j": {
                    field: sum(sample[field] for sample in samples[1:])
                    for field in ABSOLUTE_IMPULSE_WORK_FIELDS
                },
            }
        )

    return {
        "schema": SCHEMA,
        "status": "diagnostic_complete",
        "native_commit": cases[0]["summary"]["native_commit"],
        "input_commit": cases[0]["summary"]["input_commit"],
        "binary_sha256": cases[0]["summary"]["binary_sha256"],
        "physical_machine_receipt": physical_machine_receipt,
        "common_duration_nanoseconds": COMMON_DURATION_NS,
        "case_count": 4,
        "cases": case_rows,
        "comparisons": comparisons,
        "coverage": {
            "same_time_q_v": True,
            "aggregate_contact_impulse": True,
            "aggregate_equality_impulse": True,
            "aggregate_joint_limit_impulse": True,
            "post_projection_residuals": True,
            "tendon_force_and_moment_residuals": True,
            "bounded_work_terms": True,
            "constraint_stage_impulsive_work": True,
            "identical_physical_machine_identity": True,
            "complete_per_constraint_reaction_vectors": False,
            "complete_impulsive_work": False,
            "complete_physical_energy_closure": False,
        },
        "qualification": {
            "diagnostic_complete": True,
            "state_convergence": False,
            "force_convergence": False,
            "sustained_standing": False,
            "physical_m4_validation": physical_m4_validation,
        },
        "boundary": (
            "Source-identical single-machine comparison of captured state, aggregate reaction impulses, "
            "post-projection residuals, bounded force work, and all owned production constraint-stage "
            "impulse work. Physical M4 validation means only that all four cases carry one matching "
            "physical Mac mini M4 identity receipt; it is not a performance or thermal qualification. "
            "No acceptance tolerance is inferred. The trace lacks complete per-constraint "
            "reaction vectors, target-relative dissipation, and mass-consistent energy for the final "
            "exact-coordinate overwrite, so it cannot certify force convergence, sustained standing, "
            "recovery, or walking."
        ),
    }


def write_report(path: Path, report: dict[str, Any]) -> str:
    path = Path(path).resolve()
    payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    _require(not path.is_symlink(), "output path is redirected")
    if path.exists():
        _require(path.read_bytes() == payload, "output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--case",
        action="append",
        type=Path,
        required=True,
        help=(
            "case directory containing case-summary.json, stdout.txt, and stderr.txt; "
            "repeat four times"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    arguments = parser.parse_args(argv)
    try:
        report = compare_cases(arguments.case)
        digest = write_report(arguments.output, report)
    except (OSError, KeyError, TypeError, TraceRefinementError) as error:
        parser.exit(2, f"native trace refinement rejected: {error}\n")
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "output": str(arguments.output.resolve()),
                "sha256": digest,
                "force_convergence": report["qualification"]["force_convergence"],
                "physical_m4_validation": report["qualification"][
                    "physical_m4_validation"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
