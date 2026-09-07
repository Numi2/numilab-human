"""Offline admission of frozen, accepted-root Human behavior evidence.

This module does not simulate, infer missing telemetry, or authenticate its producer.
Only a native accepted-root reducer may produce the trace contract documented in
Docs/HUMAN_BEHAVIOR_QUALIFICATION.md. Unit fixtures are never runtime evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

PROTOCOL_SCHEMA = "numi.human.behavior-protocol.v1"
STACK_SCHEMA = "numi.human.behavior-stack.v1"
BUNDLE_SCHEMA = "numi.human.behavior-bundle.v1"
TRACE_SCHEMA = "numi.human.behavior-trial.v1"
REPORT_SCHEMA = "numi.human.behavior-assessment.v1"
LOWERING_SCHEMA = "numi.human.behavior-task-lowering.v1"
METRIC_CONTRACT = "accepted-root-task-reduction.v1"
ARTIFACT_ROLES = frozenset({"human_pack", "compiled_run", "task_pack", "policy_pack",
                            "native_library", "human_metallib", "matter_metallib",
                            "brain_metallib", "runner", "qualifier", "metric_program",
                            "accepted_root_proof_schema", "task_lowering_receipt"})
AUDIT_COUNTERS = ("root_assistance_steps", "direct_torque_steps", "kinematic_override_steps",
                  "unregistered_force_steps", "source_constraint_omission_steps",
                  "unaccepted_publications", "nonfinite_steps", "unexpected_reset_steps")
SECOND_NS = 1_000_000_000
SPEEDS = (0.5, 1.0, 1.5)
LOWERING_ARTIFACT_ROLES = frozenset({"human_pack", "compiled_run", "task_pack", "native_library",
                                    "human_metallib", "matter_metallib", "brain_metallib",
                                    "metric_program", "accepted_root_proof_schema"})
IMPULSE_FIELDS = frozenset({"body_semantic_id", "frame", "at_step", "point_m", "linear_impulse_ns"})


class EvidenceError(ValueError):
    """Malformed, incomplete, mixed, or inadmissible evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _object(value: Any, label: str) -> dict:
    _require(type(value) is dict, f"{label}: expected object")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    _require(type(value) is int and minimum <= value <= (1 << 64) - 1,
             f"{label}: expected unsigned 64-bit integer >= {minimum}")
    return value


def _finite_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _number(value: Any, label: str, minimum: float = 0.0) -> float:
    _require(_finite_number(value) and value >= minimum,
             f"{label}: expected finite number >= {minimum}")
    return value


def _string(value: Any, label: str) -> str:
    _require(type(value) is str and bool(value.strip()), f"{label}: expected nonempty string")
    return value


def _hash(value: Any, label: str, length: int = 64) -> str:
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) is not None,
             f"{label}: expected lowercase {length}-digit digest")
    return value


def _vector(value: Any, label: str, *, unit: bool = False) -> list:
    _require(type(value) is list and len(value) == 3 and
             all(_finite_number(x) for x in value), label + ": expected three finite numbers")
    if unit:
        _require(all(abs(x) <= 1.000001 for x in value) and
                 abs(sum(float(x) ** 2 for x in value) - 1.0) <= 1e-6,
                 label + ": expected unit axis")
    return value


def _same_frozen_value(actual: Any, expected: Any) -> bool:
    """Compare JSON values without Python's bool == integer equivalence."""
    if type(actual) in (int, float) and type(expected) in (int, float):
        return _finite_number(actual) and _finite_number(expected) and actual == expected
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(
            _same_frozen_value(actual[key], value) for key, value in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _same_frozen_value(left, right) for left, right in zip(actual, expected))
    return actual == expected


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode(raw: bytes, label: str) -> dict:
    def reject(value):
        raise EvidenceError(f"{label}: nonfinite JSON constant {value}")
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=reject)
    except (ValueError, UnicodeError) as error:
        raise EvidenceError(f"{label}: {error}") from error
    # JSON exponent overflow (1e999) bypasses parse_constant.
    def finite(item):
        if type(item) is float:
            _require(math.isfinite(item), f"{label}: nonfinite number")
        elif type(item) is list:
            for child in item:
                finite(child)
        elif type(item) is dict:
            for child in item.values():
                finite(child)
    finite(value)
    return _object(value, label)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _artifact(value: Any, base: Path, label: str) -> Path:
    entry = _object(value, label)
    path = Path(_string(entry.get("path"), label + ".path"))
    path = (base / path).resolve()
    _require(path.is_file(), f"{label}: missing artifact {path}")
    _require(path.stat().st_size == _integer(entry.get("bytes"), label + ".bytes", 1),
             f"{label}: artifact size mismatch")
    _require(digest(path) == _hash(entry.get("sha256"), label + ".sha256"),
             f"{label}: artifact hash mismatch")
    return path


def _pinned(path: Path, expected: str, schema: str) -> dict:
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == _hash(expected, schema + " pin"),
             f"{schema}: externally pinned hash mismatch")
    value = _decode(raw, str(path))
    _require(value.get("schema") == schema, f"{schema}: schema mismatch")
    return value


def _protocol(value: dict) -> dict[str, dict]:
    _require(value.get("metric_contract") == METRIC_CONTRACT, "unsupported metric contract")
    step_ns = _integer(value.get("step_ns"), "protocol.step_ns", 1)
    _require(all(duration % step_ns == 0 for duration in (5 * SECOND_NS, 60 * SECOND_NS,
                                                         120 * SECOND_NS)),
             "step_ns must exactly divide all fixed horizons")
    hold_ns = _integer(value.get("recovery_hold_ns"), "protocol.recovery_hold_ns", 1)
    _require(hold_ns <= 5 * SECOND_NS and hold_ns % step_ns == 0,
             "recovery hold must cover whole steps within the five-second horizon")
    criteria = _object(value.get("task_criteria"), "protocol.task_criteria")
    for task in ("standing", "recovery", "walking"):
        rule = _object(criteria.get(task), "task_criteria." + task)
        _number(rule.get("minimum_root_height_m"), task + ".minimum_root_height_m")
        _number(rule.get("maximum_trunk_tilt_rad"), task + ".maximum_trunk_tilt_rad")
        if task != "walking":
            _number(rule.get("maximum_planar_speed_mps"), task + ".maximum_planar_speed_mps")
        for field in ("root_body_semantic_id", "trunk_body_semantic_id"):
            _string(rule.get(field), task + "." + field)
        _vector(rule.get("world_reference_origin_m"), task + ".world_reference_origin_m")
        up = _vector(rule.get("world_up_axis"), task + ".world_up_axis", unit=True)
        forward = _vector(rule.get("world_forward_axis"), task + ".world_forward_axis", unit=True)
        _require(abs(sum(float(a) * float(b) for a, b in zip(up, forward))) <= 1e-6,
                 task + ": forward and up axes must be orthogonal")
        _vector(rule.get("trunk_up_axis_body"), task + ".trunk_up_axis_body", unit=True)
        observable = rule.get("velocity_observable")
        _require(observable in ("body_com_linear_velocity", "whole_human_com_linear_velocity"),
                 task + ": unsupported frozen velocity observable")
        if observable == "body_com_linear_velocity":
            _string(rule.get("velocity_body_semantic_id"), task + ".velocity_body_semantic_id")
        else:
            _require("velocity_body_semantic_id" in rule and rule["velocity_body_semantic_id"] is None,
                     task + ": whole-Human COM velocity has no single body owner")
        _require(rule.get("forbidden_contact_semantic_ids") and
                 type(rule["forbidden_contact_semantic_ids"]) is list,
                 task + ": frozen forbidden-contact registration required")
        contacts = rule["forbidden_contact_semantic_ids"]
        _require(all(type(item) is str and item for item in contacts) and
                 len(set(contacts)) == len(contacts), task + ": invalid contact IDs")
    cases = value.get("cases")
    _require(type(cases) is list and len(cases) == 420,
             "protocol requires 20 standing, 100 recovery and 100 walking cases per speed")
    indexed, seeds, impulses = {}, {}, set()
    for case in cases:
        case = _object(case, "case")
        key = _string(case.get("trial_id"), "case.trial_id")
        _require(key not in indexed, "duplicate protocol trial_id: " + key)
        task = case.get("task")
        _require(task in ("standing", "recovery", "walking"), "invalid case task")
        seed = _integer(case.get("seed"), "case.seed")
        _hash(case.get("reset_state_sha256"), "case.reset_state_sha256")
        speed = case.get("target_speed_mps")
        if task == "walking":
            _require(type(speed) in (int, float) and speed in SPEEDS,
                     "walking speed must be 0.5, 1.0 or 1.5 m/s")
        else:
            _require(speed is None, "non-walking case must not set a walking speed")
        group = (task, speed)
        seeds.setdefault(group, set())
        _require(seed not in seeds[group], f"duplicate seed in {group}: {seed}")
        seeds[group].add(seed)
        impulse = case.get("impulse")
        if task == "recovery":
            impulse = _object(impulse, "case.impulse")
            _require(impulse.keys() == IMPULSE_FIELDS,
                     "impulse must contain exactly the five physical descriptor fields")
            _string(impulse.get("body_semantic_id"), "impulse.body_semantic_id")
            _require(impulse.get("frame") == "world", "impulse frame must be world")
            _require(impulse.get("at_step") == 0 and type(impulse["at_step"]) is int,
                     "recovery impulse must be delivered once at reset, before root one")
            for field in ("point_m", "linear_impulse_ns"):
                vector = impulse.get(field)
                _require(type(vector) is list and len(vector) == 3 and
                         all(_finite_number(x) for x in vector),
                         "invalid impulse " + field)
            _require(any(x != 0 for x in impulse["linear_impulse_ns"]), "zero recovery impulse")
            normalized_impulse = {key: impulse[key] for key in IMPULSE_FIELDS}
            normalized_impulse["point_m"] = [float(x) + 0.0 for x in impulse["point_m"]]
            normalized_impulse["linear_impulse_ns"] = [float(x) + 0.0 for x in impulse["linear_impulse_ns"]]
            impulse_hash = hashlib.sha256(json.dumps(normalized_impulse, sort_keys=True).encode()).hexdigest()
            _require(impulse_hash not in impulses, "duplicate frozen impulse")
            impulses.add(impulse_hash)
        else:
            _require(impulse is None, "unexpected impulse in non-recovery case")
        indexed[key] = case
    expected = {("standing", None): 20, ("recovery", None): 100,
                **{("walking", speed): 100 for speed in SPEEDS}}
    _require({key: len(value) for key, value in seeds.items()} == expected,
             "protocol population does not meet fixed per-task gates")
    return indexed


def _trace(path: Path, case: dict, protocol: dict, protocol_hash: str,
           stack_hash: str, stack: dict) -> tuple[dict, str]:
    """Stream aggregate spans; memory use is independent of the number of roots."""
    step_ns = protocol["step_ns"]
    task = case["task"]
    horizon_ns = {"standing": 60, "recovery": 5, "walking": 120}[task] * SECOND_NS
    expected_steps = horizon_ns // step_ns
    accepted = elapsed = violations = speed_error_sum = 0
    settled_suffix = 0
    previous_root = None
    execution_id = None
    footer_seen = False
    initial_settled = initial_posture_valid = False
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            _require(raw.endswith(b"\n"), f"{case['trial_id']}: incomplete final line")
            entry = _decode(raw, f"{path}:{line_number}")
            if line_number == 1:
                _require(entry.get("schema") == TRACE_SCHEMA, "trace header missing")
                for key in ("trial_id", "task", "seed", "reset_state_sha256", "target_speed_mps", "impulse"):
                    _require(key in entry and _same_frozen_value(entry[key], case.get(key)),
                             f"trace {key} differs from frozen case")
                _require(type(entry.get("seed")) is int, "trace seed must be integer")
                if task == "recovery":
                    _require(type(entry["impulse"]["at_step"]) is int,
                             "trace impulse at_step must be integer")
                _require(entry.get("protocol_sha256") == protocol_hash and
                         entry.get("stack_sha256") == stack_hash, "stale or mixed trace identity")
                _require(entry.get("device") == stack["device"], "trace device differs from stack")
                _require(entry.get("os_build") == stack["os_build"], "trace OS build differs from stack")
                _require(_integer(entry.get("step_ns"), "trace.step_ns", 1) == step_ns,
                         "trace native timestep differs from frozen protocol")
                for role in ("metric_program", "accepted_root_proof_schema", "task_lowering_receipt"):
                    _require(entry.get(role + "_sha256") == stack["artifacts"][role]["sha256"],
                             "trace native program/proof identity mismatch: " + role)
                _require(entry.get("metric_contract") == METRIC_CONTRACT,
                         "trace metric contract mismatch")
                for field in ("initial_posture_valid", "initial_settled"):
                    _require(type(entry.get(field)) is bool, "trace requires native " + field)
                initial_posture_valid = entry["initial_posture_valid"]
                initial_settled = entry["initial_settled"]
                _require(not initial_settled or initial_posture_valid,
                         "initial settled state violates posture criteria")
                execution_id = _string(entry.get("execution_id"), "trace.execution_id")
                previous_root = _hash(entry.get("initial_root_sha256"), "trace.initial_root_sha256")
                continue
            _require(not footer_seen, "data follows completed trace footer")
            if entry.get("kind") == "completed":
                _require(_integer(entry.get("accepted_steps"), "footer.accepted_steps") == accepted,
                         "footer step count mismatch")
                _require(entry.get("final_root_sha256") == previous_root, "footer root mismatch")
                _require(entry.get("exit_code") == 0 and type(entry["exit_code"]) is int,
                         "native run did not complete successfully")
                footer_seen = True
                continue
            _require(entry.get("kind") == "accepted_span", "unknown trace record")
            count = _integer(entry.get("accepted_steps"), "span.accepted_steps", 1)
            _require(_integer(entry.get("first_step"), "span.first_step", 1) == accepted + 1 and
                     _integer(entry.get("last_step"), "span.last_step", 1) == accepted + count,
                     "overlapping, duplicate or missing accepted roots")
            _require(_integer(entry.get("start_ns"), "span.start_ns") == elapsed and
                     _integer(entry.get("end_ns"), "span.end_ns", 1) == elapsed + count * step_ns,
                     "accepted time is not contiguous or does not match root step count")
            _require(entry.get("start_root_sha256") == previous_root, "broken accepted-root chain")
            previous_root = _hash(entry.get("end_root_sha256"), "span.end_root_sha256")
            _require(_integer(entry.get("attempt_count"), "span.attempt_count", 1) == count +
                     _integer(entry.get("rejected_attempt_count"), "span.rejected_attempt_count"),
                     "attempt accounting does not match accepted plus rejected roots")
            for field, expected in (("metric_sample_count", count),
                                    ("audit_covered_root_count", count),
                                    ("audit_covered_attempt_count", entry["attempt_count"]),
                                    ("speed_error_sample_count", count if task == "walking" else 0)):
                _require(_integer(entry.get(field), "span." + field) == expected,
                         "incomplete native reduction coverage: " + field)
            audits = _object(entry.get("audit"), "span.audit")
            for name in AUDIT_COUNTERS:
                _require(_integer(audits.get(name), "audit." + name) == 0,
                         "inadmissible native audit: " + name)
            bad = _integer(entry.get("posture_violation_steps"), "span.posture_violation_steps")
            settled = _integer(entry.get("settled_steps"), "span.settled_steps")
            suffix = _integer(entry.get("settled_suffix_steps"), "span.settled_suffix_steps")
            _require(bad <= count and settled <= count - bad and suffix <= settled and
                     ((settled == count and suffix == count) or
                      (settled < count and suffix < count)), "inconsistent task reduction counts")
            if task == "walking":
                _require(settled == count - bad,
                         "walking settled count must cover every posture-valid accepted root")
            settled_suffix = settled_suffix + count if suffix == count else suffix
            violations += bad
            speed_error = _number(entry.get("speed_squared_error_sum_m2_per_s2"),
                                  "span.speed_squared_error_sum_m2_per_s2")
            _require(task == "walking" or speed_error == 0, "unexpected walking error in other task")
            speed_error_sum += speed_error
            _require(math.isfinite(speed_error_sum), "nonfinite aggregate speed error")
            accepted += count
            elapsed += count * step_ns
            _require(elapsed <= horizon_ns, "trace exceeds frozen horizon")
    _require(execution_id is not None and footer_seen, "missing trace header or completion footer")
    _require(accepted == expected_steps and elapsed == horizon_ns,
             "truncated trial does not cover the complete frozen horizon")
    rmse = math.sqrt(speed_error_sum / accepted) if task == "walking" else None
    # Samples describe accepted end-of-step states. A suffix of k roots begins
    # at t_end - (k - 1) * h, not one step earlier. Only a settled reset allows t=0.
    recovery_ns = None
    if task == "recovery" and settled_suffix:
        recovery_ns = (0 if initial_settled and settled_suffix == accepted else
                       elapsed - (settled_suffix - 1) * step_ns)
    if task == "standing":
        success = initial_posture_valid and initial_settled and violations == 0 and settled_suffix == accepted
    elif task == "walking":
        success = initial_posture_valid and violations == 0 and rmse <= 0.15
    else:
        success = (initial_posture_valid and violations == 0 and recovery_ns is not None
                   and elapsed - recovery_ns >= protocol["recovery_hold_ns"]
                   and recovery_ns <= 5 * SECOND_NS)
    return {"trial_id": case["trial_id"], "task": task, "seed": case["seed"],
            "target_speed_mps": case.get("target_speed_mps"), "accepted_steps": accepted,
            "accepted_seconds": elapsed / SECOND_NS, "success": success,
            "speed_rmse_mps": rmse, "recovery_seconds": recovery_ns / SECOND_NS
            if recovery_ns is not None else None}, execution_id


def evaluate(protocol_path: Path, stack_path: Path, bundle_path: Path, *,
             expected_protocol_sha256: str, expected_stack_sha256: str) -> dict:
    """Validate current bytes and return a gate assessment; invalid evidence raises.

    The two expected hashes must come from the caller's frozen protocol and current
    stack authority, independently of the evidence bundle. Self-declared identity
    in a historical receipt never selects the stack being qualified.
    """
    protocol = _pinned(protocol_path, expected_protocol_sha256, PROTOCOL_SCHEMA)
    stack = _pinned(stack_path, expected_stack_sha256, STACK_SCHEMA)
    cases = _protocol(protocol)
    revisions = _object(stack.get("revisions"), "stack.revisions")
    for owner in ("human", "native", "brain"):
        _hash(revisions.get(owner), "revision." + owner, 40)
    _require(stack.get("source_state") == "clean", "qualification requires an immutable clean stack")
    _require(stack.get("backend") == "Apple Metal", "qualification requires the Apple Metal backend")
    device = _string(stack.get("device"), "stack.device")
    _require(device.startswith("Apple "), "expected Apple device identity")
    _string(stack.get("os_build"), "stack.os_build")
    artifacts = _object(stack.get("artifacts"), "stack.artifacts")
    _require(ARTIFACT_ROLES <= artifacts.keys(), "stack omits required artifact roles")
    pinned_files = [(entry, _artifact(entry, stack_path.parent, "artifact." + key))
                    for key, entry in artifacts.items()]
    _require(artifacts["qualifier"]["sha256"] == digest(Path(__file__)),
             "stack qualifier does not match the executing evaluator")
    _require(protocol.get("task_pack_sha256") == artifacts["task_pack"]["sha256"],
             "protocol is bound to a different TaskPack")
    lowering_path = next(path for entry, path in pinned_files
                         if entry is artifacts["task_lowering_receipt"])
    lowering = _decode(lowering_path.read_bytes(), "task lowering receipt")
    _integer(lowering.get("step_ns"), "task lowering receipt.step_ns", 1)
    expected_lowering = {
        "schema": LOWERING_SCHEMA, "metric_contract": METRIC_CONTRACT,
        "protocol_sha256": expected_protocol_sha256,
        "native_revision": revisions["native"], "step_ns": protocol["step_ns"],
        "task_criteria": protocol["task_criteria"],
        "artifacts": {role: artifacts[role]["sha256"] for role in LOWERING_ARTIFACT_ROLES},
    }
    for field, expected in expected_lowering.items():
        _require(field in lowering and _same_frozen_value(lowering[field], expected),
                 "task lowering receipt does not bind current native program: " + field)
    bundle_raw = bundle_path.read_bytes()
    bundle = _decode(bundle_raw, str(bundle_path))
    _require(bundle.get("schema") == BUNDLE_SCHEMA, "bundle schema mismatch")
    _require(bundle.get("protocol_sha256") == expected_protocol_sha256 and
             bundle.get("stack_sha256") == expected_stack_sha256, "stale or mixed bundle identity")
    trial_artifacts = bundle.get("trials")
    _require(type(trial_artifacts) is list and len(trial_artifacts) == len(cases),
             "bundle must contain the complete frozen trial population")
    seen_trials, seen_paths, seen_hashes, executions = set(), set(), set(), set()
    results = []
    for item in trial_artifacts:
        item = _object(item, "trial artifact")
        key = item.get("trial_id")
        _require(type(key) is str and key in cases and key not in seen_trials,
                 "unknown or duplicate trial artifact")
        seen_trials.add(key)
        path = _artifact(item, bundle_path.parent, "trial." + key)
        _require(path not in seen_paths and item["sha256"] not in seen_hashes,
                 "duplicate trial path or content")
        seen_paths.add(path)
        seen_hashes.add(item["sha256"])
        result, execution = _trace(path, cases[key], protocol, expected_protocol_sha256,
                                   expected_stack_sha256, stack)
        _require(execution not in executions, "reused execution_id across trials")
        executions.add(execution)
        _require(digest(path) == item["sha256"], "trial changed during evaluation")
        results.append(result)
    # Revalidate live artifact bytes so concurrent rebuilds cannot produce a pass.
    for entry, path in pinned_files:
        _require(digest(path) == entry["sha256"], "stack artifact changed during evaluation")
    _require(digest(protocol_path) == expected_protocol_sha256 and
             digest(stack_path) == expected_stack_sha256 and
             digest(bundle_path) == hashlib.sha256(bundle_raw).hexdigest(),
             "manifest changed during evaluation")
    groups = []
    for task, speed, required in (("standing", None, 20), ("recovery", None, 95),
                                   *(("walking", speed, 95) for speed in SPEEDS)):
        members = [r for r in results if r["task"] == task and r["target_speed_mps"] == speed]
        successes = sum(r["success"] for r in members)
        groups.append({"task": task, "target_speed_mps": speed, "trials": len(members),
                       "successful_trials": successes, "required_successes": required,
                       "passed": successes >= required})
    return {"schema": REPORT_SCHEMA, "status": "passed" if all(g["passed"] for g in groups) else "failed",
            "protocol_sha256": expected_protocol_sha256, "stack_sha256": expected_stack_sha256,
            "bundle_sha256": hashlib.sha256(bundle_raw).hexdigest(), "groups": groups,
            "trials": results,
            "boundary": "Admission of the supplied native accepted-root metric contract only; hashes bind bytes, not producer authenticity. This does not qualify anatomy, material laws, source parity or performance."}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--stack", type=Path, required=True)
    parser.add_argument("--stack-sha256", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    try:
        report = evaluate(args.protocol, args.stack, args.bundle,
                          expected_protocol_sha256=args.protocol_sha256,
                          expected_stack_sha256=args.stack_sha256)
    except (EvidenceError, OSError) as error:
        report = {"schema": REPORT_SCHEMA, "status": "invalid", "failures": [str(error)]}
    # Evidence reports are append-only artifacts; never replace an earlier result.
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"status={report['status']} report={args.output}")
    return 0 if report["status"] == "passed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
