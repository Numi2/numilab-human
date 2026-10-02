"""Join Human subsystem evidence without promoting component results to Human qualification.

The registry is a source-hash-bound evidence join. Each subsystem's closure
conditions remain explicit in a versioned profile; missing, stale-schema, or
false evidence keeps the whole-human capability open.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from .model import ImportError as HumanImportError

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config/whole-body-capability-registry.v1.json"
PROFILE_SCHEMA = "numi.human.whole-body-capability-profile.v1"
SCHEMA = "numi.human.whole-body-capability-registry.v1"
SYSTEM_IDS = frozenset({
    "whole_body_anatomy", "internal_organs", "skin", "muscle", "tendon",
    "bloodflow", "circulation", "cardiac_electrical", "cardiac_mechanics",
    "neural_physiology", "whole_body_calibration", "whole_body_dynamics",
})
OPERATORS = frozenset({"equals", "at_least", "at_most"})


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("whole-body capability registry: " + message)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise HumanImportError("registry contains non-finite or non-JSON data") from error


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            _need(key not in result, f"{label} has duplicate key {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise HumanImportError(f"{label} contains non-finite JSON value {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           object_pairs_hook=pairs,
                           parse_constant=invalid_constant)
    except HumanImportError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise HumanImportError(f"cannot read {label}") from error
    _need(isinstance(value, dict), f"{label} must be an object")
    _canonical(value)
    return value


def _text(value: Any, label: str) -> str:
    _need(isinstance(value, str) and bool(value.strip()),
          f"{label} must be nonempty text")
    return value.strip()


def _safe_relative(root: Path, value: Any, label: str) -> tuple[Path, str]:
    raw = _text(value, label)
    relative = PurePosixPath(raw)
    _need(not relative.is_absolute() and ".." not in relative.parts and "\\" not in raw,
          f"{label} is unsafe")
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        _need(not candidate.is_symlink(), f"{label} traverses a symlink")
    resolved = candidate.resolve()
    _need(resolved.is_relative_to(root), f"{label} escapes repository root")
    return resolved, "/".join(relative.parts)


def _field(value: Any, path: str) -> tuple[bool, Any]:
    current = value
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdecimal() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, current


def _validate_profile(value: dict[str, Any]) -> None:
    _need(set(value) == {"schema", "id", "systems"}, "profile fields differ")
    _need(value.get("schema") == PROFILE_SCHEMA, "unsupported profile schema")
    _text(value.get("id"), "profile id")
    systems = value.get("systems")
    _need(isinstance(systems, list) and systems, "profile systems must be nonempty")
    ids: set[str] = set()
    for system in systems:
        expected = {"id", "domain", "scope", "evidence", "facts", "requirements"}
        _need(isinstance(system, dict) and set(system) == expected,
              "system profile fields differ")
        system_id = _text(system["id"], "system id")
        _need(system_id not in ids, f"duplicate system {system_id}")
        ids.add(system_id)
        _text(system["domain"], f"{system_id} domain")
        _text(system["scope"], f"{system_id} scope")
        evidence = system["evidence"]
        facts = system["facts"]
        requirements = system["requirements"]
        _need(isinstance(evidence, list) and evidence,
              f"{system_id} evidence must be nonempty")
        _need(isinstance(facts, list), f"{system_id} facts must be a list")
        _need(isinstance(requirements, list) and requirements,
              f"{system_id} requirements must be nonempty")
        evidence_ids: set[str] = set()
        for source in evidence:
            _need(isinstance(source, dict) and set(source) == {"id", "path", "schema"},
                  f"{system_id} evidence fields differ")
            source_id = _text(source["id"], f"{system_id} evidence id")
            _need(source_id not in evidence_ids,
                  f"{system_id} duplicate evidence id {source_id}")
            evidence_ids.add(source_id)
            _text(source["path"], f"{system_id}/{source_id} path")
            _text(source["schema"], f"{system_id}/{source_id} schema")
        fact_ids: set[str] = set()
        for fact in facts:
            _need(isinstance(fact, dict) and set(fact) == {"id", "evidence", "path"},
                  f"{system_id} fact fields differ")
            fact_id = _text(fact["id"], f"{system_id} fact id")
            _need(fact_id not in fact_ids, f"{system_id} duplicate fact {fact_id}")
            fact_ids.add(fact_id)
            _need(fact["evidence"] in evidence_ids,
                  f"{system_id}/{fact_id} refers to unknown evidence")
            _text(fact["path"], f"{system_id}/{fact_id} path")
        requirement_ids: set[str] = set()
        for requirement in requirements:
            fields = {"id", "evidence", "path", "operator", "value", "reason"}
            _need(isinstance(requirement, dict) and set(requirement) == fields,
                  f"{system_id} requirement fields differ")
            requirement_id = _text(requirement["id"], f"{system_id} requirement id")
            _need(requirement_id not in requirement_ids,
                  f"{system_id} duplicate requirement {requirement_id}")
            requirement_ids.add(requirement_id)
            _need(requirement["evidence"] in evidence_ids,
                  f"{system_id}/{requirement_id} refers to unknown evidence")
            _text(requirement["path"], f"{system_id}/{requirement_id} path")
            _need(requirement["operator"] in OPERATORS,
                  f"{system_id}/{requirement_id} has unsupported operator")
            _text(requirement["reason"], f"{system_id}/{requirement_id} reason")
            expected_value = requirement["value"]
            if requirement["operator"] == "equals":
                _canonical(expected_value)
            else:
                _need(type(expected_value) in (int, float) and math.isfinite(expected_value),
                      f"{system_id}/{requirement_id} threshold is not finite")
    _need(ids == SYSTEM_IDS,
          "profile does not cover every required whole-Human subsystem")


def _condition(operator: str, actual: Any, expected: Any) -> bool:
    if operator == "equals":
        return type(actual) is type(expected) and actual == expected
    if type(actual) not in (int, float) or not math.isfinite(actual):
        return False
    return actual >= expected if operator == "at_least" else actual <= expected


def compile_registry(*, profile: Path = PROFILE, root: Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    profile_path, profile_relative = _safe_relative(
        root, Path(profile).resolve().relative_to(root).as_posix(), "profile path")
    profile_value = _strict_json(profile_path, "capability profile")
    _validate_profile(profile_value)
    evidence_by_system: dict[str, dict[str, dict[str, Any]]] = {}
    output_systems: list[dict[str, Any]] = []
    all_sources: dict[str, dict[str, Any]] = {}

    for system in profile_value["systems"]:
        system_id = system["id"]
        parsed_sources: dict[str, dict[str, Any]] = {}
        source_outputs: list[dict[str, Any]] = []
        for source in system["evidence"]:
            path, relative = _safe_relative(root, source["path"],
                                            f"{system_id}/{source['id']} path")
            key = f"{system_id}:{source['id']}"
            if not path.is_file():
                source_record = {"path": relative, "expected_schema": source["schema"],
                                 "status": "missing", "sha256": None}
                parsed_sources[source["id"]] = {"status": "missing", "value": None}
            else:
                _need(not path.is_symlink(), f"{key} is a symlink")
                raw_sha = _file_sha256(path)
                parsed = _strict_json(path, key)
                actual_schema = parsed.get("schema")
                valid_schema = actual_schema == source["schema"]
                source_record = {"path": relative,
                                 "expected_schema": source["schema"],
                                 "schema": actual_schema,
                                 "sha256": raw_sha,
                                 "status": "verified" if valid_schema else "schema_mismatch"}
                parsed_sources[source["id"]] = {
                    "status": "verified" if valid_schema else "schema_mismatch",
                    "value": parsed if valid_schema else None,
                }
            all_sources[key] = source_record
            source_outputs.append({"id": source["id"], **source_record})
        evidence_by_system[system_id] = parsed_sources

        facts: list[dict[str, Any]] = []
        for fact in system["facts"]:
            source = parsed_sources[fact["evidence"]]
            available, actual = (False, None) if source["status"] != "verified" else _field(
                source["value"], fact["path"])
            facts.append({"id": fact["id"], "evidence": fact["evidence"],
                          "path": fact["path"], "available": available,
                          "value": actual if available else None})

        requirements: list[dict[str, Any]] = []
        for requirement in system["requirements"]:
            source = parsed_sources[requirement["evidence"]]
            available, actual = (False, None) if source["status"] != "verified" else _field(
                source["value"], requirement["path"])
            satisfied = available and _condition(
                requirement["operator"], actual, requirement["value"])
            requirements.append({
                "id": requirement["id"], "evidence": requirement["evidence"],
                "path": requirement["path"], "operator": requirement["operator"],
                "expected": requirement["value"], "available": available,
                "observed": actual if available else None, "satisfied": satisfied,
                "reason": requirement["reason"],
            })
        qualified = bool(requirements) and all(item["satisfied"] for item in requirements)
        output_systems.append({
            "id": system_id, "domain": system["domain"], "scope": system["scope"],
            "status": "qualified" if qualified else "partial",
            "evidence": source_outputs, "facts": facts,
            "requirements": requirements,
            "open_requirements": [item["id"] for item in requirements
                                  if not item["satisfied"]],
        })

    qualified_count = sum(system["status"] == "qualified" for system in output_systems)
    result = {
        "schema": SCHEMA,
        "compiler": "numilab-human.whole-body-capability-registry.1",
        "compiler_sha256": _file_sha256(Path(__file__)),
        "profile": {"id": profile_value["id"], "path": profile_relative,
                    "sha256": _file_sha256(profile_path)},
        "status": "qualified" if qualified_count == len(output_systems) else "partial",
        "systems": output_systems,
        "inputs": all_sources,
        "qualification": {
            "all_required_subsystems_qualified": qualified_count == len(output_systems),
            "whole_human_capability": qualified_count == len(output_systems),
            "clinical_human_qualification": False,
        },
        "counts": {
            "required_subsystems": len(output_systems),
            "qualified_subsystems": qualified_count,
            "partial_subsystems": len(output_systems) - qualified_count,
            "verified_evidence_sources": sum(
                item["status"] == "verified" for item in all_sources.values()),
            "missing_evidence_sources": sum(
                item["status"] == "missing" for item in all_sources.values()),
            "schema_mismatch_sources": sum(
                item["status"] == "schema_mismatch" for item in all_sources.values()),
            "open_requirements": sum(len(item["open_requirements"])
                                      for item in output_systems),
        },
        "boundary": (
            "This registry joins source-hash-bound subsystem receipts and evaluates their "
            "declared closure conditions. It does not rerun source solvers, prove the truth "
            "of their measurements, or promote visual geometry, candidates, short runs, or "
            "component checks to whole-Human, physiological, or clinical qualification."
        ),
    }
    result["report_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = _canonical(value) + b"\n"
    path = Path(path)
    _need(not path.is_symlink(), "output is redirected")
    if path.exists():
        _need(path.read_bytes() == payload, "output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = compile_registry(profile=args.profile, root=args.repository_root)
        output_sha = immutable_write(args.output, result)
    except (HumanImportError, OSError, ValueError) as error:
        parser.exit(2, f"whole-body-capability-registry: {error}\n")
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "output": str(args.output.resolve()),
                      "sha256": output_sha, **result["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
