#!/usr/bin/env python3
"""Audit the retained bounded native vascular evidence; never execute physics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "numi.human.organ-circulation-evidence.v1"
NATIVE_COMMIT = "1594f7aff5503aab96ab9de6e77b6d4f8fa4f1cd"
SHA256 = re.compile(r"[0-9a-f]{64}")
REQUIRED_ARTIFACTS = {
    "native-ctest.txt", "native-tests-detailed.txt", "native-identity.json",
    "passive-fixture.native.json", "anatomy-validation.json",
}
REQUIRED_SOURCES = {
    "src/numilab_human/physiology.py", "src/numilab_human/target_coverage.py",
    "config/physiology-passive-fixture.v1.json", "config/physiology-organ-network-template.v1.json",
    "schemas/humanpack-physiology.v1.schema.json",
}
CTEST_CASES = {
    "matter.runtime.snapshot_archive", "matter.physics.monolithic_multiphysics",
    "matter.runtime.production_transaction_rollback", "matter.physics.stateful_mpm",
    "matter.physics.stateful_fem", "numanx.integration.matter_accepted_state_2pc",
    "matter.compiler.stateful_roundtrip", "matter.compiler.vascular", "matter.metal.vascular",
    "matter.metal.human_physiology_payload", "matter.compiler.human_physiology_admission",
}
VASCULAR_CASES = {"resistive_exchange", "reverse_flow", "inertial_flow", "fem_region", "human_compiled_fixture"}


class EvidenceError(ValueError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def read_bytes(path: Path) -> bytes:
    require(path.is_file() and path.stat().st_size <= 32 * 1024 * 1024, f"missing or oversized evidence: {path.name}")
    return path.read_bytes()


def read_json(path: Path) -> dict:
    def pairs(items: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON field: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(read_bytes(path), object_pairs_hook=pairs,
                           parse_constant=lambda token: (_ for _ in ()).throw(EvidenceError(f"nonfinite JSON: {token}")))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise EvidenceError(f"invalid JSON: {path.name}") from error
    require(isinstance(value, dict), f"expected JSON object: {path.name}")
    return value


def safe_path(root: Path, relative: Any) -> Path:
    require(isinstance(relative, str) and relative and "\\" not in relative, "unsafe evidence path")
    part = PurePosixPath(relative)
    require(not part.is_absolute() and ".." not in part.parts and relative == part.as_posix() and part.parts,
            "unsafe evidence path")
    path = root / Path(*part.parts)
    require(path.resolve().is_relative_to(root.resolve()), "evidence path escapes root")
    current = root
    for component in part.parts:
        current /= component
        require(not current.is_symlink(), "symlink evidence path is not admitted")
    return path


def file_hash(path: Path) -> str:
    return hashlib.sha256(read_bytes(path)).hexdigest()


def hash_map(value: Any, context: str) -> dict[str, str]:
    require(isinstance(value, dict) and 0 < len(value) <= 512, f"missing {context} hash map")
    for name, fingerprint in value.items():
        require(isinstance(name, str) and isinstance(fingerprint, str) and SHA256.fullmatch(fingerprint),
                f"invalid {context} hash")
    return value


def records(text: str, prefix: str) -> list[dict[str, str]]:
    result = []
    for line in text.splitlines():
        if not line.startswith(prefix + "="):
            continue
        record = {}
        for token in line.split():
            require("=" in token, f"malformed {prefix} metric")
            key, value = token.split("=", 1)
            require(key and value and key not in record, f"duplicate/empty {prefix} metric")
            record[key] = value
        result.append(record)
    return result


def numeric(record: dict, key: str, *, maximum: float | None = None, positive: bool = False) -> float:
    try:
        value = float(record[key])
    except (KeyError, ValueError, TypeError) as error:
        raise EvidenceError(f"missing/invalid {key}") from error
    require(math.isfinite(value) and (value > 0 if positive else value >= 0), f"invalid {key}")
    if maximum is not None:
        require(value < maximum, f"{key} exceeds qualification gate")
    return value


def exact(record: dict, values: dict, context: str) -> None:
    require(all(record.get(key) == value for key, value in values.items()), f"{context} does not meet its required boundary")


def audit_logs(ctest: str, detailed: str) -> dict:
    summary = re.findall(r"^100% tests passed out of (\d+)\s*$", ctest, re.MULTILINE)
    require(summary == ["11"], "CTest must report exactly 11/11 passed")
    rows = re.findall(r"^\s*(\d+)/11 Test\s+#\d+:\s+(\S+)\s+\.+\s+Passed\s+[0-9.]+ sec\s*$", ctest, re.MULTILINE)
    require(len(rows) == 11 and {int(row[0]) for row in rows} == set(range(1, 12)) and
            {row[1] for row in rows} == CTEST_CASES, "CTest case coverage is missing, duplicated or failed")
    cases = records(detailed, "vascular_case")
    require(len(cases) == 5 and {row["vascular_case"] for row in cases} == VASCULAR_CASES,
            "required vascular case coverage is missing or duplicated")
    for case in cases:
        exact(case, {"result": "pass", "accepted_steps": "16", "environments": "2", "replay": "bitwise",
                     "isolated_rollback": "pass", "invalid_restore": "denied", "isolated_reset": "pass",
                     "fem_region": "bound_identity_only" if case["vascular_case"] == "fem_region" else "none"}, "vascular case")
        numeric(case, "fp64_normalized_max", maximum=8e-5)
        numeric(case, "relative_conservation_max", maximum=3e-5)
    branch = records(detailed, "vascular_branched")
    require(len(branch) == 1, "missing or duplicate branched network evidence")
    exact(branch[0], {"vascular_branched": "pass", "compartments": "3", "edges": "3", "species": "2",
                      "tissues": "2", "exchanges": "3", "replay": "bitwise"}, "branched network")
    numeric(branch[0], "relative_conservation_max", maximum=3e-5)
    refinement = records(detailed, "vascular_refinement")
    require(len(refinement) == 1, "missing or duplicate timestep refinement")
    exact(refinement[0], {"vascular_refinement": "pass", "equal_duration_s": "0.16"}, "timestep refinement")
    raw_errors = refinement[0].get("errors_m3", "").split(",")
    require(len(raw_errors) == 3, "refinement must retain all three errors")
    errors = [numeric({"refinement_error": value}, "refinement_error", positive=True) for value in raw_errors]
    ratios = [errors[index] / errors[index + 1] for index in range(2)]
    require(all(math.isfinite(ratio) and ratio > 1.7 for ratio in ratios), "timestep refinement fails decrease/rate gate")
    mutation = records(detailed, "vascular_content_mutation")
    require(mutation == [{"vascular_content_mutation": "pass"}], "missing content mutation rejection")
    admission = records(detailed, "native_input_admission")
    require(len(admission) == 1, "missing native input admission")
    exact(admission[0], {"native_input_admission": "pass", "positive_cases": "1", "negative_cases": "12"}, "native input admission")
    qualification = records(detailed, "vascular_native_qualification")
    require(len(qualification) == 2, "missing native package qualification boundaries")
    for row in qualification:
        exact(row, {"vascular_native_qualification": "pass", "runtime_input": "nmatterpack",
                    "biological_calibration": "unqualified"}, "native qualification")
    compilation = records(detailed, "physiology_compile")
    require(len(compilation) == 1, "missing physiology compilation")
    exact(compilation[0], {"physiology_compile": "pass", "compartments": "2", "species": "1",
                          "tissue_reservoirs": "1", "biological_qualification": "unqualified"}, "physiology compiler")
    return {"native_tests_passed": 11, "vascular_cases": sorted(VASCULAR_CASES),
            "maximum_fp64_normalized_error": max(float(case["fp64_normalized_max"]) for case in cases),
            "maximum_relative_conservation_error": max(float(row["relative_conservation_max"]) for row in [*cases, branch[0]]),
            "refinement_error_ratios": ratios}


def verify(receipt_path: Path, *, repository_root: Path = ROOT) -> dict:
    receipt = read_json(receipt_path)
    exact(receipt, {"schema": SCHEMA, "native_commit": NATIVE_COMMIT, "scientific_status": "unqualified"}, "receipt")
    artifacts = receipt.get("artifacts")
    require(isinstance(artifacts, list) and 5 <= len(artifacts) <= 64, "missing artifact inventory")
    paths = {}
    for record in artifacts:
        require(isinstance(record, dict) and set(record) == {"path", "sha256"}, "invalid artifact record")
        path = safe_path(receipt_path.parent, record["path"])
        require(record["path"] not in paths and path != receipt_path, "duplicate artifact path or recursive receipt")
        require(isinstance(record["sha256"], str) and SHA256.fullmatch(record["sha256"]), "invalid artifact SHA256")
        require(file_hash(path) == record["sha256"], f"artifact hash drift: {record['path']}")
        paths[record["path"]] = path
    require(REQUIRED_ARTIFACTS <= paths.keys(), "required artifact is missing")
    actual = {path.relative_to(receipt_path.parent).as_posix() for path in receipt_path.parent.rglob("*")
              if path.is_file() and path != receipt_path}
    require(actual == paths.keys(), "artifact inventory does not capture every retained file")
    source_hashes = hash_map(receipt.get("source_sha256"), "Human source")
    require(REQUIRED_SOURCES <= source_hashes.keys(), "required owning Human source pin is missing")
    for relative, expected in source_hashes.items():
        require(file_hash(safe_path(repository_root, relative)) == expected, f"Human source hash drift: {relative}")
    identity = read_json(paths["native-identity.json"])
    exact(identity, {"native_commit": NATIVE_COMMIT, "worktree_status": "", "abi": 26, "package_version": 11,
                     "snapshot_archive": 5, "accepted_proof_manifest": 5}, "native identity")
    require(type(identity["abi"]) is int and type(identity["package_version"]) is int, "native version identity must be integral")
    exact(identity.get("device", {}), {"chip": "Apple M4 Pro", "model": "Mac mini"}, "physical hardware identity")
    native_sources = hash_map(identity.get("source_sha256"), "native source")
    binaries = hash_map(identity.get("built_artifact_sha256"), "built native artifact")
    require({"lib/libmetalrobo.dylib", "matter/numi-matter-physiologyc", "matter/numi-matter-vascular-check",
             "matter/shaders/NumiMatter.metallib"} <= binaries.keys(), "missing built owner artifact identity")
    require(native_sources.get("matter/tools/fixtures/human_physiology.native.v1.json") == file_hash(paths["passive-fixture.native.json"]),
            "Human fixture does not match the qualified native input")
    fixture = read_json(paths["passive-fixture.native.json"])
    exact(fixture, {"schema": "HumanPack.physiology-native.v1", "qualification": "fixture_only",
                    "law": "closed_linear_compliance_transport_v1"}, "native fixture")
    for field, count in {"species": 1, "compartments": 2, "connections": 1, "tissue_reservoirs": 1, "exchanges": 1}.items():
        require(isinstance(fixture.get(field), list) and len(fixture[field]) == count, "compiled fixture coverage drift")
    anatomy = read_json(paths["anatomy-validation.json"])
    require(anatomy.get("calibration_required") is True and isinstance(anatomy.get("missing_parameters"), list) and
            len(anatomy["missing_parameters"]) == 276 and len(set(anatomy["missing_parameters"])) == 276,
            "anatomy template must preserve all unresolved calibration parameters")
    require(isinstance(anatomy.get("overlapping_source_memberships"), list) and len(anatomy["overlapping_source_memberships"]) == 8,
            "anatomy source overlap boundary was lost")
    exact(anatomy.get("native", {}), {"qualification": "uncalibrated", "model_id": "organ_circulation_authoring_template"}, "anatomy template")
    metrics = audit_logs(read_bytes(paths["native-ctest.txt"]).decode(), read_bytes(paths["native-tests-detailed.txt"]).decode())
    return {"status": "pass", "scientific_status": "unqualified", "native_commit": NATIVE_COMMIT,
            "boundary": "bounded_passive_native_circulation_not_full_human_or_systemic_qualification", **metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.receipt, repository_root=args.repository_root), indent=2))
    except (EvidenceError, OSError, UnicodeError, TypeError, KeyError) as error:
        print(json.dumps({"status": "fail", "error": str(error)}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
