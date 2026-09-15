"""Compile the Human calibration and unresolved-material evidence ledger.

This registry is deliberately a join, not a qualification solver.  It makes
the current one-male evidence boundary machine-readable: fitted observations,
held-out observations, source candidates and production owners are separate
fields.  A source candidate can therefore be useful for planning without
being silently promoted into mechanics, blood transfer, activation, or
subject calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from .model import ImportError as HumanImportError

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.calibration-evidence-registry.v1"
COMPILER = "numilab-human.calibration-evidence-registry.1"
PROFILE_SCHEMA = "numi.human.calibration-evidence-registry.v1"
PROFILE = ROOT / "config/calibration-evidence-registry.v1.json"
SHA256 = set("0123456789abcdef")


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("calibration evidence registry: " + message)


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise HumanImportError("registry contains non-finite or non-JSON data") from error


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(chunk)
    except OSError as error:
        raise HumanImportError(f"cannot read {path}") from error
    return result.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in items:
                _need(key not in result, f"{label} contains duplicate key {key}")
                result[key] = value
            return result

        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    except HumanImportError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise HumanImportError(f"invalid {label}") from error
    _need(isinstance(value, dict), f"{label} must be an object")
    canonical(value)
    return value


def _text(value: Any, label: str) -> str:
    _need(isinstance(value, str) and bool(value.strip()), f"{label} must be nonempty text")
    return value.strip()


def _relative(root: Path, value: Any, label: str) -> tuple[Path, str]:
    name = _text(value, label)
    relative = PurePosixPath(name)
    _need(not relative.is_absolute() and ".." not in relative.parts and "\\" not in name,
          f"{label} is unsafe")
    candidate = root / Path(*relative.parts)
    _need(not candidate.is_symlink(), f"{label} may not be a symlink")
    path = candidate.resolve()
    _need(path.is_relative_to(root.resolve()) and path.is_file() and not path.is_symlink(),
          f"{label} is missing or external")
    return path, "/".join(relative.parts)


def _sha(value: Any, label: str) -> str:
    text = _text(value, label).lower()
    _need(len(text) == 64 and set(text) <= SHA256, f"{label} is not SHA-256")
    return text


def _profile(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    value = _read_json(path, "registry profile")
    required = {"schema", "id", "rows"}
    _need(set(value) == required, "registry profile fields differ")
    _need(value["schema"] == PROFILE_SCHEMA, "unsupported registry profile schema")
    identifier = _text(value["id"], "registry profile id")
    rows = value["rows"]
    _need(isinstance(rows, list) and rows, "registry profile rows must be nonempty")
    seen: set[str] = set()
    for row in rows:
        expected = {"id", "domain", "source", "fit_evidence", "held_out_evidence",
                    "production_owner", "scope"}
        _need(isinstance(row, dict) and set(row) == expected, "registry row fields differ")
        row_id = _text(row["id"], "registry row id")
        _need(row_id not in seen, f"duplicate registry row {row_id}")
        seen.add(row_id)
        _text(row["domain"], f"registry row {row_id} domain")
        _relative(root, row["source"], f"registry row {row_id} source")
        for flag in ("fit_evidence", "held_out_evidence", "production_owner"):
            _need(type(row[flag]) is bool, f"registry row {row_id} {flag} must be boolean")
        _text(row["scope"], f"registry row {row_id} scope")
    return value, file_digest(path)


def _qualification(value: dict[str, Any], key: str, label: str) -> bool:
    qualification = value.get("qualification")
    _need(isinstance(qualification, dict), f"{label} qualification is missing")
    result = qualification.get(key)
    _need(type(result) is bool, f"{label} qualification {key} is missing")
    return result


def _false(value: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    for key in keys:
        _need(not _qualification(value, key, label),
              f"{label} promoted unresolved gate {key}")


def _validate_source(row_id: str, value: dict[str, Any]) -> dict[str, Any]:
    """Validate each source's boundary and return compact extracted facts."""
    schema = value.get("schema")
    if row_id == "cartilage_material":
        _need(schema == "HumanPack.tissue-calibration-candidate.v1", "cartilage source schema changed")
        _need(value.get("status") == "unqualified_finite_hold_candidate" and value.get("qualified") is False,
              "cartilage candidate was promoted")
        split, fit, native = value.get("split", {}), value.get("fit", {}), value.get("native_material", {})
        _need(split.get("frozen_before_fit") is True and split.get("population_holdout") is False,
              "cartilage split boundary changed")
        _need(fit.get("training", {}).get("observations") == 6 and
              fit.get("held_out", {}).get("observations") == 3,
              "cartilage fit/holdout counts changed")
        _need(value.get("anatomical_target", {}).get("production_force_owner_fraction") == 0 and
              native.get("compiled") is False and native.get("solver_validated") is False,
              "cartilage production owner boundary changed")
        return {"status": value["status"], "fit_points": 6, "held_out_points": 3,
                "production_owner": False,
                "blockers": ["stress-free reference and material parameters are not identifiable",
                             "native specimen boundary-value response is not qualified",
                             "whole-human material owner is absent"]}
    if row_id == "muscle_activation_recruitment":
        _need(schema == "HumanPack.activation-recruitment-candidate.v1" and
              value.get("status") == "partial", "activation candidate boundary changed")
        qualification = value.get("qualification", {})
        _need(qualification.get("nonmaximal_recruitment_candidate") is True and
              qualification.get("activation_calibration") is False and
              qualification.get("held_out_force_validation") is False,
              "activation candidate promoted measured calibration")
        return {"status": value["status"], "fit_points": None, "held_out_points": 0,
                "production_owner": False,
                "blockers": ["recruitment is an offline candidate, not measured activation",
                             "held-out force response is absent"]}
    if row_id == "subject_reference":
        _need(schema == "HumanPack.addbiomechanics-acquisition.v1" and
              value.get("status") == "source_acquired", "subject acquisition boundary changed")
        subject = value.get("subject", {})
        _need(subject == {"age_years": 43, "height_m": 1.78, "href": subject.get("href"),
                          "id": "Falisse2017:subject_1", "mass_kg": 65.5, "sex": "male",
                          "tags": ["healthy"]}, "subject identity changed")
        qualification = value.get("qualification", {})
        _need(qualification.get("measured_reference_tables_extracted") is True and
              qualification.get("prediction_comparison_complete") is False,
              "subject reference was promoted to prediction validation")
        return {"status": value["status"], "fit_points": 0, "held_out_points": 0,
                "production_owner": False,
                "blockers": ["reference tables have no Numi prediction comparison",
                             "standing and recovery references are absent"]}
    if row_id == "organ_tissue_mass":
        _need(schema == "HumanPack.tissue-mass-composition-candidate.v1" and
              value.get("status") == "partial", "organ tissue mass boundary changed")
        _false(value, ("material_calibration", "mechanical_mass_owner_assigned", "organ_mechanics",
                       "subject_calibration"), "organ tissue mass")
        _need(value.get("qualification", {}).get("candidate_mass_closes_against_density") is True,
              "organ tissue mass candidate lost its declared closure")
        return {"status": value["status"], "fit_points": 0, "held_out_points": 0,
                "production_owner": False,
                "blockers": ["density is an unresolved candidate", "organ mechanics and physical volume owner are absent"]}
    if row_id == "blood_tissue_transfer":
        _need(schema == "HumanPack.organ-blood-mass-transfer-candidate.v1" and
              value.get("status") == "partial", "blood transfer boundary changed")
        _false(value, ("anatomical_exchange_owner", "anatomical_vessel_lumen",
                       "mechanical_blood_mass_owner", "mechanical_tissue_mass_owner",
                       "material_density_calibrated", "subject_calibration"), "blood transfer")
        _need(value.get("qualification", {}).get("mass_and_volume_conservation") is True,
              "blood transfer conservation evidence changed")
        return {"status": value["status"], "fit_points": 0, "held_out_points": 0,
                "production_owner": False,
                "blockers": ["zeroth-moment transfer has no lumen/capillary owner",
                             "mechanical blood and tissue mass owners are absent"]}
    if row_id == "muscle_tissue_mass":
        _need(schema == "HumanPack.muscle-tissue-mass-candidate.v1" and
              value.get("status") == "partial", "muscle tissue mass boundary changed")
        _false(value, ("activation_calibration", "material_calibration", "mechanical_mass_owner",
                       "subject_calibration"), "muscle tissue mass")
        _need(value.get("counts", {}).get("unadmitted_surface_count") == 88,
              "muscle unadmitted surface count changed")
        return {"status": value["status"], "fit_points": 0, "held_out_points": 0,
                "production_owner": False,
                "blockers": ["88 muscle surfaces remain unadmitted",
                             "density and active-force calibration are unresolved"]}
    if row_id == "adipose_source_absence":
        _need(schema == "HumanPack.fat-source-absence-candidate.v1" and
              value.get("status") == "partial", "adipose absence boundary changed")
        qualification = value.get("qualification", {})
        _need(qualification.get("fat_source_absence_bound") is True and
              qualification.get("fat_geometry_present") is False and
              qualification.get("fat_mechanical_mass_owner") is False and
              qualification.get("subject_calibration") is False,
              "adipose absence was promoted")
        _need(value.get("counts", {}).get("fat_surface_count") == 0 and
              value.get("counts", {}).get("fat_mass_candidate_count") == 0,
              "adipose source counts changed")
        return {"status": value["status"], "fit_points": 0, "held_out_points": 0,
                "production_owner": False,
                "blockers": ["adipose geometry, mass and material source is absent"]}
    if row_id == "native_activation_diagnostic":
        _need(schema == "HumanPack.native-activation-sweep-audit.v1" and
              value.get("status") == "partial", "native activation diagnostic boundary changed")
        _false(value, ("activation_calibration", "internal_generalized_equilibrium",
                       "anatomical_supports_loading", "materials_resolved",
                       "subject_calibration"), "native activation diagnostic")
        _need(value.get("metrics", {}).get("internal_balanced") is False,
              "native activation diagnostic unexpectedly balanced")
        return {"status": value["status"], "fit_points": 0, "held_out_points": 0,
                "production_owner": False,
                "blockers": ["articulated generalized residual remains open",
                             "diagnostic has no measured activation target"]}
    raise HumanImportError(f"unknown calibration registry row {row_id}")


EXPECTED_ROWS = {
    "cartilage_material", "muscle_activation_recruitment", "subject_reference",
    "organ_tissue_mass", "blood_tissue_transfer", "muscle_tissue_mass",
    "adipose_source_absence", "native_activation_diagnostic",
}


def compile_registry(*, profile: Path = PROFILE, root: Path = ROOT) -> dict[str, Any]:
    profile = Path(profile)
    root = Path(root).resolve()
    value, profile_sha = _profile(root, profile)
    rows_by_id = {row["id"]: row for row in value["rows"]}
    _need(set(rows_by_id) == EXPECTED_ROWS, "registry rows do not cover the pinned domains")
    output_rows: list[dict[str, Any]] = []
    blockers: list[dict[str, str]] = []
    for row_id in sorted(rows_by_id):
        declared = rows_by_id[row_id]
        path, relative = _relative(root, declared["source"], f"registry row {row_id} source")
        source = _read_json(path, f"registry source {row_id}")
        facts = _validate_source(row_id, source)
        _need(declared["production_owner"] is False and facts["production_owner"] is False,
              f"registry row {row_id} attempted production ownership")
        output = {
            "id": row_id,
            "domain": declared["domain"],
            "source": {"path": relative, "schema": source.get("schema"), "sha256": file_digest(path)},
            "fit_evidence": declared["fit_evidence"],
            "held_out_evidence": declared["held_out_evidence"],
            "production_owner": False,
            "source_status": facts["status"],
            "scope": declared["scope"],
            "fit_points": facts["fit_points"],
            "held_out_points": facts["held_out_points"],
            "blockers": facts["blockers"],
        }
        output_rows.append(output)
        for index, reason in enumerate(facts["blockers"]):
            blockers.append({"id": f"{row_id}:{index}", "row": row_id, "reason": reason})
    result = {
        "schema": SCHEMA,
        "compiler": COMPILER,
        "compiler_sha256": file_digest(Path(__file__)),
        "status": "partial",
        "registry_id": value["id"],
        "inputs": {"profile": {"path": str(profile.relative_to(root)),
                                 "schema": value["schema"], "sha256": profile_sha}},
        "rows": output_rows,
        "blockers": blockers,
        "qualification": {
            "registry_complete": True,
            "fit_and_holdout_scopes_separated": True,
            "activation_calibration": False,
            "material_calibration": False,
            "subject_calibration": False,
            "anatomical_blood_mass_transfer": False,
            "production_owner_admission": False,
            "integrated_human_qualification": False,
        },
        "counts": {
            "domains": len(output_rows),
            "fit_rows": sum(row["fit_evidence"] for row in output_rows),
            "held_out_rows": sum(row["held_out_evidence"] for row in output_rows),
            "production_owner_rows": sum(row["production_owner"] for row in output_rows),
            "blockers": len(blockers),
        },
        "boundary": (
            "This registry joins source-bound calibration, measured reference, mass, blood/tissue "
            "and activation evidence for one adult male. Fit evidence and held-out evidence are "
            "reported independently; no row assigns a production owner. It does not qualify full "
            "generalized equilibrium, anatomical supports/loading, activation, blood mass transfer, "
            "materials, subject calibration, standing, recovery, walking, or integrated Human behavior."
        ),
    }
    result["report_sha256"] = digest(result)
    return result


def immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    path = Path(path)
    _need(not path.is_symlink(), "output is redirected")
    if path.exists():
        _need(path.read_bytes() == payload, "output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)


def run(arguments: argparse.Namespace) -> int:
    result = compile_registry(profile=arguments.profile, root=arguments.repository_root)
    output_sha = immutable_write(arguments.output, result)
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "output": str(Path(arguments.output).resolve()),
                      "sha256": output_sha, "domains": result["counts"]["domains"],
                      "blockers": result["counts"]["blockers"]}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (HumanImportError, OSError) as error:
        parser.exit(2, f"calibration-evidence-registry: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
