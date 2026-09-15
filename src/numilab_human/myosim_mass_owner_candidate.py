"""Bind the compiled MyoSim rigid-body mass ledger.

The pinned MyoSim/MuJoCo export is the authority for rigid-body mass and
inertia.  This candidate makes that authority explicit and auditable without
turning anatomical organ, blood, fat, skin, or soft-tissue candidate budgets
into dynamics.  It is source bookkeeping and a non-duplication boundary; it is
not a standing, walking, material, or subject-calibration qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical, read_json


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.myosim-rigid-body-mass-owner-candidate.v1"
SOURCE_SCHEMA = "HumanPack.myosim-rigid-body-mass-source.v1"
PROFILE_SCHEMA = "numi.human.myosim-rigid-body-mass-owner.v1"
PROFILE = ROOT / "config/myosim-mass-owner-candidate.v1.json"
SOURCE_MANIFEST = ROOT / "Docs/media/myosim-mass-owner-20260915/source-manifest-v1.json"
EXPECTED_MASS_KG = 97.13195176621338


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("MyoSim rigid-body mass owner: " + message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_number(value: Any, context: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(float(value)),
             f"{context} is not finite")
    return float(value)


def _vector(value: Any, count: int, context: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == count,
             f"{context} has the wrong length")
    return [_finite_number(item, context) for item in value]


def _read(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(),
             f"{label} is not a regular file")
    try:
        value = read_json(path)
    except (OSError, ValueError, UnicodeError) as error:
        raise HumanImportError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    return value, _sha256(path)


def _read_profile(path: Path) -> tuple[dict[str, Any], str]:
    profile, digest = _read(path, "candidate profile")
    _require(profile.get("schema") == PROFILE_SCHEMA,
             "candidate profile schema changed")
    _require(profile.get("id") == "myosim_compiled_rigid_body_mass_owner",
             "candidate profile id changed")
    _require(profile.get("manifest") == str(SOURCE_MANIFEST.relative_to(ROOT)),
             "candidate profile manifest changed")
    _require(profile.get("expected_body_count") == 103 and
             profile.get("expected_mass_bearing_body_count") == 96 and
             profile.get("expected_total_mass_kg") == EXPECTED_MASS_KG,
             "candidate profile expected mass ledger changed")
    _require(isinstance(profile.get("boundary"), str) and profile["boundary"].strip(),
             "candidate profile boundary is missing")
    return profile, digest


def _validate_source(source: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    _require(source.get("schema") == SOURCE_SCHEMA,
             "source manifest schema changed")
    source_identity = source.get("source")
    _require(isinstance(source_identity, dict), "source manifest identity is missing")
    for key in ("id", "revision", "archive", "archive_sha256", "export_schema",
                "export_sha256", "mujoco_version"):
        _require(isinstance(source_identity.get(key), str) and source_identity[key],
                 f"source manifest identity lacks {key}")
    _require(source_identity["id"] == "myosim_fullbody" and
             source_identity["revision"] ==
             "33c89c2bde282553dde3f526768eb3bdcfaa7649" and
             source_identity["archive_sha256"] ==
             "280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975" and
             source_identity["export_schema"] == "numi.human.myosim-mujoco-export.v1" and
             len(source_identity["export_sha256"]) == 64 and
             source_identity["mujoco_version"] == "3.12.0",
             "pinned MyoSim source identity changed")
    model = source.get("model")
    _require(isinstance(model, dict), "source model metadata is missing")
    _require(model == {
        "name": "myofullbody",
        "body_count_with_world": 104,
        "joint_count": 123,
        "nq": 129,
        "nv": 128,
        "nu": 416,
        "tendon_count": 424,
        "root_body": 1,
        "root_joint": 0,
        "timestep_seconds": 0.002,
    }, "compiled MyoSim model counts changed")
    bodies = source.get("bodies")
    _require(isinstance(bodies, list) and len(bodies) == profile["expected_body_count"],
             "compiled body table is incomplete")
    ids: list[int] = []
    names: list[str] = []
    total = 0.0
    positive = 0
    for row in bodies:
        _require(isinstance(row, dict), "compiled body row is not an object")
        _require(set(row) == {
            "id", "name", "parent", "mass_kg", "inertia_kg_m2",
            "inertial_position_body_m", "inertial_quaternion_body_xyzw",
        }, "compiled body row fields changed")
        body_id = row["id"]
        parent = row["parent"]
        _require(type(body_id) is int and body_id > 0 and body_id not in ids,
                 "compiled body ids repeat or are invalid")
        _require(type(parent) is int and 0 <= parent < body_id,
                 f"body {body_id} parent is not an earlier body")
        _require(isinstance(row["name"], str) and row["name"] and row["name"] not in names,
                 "compiled body names repeat or are invalid")
        mass = _finite_number(row["mass_kg"], f"body {body_id} mass")
        _require(mass >= 0.0, f"body {body_id} mass is negative")
        inertia = _vector(row["inertia_kg_m2"], 3, f"body {body_id} inertia")
        _require(all(value >= 0.0 for value in inertia),
                 f"body {body_id} inertia is negative")
        _vector(row["inertial_position_body_m"], 3, f"body {body_id} inertial position")
        quaternion = _vector(row["inertial_quaternion_body_xyzw"], 4,
                             f"body {body_id} inertial quaternion")
        _require(math.isclose(math.sqrt(sum(value * value for value in quaternion)), 1.0,
                              rel_tol=0.0, abs_tol=1.0e-9),
                 f"body {body_id} inertial quaternion is not normalized")
        if mass > 0.0:
            positive += 1
            _require(any(value > 0.0 for value in inertia),
                     f"body {body_id} has mass but no inertia")
        ids.append(body_id)
        names.append(row["name"])
        total += mass
    _require(ids == list(range(1, profile["expected_body_count"] + 1)),
             "compiled body ids are not contiguous")
    _require(ids[0] == 1 and source["bodies"][0]["parent"] == 0,
             "compiled body root changed")
    _require(positive == profile["expected_mass_bearing_body_count"],
             "compiled mass-bearing body count changed")
    _require(math.isclose(total, profile["expected_total_mass_kg"], rel_tol=0.0, abs_tol=1.0e-12),
             "compiled rigid-body total mass changed")
    return {
        "source": source_identity,
        "model": model,
        "bodies": bodies,
        "total_mass_kg": total,
        "mass_bearing_body_count": positive,
        "zero_mass_body_count": len(bodies) - positive,
    }


def compile_candidate(
    *,
    profile: Path = PROFILE,
    source_manifest: Path = SOURCE_MANIFEST,
) -> dict[str, Any]:
    profile_document, _ = _read_profile(Path(profile))
    source, source_sha256 = _read(Path(source_manifest), "source manifest")
    validated = _validate_source(source, profile_document)
    body_rows = []
    owner_ids = []
    for body in validated["bodies"]:
        body_id = body["id"]
        owner_id = f"myosim-rigid-body:{body_id}"
        owner_ids.append(owner_id)
        body_rows.append({
            **body,
            "owner_id": owner_id,
            "rigid_body_mass_owner": True,
            "physical_volume_owner": None,
            "anatomical_tissue_owner": None,
            "material_owner": None,
            "subject_calibration": False,
        })
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.myosim-mass-owner-candidate.1",
        "status": "partial",
        "subject": "one adult male pinned MyoSim source package",
        "source": {
            "manifest": str(Path(source_manifest).relative_to(ROOT))
            if Path(source_manifest).is_relative_to(ROOT) else str(source_manifest),
            "manifest_sha256": source_sha256,
            "export_sha256": validated["source"]["export_sha256"],
            "archive_sha256": validated["source"]["archive_sha256"],
            "myosim_revision": validated["source"]["revision"],
            "mujoco_version": validated["source"]["mujoco_version"],
            "model": validated["model"],
        },
        "rigid_body_mass": {
            "body_count": len(body_rows),
            "mass_bearing_body_count": validated["mass_bearing_body_count"],
            "zero_mass_body_count": validated["zero_mass_body_count"],
            "total_mass_kg": validated["total_mass_kg"],
            "owner_ids_sha256": hashlib.sha256(canonical(sorted(owner_ids))).hexdigest(),
            "bodies": body_rows,
        },
        "qualification": {
            "source_compiled_body_mass_bound": True,
            "source_compiled_body_inertia_bound": True,
            "source_body_tree_identity_bound": True,
            "single_rigid_body_mass_owner_per_body": True,
            "source_rigid_body_mass_nonduplication_checked": True,
            "whole_body_dynamic_mass_matrix_owner": False,
            "anatomical_organ_mass_owner": False,
            "mechanical_blood_mass_owner": False,
            "skeletal_muscle_tissue_mass_owner": False,
            "fat_volume_and_mass_owner": False,
            "skin_volume_and_mass_owner": False,
            "soft_tissue_material_calibration": False,
            "subject_calibration": False,
            "standing": False,
            "recovery": False,
            "walking": False,
        },
        "boundary": (
            "This record is the source owner for the 103 compiled MyoSim rigid "
            "bodies, their 96 mass-bearing rows, masses, inertias, and parent "
            "tree. It is the rigid-body mass ledger used to prevent candidate "
            "organ, blood, fat, skin, tendon, or skeletal-muscle tissue budgets "
            "from being double-counted. It does not create anatomical organ or "
            "vessel volume, tissue mass, compliant soft-tissue mechanics, a "
            "whole-body dynamic mass matrix qualification, calibrated materials, "
            "subject scaling, standing, recovery, or walking."
        ),
    }


def immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    _require(not path.is_symlink(), "output is redirected")
    if path.exists():
        _require(path.read_bytes() == payload,
                 "output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    result = compile_candidate(profile=arguments.profile,
                               source_manifest=arguments.source_manifest)
    output = arguments.output.resolve()
    digest = immutable_write(output, result)
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(output),
        "sha256": digest,
        "status": result["status"],
        "body_count": result["rigid_body_mass"]["body_count"],
        "total_mass_kg": result["rigid_body_mass"]["total_mass_kg"],
    }, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"MyoSim rigid-body mass owner: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
