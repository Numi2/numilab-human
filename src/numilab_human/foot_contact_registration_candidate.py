"""Compile a source-bound anatomical foot contact registration candidate.

This candidate joins the exact BodyParts3D calcaneus/toe meshes with the
provisional lower-limb source-to-MyoSim transforms and the existing 18-witness
support profile.  It is the hand-off needed before native anatomical contact:
source identities, frames and support witness ownership are checked together,
but the source transforms are not silently promoted to calibrated colliders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .model import (
    ImportError as HumanImportError,
    bodyparts_foot_collider_preflight,
    parse_bodyparts3d,
    parse_opensim,
    sha256,
)
from .physiology import canonical


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config/foot-contact-registration-candidate.v1.json"
SCHEMA = "HumanPack.foot-contact-registration-candidate.v1"
FOOT_BODIES = ("calcn_r", "toes_r", "calcn_l", "toes_l")


class FootRegistrationError(HumanImportError):
    """A source anatomical foot registration cannot be admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FootRegistrationError("foot contact registration: " + message)


def _read_json(path: Path, label: str, *, canonical_required: bool = False) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise FootRegistrationError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    if canonical_required:
        _require(raw == canonical(value) + b"\n", f"{label} is not canonical")
    return value, hashlib.sha256(raw).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _profile(path: Path) -> tuple[dict[str, Any], str]:
    profile, profile_sha = _read_json(path, "foot profile", canonical_required=True)
    required = {
        "schema", "id", "registration_receipt", "multi_pose_receipt",
        "support_profile", "support_receipt", "expected_foot_body_count",
        "expected_source_mesh_count", "expected_unique_source_member_count",
        "expected_registered_source_mesh_count",
        "expected_contact_count", "expected_active_contact_count",
        "expected_pose_count", "expected_source_member_count", "boundary",
    }
    _require(set(profile) == required, "foot profile fields differ")
    _require(profile["schema"] == "numi.human.foot-contact-registration-candidate.v1",
             "unsupported foot profile schema")
    _require(profile["id"] == "bodyparts3d_foot_support_registration",
             "unsupported foot profile")
    for key in ("registration_receipt", "multi_pose_receipt", "support_profile", "support_receipt"):
        value = profile[key]
        _require(isinstance(value, str) and value.strip() and not Path(value).is_absolute()
                 and ".." not in Path(value).parts and "\\" not in value,
                 f"{key} path is unsafe")
    expected = {
        "expected_foot_body_count": 4,
        "expected_source_mesh_count": 60,
        "expected_unique_source_member_count": 30,
        "expected_contact_count": 18,
        "expected_active_contact_count": 6,
        "expected_pose_count": 7,
        "expected_source_member_count": 60,
        "expected_registered_source_mesh_count": 30,
    }
    for key, value in expected.items():
        _require(profile[key] == value, f"{key} differs from the source contract")
    _require(isinstance(profile["boundary"], str) and profile["boundary"].strip(),
             "foot profile boundary is missing")
    return profile, profile_sha


def _finite(value: Any, label: str) -> float:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError as error:
            raise FootRegistrationError(f"foot contact registration: {label} is not finite") from error
    _require(type(value) in (int, float) and math.isfinite(float(value)), f"{label} is not finite")
    return float(value)


def _matrix(value: Any, body: str) -> tuple[list[list[float]], float, float]:
    _require(isinstance(value, list) and len(value) == 4
             and all(isinstance(row, list) and len(row) == 4 for row in value),
             f"{body} registration is not a 4x4 matrix")
    matrix = [[_finite(cell, f"{body} registration") for cell in row] for row in value]
    _require(matrix[3] == [0.0, 0.0, 0.0, 1.0], f"{body} registration is not affine")
    rows = [row[:3] for row in matrix[:3]]
    scale = math.sqrt(sum(cell * cell for cell in rows[0]))
    _require(scale > 0.0, f"{body} registration has zero scale")
    for row in rows:
        _require(abs(math.sqrt(sum(cell * cell for cell in row)) - scale) <= 1.0e-12,
                 f"{body} registration is anisotropic")
    normalized = [[cell / scale for cell in row] for row in rows]
    for row in normalized:
        _require(abs(sum(cell * cell for cell in row) - 1.0) <= 1.0e-10,
                 f"{body} registration rotation is not unit length")
    for first in range(3):
        for second in range(first + 1, 3):
            _require(abs(sum(normalized[first][axis] * normalized[second][axis]
                             for axis in range(3))) <= 1.0e-10,
                     f"{body} registration rotation is not orthogonal")
    determinant = (
        normalized[0][0] * (normalized[1][1] * normalized[2][2] - normalized[1][2] * normalized[2][1])
        - normalized[0][1] * (normalized[1][0] * normalized[2][2] - normalized[1][2] * normalized[2][0])
        + normalized[0][2] * (normalized[1][0] * normalized[2][1] - normalized[1][1] * normalized[2][0])
    )
    _require(abs(determinant - 1.0) <= 1.0e-10,
             f"{body} registration rotation is not proper")
    return matrix, scale, determinant


def _support_summary(profile: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    _require(profile.get("schema") == "numi.human.source-support-stance.v2",
             "unsupported source support profile")
    _require(profile.get("contact_count") == 18, "support profile contact count drifted")
    primitives = profile.get("primitive_contacts")
    active = profile.get("active_contacts")
    _require(isinstance(primitives, list) and len(primitives) == 18,
             "support primitive inventory is incomplete")
    _require(isinstance(active, list) and len(active) == 6,
             "support active-contact inventory is incomplete")
    witnesses = [row.get("witness_index") for row in primitives if isinstance(row, dict)]
    _require(len(witnesses) == 18 and set(witnesses) == set(range(18)),
             "support witness indices are not a complete permutation")
    active_witnesses = [row.get("witness_index") for row in active if isinstance(row, dict)]
    _require(len(active_witnesses) == 6 and len(set(active_witnesses)) == 6,
             "active support witnesses are duplicated")
    _require(receipt.get("schema") == "numi.human.curved-support-evidence.v1",
             "unsupported curved-support receipt")
    qualification = receipt.get("qualification")
    _require(isinstance(qualification, dict), "curved-support receipt has no qualification")
    for key in ("source_primitive_geometry_at_initialization", "static_gravity_wrench",
                "native_surface_query_parity", "matter_surface_kkt_and_rollback",
                "native_6_4ms_replay"):
        _require(qualification.get(key) is True, f"curved-support receipt lacks {key}")
    for key in ("dynamic_contact", "v5_anatomical_acceptance", "standing", "walking", "calibration"):
        _require(qualification.get(key) is False, f"curved-support receipt boundary changed for {key}")
    static = receipt.get("static_support")
    _require(isinstance(static, dict), "curved-support receipt has no static support summary")
    total = _finite(static.get("total_support_force_n"), "total support force")
    weight = _finite(static.get("expected_weight_n"), "expected weight")
    residual = _finite(static.get("max_root_force_residual"), "root force residual")
    _require(weight > 0.0 and abs(total - weight) / weight <= 1.0e-5,
             "source support wrench is not weight balanced")
    _require(0.0 <= residual <= 1.0e-3, "source support root residual is not bounded")
    return {
        "contact_count": 18,
        "active_contact_count": 6,
        "active_witnesses": sorted(active_witnesses),
        "active_source_names": sorted(str(row.get("source_name")) for row in active),
        "total_support_force_n": total,
        "expected_weight_n": weight,
        "root_force_residual_n": residual,
        "dynamic_contact_qualified": False,
        "contact_calibration_qualified": False,
    }


def compile_candidate(*, sources: Path, profile: Path = PROFILE) -> dict[str, Any]:
    profile_doc, profile_sha = _profile(Path(profile))
    sources = sources.resolve()
    registration_path = (ROOT / profile_doc["registration_receipt"]).resolve()
    multipose_path = (ROOT / profile_doc["multi_pose_receipt"]).resolve()
    support_path = (ROOT / profile_doc["support_profile"]).resolve()
    support_receipt_path = (ROOT / profile_doc["support_receipt"]).resolve()
    for path, label in ((registration_path, "registration receipt"),
                        (multipose_path, "multi-pose receipt"),
                        (support_path, "support profile"),
                        (support_receipt_path, "support receipt")):
        _require(path.is_relative_to(ROOT), f"{label} resolves outside repository")
    registration, registration_sha = _read_json(registration_path, "registration receipt")
    multipose, multipose_sha = _read_json(multipose_path, "multi-pose receipt")
    support_profile, support_profile_sha = _read_json(support_path, "support profile")
    support_receipt, support_receipt_sha = _read_json(support_receipt_path, "support receipt")
    _require(registration.get("schema") == "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2",
             "unsupported foot registration receipt")
    _require(registration.get("status") == "provisional_visual_registration_not_admitted_to_collision_or_physics",
             "foot registration receipt boundary changed")
    coordinate = registration.get("coordinate_system")
    _require(isinstance(coordinate, dict), "foot registration has no coordinate system")
    _require(coordinate.get("source") == "BodyParts3D OBJ millimetres",
             "foot registration source units changed")
    global_scale = _finite(coordinate.get("uniform_scale_after_mm_to_m"), "global source scale")
    _require(global_scale > 0.0, "global source scale is not positive")
    global_scale_m_per_mm = global_scale * 1.0e-3
    anatomy = parse_bodyparts3d(sources, ROOT / "config/anatomy-classification.v1.json")
    lower = parse_opensim(sources / "RajagopalLaiUhlrich2023.osim", "rajagopal_lai_uhlrich_2023")
    preflight = bodyparts_foot_collider_preflight(sources, anatomy, lower)
    _require(preflight.get("source_mesh_count") == profile_doc["expected_source_mesh_count"],
             "source foot mesh count drifted")
    meshes_by_body: dict[str, list[dict[str, Any]]] = {body: [] for body in FOOT_BODIES}
    for foot in preflight["per_foot"]:
        body = foot["opensim_body"]
        _require(body in meshes_by_body, f"unexpected source foot body {body}")
        # The preflight intentionally carries both is_a and part_of archive
        # candidates.  The reviewed registration receipt binds the is_a
        # source surfaces; keep the part_of duplicates as source-local context.
        meshes_by_body[body] = [
            mesh for mesh in foot["source_meshes"]
            if mesh["source"].get("hierarchy") == "is_a"
        ]
    _require(all(meshes_by_body.values()), "one or more source foot bodies have no meshes")
    unique_source_members = {
        mesh["source"]["member_id"] for meshes in meshes_by_body.values() for mesh in meshes
    }
    _require(len(unique_source_members) == profile_doc["expected_unique_source_member_count"],
             "unique source foot member count drifted")
    anchors = registration.get("anchors")
    _require(isinstance(anchors, list), "foot registration has no anchors")
    anchor_map: dict[tuple[str, str], dict[str, Any]] = {}
    for anchor in anchors:
        _require(isinstance(anchor, dict), "foot registration anchor is malformed")
        target = anchor.get("target")
        source = anchor.get("source")
        reg = anchor.get("registration")
        _require(isinstance(target, dict) and isinstance(source, dict) and isinstance(reg, dict),
                 "foot registration anchor is incomplete")
        body = target.get("name")
        member_id = source.get("member_id")
        if body not in FOOT_BODIES:
            continue
        _require(isinstance(member_id, str) and member_id,
                 f"{body} anchor has no source member")
        key = (body, member_id)
        _require(key not in anchor_map, f"duplicate {body}/{member_id} registration anchor")
        matrix, scale, determinant = _matrix(reg.get("source_obj_mm_to_core_inertial_body_m"), body)
        _require(source.get("archive_sha256") == "40665852c49f218326590e204db91064a1ecfc3c6f8cbd7bbbcaac62c7cd409e",
                 f"{body}/{member_id} source archive hash drifted")
        anchor_map[key] = {
            "core_body_index": target.get("core_body_index"),
            "member_id": member_id,
            "member_sha256": source.get("member_sha256"),
            "name": source.get("name"),
            "matrix_source_obj_mm_to_core_inertial_body_m": matrix,
            "scale_m_per_source_mm": scale,
            "rotation_determinant": determinant,
            "status": reg.get("status"),
            "held_out_metrics": reg.get("lower_limb_source_mesh_registration", {}).get("held_out_metrics", {}),
        }
    rows: list[dict[str, Any]] = []
    for body in FOOT_BODIES:
        for mesh in meshes_by_body[body]:
            source = mesh["source"]
            key = (body, source["member_id"])
            _require(key in anchor_map, f"no reviewed source-to-body transform for {body}/{source['member_id']}")
            anchor = anchor_map[key]
            _require(anchor["member_sha256"] == source["member_sha256"],
                     f"source member hash drifted for {body}/{source['member_id']}")
            rows.append({
                "opensim_body": body,
                "source_member_id": source["member_id"],
                "source_member_sha256": source["member_sha256"],
                "source_archive_sha256": source["archive_sha256"],
                "triangle_count": mesh["geometry"]["triangle_count"],
                "vertex_count": mesh["geometry"]["vertex_count"],
                "bounds_mm": mesh["geometry"]["bounds_mm"],
                "core_body_index": anchor["core_body_index"],
                "source_obj_mm_to_core_inertial_body_m": anchor["matrix_source_obj_mm_to_core_inertial_body_m"],
                "registration_status": anchor["status"],
            })
    _require(len(rows) == profile_doc["expected_registered_source_mesh_count"],
             "registered source foot mesh count drifted")
    _require(multipose.get("schema") == "numi.human.bodyparts3d-myosim-lower-limb-multi-pose-audit.v1",
             "unsupported lower-limb multi-pose receipt")
    _require(str(multipose.get("status", "")).startswith("passed_"),
             "lower-limb multi-pose receipt is not passing")
    _require(multipose.get("pose_count") == profile_doc["expected_pose_count"]
             and multipose.get("source_member_count") == profile_doc["expected_source_member_count"],
             "lower-limb multi-pose source counts drifted")
    support = _support_summary(support_profile, support_receipt)
    scales = [row["source_obj_mm_to_core_inertial_body_m"] for row in rows]
    row_scales = [math.sqrt(sum(value * value for value in matrix[0][:3])) for matrix in scales]
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.foot-contact-registration-candidate.1",
        "status": "partial",
        "source": {
            "profile": _relative(Path(profile)),
            "profile_sha256": profile_sha,
            "sources_directory": str(sources),
            "bodyparts_archive_sha256": sha256(sources / "isa_BP3D_4.0_obj_99.zip"),
            "registration_receipt": _relative(registration_path),
            "registration_receipt_sha256": registration_sha,
            "multi_pose_receipt": _relative(multipose_path),
            "multi_pose_receipt_sha256": multipose_sha,
            "support_profile": _relative(support_path),
            "support_profile_sha256": support_profile_sha,
            "support_receipt": _relative(support_receipt_path),
            "support_receipt_sha256": support_receipt_sha,
            "opensim_model_sha256": lower.get("source_sha256"),
        },
        "counts": {
            "foot_body_count": len(FOOT_BODIES),
            "source_mesh_count": preflight["source_mesh_count"],
            "registered_source_mesh_count": len(rows),
            "unique_source_member_count": len(unique_source_members),
            "registered_source_member_count": len({row["source_member_id"] for row in rows}),
            "support_witness_count": support["contact_count"],
            "active_support_witness_count": support["active_contact_count"],
        },
        "registration": {
            "global_scale_m_per_source_mm": global_scale_m_per_mm,
            "minimum_local_scale_m_per_source_mm": min(row_scales),
            "maximum_local_scale_m_per_source_mm": max(row_scales),
            "maximum_scale_deviation_m_per_source_mm": max(abs(value - global_scale_m_per_mm) for value in row_scales),
            "multi_pose_count": multipose["pose_count"],
            "continuity_evaluation_count": multipose.get("continuity_evaluation_count"),
            "bilateral_gap_parity_maximum_m": multipose.get("bilateral_gap_parity_maximum_m"),
            "source_members": rows,
        },
        "support": support,
        "qualification": {
            "source_foot_geometry_registered": True,
            "source_to_body_rest_transform_bound": True,
            "lower_limb_multi_pose_continuity": True,
            "support_witness_identity_bound": True,
            "anatomical_collider_admitted": False,
            "collision_exclusions_admitted": False,
            "contact_material_calibration": False,
            "anatomical_supports_loading": False,
            "dynamic_contact": False,
            "loaded_whole_body_equilibrium": False,
            "sustained_standing": False,
            "recovery": False,
            "walking": False,
        },
        "boundary": profile_doc["boundary"],
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    candidate = compile_candidate(sources=arguments.sources, profile=arguments.profile)
    output = arguments.output.resolve()
    encoded = json.dumps(candidate, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        _require(output.read_text(encoding="utf-8") == encoded,
                 "candidate receipt is immutable and differs")
    else:
        output.write_text(encoded, encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "status": candidate["status"],
        "source_mesh_count": candidate["counts"]["source_mesh_count"],
        "registered_source_member_count": candidate["counts"]["registered_source_member_count"],
        "active_support_witness_count": candidate["counts"]["active_support_witness_count"],
        "output": str(output),
    }, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except FootRegistrationError as error:
        parser.exit(2, f"foot-contact-registration: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
