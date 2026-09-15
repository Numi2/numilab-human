"""Compile conservative body-frame foot contact proxy candidates.

The reviewed foot registration receipt already binds BodyParts3D mesh
identities to the four MyoSim foot bodies.  This compiler carries the exact
source-coordinate AABB enclosure through those reviewed transforms so the
native contact owner has a deterministic geometry hand-off.  It deliberately
does not admit the boxes as colliders: collision exclusions, ground
registration, friction/compliance, swept-motion bounds and loaded dynamics
still require independent evidence.
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
PROFILE = ROOT / "config/foot-contact-proxy-candidate.v1.json"
REGISTRATION = ROOT / "Docs/media/foot-contact-registration-candidate-20260914/receipt-v1.json"
SCHEMA = "HumanPack.foot-contact-proxy-candidate.v1"
FOOT_BODIES = ("calcn_r", "toes_r", "calcn_l", "toes_l")
ARCHIVE_SHA256 = "40665852c49f218326590e204db91064a1ecfc3c6f8cbd7bbbcaac62c7cd409e"


class FootProxyError(HumanImportError):
    """A registered foot proxy candidate cannot be compiled."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FootProxyError("foot contact proxy: " + message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _read(path: Path, label: str, *, canonical_required: bool = False) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = read_json(path)
    except (OSError, ValueError, TypeError) as error:
        raise FootProxyError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    if canonical_required:
        _require(raw == canonical(value) + b"\n", f"{label} is not canonical")
    return value, hashlib.sha256(raw).hexdigest()


def _profile(path: Path) -> tuple[dict[str, Any], str]:
    profile, digest = _read(path, "proxy profile", canonical_required=True)
    required = {
        "schema", "id", "registration_receipt", "expected_body_count",
        "expected_registered_member_count", "expected_proxy_count",
        "expected_pose_count", "boundary",
    }
    _require(set(profile) == required, "proxy profile fields differ")
    _require(profile["schema"] == "numi.human.foot-contact-proxy-candidate.v1",
             "unsupported proxy profile schema")
    _require(profile["id"] == "bodyparts3d_registered_foot_proxy_enclosure",
             "unsupported proxy profile")
    value = profile["registration_receipt"]
    _require(isinstance(value, str) and value.strip() and not Path(value).is_absolute()
             and ".." not in Path(value).parts and "\\" not in value,
             "registration receipt path is unsafe")
    expected = {
        "expected_body_count": 4,
        "expected_registered_member_count": 30,
        "expected_proxy_count": 30,
        "expected_pose_count": 7,
    }
    for key, expected_value in expected.items():
        _require(profile[key] == expected_value, f"{key} differs from the source contract")
    _require(isinstance(profile["boundary"], str) and profile["boundary"].strip(),
             "proxy boundary is missing")
    return profile, digest


def _finite(value: Any, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(float(value)),
             f"{label} is not finite")
    return float(value)


def _matrix(value: Any, body: str) -> list[list[float]]:
    _require(isinstance(value, list) and len(value) == 4
             and all(isinstance(row, list) and len(row) == 4 for row in value),
             f"{body} registration is not a 4x4 matrix")
    matrix = [[_finite(cell, f"{body} registration") for cell in row] for row in value]
    _require(matrix[3] == [0.0, 0.0, 0.0, 1.0], f"{body} registration is not affine")
    rows = [row[:3] for row in matrix[:3]]
    scale = math.sqrt(sum(cell * cell for cell in rows[0]))
    _require(scale > 0.0, f"{body} registration has zero scale")
    for row in rows:
        _require(abs(math.sqrt(sum(cell * cell for cell in row)) - scale) <= 1.0e-10,
                 f"{body} registration is anisotropic")
    normalized = [[cell / scale for cell in row] for row in rows]
    for row in normalized:
        _require(abs(sum(cell * cell for cell in row) - 1.0) <= 1.0e-9,
                 f"{body} registration rotation is not unit length")
    for first in range(3):
        for second in range(first + 1, 3):
            _require(abs(sum(normalized[first][axis] * normalized[second][axis]
                             for axis in range(3))) <= 1.0e-9,
                     f"{body} registration rotation is not orthogonal")
    determinant = (
        normalized[0][0] * (normalized[1][1] * normalized[2][2] - normalized[1][2] * normalized[2][1])
        - normalized[0][1] * (normalized[1][0] * normalized[2][2] - normalized[1][2] * normalized[2][0])
        + normalized[0][2] * (normalized[1][0] * normalized[2][1] - normalized[1][1] * normalized[2][0])
    )
    _require(abs(determinant - 1.0) <= 1.0e-9,
             f"{body} registration rotation is not proper")
    return matrix


def _corners(bounds: dict[str, Any]) -> list[tuple[float, float, float]]:
    minimum = bounds.get("minimum")
    maximum = bounds.get("maximum")
    _require(isinstance(minimum, list) and len(minimum) == 3
             and isinstance(maximum, list) and len(maximum) == 3,
             "source bounds are not three-dimensional")
    lower = [_finite(value, "source lower bound") for value in minimum]
    upper = [_finite(value, "source upper bound") for value in maximum]
    _require(all(lo < hi for lo, hi in zip(lower, upper)), "source bounds are degenerate")
    return [
        (x, y, z)
        for x in (lower[0], upper[0])
        for y in (lower[1], upper[1])
        for z in (lower[2], upper[2])
    ]


def _transform(matrix: list[list[float]], point: tuple[float, float, float]) -> list[float]:
    return [
        sum(matrix[row][column] * point[column] for column in range(3)) + matrix[row][3]
        for row in range(3)
    ]


def _bounds(points: list[list[float]]) -> dict[str, list[float]]:
    return {
        "minimum": [min(point[axis] for point in points) for axis in range(3)],
        "maximum": [max(point[axis] for point in points) for axis in range(3)],
    }


def _volume(bounds: dict[str, list[float]]) -> float:
    return math.prod(hi - lo for lo, hi in zip(bounds["minimum"], bounds["maximum"]))


def compile_candidate(*, registration: Path = REGISTRATION,
                      profile: Path = PROFILE) -> dict[str, Any]:
    profile_doc, profile_sha = _profile(Path(profile))
    registration_path = Path(registration)
    registration_doc, registration_sha = _read(registration_path, "foot registration receipt")
    _require(registration_doc.get("schema") == "HumanPack.foot-contact-registration-candidate.v1",
             "unsupported foot registration receipt")
    _require(registration_doc.get("status") == "partial", "foot registration status changed")
    source = registration_doc.get("source")
    registration = registration_doc.get("registration")
    counts = registration_doc.get("counts")
    qualification = registration_doc.get("qualification")
    _require(isinstance(source, dict) and isinstance(registration, dict)
             and isinstance(counts, dict) and isinstance(qualification, dict),
             "foot registration receipt is incomplete")
    _require(source.get("bodyparts_archive_sha256") == ARCHIVE_SHA256,
             "BodyParts3D archive hash changed")
    _require(counts.get("foot_body_count") == len(FOOT_BODIES)
             and counts.get("registered_source_mesh_count") == 30
             and counts.get("registered_source_member_count") == 30,
             "foot registration counts changed")
    _require(counts.get("support_witness_count") == 18
             and counts.get("active_support_witness_count") == 6,
             "support witness counts changed")
    _require(qualification.get("source_foot_geometry_registered") is True
             and qualification.get("source_to_body_rest_transform_bound") is True
             and qualification.get("lower_limb_multi_pose_continuity") is True
             and qualification.get("anatomical_collider_admitted") is False
             and qualification.get("dynamic_contact") is False,
             "foot registration qualification boundary changed")

    rows = registration.get("source_members")
    _require(isinstance(rows, list) and len(rows) == profile_doc["expected_proxy_count"],
             "registered source-member rows are incomplete")
    proxies: list[dict[str, Any]] = []
    seen_members: set[str] = set()
    body_rows: dict[str, list[dict[str, Any]]] = {body: [] for body in FOOT_BODIES}
    for row in rows:
        _require(isinstance(row, dict), "registered source-member row is malformed")
        body = row.get("opensim_body")
        member = row.get("source_member_id")
        member_sha = row.get("source_member_sha256")
        archive_sha = row.get("source_archive_sha256")
        _require(body in body_rows, f"unexpected foot body {body!r}")
        _require(isinstance(member, str) and member and member not in seen_members,
                 "source member identity is invalid or duplicated")
        _require(isinstance(member_sha, str) and len(member_sha) == 64,
                 f"source member hash is invalid for {member}")
        _require(archive_sha == ARCHIVE_SHA256, f"source archive hash changed for {member}")
        bounds_mm = row.get("bounds_mm")
        matrix = _matrix(row.get("source_obj_mm_to_core_inertial_body_m"), str(body))
        source_corners = _corners(bounds_mm)
        transformed = [_transform(matrix, point) for point in source_corners]
        body_bounds = _bounds(transformed)
        half_extents = [
            (hi - lo) * 0.5
            for lo, hi in zip(body_bounds["minimum"], body_bounds["maximum"])
        ]
        center = [
            (lo + hi) * 0.5
            for lo, hi in zip(body_bounds["minimum"], body_bounds["maximum"])
        ]
        proxy = {
            "opensim_body": body,
            "core_body_index": row.get("core_body_index"),
            "source_member_id": member,
            "source_member_sha256": member_sha,
            "source_archive_sha256": archive_sha,
            "source_bounds_mm": bounds_mm,
            "body_frame_bounds_m": body_bounds,
            "body_frame_aabb_proxy": {
                "shape": "axis_aligned_box",
                "center_m": center,
                "half_extents_m": half_extents,
                "encloses_transformed_source_aabb": True,
                "status": "candidate_not_admitted_as_collider",
            },
            "transform": matrix,
            "registration_status": row.get("registration_status"),
        }
        seen_members.add(member)
        proxies.append(proxy)
        body_rows[body].append(proxy)
    _require(len(seen_members) == profile_doc["expected_registered_member_count"],
             "registered source-member identity count changed")
    _require(all(body_rows.values()), "one or more foot bodies have no proxy rows")

    body_summaries: list[dict[str, Any]] = []
    for body in FOOT_BODIES:
        members = body_rows[body]
        minimum = [min(row["body_frame_bounds_m"]["minimum"][axis] for row in members)
                   for axis in range(3)]
        maximum = [max(row["body_frame_bounds_m"]["maximum"][axis] for row in members)
                   for axis in range(3)]
        union = {"minimum": minimum, "maximum": maximum}
        body_summaries.append({
            "opensim_body": body,
            "core_body_index": members[0]["core_body_index"],
            "registered_member_count": len(members),
            "union_body_frame_bounds_m": union,
            "union_aabb_volume_m3": _volume(union),
            "support_patch": {
                "status": "requires_ground_frame_and_contact_data",
                "source_member_ids": [row["source_member_id"] for row in members],
                "normal_world": None,
                "loaded_force_n": None,
            },
        })

    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.foot-contact-proxy-candidate.1",
        "status": "partial",
        "source": {
            "profile": _relative(Path(profile)),
            "profile_sha256": profile_sha,
            "registration_receipt": _relative(registration_path),
            "registration_receipt_sha256": registration_sha,
            "bodyparts_archive_sha256": ARCHIVE_SHA256,
            "subject": "one adult male source package",
        },
        "counts": {
            "foot_body_count": len(FOOT_BODIES),
            "registered_source_member_count": len(seen_members),
            "proxy_count": len(proxies),
            "support_witness_count": counts["support_witness_count"],
            "active_support_witness_count": counts["active_support_witness_count"],
            "multi_pose_count": profile_doc["expected_pose_count"],
        },
        "body_summaries": body_summaries,
        "proxies": proxies,
        "contact_prerequisites": {
            "ground_frame_registration": "required",
            "collision_exclusions": "required",
            "friction_compliance_restitution": "required",
            "swept_motion_bounds": "required",
            "support_jacobian_control": "required",
            "loaded_whole_body_equilibrium": "required",
        },
        "qualification": {
            "source_registered_geometry_bound": True,
            "conservative_body_frame_proxy_bounds": True,
            "source_triangle_enclosure_preserved": True,
            "support_witness_identity_carried": True,
            "anatomical_collider_admitted": False,
            "collision_exclusions_admitted": False,
            "contact_material_calibration": False,
            "swept_motion_bounds": False,
            "anatomical_supports_loading": False,
            "dynamic_contact": False,
            "loaded_whole_body_equilibrium": False,
            "standing": False,
            "recovery": False,
            "walking": False,
        },
        "boundary": profile_doc["boundary"],
    }


def _immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    _require(not path.is_symlink(), "output is redirected")
    if path.exists():
        _require(path.read_bytes() == payload, "output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def run(arguments: argparse.Namespace) -> int:
    candidate = compile_candidate(registration=arguments.registration, profile=arguments.profile)
    output = arguments.output.resolve()
    digest = _immutable_write(output, candidate)
    print(json.dumps({
        "schema": SCHEMA,
        "status": candidate["status"],
        "proxy_count": candidate["counts"]["proxy_count"],
        "foot_body_count": candidate["counts"]["foot_body_count"],
        "sha256": digest,
        "output": str(output),
    }, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registration", type=Path, default=REGISTRATION)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (FootProxyError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"foot-contact-proxy: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
