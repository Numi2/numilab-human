"""Validate the source-bound full-skin shell candidate.

The BodyParts3D shell is useful for anatomical presentation and for the next
registration step.  It is deliberately kept separate from physical tissue
volume, shell thickness, constitutive material, collision/contact and mass
ownership.  The compiler verifies the binary payload and its source archives
without promoting a visual skinning artifact to mechanics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical, read_json


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_V1 = "HumanPack.skin-shell-candidate.v1"
SCHEMA_V2 = "HumanPack.skin-shell-candidate.v2"
SCHEMA = SCHEMA_V2
PROFILE = ROOT / "config/skin-shell-candidate.v2.json"
MANIFEST = ROOT / "Docs/media/skin-shell-candidate-native-v2-20260914/bodyparts3d-myosim-skinned-shell-native.manifest.json"
PAYLOAD = ROOT / "Docs/media/skin-shell-candidate-native-v2-20260914/bodyparts3d-myosim-skinned-shell-native.nhskin"
REGISTRATION = ROOT / "Docs/media/skin-shell-candidate-20260914/registration.json"
NATIVE_BONE_MANIFEST = ROOT / "Docs/media/numi-human-lower-joint-focus-v1/receipts/nhbones1.manifest.json"
MYOSIM_ARCHIVE = ROOT / "Sources/myosim/myo_sim-33c89c2b.tar.gz"
ISA_ARCHIVE = ROOT / "Sources/isa_BP3D_4.0_obj_99.zip"
PARTOF_ARCHIVE = ROOT / "Sources/partof_BP3D_4.0_obj_99.zip"
SKIN_MEMBER = "isa_BP3D_4.0_obj_99/FJ2810.obj"


class SkinShellCandidateError(HumanImportError):
    """A skin shell candidate cannot be admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SkinShellCandidateError("skin shell candidate: " + message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _read(path: Path, label: str) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    value = read_json(path)
    _require(isinstance(value, dict), f"{label} is not an object")
    return value, _sha256_bytes(raw)


def _profile(path: Path) -> dict[str, Any]:
    profile, _ = _read(path, "skin shell profile")
    _require(path.read_bytes() == canonical(profile) + b"\n", "skin shell profile is not canonical")
    schema = profile.get("schema")
    _require(schema in {"numi.human.skin-shell-candidate.v1", "numi.human.skin-shell-candidate.v2"},
             "unsupported skin shell profile schema")
    expected_id = (
        "bodyparts3d_full_skin_visual_shell_candidate"
        if schema == "numi.human.skin-shell-candidate.v1"
        else "bodyparts3d_full_skin_visual_shell_native_candidate"
    )
    _require(profile.get("id") == expected_id, "unsupported skin shell profile")
    expected_fields = {
        "boundary", "expected_payload_abi", "expected_registration_anchor_count",
        "expected_skin_member_id", "id", "manifest", "payload", "registration", "schema",
    }
    if schema == "numi.human.skin-shell-candidate.v2":
        expected_fields.update({"expected_native_registration_fingerprint32", "native_bone_manifest"})
    _require(set(profile) == expected_fields, "skin shell profile fields differ")
    path_fields = ["manifest", "payload", "registration"]
    if schema == "numi.human.skin-shell-candidate.v2":
        path_fields.append("native_bone_manifest")
    for key in path_fields:
        value = profile.get(key)
        _require(isinstance(value, str) and value and not Path(value).is_absolute()
                 and ".." not in Path(value).parts and "\\" not in value,
                 f"skin shell {key} path is unsafe")
    _require(profile.get("expected_payload_abi") == 4 and
             profile.get("expected_registration_anchor_count") == 185 and
             profile.get("expected_skin_member_id") == "FJ2810",
             "skin shell profile constants differ")
    if schema == "numi.human.skin-shell-candidate.v2":
        fingerprint = profile.get("expected_native_registration_fingerprint32")
        _require(isinstance(fingerprint, str) and len(fingerprint) == 8 and
                 all(character in "0123456789abcdef" for character in fingerprint),
                 "native skin registration fingerprint is malformed")
    return profile


def _safe_repo_path(relative: str, label: str) -> Path:
    path = (ROOT / relative).resolve()
    _require(path.is_relative_to(ROOT), f"{label} resolves outside the repository")
    return path


def _zip_member_sha256(path: Path, member: str) -> tuple[int, str]:
    _require(path.is_file() and not path.is_symlink(), f"source archive is unavailable: {path.name}")
    with zipfile.ZipFile(path) as archive:
        try:
            info = archive.getinfo(member)
        except KeyError as error:
            raise SkinShellCandidateError(f"skin source member is missing: {member}") from error
        _require(not member.endswith("/") and info.file_size > 0, "skin source member is empty")
        digest = hashlib.sha256()
        with archive.open(info) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return info.file_size, digest.hexdigest()


def compile_candidate(
    *, profile: Path = PROFILE, manifest: Path = MANIFEST, payload: Path = PAYLOAD,
    registration: Path = REGISTRATION, myosim_archive: Path = MYOSIM_ARCHIVE,
    isa_archive: Path = ISA_ARCHIVE, partof_archive: Path = PARTOF_ARCHIVE,
) -> dict[str, Any]:
    profile_doc = _profile(Path(profile))
    manifest_doc, manifest_sha = _read(Path(manifest), "skin shell payload manifest")
    registration_doc, registration_sha = _read(Path(registration), "skin shell registration")
    payload_path = Path(payload)
    _require(payload_path.is_file() and not payload_path.is_symlink(),
             "skin shell payload is not a regular file")
    payload_sha = _sha256(payload_path)
    payload_bytes = payload_path.read_bytes()
    _require(manifest_doc.get("schema") == "numi.human.bodyparts3d-myosim-skinned-shell-visual-payload.v4",
             "unsupported skin shell payload manifest")
    _require(manifest_doc.get("status") ==
             "native_four_body_source_surface_local_linear_blend_skin_shell_visual_input_not_collision_or_physics",
             "skin shell payload promoted a physical owner")
    payload_doc = manifest_doc.get("payload")
    source_doc = manifest_doc.get("source")
    coverage = manifest_doc.get("coverage")
    _require(isinstance(payload_doc, dict) and isinstance(source_doc, dict) and
             isinstance(coverage, dict), "skin shell payload manifest is incomplete")
    _require(payload_doc.get("file") == payload_path.name and
             payload_doc.get("sha256") == payload_sha and
             payload_doc.get("bytes") == len(payload_bytes),
             "skin shell payload hash or size differs")
    _require(payload_doc.get("payload_abi") == profile_doc["expected_payload_abi"] and
             payload_doc.get("magic") == "NHSKIN1" and
             payload_doc.get("binding_count") == 86 and
             payload_doc.get("vertex_count") == 54949 and
             payload_doc.get("triangle_count") == 109183 and
             payload_doc.get("index_count") == 327549,
             "skin shell payload counts differ")
    _require(len(payload_bytes) >= 60, "skin shell payload is truncated")
    magic, abi, bindings, vertices, indices, registration_fingerprint, source_hash = struct.unpack(
        "<8s5I32s", payload_bytes[:60]
    )
    _require(magic == b"NHSKIN1\0" and abi == 4 and bindings == payload_doc["binding_count"] and
             vertices == payload_doc["vertex_count"] and indices == payload_doc["index_count"],
             "skin shell binary header differs from its manifest")
    _require(f"{registration_fingerprint:08x}" == payload_doc["registration_fingerprint32"],
             "skin shell registration fingerprint differs")
    _require(source_hash.hex() == source_doc.get("myosim_source_archive_sha256"),
             "skin shell MyoSim source hash differs")
    _require(registration_doc.get("schema") ==
             "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2" and
             registration_doc.get("status") == "provisional_visual_registration_not_admitted_to_collision_or_physics" and
             len(registration_doc.get("anchors", [])) == profile_doc["expected_registration_anchor_count"],
             "skin shell registration is incomplete or promoted")
    _require(source_doc.get("registration", {}).get("file") == registration_path_name(registration) and
             source_doc.get("registration", {}).get("sha256") == registration_sha,
             "skin shell registration provenance differs")
    bodyparts = source_doc.get("bodyparts")
    _require(bodyparts == {
        "id": "bodyparts3d_4", "version": "4.0",
        "archives": [
            {"file": "isa_BP3D_4.0_obj_99.zip", "hierarchy": "is_a", "sha256": _sha256(isa_archive)},
            {"file": "partof_BP3D_4.0_obj_99.zip", "hierarchy": "part_of", "sha256": _sha256(partof_archive)},
        ],
    }, "skin shell BodyParts3D source archive identity differs")
    myosim_source_hash = _sha256(myosim_archive)
    _require(myosim_source_hash == source_doc.get("myosim_source_archive_sha256"),
             "skin shell MyoSim source archive differs")
    native_registration_bound = False
    native_bone_manifest_sha: str | None = None
    native_bone_payload_sha: str | None = None
    if profile_doc["schema"] == "numi.human.skin-shell-candidate.v2":
        native_manifest_path = _safe_repo_path(
            profile_doc["native_bone_manifest"], "native BodyParts3D bone manifest"
        )
        native_manifest, native_bone_manifest_sha = _read(
            native_manifest_path, "native BodyParts3D bone manifest"
        )
        _require(native_manifest.get("schema") ==
                 "numi.human.bodyparts3d-myosim-major-bone-visual-payload.v1",
                 "unsupported native BodyParts3D bone manifest")
        native_payload = native_manifest.get("payload")
        _require(isinstance(native_payload, dict) and
                 native_payload.get("magic") == "NHBONES1" and
                 native_payload.get("payload_abi") == 2 and
                 native_payload.get("bone_count") == 185 and
                 native_payload.get("registration_fingerprint32") ==
                 profile_doc["expected_native_registration_fingerprint32"] and
                 native_payload.get("sha256") and
                 native_payload.get("vertex_count") == 252167 and
                 native_payload.get("index_count") == 1378566,
                 "native BodyParts3D bone identity differs")
        native_bone_payload_sha = native_payload["sha256"]
        _require(payload_doc.get("registration_fingerprint32") ==
                 profile_doc["expected_native_registration_fingerprint32"],
                 "skin shell registration fingerprint is not compatible with native NHBONES1")
        native_registration_bound = True
    skin = source_doc.get("skin")
    _require(isinstance(skin, dict) and skin.get("member_id") == profile_doc["expected_skin_member_id"] and
             skin.get("member") == SKIN_MEMBER and skin.get("archive") == isa_archive.name and
             skin.get("archive_sha256") == _sha256(isa_archive),
             "skin shell source member identity differs")
    source_size, source_member_sha = _zip_member_sha256(isa_archive, SKIN_MEMBER)
    _require(source_member_sha == skin.get("member_sha256") and
             skin.get("source_vertex_count") == 102467 and
             skin.get("source_triangle_count") == 203382 and source_size > 0,
             "skin shell source member hash or counts differ")
    outer = coverage.get("outer_source_surface")
    _require(isinstance(outer, dict) and outer.get("method") ==
             "exact_outer_connected_component_of_bodyparts3d_compound_skin_solid" and
             outer.get("source_vertex_count") == 102467 and
             outer.get("source_triangle_count") == 203382 and
             outer.get("retained_vertex_count") == 54949 and
             outer.get("retained_triangle_count") == 109183 and
             outer.get("component_count") == 100,
             "skin shell outer-surface audit differs")
    _require(coverage.get("influences_per_vertex") == 4 and
             coverage.get("source_bone_surface_sample_count") == 7040 and
             type(coverage.get("rest_pose_reconstruction_max_error_m")) in (int, float) and
             coverage["rest_pose_reconstruction_max_error_m"] <= 2.0e-5,
             "skin shell registration coverage is not qualified")
    result = {
        "schema": SCHEMA_V2 if profile_doc["schema"] == "numi.human.skin-shell-candidate.v2" else SCHEMA_V1,
        "compiler": "numilab-human.skin-shell-candidate.1",
        "status": "partial",
        "subject": "one adult male source package",
        "inputs": {
            "profile": {"path": _relative(Path(profile)), "sha256": _sha256(Path(profile))},
            "manifest": {"path": _relative(Path(manifest)), "sha256": manifest_sha},
            "payload": {"path": _relative(payload_path), "sha256": payload_sha},
            "registration": {"path": _relative(Path(registration)), "sha256": registration_sha},
        },
        "source": {
            "bodyparts3d_archive_sha256": _sha256(isa_archive),
            "bodyparts3d_member_id": skin["member_id"],
            "bodyparts3d_member_sha256": skin["member_sha256"],
            "myosim_archive_sha256": myosim_source_hash,
            "source_vertex_count": skin["source_vertex_count"],
            "source_triangle_count": skin["source_triangle_count"],
            "outer_surface_vertex_count": outer["retained_vertex_count"],
            "outer_surface_triangle_count": outer["retained_triangle_count"],
        },
        "coverage": {
            "registered_body_influence_count": payload_doc["binding_count"],
            "influences_per_vertex": coverage["influences_per_vertex"],
            "source_bone_surface_sample_count": coverage["source_bone_surface_sample_count"],
            "rest_pose_reconstruction_max_error_m": coverage["rest_pose_reconstruction_max_error_m"],
            "outer_component_count": outer["component_count"],
        },
        "ownership": {
            "skin_visual_shell_owner": True,
            "skin_physical_volume_owner": False,
            "skin_mechanical_mass_owner": False,
            "skin_thickness_owner": False,
            "skin_material_owner": False,
            "skin_collision_owner": False,
            "skin_self_contact_owner": False,
            "fat_geometry_owner": False,
            "fat_mass_owner": False,
        },
        "qualification": {
            "source_archive_bound": True,
            "source_skin_member_bound": True,
            "outer_surface_topology_selection_bound": True,
            "registered_visual_influences_bound": True,
            "rest_pose_reconstruction_bound": True,
            "native_registration_fingerprint_bound": native_registration_bound,
            "skin_physical_volume": False,
            "skin_shell_thickness": False,
            "skin_material_calibration": False,
            "skin_collision_or_contact": False,
            "skin_self_contact": False,
            "fat_geometry": False,
            "fat_mass": False,
            "subject_calibration": False,
            "integrated_human_qualification": False,
        },
        "boundary": (
            "The candidate binds the exact BodyParts3D FJ2810 full-skin source, "
            "its selected outer connected component, and four-body visual "
            "influences to the source-bound MyoSim rest frame"
            + (" and the native NHBONES1 registration fingerprint. "
               if native_registration_bound else ". ")
            + "It is a visual "
            "registration artifact only. No shell thickness, physical volume, "
            "mechanical mass, constitutive material, collision/contact, "
            "self-contact, muscle sliding, adipose domain, or subject-specific "
            "calibration is assigned. Fat remains absent from the pinned source "
            "inventory and must not be inferred from the skin shell."
        ),
    }
    if native_registration_bound:
        result["inputs"]["native_bone_manifest"] = {
            "path": _relative(_safe_repo_path(profile_doc["native_bone_manifest"], "native BodyParts3D bone manifest")),
            "sha256": native_bone_manifest_sha,
        }
        result["source"]["native_bone_registration_fingerprint32"] = profile_doc[
            "expected_native_registration_fingerprint32"
        ]
        result["source"]["native_bone_payload_sha256"] = native_bone_payload_sha
    return result


def registration_path_name(path: Path) -> str:
    return path.name


def _immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    _require(not path.is_symlink(), "output is redirected")
    if path.exists():
        _require(path.read_bytes() == payload, "output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return _sha256_bytes(payload)


def run(arguments: argparse.Namespace) -> int:
    result = compile_candidate(
        profile=arguments.profile, manifest=arguments.manifest, payload=arguments.payload,
        registration=arguments.registration, myosim_archive=arguments.myosim_archive,
        isa_archive=arguments.isa_archive, partof_archive=arguments.partof_archive,
    )
    output = arguments.output.resolve()
    digest = _immutable_write(output, result)
    print(json.dumps({"schema": result["schema"], "output": str(output), "sha256": digest,
                      "status": result["status"], "coverage": result["coverage"],
                      "qualification": result["qualification"]}, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--payload", type=Path, default=PAYLOAD)
    parser.add_argument("--registration", type=Path, default=REGISTRATION)
    parser.add_argument("--myosim-archive", type=Path, default=MYOSIM_ARCHIVE)
    parser.add_argument("--isa-archive", type=Path, default=ISA_ARCHIVE)
    parser.add_argument("--partof-archive", type=Path, default=PARTOF_ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the source-bound Human skin-shell candidate")
    add_arguments(parser)
    arguments = parser.parse_args()
    raise SystemExit(arguments.handler(arguments))
