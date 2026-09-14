"""Admit the native four-view skin-shell visual capture.

The native probe is a source-to-renderer admission check.  It binds the
native-compatible ``NHSKIN1`` payload to the official ``NHBONES1`` source
registration and records the exact M4 Pro capture, while keeping skin volume,
mass, material, contact, deformation, and subject calibration out of scope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical, read_json


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.skin-shell-native-visual.v1"
OUTPUT_DIR = ROOT / "Docs/media/skin-shell-native-visual-20260914"
SKIN_RECEIPT = ROOT / "Docs/media/skin-shell-candidate-native-v2-20260914/receipt-v2.json"
SKIN_PAYLOAD = ROOT / "Docs/media/skin-shell-candidate-native-v2-20260914/bodyparts3d-myosim-skinned-shell-native.nhskin"
NATIVE_BONE_MANIFEST = ROOT / "Docs/media/numi-human-lower-joint-focus-v1/receipts/nhbones1.manifest.json"
STDOUT = OUTPUT_DIR / "native.stdout.log"
STDERR = OUTPUT_DIR / "native.stderr.log"
VISUAL_MANIFEST = OUTPUT_DIR / "myosim-fullbody-articulated-bodyparts-bones-source-skinned-shell.visual.v3.json"
VISUAL_PACK = OUTPUT_DIR / "myosim-fullbody-articulated-bodyparts-bones-source-skinned-shell.mrvpack"
VIEWS = (
    "front",
    "oblique",
    "side",
    "rear",
)
FRAME_NAMES = {
    view: f"myosim-fullbody-articulated-bodyparts-bones-source-skinned-shell-{view}.png"
    for view in VIEWS
}
VIEW_RE = re.compile(
    r"^view=(?P<view>\S+)\s+.*?skin_shell_pixels=(?P<pixels>[0-9]+)\s+"
    r".*?frame=(?P<frame>\S+)$"
)


class SkinShellNativeVisualError(HumanImportError):
    """The native skin-shell visual capture cannot be admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SkinShellNativeVisualError("skin shell native visual: " + message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _file(path: Path, label: str) -> str:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    _require(path.stat().st_size > 0 or path == STDERR, f"{label} is empty")
    return _sha256(path)


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    digest = _file(path, label)
    value = read_json(path)
    _require(isinstance(value, dict), f"{label} is not an object")
    return value, digest


def _input(path: Path, schema: str | None, digest: str) -> dict[str, Any]:
    value: dict[str, Any] = {"path": _relative(path), "sha256": digest}
    if schema is not None:
        value["schema"] = schema
    return value


def _parse_native_capture(stdout: Path, stderr: Path) -> tuple[list[dict[str, Any]], str]:
    stdout_sha = _file(stdout, "native stdout")
    _require(stderr.is_file() and not stderr.is_symlink(), "native stderr is not a regular file")
    _require(stderr.read_bytes() == b"", "native stderr is not empty")
    rows: list[dict[str, Any]] = []
    summary: str | None = None
    for line in stdout.read_text(encoding="utf-8").splitlines():
        match = VIEW_RE.match(line)
        if match:
            rows.append({
                "view": match.group("view"),
                "skin_shell_pixels": int(match.group("pixels")),
                "frame": Path(match.group("frame")).name,
            })
        elif line.startswith("myosim_articulated_bodyparts_bone_visual="):
            summary = line
    _require([row["view"] for row in rows] == list(VIEWS),
             "native capture does not contain the four ordered views")
    _require(all(row["skin_shell_pixels"] > 0 for row in rows),
             "native capture has a view with no skin-shell pixels")
    _require(summary is not None and
             "myosim_articulated_bodyparts_bone_visual=ok" in summary and
             'metal_pose_device="Apple M4 Pro"' in summary and
             'renderer_device="Apple M4 Pro"' in summary and
             "frame_dimension=512" in summary and
             "core_bodies=157" in summary and
             "bodyparts_skin_shells=1" in summary and
             "skin_shell_binding=four_body_registered_source_bone_surface_local_linear_blend_world_rest_normals" in summary,
             "native visual summary is not the expected M4 Pro skin-shell admission")
    return rows, stdout_sha


def compile_candidate(
    *,
    skin_receipt: Path = SKIN_RECEIPT,
    skin_payload: Path = SKIN_PAYLOAD,
    native_bone_manifest: Path = NATIVE_BONE_MANIFEST,
    stdout: Path = STDOUT,
    stderr: Path = STDERR,
    visual_manifest: Path = VISUAL_MANIFEST,
    visual_pack: Path = VISUAL_PACK,
    frame_directory: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    skin, skin_sha = _read_json(Path(skin_receipt), "skin shell candidate receipt")
    _require(skin.get("schema") == "HumanPack.skin-shell-candidate.v2" and
             skin.get("qualification", {}).get("native_registration_fingerprint_bound") is True,
             "skin shell candidate is not the native-compatible v2 receipt")
    payload_sha = _file(Path(skin_payload), "skin shell payload")
    _require(skin.get("inputs", {}).get("payload", {}).get("sha256") == payload_sha,
             "skin shell payload differs from its candidate receipt")
    native_manifest, native_manifest_sha = _read_json(
        Path(native_bone_manifest), "native BodyParts3D bone manifest"
    )
    _require(native_manifest.get("schema") ==
             "numi.human.bodyparts3d-myosim-major-bone-visual-payload.v1",
             "unsupported native bone manifest")
    native_payload = native_manifest.get("payload")
    _require(isinstance(native_payload, dict) and
             native_payload.get("magic") == "NHBONES1" and
             native_payload.get("payload_abi") == 2 and
             native_payload.get("bone_count") == 185 and
             native_payload.get("registration_fingerprint32") == "6a48e223" and
             native_payload.get("vertex_count") == 252167 and
             native_payload.get("index_count") == 1378566 and
             isinstance(native_payload.get("sha256"), str),
             "native bone manifest identity differs")

    rows, stdout_sha = _parse_native_capture(Path(stdout), Path(stderr))
    visual, visual_sha = _read_json(Path(visual_manifest), "native visual manifest")
    _require(visual.get("schema_version") == 3 and
             visual.get("render_scene", {}).get("body_count") == 157 and
             visual.get("render_scene", {}).get("visual_pack_count") == 1 and
             visual.get("visual_pack_hashes") and
             len(visual["visual_pack_hashes"]) == 1,
             "native visual manifest is not the expected full-body scene")
    visual_pack_path = Path(visual_pack)
    visual_pack_sha = _file(visual_pack_path, "native visual pack")
    visual_pack_bytes = visual_pack_path.read_bytes()
    _require(len(visual_pack_bytes) >= 64 and visual_pack_bytes[:8] == b"MRVPACK2",
             "native visual pack header is not MRVPACK2")
    visual_pack_content_sha = visual_pack_bytes[32:64].hex()
    _require(visual["visual_pack_hashes"] == [f"sha256:{visual_pack_content_sha}"],
             "native visual pack content hash differs from its manifest")
    frame_hashes: dict[str, str] = {}
    for row in rows:
        expected_name = FRAME_NAMES[row["view"]]
        _require(row["frame"] == expected_name,
                 f"native {row['view']} frame identity differs")
        frame_hashes[row["view"]] = _file(Path(frame_directory) / expected_name,
                                            f"native {row['view']} frame")

    result = {
        "schema": SCHEMA,
        "compiler": "numilab-human.skin-shell-native-visual.1",
        "status": "qualified",
        "subject": "one adult male source package",
        "inputs": {
            "skin_shell_candidate": _input(Path(skin_receipt), skin["schema"], skin_sha),
            "skin_shell_payload": _input(Path(skin_payload), "NHSKIN1-ABI4", payload_sha),
            "native_bone_manifest": _input(Path(native_bone_manifest), native_manifest["schema"], native_manifest_sha),
            "native_stdout": _input(Path(stdout), None, stdout_sha),
            "native_stderr": _input(Path(stderr), None, _sha256(Path(stderr))),
            "visual_manifest": _input(Path(visual_manifest), "myosim.visual-scene-manifest.v3", visual_sha),
            "visual_pack": _input(Path(visual_pack), "myosim.visual-pack.mrvpack", visual_pack_sha),
        },
        "source": {
            "bodyparts3d_member_id": skin["source"]["bodyparts3d_member_id"],
            "skin_payload_sha256": payload_sha,
            "native_bone_payload_sha256": native_payload["sha256"],
            "native_registration_fingerprint32": native_payload["registration_fingerprint32"],
            "core_body_count": 157,
            "rendered_skin_shell_count": 1,
        },
        "capture": {
            "metal_pose_device": "Apple M4 Pro",
            "renderer_device": "Apple M4 Pro",
            "frame_dimension": 512,
            "views": rows,
            "frame_sha256": frame_hashes,
            "visual_pack_sha256": visual_pack_sha,
            "visual_pack_content_sha256": visual_pack_content_sha,
            "visual_manifest_sha256": visual_sha,
        },
        "ownership": {
            "skin_visual_shell_owner": True,
            "skin_physical_volume_owner": False,
            "skin_mechanical_mass_owner": False,
            "skin_material_owner": False,
            "skin_collision_owner": False,
            "skin_deformation_owner": False,
            "fat_geometry_owner": False,
        },
        "qualification": {
            "native_visual_admission": True,
            "source_payload_hash_bound": True,
            "native_registration_fingerprint_bound": True,
            "native_renderer_device_bound": True,
            "four_view_capture": True,
            "skin_shell_pixels_positive": True,
            "skin_physical_volume": False,
            "skin_mechanical_mass": False,
            "skin_material_calibration": False,
            "skin_collision_or_contact": False,
            "skin_deformation": False,
            "fat_geometry": False,
            "subject_calibration": False,
            "integrated_human_qualification": False,
        },
        "boundary": (
            "The native Apple M4 Pro probe rendered the native-compatible "
            "BodyParts3D FJ2810 shell through the 157-body source pose in four "
            "views and bound the result to the official 185-bone NHBONES1 "
            "registration fingerprint. This is a visual admission only. It "
            "does not create shell thickness, physical volume, mass, material, "
            "collision/contact, deformation, adipose geometry, subject "
            "calibration, or live standing/walking mechanics."
        ),
    }
    return result


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
    result = compile_candidate(
        skin_receipt=arguments.skin_receipt,
        skin_payload=arguments.skin_payload,
        native_bone_manifest=arguments.native_bone_manifest,
        stdout=arguments.stdout,
        stderr=arguments.stderr,
        visual_manifest=arguments.visual_manifest,
        visual_pack=arguments.visual_pack,
        frame_directory=arguments.frame_directory,
    )
    output = arguments.output.resolve()
    digest = _immutable_write(output, result)
    print(json.dumps({"schema": SCHEMA, "output": str(output), "sha256": digest,
                      "status": result["status"], "views": result["capture"]["views"]},
                     sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--skin-receipt", type=Path, default=SKIN_RECEIPT)
    parser.add_argument("--skin-payload", type=Path, default=SKIN_PAYLOAD)
    parser.add_argument("--native-bone-manifest", type=Path, default=NATIVE_BONE_MANIFEST)
    parser.add_argument("--stdout", type=Path, default=STDOUT)
    parser.add_argument("--stderr", type=Path, default=STDERR)
    parser.add_argument("--visual-manifest", type=Path, default=VISUAL_MANIFEST)
    parser.add_argument("--visual-pack", type=Path, default=VISUAL_PACK)
    parser.add_argument("--frame-directory", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    arguments = parser.parse_args()
    raise SystemExit(arguments.handler(arguments))
