"""Recompute hash-locked geometry evidence for the NHTISS4 muscle surfaces.

This audit reads the original BodyParts3D OBJ members rather than trusting the
visual payload's triangle and vertex counts.  It applies the same exact
authored-coordinate quotient used by the organ geometry compiler, reports
surface area and topology, and computes algebraic volume moments only for a
single closed component.  The result remains an anatomy candidate: no volume,
mass, constitutive material, activation force, or body owner is promoted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zipfile
from typing import Any

from . import cvsim21_anatomy
from .cardiac_cavity_geometry import exact_coordinate_quotient, parse_obj
from .model import ImportError as HumanImportError
from .organ_geometry import _topology_summary
from .physiology import canonical


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config/muscle-surface-geometry-audit.v1.json"
SURFACE_RECEIPT = ROOT / "Docs/media/soft-tissue-surface-candidate-20260914/receipt-v1.json"
SURFACE_MANIFEST = ROOT / (
    "Docs/media/numi-human-toe-enthesis-v5-2048/manifests/"
    "bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json"
)
SOURCE_LOCK = ROOT / "sources.lock.json"
ARCHIVE = ROOT / "Sources/isa_BP3D_4.0_obj_99.zip"
ARCHIVE_NAME = "isa_BP3D_4.0_obj_99.zip"
ARCHIVE_MEMBER_ROOT = "isa_BP3D_4.0_obj_99"
SCHEMA = "HumanPack.muscle-surface-geometry-audit.v1"


class GeometryAuditError(HumanImportError):
    """A source muscle-surface geometry audit cannot be admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GeometryAuditError("muscle surface geometry: " + message)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _read_json(path: Path, label: str, *, canonical_required: bool = False) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise GeometryAuditError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    if canonical_required:
        _require(raw == canonical(value) + b"\n", f"{label} is not canonical")
    return value, _sha256_bytes(raw)


def _load_profile(path: Path) -> tuple[dict[str, Any], str]:
    profile, profile_sha = _read_json(path, "geometry audit profile", canonical_required=True)
    required = {
        "archive", "boundary", "expected_closed_multi_component_count",
        "expected_closed_single_component_count", "expected_muscle_surface_count",
        "expected_surface_count", "expected_tendon_surface_count", "id", "schema",
        "source_lock", "surface_manifest", "surface_receipt",
    }
    _require(set(profile) == required, "geometry audit profile fields differ")
    _require(profile["schema"] == "numi.human.muscle-surface-geometry-audit.v1",
             "unsupported geometry audit profile schema")
    _require(profile["id"] == "bodyparts3d_muscle_surface_geometry_audit",
             "unsupported geometry audit profile")
    for key in ("archive", "source_lock", "surface_manifest", "surface_receipt"):
        value = profile[key]
        _require(isinstance(value, str) and value.strip() and not Path(value).is_absolute()
                 and ".." not in Path(value).parts and "\\" not in value,
                 f"{key} path is unsafe")
    for key in (
        "expected_closed_multi_component_count", "expected_closed_single_component_count",
        "expected_muscle_surface_count", "expected_surface_count",
        "expected_tendon_surface_count",
    ):
        _require(type(profile[key]) is int and profile[key] >= 0, f"{key} is invalid")
    _require(profile["expected_muscle_surface_count"]
             + profile["expected_tendon_surface_count"] == profile["expected_surface_count"],
             "surface layer counts do not add up")
    _require(isinstance(profile["boundary"], str) and profile["boundary"].strip(),
             "geometry audit boundary is missing")
    return profile, profile_sha


def _locked_archive(sources: Path, source_lock: Path) -> tuple[Path, dict[str, Any], str]:
    lock, _ = _read_json(source_lock, "source lock")
    record = lock.get("sources", {}).get("bodyparts3d_4", {}).get("files", {}).get(ARCHIVE_NAME)
    _require(isinstance(record, dict), "source lock has no is_a BodyParts3D archive record")
    expected_sha = record.get("sha256")
    expected_bytes = record.get("bytes")
    _require(isinstance(expected_sha, str) and len(expected_sha) == 64,
             "source archive hash lock is invalid")
    archive = Path(sources) / ARCHIVE_NAME
    _require(archive.is_file() and not archive.is_symlink(),
             "source archive is missing or redirected")
    actual_sha = _sha256_file(archive)
    _require(actual_sha == expected_sha, "source archive hash differs from sources.lock.json")
    if expected_bytes is not None:
        _require(type(expected_bytes) is int and archive.stat().st_size == expected_bytes,
                 "source archive byte count differs from sources.lock.json")
    return archive, {"file": ARCHIVE_NAME, "sha256": actual_sha,
                     "bytes": archive.stat().st_size}, actual_sha


def _surface_area(vertices_m: list[list[float]], triangles: list[list[int]]) -> float:
    area = 0.0
    for face in triangles:
        a, b, c = (vertices_m[index] for index in face)
        u = [b[index] - a[index] for index in range(3)]
        v = [c[index] - a[index] for index in range(3)]
        cross = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
        area += 0.5 * math.sqrt(math.fsum(value * value for value in cross))
    _require(math.isfinite(area) and area > 0.0, "surface area is not positive")
    return area


def _status(topology: dict[str, Any], quotient: dict[str, Any]) -> str:
    if (topology["closed_oriented_manifold_candidate"]
            and topology["face_component_count"] == 1
            and not quotient["topology"]["unused_vertex_ids"]):
        return "single_closed_component"
    if topology["closed_oriented_manifold_candidate"]:
        return "closed_multi_component"
    return "topology_defective"


def compile_audit(
    *, profile: Path = PROFILE, sources: Path | None = None,
    source_lock: Path | None = None, surface_receipt: Path | None = None,
    surface_manifest: Path | None = None,
) -> dict[str, Any]:
    profile_doc, profile_sha = _load_profile(Path(profile))
    sources = Path(sources) if sources is not None else ROOT / Path(profile_doc["archive"]).parent
    source_lock = Path(source_lock) if source_lock is not None else ROOT / profile_doc["source_lock"]
    surface_receipt = (Path(surface_receipt) if surface_receipt is not None
                       else ROOT / profile_doc["surface_receipt"])
    surface_manifest = (Path(surface_manifest) if surface_manifest is not None
                        else ROOT / profile_doc["surface_manifest"])
    receipt, receipt_sha = _read_json(surface_receipt, "soft-tissue surface receipt",
                                      canonical_required=True)
    _require(receipt.get("schema") == "HumanPack.soft-tissue-surface-candidate.v1",
             "unsupported soft-tissue surface receipt schema")
    source = receipt.get("source")
    _require(isinstance(source, dict), "soft-tissue receipt has no source record")
    manifest, manifest_sha = _read_json(surface_manifest, "surface manifest")
    _require(manifest.get("schema")
             == "numi.human.bodyparts3d-myosim-fullbody-muscle-surface-visual-payload.v1",
             "unsupported surface manifest schema")
    _require(source.get("surface_manifest") == _relative(surface_manifest)
             and source.get("surface_manifest_sha256") == manifest_sha,
             "surface receipt does not bind the surface manifest hash")
    coverage = manifest.get("coverage")
    payload = manifest.get("payload")
    source_doc = manifest.get("source")
    _require(isinstance(coverage, dict) and isinstance(payload, dict)
             and isinstance(source_doc, dict), "surface manifest is incomplete")
    rows = source_doc.get("surfaces")
    _require(isinstance(rows, list) and len(rows) == profile_doc["expected_surface_count"],
             "surface manifest row count differs from profile")
    _require(coverage.get("emitted_surface_count") == len(rows)
             and payload.get("surface_count") == len(rows),
             "surface manifest coverage count differs")
    source_by_id: dict[int, dict[str, Any]] = {}
    member_ids: set[str] = set()
    layers = Counter()
    for row in rows:
        _require(isinstance(row, dict), "surface manifest row is malformed")
        stable_id, member_id, layer = row.get("stable_id"), row.get("member_id"), row.get("layer")
        _require(type(stable_id) is int and stable_id > 0 and stable_id not in source_by_id,
                 "surface stable identity is invalid or duplicated")
        _require(isinstance(member_id, str) and member_id and member_id not in member_ids,
                 "surface member identity is invalid or duplicated")
        _require(layer in {"muscle", "tendon"}, f"surface layer is invalid: {member_id}")
        _require(type(row.get("triangle_count")) is int and row["triangle_count"] > 0
                 and type(row.get("vertex_count")) is int and row["vertex_count"] > 0,
                 f"surface counts are invalid: {member_id}")
        member_sha = row.get("member_sha256")
        _require(isinstance(member_sha, str) and len(member_sha) == 64
                 and all(character in "0123456789abcdef" for character in member_sha),
                 f"surface source hash is invalid: {member_id}")
        source_by_id[stable_id] = row
        member_ids.add(member_id)
        layers[layer] += 1
    _require(layers["muscle"] == profile_doc["expected_muscle_surface_count"]
             and layers["tendon"] == profile_doc["expected_tendon_surface_count"],
             "surface layer counts differ from profile")

    archive, archive_record, archive_sha = _locked_archive(sources, source_lock)
    audit_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    total_area = 0.0
    total_vertices = 0
    total_triangles = 0
    total_duplicate_faces = 0
    total_vertex_defects = 0
    with zipfile.ZipFile(archive) as archive_file:
        archive_names = {info.filename for info in archive_file.infolist()}
        for stable_id in sorted(source_by_id):
            manifest_row = source_by_id[stable_id]
            member_id = manifest_row["member_id"]
            archive_member = f"{ARCHIVE_MEMBER_ROOT}/{member_id}.obj"
            _require(archive_member in archive_names, f"missing source archive member: {archive_member}")
            data = archive_file.read(archive_member)
            member_sha = _sha256_bytes(data)
            _require(member_sha == manifest_row["member_sha256"],
                     f"source member hash drift: {member_id}")
            parsed = parse_obj(data, archive_member)
            source_selection = manifest_row.get("source_component_selection")
            expected_source_vertex_count = (
                source_selection.get("source_vertex_count")
                if isinstance(source_selection, dict) else manifest_row["vertex_count"]
            )
            expected_source_triangle_count = (
                source_selection.get("source_triangle_count")
                if isinstance(source_selection, dict) else manifest_row["triangle_count"]
            )
            _require(len(parsed["vertices_mm"]) == expected_source_vertex_count
                     and len(parsed["triangles"]) == expected_source_triangle_count,
                     f"source geometry counts drifted: {member_id}")
            raw_topology = cvsim21_anatomy.geometry.analyze_topology(
                parsed["vertices_mm"], parsed["triangles"]
            )
            quotient = exact_coordinate_quotient(parsed)
            topology = _topology_summary(raw_topology, quotient)
            status = _status(topology, quotient)
            area = _surface_area(quotient["vertices_m"], quotient["triangles"])
            volume_moments: dict[str, Any] | None = None
            if status == "single_closed_component":
                volume_moments = cvsim21_anatomy.geometry_moments(
                    quotient["vertices_m"], quotient["triangles"]
                )
                _require(volume_moments.get("physical_volume_m3") is None
                         and volume_moments.get("mechanical_mass_kg") is None,
                         f"source volume promoted an owner: {member_id}")
            status_counts[status] += 1
            total_area += area
            total_vertices += len(quotient["vertices_m"])
            total_triangles += len(quotient["triangles"])
            total_duplicate_faces += topology["duplicate_face_count"]
            total_vertex_defects += topology["vertex_manifold_defect_count"]
            audit_rows.append({
                "stable_id": stable_id,
                "member_id": member_id,
                "layer": manifest_row["layer"],
                "member_sha256": member_sha,
                "emitted_vertex_count": manifest_row["vertex_count"],
                "emitted_triangle_count": manifest_row["triangle_count"],
                "source_vertex_count": len(parsed["vertices_mm"]),
                "source_triangle_count": len(parsed["triangles"]),
                "quotient_vertex_count": len(quotient["vertices_m"]),
                "quotient_triangle_count": len(quotient["triangles"]),
                "surface_area_m2": area,
                "topology": topology,
                "status": status,
                "algebraic_volume_candidate_m3": (
                    volume_moments["absolute_signed_volume_m3"]
                    if volume_moments is not None else None
                ),
                "physical_volume_owner": None,
                "mechanical_mass_owner": None,
                "material_owner": None,
                "volumetric_active_force_owner": None,
            })
    _require(status_counts["single_closed_component"]
             == profile_doc["expected_closed_single_component_count"]
             and status_counts["closed_multi_component"]
             == profile_doc["expected_closed_multi_component_count"]
             and sum(status_counts.values()) == profile_doc["expected_surface_count"],
             f"source topology status counts differ: {dict(status_counts)}")
    result = {
        "schema": SCHEMA,
        "compiler": "numilab-human.muscle-surface-geometry-audit.1",
        "status": "partial",
        "source": {
            "profile": _relative(Path(profile)),
            "profile_sha256": profile_sha,
            "surface_receipt": _relative(surface_receipt),
            "surface_receipt_sha256": receipt_sha,
            "surface_manifest": _relative(surface_manifest),
            "surface_manifest_sha256": manifest_sha,
            "source_lock": _relative(source_lock),
            "source_archive": _relative(archive),
            "source_archive_sha256": archive_sha,
            "source_archive_bytes": archive.stat().st_size,
            "nhtiss4_payload_sha256": source.get("nhtiss4_payload_sha256"),
            "subject": "one adult male source package",
        },
        "coordinate_frame": "BodyParts3D OBJ coordinates, exact authored-coordinate quotient",
        "counts": {
            "source_surface_count": len(audit_rows),
            "muscle_surface_count": layers["muscle"],
            "tendon_surface_count": layers["tendon"],
            "topology_recomputed_surface_count": len(audit_rows),
            "single_closed_component_count": status_counts["single_closed_component"],
            "closed_multi_component_count": status_counts["closed_multi_component"],
            "topology_defective_count": status_counts["topology_defective"],
            "surface_volume_candidate_count": sum(
                row["algebraic_volume_candidate_m3"] is not None for row in audit_rows
            ),
            "physical_volume_owner_count": 0,
            "mechanical_mass_owner_count": 0,
            "material_owner_count": 0,
            "volumetric_active_force_owner_count": 0,
            "total_quotient_vertex_count": total_vertices,
            "total_quotient_triangle_count": total_triangles,
            "total_duplicate_face_count": total_duplicate_faces,
            "total_vertex_manifold_defect_count": total_vertex_defects,
        },
        "geometry": {
            "surface_area_total_m2": total_area,
            "surface_area_units": "m^2",
            "algebraic_volume_total_m3": math.fsum(
                row["algebraic_volume_candidate_m3"] or 0.0 for row in audit_rows
            ),
            "algebraic_volume_units": "m^3",
            "volume_is_physical_owner": False,
        },
        "surfaces": audit_rows,
        "qualification": {
            "source_member_hashes_bound": True,
            "source_counts_recomputed": True,
            "exact_coordinate_quotient_recomputed": True,
            "surface_area_candidate_recomputed": True,
            "single_closed_surface_volume_candidates_recomputed": True,
            "muscle_tendon_surface_identity_bound": True,
            "physical_tissue_volume_owner": False,
            "skeletal_muscle_tissue_mass_owner": False,
            "tendon_fascia_mass_owner": False,
            "material_calibration": False,
            "subject_calibration": False,
            "volumetric_active_force_owner": False,
            "activation_force_transfer": False,
            "anatomical_supports_loading": False,
            "standing_walking": False,
        },
        "boundary": profile_doc["boundary"],
    }
    canonical(result)
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
    return _sha256_bytes(payload)


def run(arguments: argparse.Namespace) -> int:
    result = compile_audit(
        profile=arguments.profile,
        sources=arguments.sources,
        source_lock=arguments.source_lock,
        surface_receipt=arguments.surface_receipt,
        surface_manifest=arguments.surface_manifest,
    )
    output = arguments.output.resolve()
    digest = _immutable_write(output, result)
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(output),
        "sha256": digest,
        "status": result["status"],
        **result["counts"],
    }, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument("--surface-receipt", type=Path, default=SURFACE_RECEIPT)
    parser.add_argument("--surface-manifest", type=Path, default=SURFACE_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (HumanImportError, OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(2, f"muscle-surface-geometry: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
