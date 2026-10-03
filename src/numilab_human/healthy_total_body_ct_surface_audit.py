"""Independently audit retained voxel-boundary PLY candidates.

This auditor reads only the pinned source intake, the compiler receipt, and the
serialized PLY files. It does not call the surface compiler or repair topology.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import zipfile
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical


SCHEMA = "numi.healthy-total-body-ct-surface-independent-audit.v1"
PLAN_SCHEMA = "numi.healthy-total-body-ct-surface-trial-plan.v2"
INTAKE_SCHEMA = "HumanPack.external-segmentation-source-ingest.v2"
COMPILER_SCHEMA = "HumanPack.external-segmentation-voxel-surface-candidates.v2"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("healthy total-body CT surface audit: " + message)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(
    path: Path, label: str, *, require_canonical: bool = False
) -> tuple[dict[str, Any], bytes]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise HumanImportError(
            f"healthy total-body CT surface audit: {label} is malformed JSON"
        ) from error
    _require(isinstance(value, dict), f"{label} is not an object")
    if require_canonical:
        _require(raw == canonical(value) + b"\n", f"{label} is not canonical JSON")
    return value, raw


def _hash_nifti_from_archive(archive_path: Path, member_name: str) -> str:
    digest = hashlib.sha256()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            _require(
                archive.namelist().count(member_name) == 1,
                "selected NIfTI archive member is absent or duplicated",
            )
            with archive.open(member_name, "r") as member:
                with gzip.GzipFile(fileobj=member, mode="rb") as nifti:
                    for block in iter(lambda: nifti.read(16 * 1024 * 1024), b""):
                        digest.update(block)
    except (zipfile.BadZipFile, EOFError, gzip.BadGzipFile) as error:
        raise HumanImportError(
            "healthy total-body CT surface audit: archive or NIfTI member is corrupt"
        ) from error
    return digest.hexdigest()


def _edge_incidence_histogram(faces: Any, np: Any) -> dict[str, int]:
    edges = np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]))
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    values, frequencies = np.unique(counts, return_counts=True)
    return {
        str(int(value)): int(count)
        for value, count in zip(values, frequencies, strict=True)
    }


def _bad_vertex_links(faces: Any, vertex_count: int) -> int:
    """Count empty, disconnected, or non-cycle triangle links per vertex."""
    links: list[list[tuple[int, int]]] = [[] for _ in range(vertex_count)]
    for a_value, b_value, c_value in faces:
        a, b, c = int(a_value), int(b_value), int(c_value)
        links[a].append((b, c))
        links[b].append((c, a))
        links[c].append((a, b))
    invalid = 0
    for link_edges in links:
        adjacency: dict[int, list[int]] = {}
        for left, right in link_edges:
            adjacency.setdefault(left, []).append(right)
            adjacency.setdefault(right, []).append(left)
        if not adjacency or any(
            len(neighbors) != 2 for neighbors in adjacency.values()
        ):
            invalid += 1
            continue
        first = next(iter(adjacency))
        visited, stack = {first}, [first]
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        invalid += len(visited) != len(adjacency)
    return invalid


def _read_binary_ply_gzip(path: Path, np: Any) -> tuple[Any, Any, dict[str, int]]:
    _require(path.is_file() and not path.is_symlink(), "mesh is not a regular file")
    try:
        with gzip.open(path, "rb") as stream:
            header = bytearray()
            while not header.endswith(b"end_header\n"):
                line = stream.readline()
                _require(
                    bool(line) and len(header) + len(line) <= 65536,
                    "PLY header is truncated or unreasonably large",
                )
                header.extend(line)
            payload = stream.read()
    except (OSError, EOFError, gzip.BadGzipFile) as error:
        raise HumanImportError(
            "healthy total-body CT surface audit: mesh gzip payload is corrupt"
        ) from error
    lines = bytes(header).decode("ascii", errors="strict").splitlines()
    _require(
        lines[:2] == ["ply", "format binary_little_endian 1.0"],
        "mesh is not binary little-endian PLY 1.0",
    )
    vertex_count = face_count = None
    vertex_properties: list[str] = []
    face_properties: list[str] = []
    element = None
    for line in lines[2:]:
        fields = line.split()
        if len(fields) == 3 and fields[0] == "element":
            element = fields[1]
            if element == "vertex":
                vertex_count = int(fields[2])
            elif element == "face":
                face_count = int(fields[2])
        elif fields and fields[0] == "property":
            if element == "vertex":
                vertex_properties.append(" ".join(fields[1:]))
            elif element == "face":
                face_properties.append(" ".join(fields[1:]))
    _require(
        vertex_count is not None
        and face_count is not None
        and vertex_properties == ["double x", "double y", "double z"]
        and face_properties == ["list uchar int vertex_indices"],
        "mesh PLY element declarations differ",
    )
    vertex_bytes = int(vertex_count) * 24
    face_bytes = int(face_count) * 13
    _require(
        len(payload) == vertex_bytes + face_bytes,
        "mesh PLY payload length differs from its declarations",
    )
    vertices = np.frombuffer(payload[:vertex_bytes], dtype="<f8").reshape(
        int(vertex_count), 3
    )
    records = np.frombuffer(
        payload[vertex_bytes:],
        dtype=np.dtype([("count", "u1"), ("indices", "<i4", (3,))]),
    )
    _require(bool(np.all(records["count"] == 3)), "mesh contains a non-triangle face")
    faces = records["indices"]
    metadata = {
        "vertex_count": int(vertex_count),
        "triangle_count": int(face_count),
        "payload_bytes": len(payload),
    }
    return vertices, faces, metadata


def _envelope_from_geometry(
    geometry: dict[str, Any], affine: list[list[float]], np: Any
) -> tuple[Any, Any]:
    bounds = geometry.get("voxel_center_bounds_ijk")
    _require(isinstance(bounds, dict), "intake label lacks voxel-center bounds")
    lower = np.asarray(bounds.get("minimum_inclusive"), dtype=np.float64) - 0.5
    upper = np.asarray(bounds.get("maximum_inclusive"), dtype=np.float64) + 0.5
    _require(
        lower.shape == (3,) and upper.shape == (3,),
        "intake voxel bounds have invalid dimensions",
    )
    corners = np.asarray(
        [
            [x, y, z, 1.0]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float64,
    )
    world = corners @ np.asarray(affine, dtype=np.float64).T
    return world[:, :3].min(axis=0), world[:, :3].max(axis=0)


def _audit_mesh(
    mesh_path: Path,
    mesh: dict[str, Any],
    geometry: dict[str, Any],
    affine: list[list[float]],
    expected_voxel_count: int,
    np: Any,
) -> dict[str, Any]:
    digest = _sha256_file(mesh_path)
    _require(
        digest == mesh.get("mesh_file_sha256"),
        f"{mesh_path.name} SHA-256 differs from compiler receipt",
    )
    _require(
        mesh_path.stat().st_size == mesh.get("mesh_file_bytes"),
        f"{mesh_path.name} byte size differs from compiler receipt",
    )
    vertices, faces, payload = _read_binary_ply_gzip(mesh_path, np)
    _require(
        bool(np.isfinite(vertices).all()),
        f"{mesh_path.name} has non-finite vertex coordinates",
    )
    _require(
        bool(np.all((faces >= 0) & (faces < len(vertices)))),
        f"{mesh_path.name} has out-of-range vertex indices",
    )
    _require(
        bool(
            np.all(
                (faces[:, 0] != faces[:, 1])
                & (faces[:, 1] != faces[:, 2])
                & (faces[:, 2] != faces[:, 0])
            )
        ),
        f"{mesh_path.name} has repeated triangle indices",
    )
    triangles = vertices[faces]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edge_a, edge_b)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    _require(bool(np.all(area > 0.0)), f"{mesh_path.name} has a zero-area triangle")
    origin = vertices.mean(axis=0)
    relative = triangles - origin
    signed_volume = float(
        np.einsum(
            "ij,ij->i",
            relative[:, 0],
            np.cross(relative[:, 1], relative[:, 2]),
        ).sum(dtype=np.float64)
        / 6.0
    )
    edge_histogram = _edge_incidence_histogram(faces, np)
    bad_vertices = _bad_vertex_links(faces, len(vertices))
    closed_two_manifold = bool(
        edge_histogram and set(edge_histogram) == {"2"} and bad_vertices == 0
    )
    envelope_min, envelope_max = _envelope_from_geometry(geometry, affine, np)
    coordinate_min, coordinate_max = vertices.min(axis=0), vertices.max(axis=0)
    tolerance = max(1e-9, float(np.max(np.abs(envelope_max))) * 1e-12)
    inside_envelope = bool(
        np.all(coordinate_min >= envelope_min - tolerance)
        and np.all(coordinate_max <= envelope_max + tolerance)
    )
    metrics = mesh["mesh_metrics"]
    source_volume = float(
        expected_voxel_count * abs(np.linalg.det(np.asarray(affine)[:3, :3]))
    )
    relative_volume_error = abs(signed_volume - source_volume) / source_volume
    _require(
        payload["vertex_count"] == metrics.get("vertex_count")
        and payload["triangle_count"] == metrics.get("triangle_count"),
        f"{mesh_path.name} element counts differ from compiler receipt",
    )
    _require(
        expected_voxel_count == metrics.get("occupied_voxel_count"),
        f"{mesh_path.name} source voxel count differs from intake",
    )
    _require(
        inside_envelope, f"{mesh_path.name} vertices leave the source voxel envelope"
    )
    _require(
        relative_volume_error <= 1e-9,
        f"{mesh_path.name} signed volume differs from source voxel occupancy",
    )
    _require(
        abs(signed_volume - float(metrics.get("signed_volume_mm3")))
        <= max(1e-9, abs(signed_volume) * 1e-12),
        f"{mesh_path.name} signed volume differs from compiler measurement",
    )
    _require(
        edge_histogram == metrics.get("edge_incidence", {}).get("incidence_histogram")
        and bad_vertices == metrics.get("nonmanifold_vertex_count")
        and closed_two_manifold == metrics.get("closed_two_manifold"),
        f"{mesh_path.name} independently measured topology differs from compiler receipt",
    )
    result = {
        "mesh_file": mesh_path.name,
        "mesh_sha256": digest,
        "vertex_count": payload["vertex_count"],
        "triangle_count": payload["triangle_count"],
        "all_coordinates_finite": True,
        "all_indices_in_range": True,
        "all_triangles_have_positive_area": True,
        "vertices_within_source_voxel_envelope": inside_envelope,
        "independent_edge_incidence_histogram": edge_histogram,
        "independent_nonmanifold_vertex_count": bad_vertices,
        "closed_two_manifold": closed_two_manifold,
        "signed_volume_mm3": signed_volume,
        "source_voxel_occupancy_volume_mm3": source_volume,
        "relative_signed_volume_error": relative_volume_error,
        "compiler_topology_measurements_reproduced": True,
    }
    return result


def _verify_manifest(mesh_directory: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    sums_path = mesh_directory / "SHA256SUMS"
    _require(
        sums_path.is_file() and not sums_path.is_symlink(),
        "mesh SHA256SUMS manifest is missing",
    )
    expected = {"receipt.json": _sha256_bytes(canonical(receipt) + b"\n")}
    for mesh in receipt.get("meshes", []):
        expected[mesh["mesh_file"]] = mesh["mesh_file_sha256"]
    actual: dict[str, str] = {}
    for line in sums_path.read_text(encoding="ascii").splitlines():
        fields = line.split("  ", 1)
        _require(
            len(fields) == 2 and bool(SHA256.fullmatch(fields[0])),
            "checksum manifest line is malformed",
        )
        _require(fields[1] not in actual, "checksum manifest contains duplicate paths")
        actual[fields[1]] = fields[0]
    _require(
        actual == expected, "checksum manifest contents differ from the exact run files"
    )
    for name, digest in expected.items():
        path = mesh_directory / name
        _require(
            path.is_file() and not path.is_symlink() and _sha256_file(path) == digest,
            f"checksum mismatch for {name}",
        )
    disk_files = {path.name for path in mesh_directory.iterdir() if path.is_file()}
    _require(
        disk_files == set(expected) | {"SHA256SUMS"},
        "mesh directory contains unbound files",
    )
    return {
        "valid": True,
        "file_count": len(expected),
        "sha256sums_sha256": _sha256_file(sums_path),
    }


def audit_run(
    *,
    intake_path: Path,
    archive_path: Path,
    trial_plan_path: Path,
    compiler_receipt_path: Path,
    np: Any,
) -> dict[str, Any]:
    intake, intake_raw = _read_json(
        intake_path, "source intake receipt", require_canonical=True
    )
    plan, plan_raw = _read_json(
        trial_plan_path, "preregistered trial plan", require_canonical=True
    )
    receipt, receipt_raw = _read_json(
        compiler_receipt_path, "compiler receipt", require_canonical=True
    )
    _require(
        intake.get("schema") == INTAKE_SCHEMA
        and plan.get("schema") == PLAN_SCHEMA
        and receipt.get("schema") == COMPILER_SCHEMA,
        "input schema differs",
    )
    _require(
        archive_path.is_file() and not archive_path.is_symlink(),
        "source archive is not a regular file",
    )
    intake_sha, archive_sha = _sha256_bytes(intake_raw), _sha256_file(archive_path)
    _require(
        archive_sha == intake.get("source", {}).get("archive_sha256"),
        "source archive SHA-256 differs from intake",
    )
    scan_id = receipt.get("scan", {}).get("scan_id")
    meshes = receipt.get("meshes")
    _require(
        isinstance(meshes, list) and bool(meshes), "compiler receipt has no meshes"
    )
    label_ids = sorted(int(mesh["label_id"]) for mesh in meshes)
    plan_sha = _sha256_bytes(plan_raw)
    _require(
        plan.get("input_intake_receipt_sha256") == intake_sha
        and plan.get("source_archive_sha256") == archive_sha
        and plan.get("scan_id") == scan_id
        and plan.get("label_ids") == label_ids
        and plan.get("compiler_sources_sha256")
        == receipt.get("compiler_sources_sha256")
        and plan.get("runtime") == receipt.get("runtime")
        and receipt.get("source", {}).get("intake_receipt_sha256") == intake_sha
        and receipt.get("source", {}).get("archive_sha256") == archive_sha
        and receipt.get("trial_plan_sha256") == plan_sha,
        "trial plan, intake, archive, and compiler receipt identities do not join",
    )
    scans = {row["scan_id"]: row for row in intake.get("scans", [])}
    labels = {int(row["label_id"]): row for row in intake.get("labels", [])}
    scan = scans.get(scan_id)
    _require(
        isinstance(scan, dict)
        and receipt.get("scan", {}).get("nifti_member") == scan.get("nifti_member")
        and receipt.get("scan", {}).get("nifti_uncompressed_sha256")
        == scan.get("nifti_uncompressed_sha256")
        and _hash_nifti_from_archive(archive_path, scan["nifti_member"])
        == scan["nifti_uncompressed_sha256"],
        "selected NIfTI member identity differs from intake",
    )
    mesh_directory = Path(compiler_receipt_path).parent
    manifest = _verify_manifest(mesh_directory, receipt)
    affine = scan["voxel_to_world_affine"]
    mesh_results = []
    for mesh in meshes:
        label_id = int(mesh["label_id"])
        label = labels.get(label_id)
        geometry = scan.get("label_spatial_geometry_candidates", {}).get(str(label_id))
        _require(
            isinstance(label, dict) and isinstance(geometry, dict),
            f"label {label_id} is absent from the pinned intake",
        )
        _require(
            mesh.get("source_label_name") == label.get("name")
            and mesh.get("source_semantic_id")
            == f"{intake['source']['source_id']}:segmentation-label:{label_id}",
            f"label {label_id} semantic identity differs from intake",
        )
        voxel_count = int(scan["label_voxel_counts"].get(str(label_id), 0))
        _require(
            voxel_count > 0 and int(geometry.get("voxel_count", -1)) == voxel_count,
            f"label {label_id} voxel count differs from intake",
        )
        mesh_path = mesh_directory / mesh["mesh_file"]
        _require(
            mesh_path.parent == mesh_directory and mesh_path.name == mesh["mesh_file"],
            "mesh path escapes the run directory",
        )
        result = _audit_mesh(mesh_path, mesh, geometry, affine, voxel_count, np)
        result.update(
            {
                "label_id": label_id,
                "label_name": label["name"],
                "source_voxel_count": voxel_count,
            }
        )
        mesh_results.append(result)
    _require(
        len(mesh_results) == len(label_ids), "mesh identity count changed during audit"
    )
    all_geometry_checks_pass = all(
        row["all_coordinates_finite"]
        and row["all_indices_in_range"]
        and row["all_triangles_have_positive_area"]
        and row["vertices_within_source_voxel_envelope"]
        and row["relative_signed_volume_error"] <= 1e-9
        and row["compiler_topology_measurements_reproduced"]
        for row in mesh_results
    )
    return {
        "schema": SCHEMA,
        "status": "independent_source_and_mesh_geometry_checks_passed_topology_defects_retained",
        "intake_receipt_sha256": intake_sha,
        "source_archive_sha256": archive_sha,
        "trial_plan_sha256": plan_sha,
        "compiler_receipt_sha256": _sha256_bytes(receipt_raw),
        "compiler": receipt["compiler"],
        "compiler_sources_sha256": receipt["compiler_sources_sha256"],
        "runtime": receipt["runtime"],
        "scan_id": scan_id,
        "source_nifti_member": scan["nifti_member"],
        "source_nifti_uncompressed_sha256": scan["nifti_uncompressed_sha256"],
        "checksum_manifest": manifest,
        "mesh_count": len(mesh_results),
        "closed_two_manifold_mesh_count": sum(
            row["closed_two_manifold"] for row in mesh_results
        ),
        "nonmanifold_meshes": [
            row["label_name"] for row in mesh_results if not row["closed_two_manifold"]
        ],
        "all_independent_source_and_mesh_geometry_checks_pass": bool(
            all_geometry_checks_pass
        ),
        "meshes": mesh_results,
        "qualification": {
            "source_archive_and_nifti_hashes_reverified": True,
            "serialized_ply_payloads_independently_reparsed": True,
            "mesh_topology_recomputed_without_surface_compiler": True,
            "source_occupancy_volume_reproduced_from_intake_voxels_and_affine": True,
            "cross_scan_registration": False,
            "expert_segmentation_accuracy": False,
            "current_numi_subject_binding": False,
            "physical_tissue_volume_owner": False,
            "mechanics_or_physiology": False,
            "clinical_anatomy": False,
        },
        "boundary": (
            "Independent archive/NIfTI identity, serialized PLY integrity, source-envelope, "
            "topology, and voxel-occupancy volume audit. A passing audit validates these "
            "source geometry artifacts only; automatic segmentation accuracy, cross-scan "
            "registration, subject binding, physical ownership, mechanics, physiology, and "
            "clinical anatomy remain unqualified."
        ),
    }


def run(arguments: argparse.Namespace) -> int:
    from .healthy_total_body_ct_source import _load_numpy

    np = _load_numpy()
    receipt_paths = list(arguments.receipt)
    plan_paths = list(arguments.plan)
    _require(
        len(receipt_paths) == len(plan_paths) and bool(receipt_paths),
        "provide one trial plan for each compiler receipt",
    )
    report = {
        "schema": SCHEMA,
        "runs": [
            audit_run(
                intake_path=arguments.intake,
                archive_path=arguments.archive,
                trial_plan_path=plan,
                compiler_receipt_path=receipt,
                np=np,
            )
            for receipt, plan in zip(receipt_paths, plan_paths, strict=True)
        ],
    }
    scans = {row["scan_id"]: row for row in report["runs"]}
    all_label_names = sorted(
        set.intersection(
            *[{row["label_name"] for row in scan["meshes"]} for scan in report["runs"]]
        )
    )
    report["cross_scan_comparison"] = {
        "registered_same_scan_frame": False,
        "shared_label_count": len(all_label_names),
        "labels": [
            {
                "label_name": name,
                "per_scan_topology": {
                    scan_id: next(
                        row
                        for row in scans[scan_id]["meshes"]
                        if row["label_name"] == name
                    )["closed_two_manifold"]
                    for scan_id in sorted(scans)
                },
            }
            for name in all_label_names
        ],
        "boundary": "Per-scan topology only; no pointwise or subject-frame comparison is performed.",
    }
    report["all_independent_source_and_mesh_geometry_checks_pass"] = all(
        row["all_independent_source_and_mesh_geometry_checks_pass"]
        for row in report["runs"]
    )
    output = Path(arguments.output)
    _require(
        not output.exists() and not output.is_symlink(),
        "output exists; reports are immutable",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(report) + b"\n"
    output.write_bytes(raw)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "sha256": _sha256_bytes(raw),
                "scan_ids": [row["scan_id"] for row in report["runs"]],
                "independently_checked_meshes": sum(
                    row["mesh_count"] for row in report["runs"]
                ),
                "all_independent_source_and_mesh_geometry_checks_pass": report[
                    "all_independent_source_and_mesh_geometry_checks_pass"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--intake", type=Path, required=True, help="pinned CT source intake receipt"
    )
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="registered TCIA segmentation archive",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        nargs="+",
        required=True,
        help="one or more compiler receipts, each beside its mesh files",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        nargs="+",
        required=True,
        help="one preregistered surface plan for each compiler receipt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new immutable independent audit JSON",
    )
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    arguments = parser.parse_args(argv)
    try:
        return run(arguments)
    except (
        HumanImportError,
        OSError,
        TypeError,
        ValueError,
        KeyError,
        UnicodeError,
    ) as error:
        print(f"healthy total-body CT surface audit: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
