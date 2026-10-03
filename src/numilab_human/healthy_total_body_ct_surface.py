"""Build exact voxel-boundary meshes from source-labelled CT masks.

Meshes are lossless cubical boundaries in each scan's own NIfTI RAS+ frame.
They are geometry candidates, not expert-reviewed or mechanics-ready anatomy.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, BinaryIO
import zipfile

from .healthy_total_body_ct_source import (
    NIFTI_HEADER_BYTES,
    _load_numpy,
    parse_nifti_header,
)
from .model import ImportError as HumanImportError
from .physiology import canonical


SCHEMA = "HumanPack.external-segmentation-voxel-surface-candidates.v2"
COMPILER = "numilab-human.healthy-total-body-ct-surface.2"
INTAKE_SCHEMA = "HumanPack.external-segmentation-source-ingest.v2"
PLAN_SCHEMA = "numi.healthy-total-body-ct-surface-trial-plan.v2"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COPY_CHUNK_BYTES = 16 * 1024 * 1024
COMPILER_SOURCE_FILES = (
    "healthy_total_body_ct_surface.py",
    "healthy_total_body_ct_source.py",
    "physiology.py",
    "cli.py",
)

# Each row gives an outward-oriented quad's four offsets from its voxel centre
# in right-handed IJK coordinates. A reflected NIfTI affine reverses winding.
_FACE_CORNERS = (
    (0, 1, ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1))),
    (0, -1, ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1))),
    (1, 1, ((-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1))),
    (1, -1, ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1))),
    (2, 1, ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))),
    (2, -1, ((-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1))),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("healthy total-body CT surface: " + message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _edge_incidence(triangles: Any, np: Any) -> dict[str, Any]:
    edges = np.concatenate((triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]))
    edges.sort(axis=1)
    _, incidence = np.unique(edges, axis=0, return_counts=True)
    values, counts = np.unique(incidence, return_counts=True)
    histogram = {str(int(value)): int(count) for value, count in zip(values, counts, strict=True)}
    return {
        "unique_edge_count": int(incidence.size),
        "incidence_histogram": histogram,
        "all_edges_have_two_incident_triangles": bool(incidence.size and np.all(incidence == 2)),
    }


def _nonmanifold_vertex_count(triangles: Any, vertex_count: int) -> int:
    """Count vertices whose triangle link is not one connected cycle."""
    links: list[list[tuple[int, int]]] = [[] for _ in range(vertex_count)]
    for a, b, c in triangles:
        a, b, c = int(a), int(b), int(c)
        links[a].append((b, c))
        links[b].append((c, a))
        links[c].append((a, b))
    invalid = 0
    for link_edges in links:
        adjacency: dict[int, list[int]] = {}
        for left, right in link_edges:
            adjacency.setdefault(left, []).append(right)
            adjacency.setdefault(right, []).append(left)
        if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
            invalid += 1
            continue
        first = next(iter(adjacency))
        visited = {first}
        stack = [first]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        if len(visited) != len(adjacency):
            invalid += 1
    return invalid


def build_voxel_boundary_mesh(
    binary_mask: Any,
    affine: list[list[float]],
    *,
    voxel_offset_ijk: tuple[int, int, int] = (0, 0, 0),
    np: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Mesh the exact union of occupied voxels without smoothing or decimation."""
    if np is None:
        np = _load_numpy()
    mask = np.asarray(binary_mask, dtype=np.bool_)
    require(mask.ndim == 3 and bool(mask.any()), "mask must be a nonempty 3-D array")
    require(len(affine) == 4 and all(len(row) == 4 for row in affine),
            "voxel-to-world affine must be 4 by 4")
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1, 1:-1]
    face_corners = []
    exposed_face_count = 0
    for axis, direction, corner_offsets in _FACE_CORNERS:
        neighbor_slice = [slice(1, -1)] * 3
        neighbor_slice[axis] = slice(2, None) if direction > 0 else slice(None, -2)
        exposed = center & ~padded[tuple(neighbor_slice)]
        centers = np.argwhere(exposed).astype(np.int32, copy=False)
        exposed_face_count += int(centers.shape[0])
        if not centers.size:
            continue
        offset = np.asarray(voxel_offset_ijk, dtype=np.int32)
        global_centers = centers + offset
        corners = 2 * global_centers[:, None, :] + np.asarray(corner_offsets, dtype=np.int32)[None, :, :]
        face_corners.append(corners.reshape(-1, 3))
    require(exposed_face_count > 0 and face_corners, "mask has no exposed voxel faces")

    half_grid_vertices, inverse = np.unique(
        np.concatenate(face_corners, axis=0), axis=0, return_inverse=True,
    )
    quads = inverse.reshape(exposed_face_count, 4).astype(np.int64, copy=False)
    triangles = np.empty((exposed_face_count * 2, 3), dtype=np.int64)
    triangles[0::2] = quads[:, (0, 1, 2)]
    triangles[1::2] = quads[:, (0, 2, 3)]

    linear = np.asarray([row[:3] for row in affine[:3]], dtype=np.float64)
    translation = np.asarray([row[3] for row in affine[:3]], dtype=np.float64)
    determinant = float(np.linalg.det(linear))
    require(math.isfinite(determinant) and abs(determinant) > 0.0,
            "voxel-to-world affine is singular")
    if determinant < 0.0:
        triangles[:, (1, 2)] = triangles[:, (2, 1)]
    vertices_ras = (half_grid_vertices.astype(np.float64) * 0.5) @ linear.T + translation

    triangle_points = vertices_ras[triangles]
    edge_a = triangle_points[:, 1] - triangle_points[:, 0]
    edge_b = triangle_points[:, 2] - triangle_points[:, 0]
    triangle_areas = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    area_mm2 = float(triangle_areas.sum(dtype=np.float64))
    origin = vertices_ras.mean(axis=0)
    relative_points = triangle_points - origin
    signed_volume_mm3 = float(np.einsum(
        "ij,ij->i",
        relative_points[:, 0],
        np.cross(relative_points[:, 1], relative_points[:, 2]),
    ).sum(dtype=np.float64) / 6.0)
    edges = _edge_incidence(triangles, np)
    nonmanifold_vertices = _nonmanifold_vertex_count(triangles, len(vertices_ras))
    metrics = {
        "occupied_voxel_count": int(mask.sum(dtype=np.int64)),
        "exposed_voxel_face_count": exposed_face_count,
        "vertex_count": int(len(vertices_ras)),
        "triangle_count": int(len(triangles)),
        "surface_area_mm2": area_mm2,
        "signed_volume_mm3": signed_volume_mm3,
        "edge_incidence": edges,
        "nonmanifold_vertex_count": nonmanifold_vertices,
        "closed_two_manifold": bool(
            edges["all_edges_have_two_incident_triangles"] and nonmanifold_vertices == 0
        ),
        "affine_determinant_mm3_per_voxel": abs(determinant),
    }
    return vertices_ras, triangles, metrics


def _write_binary_ply_gzip(path: Path, vertices: Any, triangles: Any, np: Any) -> None:
    vertex_dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
    vertex_records = np.empty(len(vertices), dtype=vertex_dtype)
    vertex_records["x"], vertex_records["y"], vertex_records["z"] = vertices.T
    face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
    face_records = np.empty(len(triangles), dtype=face_dtype)
    face_records["count"] = 3
    require(len(vertices) < 2**31 and (not len(triangles) or int(triangles.max()) < 2**31),
            "PLY index range exceeds signed 32-bit format")
    face_records["indices"] = triangles.astype(np.int32, copy=False)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property double x\nproperty double y\nproperty double z\n"
        f"element face {len(triangles)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    partial = path.with_name(path.name + ".partial")
    try:
        with partial.open("xb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as stream:
                stream.write(header)
                stream.write(vertex_records.tobytes(order="C"))
                stream.write(face_records.tobytes(order="C"))
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _nifti_payload_to_file(
    member_stream: BinaryIO,
    target: Path,
    *,
    member_name: str,
) -> tuple[dict[str, Any], str]:
    digest = hashlib.sha256()
    with gzip.GzipFile(fileobj=member_stream, mode="rb") as image:
        header = image.read(NIFTI_HEADER_BYTES)
        require(len(header) == NIFTI_HEADER_BYTES, f"{member_name} has a truncated NIfTI header")
        digest.update(header)
        info = parse_nifti_header(header)
        expected = int(info["voxel_count"]) * 4
        payload_count = 0
        with target.open("xb") as payload:
            while True:
                block = image.read(COPY_CHUNK_BYTES)
                if not block:
                    break
                payload_count += len(block)
                require(payload_count <= expected, f"{member_name} has excess voxel payload")
                digest.update(block)
                payload.write(block)
        require(payload_count == expected, f"{member_name} voxel payload length is invalid")
    return info, digest.hexdigest()


def _compiler_source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {
        name: _sha256_file(package / name)
        for name in COMPILER_SOURCE_FILES
    }


def _runtime_info(np: Any) -> dict[str, str]:
    return {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
    }


def _read_trial_plan(
    path: Path,
    *,
    receipt_sha256: str,
    source_archive_sha256: str,
    scan_id: str,
    label_ids: list[int],
    compiler_sources_sha256: dict[str, str],
    runtime: dict[str, str],
) -> tuple[dict[str, Any], str]:
    require(path.is_file() and not path.is_symlink(), "trial plan is not a regular file")
    raw = path.read_bytes()
    try:
        plan = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise HumanImportError("healthy total-body CT surface: trial plan is malformed JSON") from error
    require(isinstance(plan, dict) and plan.get("schema") == PLAN_SCHEMA,
            "trial plan schema is unsupported")
    require(plan.get("input_intake_receipt_sha256") == receipt_sha256 and
            plan.get("source_archive_sha256") == source_archive_sha256 and
            plan.get("scan_id") == scan_id and
            plan.get("label_ids") == sorted(label_ids) and
            plan.get("compiler_sources_sha256") == compiler_sources_sha256 and
            plan.get("runtime") == runtime,
            "trial plan does not bind this exact intake, source, scan, labels, and compiler")
    return plan, hashlib.sha256(raw).hexdigest()


def compile_surface_candidates(
    *,
    receipt_path: Path,
    archive_path: Path,
    scan_id: str,
    label_ids: list[int],
    trial_plan_path: Path | None = None,
) -> tuple[dict[str, Any], list[tuple[str, Any, Any]]]:
    np = _load_numpy()
    runtime = _runtime_info(np)
    receipt_path, archive_path = Path(receipt_path), Path(archive_path)
    require(receipt_path.is_file() and not receipt_path.is_symlink(),
            "intake receipt is not a regular file")
    raw_receipt = receipt_path.read_bytes()
    receipt_sha = hashlib.sha256(raw_receipt).hexdigest()
    try:
        intake = json.loads(raw_receipt)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise HumanImportError("healthy total-body CT surface: intake receipt is malformed JSON") from error
    require(isinstance(intake, dict) and intake.get("schema") == INTAKE_SCHEMA,
            "intake receipt schema is unsupported")
    source = intake.get("source")
    require(isinstance(source, dict) and source.get("source_id") == "healthy_total_body_cts_v3",
            "intake does not name the registered TCIA source")
    require(archive_path.is_file() and not archive_path.is_symlink(),
            "registered source archive is not a regular file")
    archive_sha = source.get("archive_sha256")
    require(archive_path.stat().st_size == source.get("archive_bytes") and
            _sha256_file(archive_path) == archive_sha,
            "source archive differs from the pinned intake identity")
    compiler_sources_sha256 = _compiler_source_hashes()
    require(isinstance(scan_id, str) and re.fullmatch(r"\d{3}", scan_id),
            "scan ID must be three digits")
    require(label_ids and all(type(value) is int and value > 0 for value in label_ids) and
            len(label_ids) == len(set(label_ids)), "label IDs are empty, invalid, or duplicated")
    labels = {row["label_id"]: row for row in intake["labels"]}
    scans = {row["scan_id"]: row for row in intake["scans"]}
    require(scan_id in scans and all(label_id in labels for label_id in label_ids),
            "selected scan or label is absent from the intake")
    plan = None
    plan_sha = None
    if trial_plan_path is not None:
        plan, plan_sha = _read_trial_plan(
            Path(trial_plan_path), receipt_sha256=receipt_sha, source_archive_sha256=archive_sha,
            scan_id=scan_id,
            label_ids=label_ids, compiler_sources_sha256=compiler_sources_sha256,
            runtime=runtime,
        )
    scan = scans[scan_id]
    affine = scan["voxel_to_world_affine"]
    geometry_by_label = scan["label_spatial_geometry_candidates"]
    counts_by_label = scan["label_voxel_counts"]
    candidates: list[tuple[str, Any, Any, dict[str, Any]]] = []

    with tempfile.TemporaryDirectory(prefix="numi-ct-surface-") as temporary_dir:
        payload_path = Path(temporary_dir) / "nifti-float32-payload.raw"
        try:
            with zipfile.ZipFile(archive_path) as archive:
                member_name = scan.get("nifti_member")
                require(isinstance(member_name, str) and member_name in archive.namelist(),
                        "selected NIfTI member is missing from the registered archive")
                with archive.open(member_name, "r") as member_stream:
                    info, nifti_sha = _nifti_payload_to_file(
                        member_stream, payload_path, member_name=member_name,
                    )
        except (zipfile.BadZipFile, EOFError, gzip.BadGzipFile) as error:
            raise HumanImportError("healthy total-body CT surface: selected NIfTI archive member is corrupt") from error
        require(nifti_sha == scan.get("nifti_uncompressed_sha256"),
                "selected decompressed NIfTI hash differs from the intake")
        require(info["shape_ijk"] == intake["source"].get("voxel_array_shape_ijk") and
                info["spatial_units"] == "mm" and info["voxel_to_world_affine"] == affine,
                "selected NIfTI header differs from its recorded geometry")
        np_dtype = "<f4" if info["byte_order"] == "little" else ">f4"
        values = np.memmap(payload_path, dtype=np_dtype, mode="r", shape=tuple(info["shape_ijk"]), order="F")
        slope = info["label_scaling_slope"]
        intercept = info["label_scaling_intercept"]
        for label_id in sorted(label_ids):
            expected_count = int(counts_by_label.get(str(label_id), 0))
            geometry = geometry_by_label.get(str(label_id))
            require(expected_count > 0 and isinstance(geometry, dict),
                    f"label {label_id} is absent from scan {scan_id}")
            lower = geometry["voxel_center_bounds_ijk"]["minimum_inclusive"]
            upper = geometry["voxel_center_bounds_ijk"]["maximum_inclusive"]
            shape = info["shape_ijk"]
            crop_lower = [max(0, int(lower[axis]) - 1) for axis in range(3)]
            crop_upper = [min(shape[axis] - 1, int(upper[axis]) + 1) for axis in range(3)]
            region = np.asarray(values[
                crop_lower[0]:crop_upper[0] + 1,
                crop_lower[1]:crop_upper[1] + 1,
                crop_lower[2]:crop_upper[2] + 1,
            ])
            scaled = region.astype(np.float64, copy=False) * float(slope) + float(intercept)
            require(bool(np.isfinite(scaled).all()) and
                    bool(np.equal(scaled, np.rint(scaled)).all()),
                    f"scan {scan_id} crop for label {label_id} contains invalid labels")
            binary = np.equal(np.rint(scaled), label_id)
            voxel_count = int(binary.sum(dtype=np.int64))
            require(voxel_count == expected_count,
                    f"scan {scan_id} label {label_id} cropped voxel count differs from intake")
            vertices, triangles, metrics = build_voxel_boundary_mesh(
                binary, affine, voxel_offset_ijk=tuple(crop_lower), np=np,
            )
            expected_volume_mm3 = voxel_count * float(info["voxel_volume_in_source_units_cubed"])
            volume_error = abs(metrics["signed_volume_mm3"] - expected_volume_mm3) / expected_volume_mm3
            require(volume_error <= 1.0e-9,
                    f"scan {scan_id} label {label_id} mesh volume differs from source occupancy")
            label = labels[label_id]
            safe_name = re.sub(r"[^a-z0-9]+", "-", label["name"].casefold()).strip("-")
            mesh_name = f"scan-{scan_id}-label-{label_id:03d}-{safe_name}-boundary.ply.gz"
            metrics.update({
                "source_voxel_occupancy_volume_candidate_mm3": expected_volume_mm3,
                "relative_signed_volume_error": volume_error,
                "surface_area_per_occupancy_volume_mm_inv": metrics["surface_area_mm2"] / expected_volume_mm3,
            })
            candidates.append((mesh_name, vertices, triangles, {
                "label_id": label_id,
                "source_semantic_id": f"{source['source_id']}:segmentation-label:{label_id}",
                "source_label_name": label["name"],
                "source_voxel_count": voxel_count,
                "source_nifti_uncompressed_sha256": nifti_sha,
                "mesh_metrics": metrics,
            }))
        del values

    result = {
        "schema": SCHEMA,
        "compiler": COMPILER,
        "compiler_sources_sha256": compiler_sources_sha256,
        "runtime": runtime,
        "status": "source_voxel_boundary_mesh_candidates",
        "source": {
            "source_id": source["source_id"],
            "release": source["release"],
            "doi": source["doi"],
            "license": source["license"],
            "archive_sha256": archive_sha,
            "intake_receipt_sha256": receipt_sha,
            "label_dictionary_sha256": source["label_dictionary_sha256"],
            "segmentation_generation": source["segmentation_generation"],
        },
        "scan": {
            "scan_id": scan_id,
            "nifti_member": scan["nifti_member"],
            "nifti_uncompressed_sha256": scan["nifti_uncompressed_sha256"],
            "affine_source": scan["affine_source"],
            "voxel_spacing_mm": scan["voxel_spacing_mm"],
            "voxel_to_world_affine_ras_mm": affine,
            "coordinate_convention": "NIfTI RAS+ millimetres; this scan's affine only",
        },
        "trial_plan_sha256": plan_sha,
        "meshes": [candidate[3] for candidate in candidates],
        "qualification": {
            "archive_and_decompressed_nifti_hashes_match_intake": True,
            "header_affine_and_units_match_intake": True,
            "selected_voxel_counts_match_intake": True,
            "mesh_signed_volumes_match_voxel_occupancy_within_1e-9_relative": True,
            "all_selected_meshes_closed_two_manifolds": all(
                candidate[3]["mesh_metrics"]["closed_two_manifold"] for candidate in candidates
            ),
            "cross_scan_registration": False,
            "expert_segmentation_accuracy_qualified": False,
            "current_numi_subject_binding": False,
            "physical_tissue_volume_owner": False,
            "material_or_mechanical_admission": False,
            "physiological_or_clinical_qualification": False,
        },
        "boundary": (
            "Exact boundary of the registered label's occupied voxel union, without "
            "smoothing, interpolation, or decimation. Source labels are automatic; "
            "meshes remain scan-specific geometry candidates and are not mechanics-ready."
        ),
    }
    return result, candidates


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--receipt", type=Path, required=True,
                        help="validated healthy-total-body-ct-source intake receipt")
    parser.add_argument("--archive", type=Path, required=True,
                        help="exact TCIA archive registered by the intake receipt")
    parser.add_argument("--scan-id", required=True, help="three-digit registered scan ID")
    parser.add_argument("--label-ids", type=int, nargs="+", required=True,
                        help="source dictionary IDs to mesh from this scan")
    parser.add_argument("--plan", type=Path,
                        help="optional preregistered plan bound to this receipt, scan, and labels")
    parser.add_argument("--output", type=Path, required=True,
                        help="new output directory; existing directories are not overwritten")
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    output = Path(arguments.output)
    require(not output.exists() and not output.is_symlink(),
            "output directory already exists; choose a new path")
    receipt, candidates = compile_surface_candidates(
        receipt_path=arguments.receipt,
        archive_path=arguments.archive,
        scan_id=arguments.scan_id,
        label_ids=arguments.label_ids,
        trial_plan_path=arguments.plan,
    )
    output.mkdir(parents=True, exist_ok=False)
    np = _load_numpy()
    for mesh_name, vertices, triangles, detail in candidates:
        mesh_path = output / mesh_name
        _write_binary_ply_gzip(mesh_path, vertices, triangles, np)
        detail["mesh_file"] = mesh_name
        detail["mesh_file_sha256"] = _sha256_file(mesh_path)
        detail["mesh_file_bytes"] = mesh_path.stat().st_size
    manifest = canonical(receipt) + b"\n"
    (output / "receipt.json").write_bytes(manifest)
    receipt_sha = hashlib.sha256(manifest).hexdigest()
    sums = []
    for path in sorted(output.iterdir()):
        if path.is_file():
            sums.append(f"{_sha256_file(path)}  {path.name}\n")
    (output / "SHA256SUMS").write_text("".join(sums), encoding="ascii")
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(output.resolve()),
        "receipt_sha256": receipt_sha,
        "mesh_count": len(candidates),
        "all_selected_meshes_closed_two_manifolds": receipt["qualification"][
            "all_selected_meshes_closed_two_manifolds"
        ],
        "cross_scan_registration": False,
        "material_or_mechanical_admission": False,
    }, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    arguments = parser.parse_args(argv)
    try:
        return run(arguments)
    except (HumanImportError, OSError, TypeError, ValueError, KeyError) as error:
        print(f"healthy total-body CT surface: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
