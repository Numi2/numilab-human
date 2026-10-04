"""Build voxel-exact volume geometry for a pinned whole-body CT organ label set."""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
import zipfile

from . import healthy_total_body_ct_source as source
from . import healthy_total_body_ct_voxel_tets as voxel
from .model import ImportError as HumanImportError

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.external-segmentation-organ-voxel-tet-volume-candidates.v2"
PLAN_SCHEMA = "numi.healthy-total-body-ct-organ-voxel-tets-plan.v2"
SLUG = re.compile(r"[^a-z0-9]+")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("healthy total-body CT organ voxel tetrahedra: " + message)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _linear_indices_to_voxels(linear_indices: Any, dimensions: list[int]) -> set[tuple[int, int, int]]:
    size_i, size_j, size_k = dimensions
    count = size_i * size_j * size_k
    voxels = set()
    for raw_index in linear_indices:
        index = int(raw_index)
        require(0 <= index < count, "source NIfTI label index is out of bounds")
        i = index % size_i
        j = (index // size_i) % size_j
        k = index // (size_i * size_j)
        voxels.add((i, j, k))
    return voxels


def _read_source_label_indices(*, archive_path: Path, member_name: str,
                               expected_nifti_sha256: str,
                               expected_counts: dict[int, int],
                               allowed_source_label_ids: set[int]) -> tuple[dict[int, list[Any]], list[int]]:
    """Read selected label indices directly from the registered source NIfTI."""
    np = source._load_numpy()
    chunks: dict[int, list[Any]] = {label_id: [] for label_id in expected_counts}
    digest = hashlib.sha256()
    data_bytes = 0
    with zipfile.ZipFile(archive_path) as archive:
        require(member_name in archive.namelist(), "registered scan member is missing from source archive")
        with archive.open(member_name, "r") as member:
            with gzip.GzipFile(fileobj=member, mode="rb") as image:
                header = image.read(source.NIFTI_HEADER_BYTES)
                require(len(header) == source.NIFTI_HEADER_BYTES, "source NIfTI header is truncated")
                digest.update(header)
                info = source.parse_nifti_header(header)
                dimensions = info["shape_ijk"]
                size_i, size_j, _ = dimensions
                data_bytes_expected = info["voxel_count"] * (info["bitpix"] // 8)
                dtype = "<f4" if info["byte_order"] == "little" else ">f4"
                while True:
                    block = image.read(source.NIFTI_CHUNK_BYTES)
                    if not block:
                        break
                    require(len(block) % 4 == 0, "source NIfTI ends within a float32 voxel")
                    first_voxel = data_bytes // 4
                    data_bytes += len(block)
                    require(data_bytes <= data_bytes_expected, "source NIfTI has excess voxel data")
                    digest.update(block)
                    values = np.frombuffer(block, dtype=dtype)
                    if info["label_scaling_slope"] != 1.0 or info["label_scaling_intercept"] != 0.0:
                        values = values * info["label_scaling_slope"] + info["label_scaling_intercept"]
                    require(bool(np.isfinite(values).all()), "source NIfTI contains nonfinite labels")
                    rounded = np.rint(values)
                    require(bool(np.equal(values, rounded).all()), "source NIfTI contains noninteger labels")
                    observed = {int(label_id) for label_id in np.unique(rounded)}
                    require(observed <= allowed_source_label_ids | {0},
                            "source NIfTI contains labels outside the registered source dictionary")
                    for label_id in (observed - {0}) & set(expected_counts):
                        local = np.flatnonzero(rounded == label_id)
                        if local.size:
                            linear = local.astype(np.uint64, copy=False) + first_voxel
                            chunks[label_id].append(linear.astype(np.uint32, copy=False))
                require(data_bytes == data_bytes_expected,
                        "source NIfTI voxel payload length disagrees with its header")
    require(digest.hexdigest() == expected_nifti_sha256,
            "uncompressed source NIfTI hash differs from the intake receipt")
    return chunks, dimensions


def compile_candidates(*, plan_path: Path, output_dir: Path) -> dict[str, Any]:
    plan_path, output_dir = Path(plan_path), Path(output_dir)
    plan = voxel._read_json(plan_path)
    require(plan.get("schema") == PLAN_SCHEMA and plan.get("status") == "preregistered",
            "unsupported or used organ volume plan")
    require(plan.get("method") == "source_nifti_label_voxels_freudenthal_6tet",
            "unsupported organ volume method")
    require(plan.get("maximum_source_voxels_per_label") == 1_800_000,
            "plan source-label voxel bound differs from the compiler")
    require(plan.get("maximum_total_source_voxels") == 4_100_000,
            "plan total source voxel bound differs from the compiler")
    source_hashes = plan.get("instrument_sources")
    require(isinstance(source_hashes, dict) and source_hashes, "plan does not bind compiler sources")
    for relative, expected in source_hashes.items():
        require(voxel._sha_file(voxel._root_path(relative)) == expected,
                f"compiler source hash drifted: {relative}")
    require(source_hashes.get("src/numilab_human/healthy_total_body_ct_organ_voxel_tets.py") ==
            voxel._sha_file(Path(__file__)), "plan does not bind this organ compiler")
    require(source_hashes.get("src/numilab_human/healthy_total_body_ct_voxel_tets.py") ==
            voxel._sha_file(Path(voxel.__file__)), "plan does not bind the shared voxel-mesh kernel")
    require(not output_dir.is_symlink(), "output directory is redirected")

    intake_path = voxel._root_path(plan["intake_receipt_path"])
    surface_receipt_path = voxel._root_path(plan["surface_receipt_path"])
    archive_path = voxel._root_path(plan["source_archive_path"])
    require(voxel._sha_file(intake_path) == plan["intake_receipt_sha256"], "source intake hash drifted")
    require(voxel._sha_file(surface_receipt_path) == plan["surface_receipt_sha256"],
            "whole-body organ surface receipt hash drifted")
    intake = voxel._read_json(intake_path)
    surface_receipt = voxel._read_json(surface_receipt_path)
    source_config_path = voxel._root_path(plan["source_config_path"])
    require(voxel._sha_file(source_config_path) == plan["source_config_sha256"],
            "registered source configuration hash drifted")
    source_config = voxel._read_json(source_config_path)
    require(source_config.get("source_id") == "healthy_total_body_cts_v3" and
            source_config.get("archive_sha256") == plan.get("source_archive_sha256") and
            archive_path.stat().st_size == source_config.get("archive_bytes") and
            voxel._sha_file(archive_path) == plan.get("source_archive_sha256"),
            "licensed source archive does not match its pinned registration")
    require(intake.get("schema") == "HumanPack.external-segmentation-source-ingest.v2" and
            surface_receipt.get("schema") == "HumanPack.external-segmentation-voxel-surface-candidates.v2",
            "unsupported whole-body CT intake or surface receipt")
    require(surface_receipt.get("qualification", {}).get("all_selected_meshes_closed_two_manifolds") is True and
            surface_receipt.get("qualification", {}).get("selected_voxel_counts_match_intake") is True,
            "whole-body source surfaces are not closed and voxel-count bound")
    require(surface_receipt.get("source", {}).get("archive_sha256") == plan.get("source_archive_sha256"),
            "surface receipt does not bind the pinned source archive")
    scan_id = plan["scan_id"]
    surface_scan = surface_receipt.get("scan", {})
    scan = next((row for row in intake.get("scans", []) if row.get("scan_id") == scan_id), None)
    require(scan is not None and surface_scan.get("scan_id") == scan_id,
            "intake and surface receipt do not identify the same scan")
    require(surface_scan.get("nifti_uncompressed_sha256") == scan.get("nifti_uncompressed_sha256"),
            "surface receipt does not bind this scan's exact source NIfTI")
    matrix, voxel_volume_mm3 = voxel._matrix(scan)
    require(matrix == [[float(x) for x in row] for row in surface_scan["voxel_to_world_affine_ras_mm"]],
            "source intake and organ surface affine disagree")
    surface_rows = {row["label_id"]: row for row in surface_receipt.get("meshes", [])}
    label_counts = scan.get("label_voxel_counts", {})
    labels = plan.get("labels")
    require(isinstance(labels, list) and len(labels) == plan.get("expected_label_count"),
            "plan organ-label count is invalid")
    expected_total = sum(int(row["source_voxel_count"]) for row in labels)
    require(expected_total <= plan["maximum_total_source_voxels"] and
            all(int(row["source_voxel_count"]) <= plan["maximum_source_voxels_per_label"] for row in labels),
            "pinned source voxel count exceeds the registered compiler resource bound")
    expected_counts = {int(row["label_id"]): int(row["source_voxel_count"]) for row in labels}
    allowed_source_label_ids = {int(label_id) for label_id in label_counts}
    index_chunks, dimensions = _read_source_label_indices(
        archive_path=archive_path, member_name=scan["nifti_member"],
        expected_nifti_sha256=scan["nifti_uncompressed_sha256"], expected_counts=expected_counts,
        allowed_source_label_ids=allowed_source_label_ids)
    require(dimensions == [int(value) for value in source_config["expected_shape_ijk"]],
            "source NIfTI dimensions differ from the registered release")
    np = source._load_numpy()

    meshes = []
    # Mesh the largest mask first and release each point/voxel field before
    # moving on to the next label.
    for label in sorted(labels, key=lambda row: (-row["source_voxel_count"], row["label_id"])):
        label_id = label["label_id"]
        source_row = surface_rows.get(label_id)
        require(source_row is not None and source_row.get("source_label_name") == label["name"],
                f"source label identity differs: {label_id}")
        require(source_row.get("mesh_file") == label["mesh_file"] and
                source_row.get("mesh_file_sha256") == label["mesh_sha256"] and
                source_row.get("source_voxel_count") == label["source_voxel_count"] and
                source_row.get("source_semantic_id") == label["source_semantic_id"],
                f"source mesh receipt row differs: {label['name']}")
        require(source_row.get("source_nifti_uncompressed_sha256") == scan["nifti_uncompressed_sha256"] and
                int(label_counts.get(str(label_id), -1)) == label["source_voxel_count"],
                f"source NIfTI voxel count differs: {label['name']}")
        metrics = source_row.get("mesh_metrics", {})
        require(metrics.get("closed_two_manifold") is True and
                metrics.get("edge_incidence", {}).get("all_edges_have_two_incident_triangles") is True and
                metrics.get("nonmanifold_vertex_count") == 0,
                f"source surface is not a closed two-manifold: {label['name']}")
        surface_path = surface_receipt_path.parent / source_row["mesh_file"]
        require(surface_path.is_file() and not surface_path.is_symlink() and
                voxel._sha_file(surface_path) == label["mesh_sha256"],
                f"source surface file hash differs: {label['name']}")
        vertices, faces = voxel._ply(surface_path)
        require(len(faces) == metrics.get("triangle_count") and
                len(vertices) == metrics.get("vertex_count"),
                f"source surface counts differ: {label['name']}")
        surface_volume_mm3 = voxel._signed_surface_volume(vertices, faces)
        require(surface_volume_mm3 > 0.0 and math.isfinite(surface_volume_mm3),
                f"source surface winding or volume is invalid: {label['name']}")
        chunks = index_chunks.pop(label_id)
        linear_indices = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint32)
        del chunks
        voxels = _linear_indices_to_voxels(linear_indices, dimensions)
        del linear_indices
        require(len(voxels) == label["source_voxel_count"],
                f"direct source NIfTI voxel count differs from the intake: {label['name']}")
        source_boundary, source_triangles = voxel._surface_boundary_cells(vertices, faces, matrix)
        mask_triangles = voxel._voxel_boundary_triangles(voxels)
        require(mask_triangles == source_triangles,
                f"direct source mask boundary differs from the registered surface: {label['name']}")
        require(source_boundary <= voxels, f"registered source surface exceeds the direct mask: {label['name']}")
        raster_volume_mm3 = len(voxels) * voxel_volume_mm3
        slug = SLUG.sub("-", label["name"].casefold()).strip("-")
        output_path = output_dir / f"scan-{scan_id}-label-{label_id:03d}-{slug}.vtk.gz"
        mesh = voxel._write_vtk(output_path, voxels, matrix)
        expected_volume_m3 = len(voxels) * voxel_volume_mm3 * 1e-9
        volume_error = abs(mesh["tetrahedron_signed_volume_sum_m3"] - expected_volume_m3) / expected_volume_m3
        require(volume_error <= plan["maximum_tetrahedral_volume_relative_error"] and
                mesh["tetrahedron_count"] == 6 * len(voxels) and
                mesh["tetrahedron_volume_min_m3"] > 0.0,
                f"tetrahedral volume gate failed: {label['name']}")
        mesh.update({
            "scan_id": scan_id,
            "label_id": label_id,
            "source_label_name": label["name"],
            "source_semantic_id": source_row["source_semantic_id"],
            "source_voxel_count": len(voxels),
            "source_voxel_volume_m3": expected_volume_m3,
            "source_surface_volume_mm3": surface_volume_mm3,
            "source_raster_volume_mm3": raster_volume_mm3,
            "source_surface_volume_relative_difference": abs(raster_volume_mm3 - surface_volume_mm3) / surface_volume_mm3,
            "tetrahedral_volume_relative_error": volume_error,
            "source_surface_path": _relative(surface_path),
            "source_surface_sha256": label["mesh_sha256"],
            "source_boundary_triangle_count": sum(source_triangles.values()),
            "direct_mask_boundary_triangle_count": sum(mask_triangles.values()),
            "exact_source_surface_and_mask_boundary_triangle_multisets": True,
            "source_nifti_uncompressed_sha256": scan["nifti_uncompressed_sha256"],
            "all_tetrahedra_positive": mesh["tetrahedron_volume_min_m3"] > 0.0,
        })
        meshes.append(mesh)
        del vertices, faces, source_boundary, source_triangles, mask_triangles, voxels
        gc.collect()

    result = {
        "schema": SCHEMA,
        "status": "geometry_candidates_passed",
        "compiler": "numilab-human.healthy-total-body-ct-organ-voxel-tets.2",
        "compiler_source_sha256": voxel._sha_file(Path(__file__)),
        "shared_voxel_kernel_source_sha256": voxel._sha_file(Path(voxel.__file__)),
        "plan_path": _relative(plan_path),
        "plan_sha256": voxel._sha_file(plan_path),
        "source_intake_path": _relative(intake_path),
        "source_intake_sha256": voxel._sha_file(intake_path),
        "source_surface_receipt_path": _relative(surface_receipt_path),
        "source_surface_receipt_sha256": voxel._sha_file(surface_receipt_path),
        "source_archive_path": _relative(archive_path),
        "source_archive_sha256": plan["source_archive_sha256"],
        "source_nifti_member": scan["nifti_member"],
        "source_nifti_uncompressed_sha256": scan["nifti_uncompressed_sha256"],
        "scan_id": scan_id,
        "meshes": meshes,
        "qualification": {
            "source_intake_receipt_and_surface_hashes_bound": True,
            "source_voxel_counts_read_directly_from_hash_bound_nifti": True,
            "exact_source_mask_boundary_triangle_multisets": True,
            "all_selected_registered_surfaces_match_source_mask_boundaries": True,
            "six_positive_tetrahedra_per_source_voxel": True,
            "voxel_occupancy_volume_closed": True,
            "expert_segmentation_review": False,
            "independent_participant_replication": False,
            "numi_subject_binding": False,
            "organ_physical_volume_owners": False,
            "material_or_mass_ownership": False,
            "mechanics_admitted": False,
            "vascular_lumens_or_perfusion": False,
            "physiological_or_clinical_validation": False,
        },
        "boundary": (
            "Named whole-body CT labels are represented by scan-specific "
            "voxel-exact volume geometry. These automatic segmentations are not "
            "expert-reviewed or bound to the Numi Human subject; vessel masks do "
            "not define lumens, and no tissue mechanics, material, mass, or "
            "physiology is assigned."
        ),
    }
    receipt_path = output_dir / "receipt.json"
    result["receipt_sha256"] = voxel._immutable_json(receipt_path, result)
    return result


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    try:
        result = compile_candidates(plan_path=arguments.plan, output_dir=arguments.output)
        print(json.dumps({"schema": SCHEMA, "receipt_sha256": result["receipt_sha256"],
                          "scan_id": result["scan_id"], "labels": len(result["meshes"]),
                          "source_voxels": sum(row["source_voxel_count"] for row in result["meshes"]),
                          "tetrahedra": sum(row["tetrahedron_count"] for row in result["meshes"]),
                          "all_source_boundaries_exact": result["qualification"]["all_selected_registered_surfaces_match_source_mask_boundaries"],
                          "mechanics_admitted": False}, sort_keys=True))
        return 0
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"healthy total-body CT organ voxel tetrahedra: {error}")
        return 2
