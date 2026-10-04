"""Create and independently audit side-specific CT organ surface candidates.

The operation partitions occupied source voxels by the scan's RAS-X midplane.
It preserves voxel occupancy and disconnected components; it does not repair or
validate the automatic segmentation, register it to Numi Human, or admit it to
mechanics.
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
import tempfile
from typing import Any
import zipfile

from .healthy_total_body_ct_source import (
    NIFTI_HEADER_BYTES,
    _load_numpy,
    parse_nifti_header,
)
from .healthy_total_body_ct_surface import (
    _nifti_payload_to_file,
    _runtime_info,
    _sha256_file,
    _write_binary_ply_gzip,
    build_voxel_boundary_mesh,
    require,
)
from .healthy_total_body_ct_surface_audit import (
    _bad_vertex_links,
    _edge_incidence_histogram,
    _read_binary_ply_gzip,
)
from .model import ImportError as HumanImportError
from .physiology import canonical


PLAN_SCHEMA = "numi.healthy-total-body-ct-ras-x-partition-plan.v1"
RECEIPT_SCHEMA = "numi.healthy-total-body-ct-ras-x-partition.v1"
AUDIT_SCHEMA = "numi.healthy-total-body-ct-ras-x-partition-audit.v1"
INTAKE_SCHEMA = "HumanPack.external-segmentation-source-ingest.v2"
SURFACE_SCHEMA = "HumanPack.external-segmentation-voxel-surface-candidates.v2"
COPY_CHUNK_BYTES = 16 * 1024 * 1024
PLANE_TOLERANCE_MM = 1e-9
SOURCE_FILES = (
    "healthy_total_body_ct_ras_x_partition.py",
    "healthy_total_body_ct_surface.py",
    "healthy_total_body_ct_surface_audit.py",
    "healthy_total_body_ct_source.py",
    "physiology.py",
    "cli.py",
)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    paths = {name: package / name for name in SOURCE_FILES}
    paths["pyproject.toml"] = package.parent.parent / "pyproject.toml"
    return {name: _sha256_file(path) for name, path in paths.items()}


def _read_json(path: Path, label: str, *, require_canonical: bool = False) -> tuple[dict[str, Any], bytes]:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise HumanImportError(f"CT RAS-X partition: {label} is malformed JSON") from error
    require(isinstance(value, dict), f"{label} is not a JSON object")
    if require_canonical:
        require(raw == canonical(value) + b"\n", f"{label} is not canonical JSON")
    return value, raw


def _ras_x_plane(affine: list[list[float]], np: Any) -> tuple[int, float, float]:
    require(len(affine) == 4 and all(len(row) == 4 for row in affine), "affine is not 4 by 4")
    row = np.asarray(affine[0][:3], dtype=np.float64)
    active = np.flatnonzero(np.abs(row) > 1e-12)
    require(active.size == 1, "RAS-X midplane must align with one voxel-grid axis")
    axis = int(active[0])
    coefficient = float(row[axis])
    plane_index = -float(affine[0][3]) / coefficient
    nearest_face = math.floor(plane_index) + 0.5
    require(
        abs(plane_index - nearest_face) <= 1e-9,
        "RAS-X zero plane does not lie on a voxel face",
    )
    return axis, coefficient, plane_index


def _partition_mask_by_ras_x(
    binary: Any,
    affine: list[list[float]],
    voxel_offset_ijk: tuple[int, int, int],
    np: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    axis, coefficient, plane_index = _ras_x_plane(affine, np)
    centers = np.arange(binary.shape[axis], dtype=np.int64) + voxel_offset_ijk[axis]
    ras_x = coefficient * centers.astype(np.float64) + float(affine[0][3])
    require(
        bool(np.all(np.abs(ras_x) > PLANE_TOLERANCE_MM)),
        "selected mask contains voxel centers on the RAS-X partition plane",
    )
    side_shape = [1, 1, 1]
    side_shape[axis] = len(ras_x)
    positive = (ras_x > 0.0).reshape(side_shape)
    negative = (ras_x < 0.0).reshape(side_shape)
    left = np.asarray(binary & positive, dtype=np.bool_)
    right = np.asarray(binary & negative, dtype=np.bool_)
    source_count = int(binary.sum(dtype=np.int64))
    left_count = int(left.sum(dtype=np.int64))
    right_count = int(right.sum(dtype=np.int64))
    require(source_count > 0 and left_count > 0 and right_count > 0,
            "selected label does not occupy both RAS-X sides")
    require(left_count + right_count == source_count,
            "RAS-X partition lost or duplicated source voxels")
    return left, right, {
        "axis_ijk": axis,
        "ras_x_coefficient_mm": coefficient,
        "ras_x_translation_mm": float(affine[0][3]),
        "plane_index_coordinate": plane_index,
        "split_coordinate_ras_x_mm": 0.0,
        "positive_ras_x_side": "left",
        "negative_ras_x_side": "right",
        "source_voxel_count": source_count,
        "left_voxel_count": left_count,
        "right_voxel_count": right_count,
        "voxel_partition_exact": True,
        "voxel_centers_on_plane": 0,
    }


def _crop_bounds(geometry: dict[str, Any], shape: list[int]) -> tuple[list[int], list[int]]:
    bounds = geometry.get("voxel_center_bounds_ijk")
    require(isinstance(bounds, dict), "source label has no voxel-center bounds")
    lower = [max(0, int(bounds["minimum_inclusive"][axis]) - 1) for axis in range(3)]
    upper = [min(shape[axis] - 1, int(bounds["maximum_inclusive"][axis]) + 1) for axis in range(3)]
    return lower, upper


def _label_mask(
    payload_path: Path,
    info: dict[str, Any],
    geometry: dict[str, Any],
    label_id: int,
    np: Any,
) -> tuple[Any, tuple[int, int, int]]:
    shape = [int(value) for value in info["shape_ijk"]]
    lower, upper = _crop_bounds(geometry, shape)
    volume = np.memmap(
        payload_path,
        dtype="<f4" if info["byte_order"] == "little" else ">f4",
        mode="r",
        shape=tuple(shape),
        order="F",
    )
    region = volume[
        lower[0] : upper[0] + 1,
        lower[1] : upper[1] + 1,
        lower[2] : upper[2] + 1,
    ]
    slope = float(info["label_scaling_slope"])
    intercept = float(info["label_scaling_intercept"])
    if slope == 1.0 and intercept == 0.0:
        require(bool(np.isfinite(region).all()), f"label {label_id} crop has non-finite values")
        binary = np.equal(region, label_id)
    else:
        scaled = np.asarray(region, dtype=np.float64) * slope + intercept
        rounded = np.rint(scaled)
        require(bool(np.isfinite(scaled).all()) and bool(np.equal(scaled, rounded).all()),
                f"label {label_id} crop has non-integer scaled values")
        binary = np.equal(rounded, label_id)
        del scaled, rounded
    del region
    require(int(binary.sum(dtype=np.int64)) == int(geometry["voxel_count"]),
            f"label {label_id} crop count differs from intake")
    return binary, tuple(lower)


def _verify_plan(
    path: Path,
    *,
    intake_sha256: str,
    archive_sha256: str,
    source_surface_receipt_sha256: str,
    source_surface_plan_sha256: str,
    scan_id: str,
    label_ids: list[int],
    compiler_sources_sha256: dict[str, str],
    runtime: dict[str, str | None],
) -> tuple[dict[str, Any], str]:
    plan, raw = _read_json(Path(path), "trial plan", require_canonical=True)
    require(plan.get("schema") == PLAN_SCHEMA, "trial plan schema is unsupported")
    require(
        plan.get("input_intake_receipt_sha256") == intake_sha256
        and plan.get("source_archive_sha256") == archive_sha256
        and plan.get("source_surface_receipt_sha256") == source_surface_receipt_sha256
        and plan.get("source_surface_plan_sha256") == source_surface_plan_sha256
        and plan.get("scan_id") == scan_id
        and plan.get("label_ids") == sorted(label_ids)
        and plan.get("compiler_sources_sha256") == compiler_sources_sha256
        and plan.get("runtime") == runtime
        and plan.get("partition") == {
            "coordinate_system": "NIfTI RAS+",
            "axis": "x",
            "coordinate_mm": 0.0,
            "positive_side": "left",
            "negative_side": "right",
            "voxel_center_rule": "assign each occupied voxel by the sign of its RAS-X center",
            "voxel_face_aligned": True,
            "split_contact_topology": True,
        }
        and plan.get("acceptance")
        == [
            "source label voxels partition exactly once by RAS-X center sign",
            "left and right candidates are independently closed two-manifolds",
            "each side mesh volume matches its source voxel occupancy",
            "all disconnected source fragments remain represented",
        ]
        and plan.get("predictions")
        == [
            "adrenal label 1 occupies both RAS-X sides in scan 002",
            "lung label 12 occupies both RAS-X sides in scan 002",
            "midline assignment loses or duplicates no occupied source voxel",
        ]
        and plan.get("boundary")
        == "Automatic segmentation candidates only; no clinical anatomy, registration, Numi subject, mechanics, or physiology claim.",
        "trial plan does not bind the exact source, scan, compiler, runtime, and RAS-X partition",
    )
    return plan, _sha256_bytes(raw)


def compile_ras_x_candidates(
    *,
    intake_path: Path,
    archive_path: Path,
    source_surface_receipt_path: Path,
    source_surface_plan_path: Path,
    scan_id: str,
    label_ids: list[int],
    trial_plan_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    np = _load_numpy()
    intake, intake_raw = _read_json(intake_path, "source intake", require_canonical=True)
    require(intake.get("schema") == INTAKE_SCHEMA, "source intake schema is unsupported")
    archive_path = Path(archive_path)
    require(archive_path.is_file() and not archive_path.is_symlink(), "source archive is missing")
    source = intake.get("source", {})
    archive_sha = _sha256_file(archive_path)
    require(archive_sha == source.get("archive_sha256"), "archive SHA-256 differs from intake")
    source_surface, source_surface_raw = _read_json(
        source_surface_receipt_path, "source surface receipt", require_canonical=True
    )
    source_surface_sha = _sha256_bytes(source_surface_raw)
    surface_plan, surface_plan_raw = _read_json(
        source_surface_plan_path, "source surface plan", require_canonical=True
    )
    require(source_surface.get("schema") == SURFACE_SCHEMA,
            "source surface receipt schema is unsupported")
    reference_source = source_surface.get("source", {})
    reference_scan = source_surface.get("scan", {})
    scans = {row["scan_id"]: row for row in intake.get("scans", [])}
    require(scan_id in scans, "selected scan is absent from intake")
    selected_scan = scans[scan_id]
    require(
        reference_source.get("source_id") == source.get("source_id")
        and reference_source.get("release") == source.get("release")
        and reference_source.get("license") == source.get("license")
        and reference_source.get("archive_sha256") == archive_sha
        and reference_source.get("intake_receipt_sha256") == _sha256_bytes(intake_raw)
        and reference_scan.get("scan_id") == scan_id
        and reference_scan.get("nifti_member") == selected_scan.get("nifti_member")
        and reference_scan.get("nifti_uncompressed_sha256")
        == selected_scan.get("nifti_uncompressed_sha256")
        and reference_scan.get("voxel_to_world_affine_ras_mm")
        == selected_scan.get("voxel_to_world_affine"),
        "source surface receipt does not bind the selected source, scan, and intake",
    )
    require(
        source_surface.get("trial_plan_sha256") == _sha256_bytes(surface_plan_raw),
        "source surface receipt does not bind the provided surface plan",
    )
    require(
        surface_plan.get("schema") == "numi.healthy-total-body-ct-surface-trial-plan.v2"
        and surface_plan.get("source_archive_sha256") == archive_sha
        and surface_plan.get("input_intake_receipt_sha256") == _sha256_bytes(intake_raw)
        and surface_plan.get("scan_id") == scan_id
        and surface_plan.get("compiler_sources_sha256")
        == source_surface.get("compiler_sources_sha256")
        and surface_plan.get("runtime") == source_surface.get("runtime"),
        "source surface plan does not bind the reference receipt inputs",
    )
    source_surface_labels = {
        int(row["label_id"]) for row in source_surface.get("meshes", [])
    }
    require(
        surface_plan.get("label_ids") == sorted(source_surface_labels)
        and len(source_surface_labels) == len(source_surface.get("meshes", [])),
        "source surface plan and receipt label identities differ",
    )
    require(set(label_ids) <= source_surface_labels,
            "source surface receipt does not contain every selected label")
    _verify_reference_surface_manifest(
        Path(source_surface_receipt_path).parent,
        source_surface,
        expected_receipt_sha256=source_surface_sha,
    )
    intake_sha = _sha256_bytes(intake_raw)
    source_surface_plan_sha = _sha256_bytes(surface_plan_raw)
    compiler_sources = _source_hashes()
    runtime = _runtime_info(np)
    _verify_plan(
        Path(trial_plan_path),
        intake_sha256=intake_sha,
        archive_sha256=archive_sha,
        source_surface_receipt_sha256=source_surface_sha,
        source_surface_plan_sha256=source_surface_plan_sha,
        scan_id=scan_id,
        label_ids=label_ids,
        compiler_sources_sha256=compiler_sources,
        runtime=runtime,
    )
    output_directory = Path(output_directory)
    require(not output_directory.exists() and not output_directory.is_symlink(),
            "output directory exists; choose a new immutable path")
    require(re.fullmatch(r"\d{3}", scan_id) is not None, "scan ID must be three digits")
    require(label_ids and len(label_ids) == len(set(label_ids))
            and all(type(value) is int and value > 0 for value in label_ids),
            "selected label IDs are empty, invalid, or duplicated")
    scans = {row["scan_id"]: row for row in intake.get("scans", [])}
    labels = {int(row["label_id"]): row for row in intake.get("labels", [])}
    require(scan_id in scans and all(value in labels for value in label_ids),
            "scan or label is absent from intake")
    scan = scans[scan_id]
    shape = [int(value) for value in intake["source"]["voxel_array_shape_ijk"]]
    affine = scan["voxel_to_world_affine"]
    _ras_x_plane(affine, np)
    counts_by_label = scan["label_voxel_counts"]
    geometry_by_label = scan["label_spatial_geometry_candidates"]

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".ras-x-partition-", dir=output_directory.parent))
    try:
        with tempfile.TemporaryDirectory(prefix="numi-ct-rasx-") as temporary_dir:
            payload_path = Path(temporary_dir) / "nifti-payload.raw"
            with zipfile.ZipFile(archive_path) as archive:
                member_name = scan.get("nifti_member")
                require(isinstance(member_name, str) and archive.namelist().count(member_name) == 1,
                        "selected NIfTI member is missing or duplicated")
                with archive.open(member_name, "r") as member_stream:
                    info, nifti_sha = _nifti_payload_to_file(
                        member_stream, payload_path, member_name=member_name
                    )
            require(nifti_sha == scan.get("nifti_uncompressed_sha256"),
                    "decompressed NIfTI SHA-256 differs from intake")
            require(info["shape_ijk"] == shape and info["spatial_units"] == "mm"
                    and info["voxel_to_world_affine"] == affine,
                    "NIfTI shape, units, or affine differs from intake")
            meshes = []
            for label_id in sorted(label_ids):
                geometry = geometry_by_label.get(str(label_id))
                require(isinstance(geometry, dict), f"label {label_id} has no intake geometry")
                binary, crop_lower = _label_mask(
                    payload_path, info, geometry, label_id, np
                )
                left_mask, right_mask, split = _partition_mask_by_ras_x(
                    binary, affine, crop_lower, np
                )
                require(split["source_voxel_count"] == int(counts_by_label[str(label_id)]),
                        f"label {label_id} partition differs from intake voxel count")
                del binary
                for side, side_sign, side_mask, side_count in (
                    ("left", "positive", left_mask, split["left_voxel_count"]),
                    ("right", "negative", right_mask, split["right_voxel_count"]),
                ):
                    vertices, triangles, metrics = build_voxel_boundary_mesh(
                        side_mask,
                        affine,
                        voxel_offset_ijk=crop_lower,
                        split_contact_topology=True,
                        np=np,
                    )
                    voxel_volume = float(info["voxel_volume_in_source_units_cubed"])
                    source_volume = float(side_count) * voxel_volume
                    relative_error = abs(metrics["signed_volume_mm3"] - source_volume) / source_volume
                    require(relative_error <= 1e-9,
                            f"label {label_id} {side} mesh volume differs from source occupancy")
                    safe_name = re.sub(
                        r"[^a-z0-9]+", "-", labels[label_id]["name"].casefold()
                    ).strip("-")
                    mesh_name = f"scan-{scan_id}-label-{label_id:03d}-{safe_name}-{side}-ras-x-boundary.ply.gz"
                    mesh_path = stage / mesh_name
                    _write_binary_ply_gzip(mesh_path, vertices, triangles, np)
                    record = {
                        "label_id": label_id,
                        "source_semantic_id": f"{source['source_id']}:segmentation-label:{label_id}",
                        "source_label_name": labels[label_id]["name"],
                        "ras_x_side": side,
                        "ras_x_sign": side_sign,
                        "source_voxel_count": side_count,
                        "source_voxel_occupancy_volume_candidate_mm3": source_volume,
                        "mesh_file": mesh_name,
                        "mesh_file_sha256": _sha256_file(mesh_path),
                        "mesh_file_bytes": mesh_path.stat().st_size,
                        "mesh_metrics": metrics | {
                            "relative_signed_volume_error": relative_error,
                            "source_voxel_occupancy_volume_candidate_mm3": source_volume,
                        },
                        "voxel_partition": split,
                    }
                    meshes.append(record)
                    del side_mask, vertices, triangles
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "scan_specific_ras_x_side_surface_candidates",
            "compiler": "numilab-human.healthy-total-body-ct-ras-x-partition.1",
            "compiler_sources_sha256": compiler_sources,
            "runtime": runtime,
            "source": {
                "source_id": source["source_id"],
                "release": source["release"],
                "license": source["license"],
                "archive_sha256": archive_sha,
                "intake_receipt_sha256": intake_sha,
                "nifti_member": scan["nifti_member"],
                "nifti_uncompressed_sha256": nifti_sha,
                "reference_surface_receipt_sha256": source_surface_sha,
                "reference_surface_plan_sha256": source_surface_plan_sha,
            },
            "scan": {
                "scan_id": scan_id,
                "coordinate_convention": "NIfTI RAS+ millimetres; this scan affine only",
                "voxel_to_world_affine_ras_mm": affine,
                "voxel_spacing_mm": scan["voxel_spacing_mm"],
                "voxel_volume_mm3": float(info["voxel_volume_in_source_units_cubed"]),
            },
            "trial_plan_sha256": _sha256_file(Path(trial_plan_path)),
            "partition": {
                "axis": "RAS-X",
                "coordinate_mm": 0.0,
                "plane_index_coordinate": _ras_x_plane(affine, np)[2],
                "positive_side": "left",
                "negative_side": "right",
                "method": "occupied-voxel-center-sign; exact voxel partition; no surface smoothing",
                "source_voxel_centers_on_plane": 0,
                "all_selected_source_voxels_partitioned_exactly_once": True,
            },
            "meshes": meshes,
            "qualification": {
                "source_archive_and_nifti_hashes_match_intake": True,
                "ras_x_plane_aligned_to_voxel_face": True,
                "left_and_right_source_voxel_sets_disjoint": True,
                "left_and_right_voxel_counts_sum_to_source": True,
                "mesh_volume_matches_each_side_voxel_occupancy": True,
                "all_meshes_closed_two_manifolds": all(
                    row["mesh_metrics"]["closed_two_manifold"] for row in meshes
                ),
                "expert_segmentation_accuracy": False,
                "cross_scan_registration": False,
                "numi_human_subject_binding": False,
                "physical_tissue_owner": False,
                "mechanics_or_physiology": False,
                "clinical_anatomy": False,
            },
            "boundary": (
                "This output splits automatic scan-label voxels by RAS-X center sign and preserves all source "
                "voxels, including disconnected fragments. A passing geometric audit does not validate the "
                "automatic segmentation, anatomical boundaries, clinical laterality, subject registration, "
                "physical ownership, mechanics, or physiology."
            ),
        }
        receipt_path = stage / "receipt.json"
        receipt_path.write_bytes(canonical(receipt) + b"\n")
        sums = []
        for path in sorted(stage.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                sums.append(f"{_sha256_file(path)}  {path.name}\n")
        (stage / "SHA256SUMS").write_text("".join(sums), encoding="ascii")
        os.replace(stage, output_directory)
        return receipt
    finally:
        if stage.exists():
            import shutil

            shutil.rmtree(stage)


def _audit_source_counts(
    archive_path: Path,
    *,
    member_name: str,
    scan: dict[str, Any],
    expected_nifti_sha256: str,
    label_ids: list[int],
    np: Any,
) -> tuple[dict[int, dict[str, int]], str, dict[str, Any]]:
    """Independently count side voxels by streaming source NIfTI values."""
    counts = {label_id: {"left": 0, "right": 0, "on_plane": 0} for label_id in label_ids}
    digest = hashlib.sha256()
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.namelist().count(member_name) == 1,
                "source NIfTI member is missing or duplicated")
        with archive.open(member_name, "r") as member_stream:
            with gzip.GzipFile(fileobj=member_stream, mode="rb") as image:
                header = image.read(NIFTI_HEADER_BYTES)
                require(len(header) == NIFTI_HEADER_BYTES, "source NIfTI header is truncated")
                digest.update(header)
                info = parse_nifti_header(header)
                shape = [int(value) for value in info["shape_ijk"]]
                require(info["spatial_units"] == "mm"
                        and info["voxel_to_world_affine"] == scan["voxel_to_world_affine"],
                        "source NIfTI affine or units differ from intake")
                axis, coefficient, _plane = _ras_x_plane(info["voxel_to_world_affine"], np)
                strides = [1, shape[0], shape[0] * shape[1]]
                expected_payload = int(info["voxel_count"]) * 4
                payload_bytes = 0
                while True:
                    block = image.read(COPY_CHUNK_BYTES)
                    if not block:
                        break
                    require(len(block) % 4 == 0, "NIfTI payload ends within a voxel")
                    digest.update(block)
                    first = payload_bytes // 4
                    payload_bytes += len(block)
                    require(payload_bytes <= expected_payload, "NIfTI payload is longer than declared")
                    values = np.frombuffer(
                        block,
                        dtype="<f4" if info["byte_order"] == "little" else ">f4",
                    )
                    if info["label_scaling_slope"] != 1.0 or info["label_scaling_intercept"] != 0.0:
                        values = values.astype(np.float64) * float(info["label_scaling_slope"]) + float(info["label_scaling_intercept"])
                    require(bool(np.isfinite(values).all()), "NIfTI payload contains non-finite labels")
                    rounded = np.rint(values)
                    require(bool(np.equal(values, rounded).all()), "NIfTI payload contains non-integer labels")
                    linear = np.arange(first, first + len(values), dtype=np.int64)
                    axis_index = (linear // strides[axis]) % shape[axis]
                    ras_x = coefficient * axis_index.astype(np.float64) + float(info["voxel_to_world_affine"][0][3])
                    left_voxels, right_voxels, on_plane = ras_x > 0.0, ras_x < 0.0, np.abs(ras_x) <= PLANE_TOLERANCE_MM
                    for label_id in label_ids:
                        selected = rounded == label_id
                        counts[label_id]["left"] += int(np.count_nonzero(selected & left_voxels))
                        counts[label_id]["right"] += int(np.count_nonzero(selected & right_voxels))
                        counts[label_id]["on_plane"] += int(np.count_nonzero(selected & on_plane))
                    del values, rounded, linear, axis_index, ras_x
                require(payload_bytes == expected_payload, "NIfTI payload length differs from header")
    require(digest.hexdigest() == expected_nifti_sha256,
            "independently streamed NIfTI SHA-256 differs from intake")
    require(all(row["on_plane"] == 0 for row in counts.values()),
            "selected source has voxel centers on the RAS-X plane")
    return counts, digest.hexdigest(), info


def _verify_output_manifest(directory: Path, receipt: dict[str, Any]) -> str:
    sums_path = directory / "SHA256SUMS"
    require(sums_path.is_file() and not sums_path.is_symlink(), "output checksum manifest is missing")
    expected = {"receipt.json": _sha256_bytes(canonical(receipt) + b"\n")}
    expected.update({row["mesh_file"]: row["mesh_file_sha256"] for row in receipt["meshes"]})
    actual: dict[str, str] = {}
    for line in sums_path.read_text(encoding="ascii").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and len(fields[0]) == 64,
                "checksum manifest line is malformed")
        require(fields[1] not in actual, "checksum manifest has duplicate filenames")
        actual[fields[1]] = fields[0]
    require(actual == expected, "checksum manifest does not match exact candidate files")
    for name, digest in expected.items():
        path = directory / name
        require(path.is_file() and not path.is_symlink() and _sha256_file(path) == digest,
                f"output checksum mismatch for {name}")
    require({path.name for path in directory.iterdir() if path.is_file()} == set(expected) | {"SHA256SUMS"},
            "output directory contains unbound files")
    return _sha256_file(sums_path)


def _verify_reference_surface_manifest(
    directory: Path,
    receipt: dict[str, Any],
    *,
    expected_receipt_sha256: str,
) -> None:
    """Check the retained source-surface payload identities used as provenance."""
    require(directory.is_dir() and not directory.is_symlink(),
            "source surface output directory is missing")
    expected = {"receipt.json": expected_receipt_sha256}
    expected.update({row["mesh_file"]: row["mesh_file_sha256"] for row in receipt["meshes"]})
    manifest = directory / "SHA256SUMS"
    require(manifest.is_file() and not manifest.is_symlink(),
            "source surface checksum manifest is missing")
    actual: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and len(fields[0]) == 64,
                "source surface checksum manifest line is malformed")
        require(fields[1] not in actual, "source surface checksum manifest has duplicates")
        actual[fields[1]] = fields[0]
    require(actual == expected, "source surface checksum manifest differs from receipt")
    for name, digest in expected.items():
        require(Path(name).name == name, "source surface filename escapes its output directory")
        path = directory / name
        require(path.is_file() and not path.is_symlink() and _sha256_file(path) == digest,
                f"source surface payload checksum mismatch for {name}")
    require(
        {path.name for path in directory.iterdir() if path.is_file()}
        == set(expected) | {"SHA256SUMS"},
        "source surface output directory contains unbound files",
    )


def _mesh_source_voxel_geometry(
    vertices: Any,
    *,
    affine: list[list[float]],
    voxel_center_bounds_ijk: dict[str, Any],
    np: Any,
) -> dict[str, Any]:
    transform = np.asarray(affine, dtype=np.float64)
    vertices_ijk = np.linalg.solve(
        transform[:3, :3], (vertices - transform[:3, 3]).T
    ).T
    lower = np.asarray(voxel_center_bounds_ijk["minimum_inclusive"], dtype=np.float64) - 0.5
    upper = np.asarray(voxel_center_bounds_ijk["maximum_inclusive"], dtype=np.float64) + 0.5
    tolerance = 1e-4
    within_source_envelope = bool(
        np.all(vertices_ijk >= lower - tolerance)
        and np.all(vertices_ijk <= upper + tolerance)
    )
    on_source_voxel_grid = bool(
        np.all(np.abs((vertices_ijk - 0.5) - np.rint(vertices_ijk - 0.5)) <= tolerance)
    )
    return {
        "vertices_within_source_voxel_envelope": within_source_envelope,
        "vertices_on_source_voxel_grid": on_source_voxel_grid,
    }


def _audit_mesh(
    mesh_path: Path,
    record: dict[str, Any],
    *,
    expected_voxel_count: int,
    voxel_volume_mm3: float,
    voxel_center_bounds_ijk: dict[str, Any],
    affine: list[list[float]],
    side: str,
    np: Any,
) -> dict[str, Any]:
    require(expected_voxel_count > 0, f"{mesh_path.name} represents an empty source side")
    require(_sha256_file(mesh_path) == record.get("mesh_file_sha256"),
            f"{mesh_path.name} SHA-256 differs from receipt")
    require(mesh_path.stat().st_size == record.get("mesh_file_bytes"),
            f"{mesh_path.name} size differs from receipt")
    vertices, faces, payload = _read_binary_ply_gzip(mesh_path, np)
    require(len(vertices) > 0 and len(faces) > 0,
            f"{mesh_path.name} has no serialized surface elements")
    require(bool(np.isfinite(vertices).all()), f"{mesh_path.name} has non-finite coordinates")
    require(bool(np.all((faces >= 0) & (faces < len(vertices)))),
            f"{mesh_path.name} has invalid triangle indices")
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    require(bool(np.all(areas > 0.0)), f"{mesh_path.name} contains zero-area triangles")
    origin = vertices.mean(axis=0)
    relative = triangles - origin
    signed_volume = float(np.einsum(
        "ij,ij->i", relative[:, 0], np.cross(relative[:, 1], relative[:, 2])
    ).sum(dtype=np.float64) / 6.0)
    source_volume = expected_voxel_count * voxel_volume_mm3
    relative_error = abs(signed_volume - source_volume) / source_volume
    edge_histogram = _edge_incidence_histogram(faces, np)
    bad_vertices = _bad_vertex_links(faces, len(vertices))
    closed = bool(edge_histogram and set(edge_histogram) == {"2"} and bad_vertices == 0)
    tolerance = max(1e-9, float(np.max(np.abs(vertices[:, 0]))) * 1e-12)
    halfspace = bool(
        np.all(vertices[:, 0] >= -tolerance)
        if side == "left"
        else np.all(vertices[:, 0] <= tolerance)
    )
    voxel_geometry = _mesh_source_voxel_geometry(
        vertices,
        affine=affine,
        voxel_center_bounds_ijk=voxel_center_bounds_ijk,
        np=np,
    )
    metrics = record.get("mesh_metrics", {})
    require(expected_voxel_count == record.get("source_voxel_count"),
            f"{mesh_path.name} source-side voxel count differs from NIfTI")
    require(payload["vertex_count"] == metrics.get("vertex_count")
            and payload["triangle_count"] == metrics.get("triangle_count"),
            f"{mesh_path.name} element counts differ from compiler receipt")
    require(edge_histogram == metrics.get("edge_incidence", {}).get("incidence_histogram")
            and bad_vertices == metrics.get("nonmanifold_vertex_count")
            and closed == metrics.get("closed_two_manifold"),
            f"{mesh_path.name} independently measured topology differs from receipt")
    require(closed, f"{mesh_path.name} is not an independently verified closed two-manifold")
    require(signed_volume > 0.0 and relative_error <= 1e-9,
            f"{mesh_path.name} signed volume differs from side voxel occupancy")
    require(halfspace, f"{mesh_path.name} crosses the RAS-X side plane")
    require(voxel_geometry["vertices_within_source_voxel_envelope"],
            f"{mesh_path.name} exceeds its source label voxel envelope")
    require(voxel_geometry["vertices_on_source_voxel_grid"],
            f"{mesh_path.name} contains vertices away from source voxel boundaries")
    return {
        "mesh_file": mesh_path.name,
        "mesh_sha256": record["mesh_file_sha256"],
        "ras_x_side": side,
        "source_voxel_count": expected_voxel_count,
        "vertices_in_declared_ras_x_halfspace": halfspace,
        **voxel_geometry,
        "independent_edge_incidence_histogram": edge_histogram,
        "independent_nonmanifold_vertex_count": bad_vertices,
        "closed_two_manifold": closed,
        "signed_volume_mm3": signed_volume,
        "source_side_voxel_occupancy_volume_mm3": source_volume,
        "relative_signed_volume_error": relative_error,
    }


def audit_ras_x_candidates(
    *,
    intake_path: Path,
    archive_path: Path,
    trial_plan_path: Path,
    compiler_receipt_path: Path,
) -> dict[str, Any]:
    np = _load_numpy()
    intake, intake_raw = _read_json(intake_path, "source intake", require_canonical=True)
    plan, plan_raw = _read_json(trial_plan_path, "trial plan", require_canonical=True)
    receipt, receipt_raw = _read_json(compiler_receipt_path, "compiler receipt", require_canonical=True)
    require(intake.get("schema") == INTAKE_SCHEMA
            and plan.get("schema") == PLAN_SCHEMA
            and receipt.get("schema") == RECEIPT_SCHEMA,
            "audit input schema differs")
    archive_path = Path(archive_path)
    require(archive_path.is_file() and not archive_path.is_symlink(), "source archive is missing")
    archive_sha = _sha256_file(archive_path)
    intake_sha = _sha256_bytes(intake_raw)
    plan_sha = _sha256_bytes(plan_raw)
    source = intake["source"]
    require(archive_sha == source.get("archive_sha256")
            and receipt.get("source", {}).get("archive_sha256") == archive_sha
            and receipt.get("source", {}).get("intake_receipt_sha256") == intake_sha,
            "archive, intake, and output receipt identities do not join")
    require(receipt.get("trial_plan_sha256") == plan_sha,
            "output receipt does not bind this plan")
    require(plan.get("input_intake_receipt_sha256") == intake_sha
            and plan.get("source_archive_sha256") == archive_sha
            and plan.get("compiler_sources_sha256") == receipt.get("compiler_sources_sha256")
            and plan.get("runtime") == receipt.get("runtime")
            and plan.get("scan_id") == receipt.get("scan", {}).get("scan_id"),
            "plan and output receipt identities do not join")
    scan_id = receipt["scan"]["scan_id"]
    scans = {row["scan_id"]: row for row in intake.get("scans", [])}
    require(scan_id in scans, "output scan is absent from intake")
    scan = scans[scan_id]
    require(scan["nifti_member"] == receipt["source"].get("nifti_member")
            and scan["nifti_uncompressed_sha256"] == receipt["source"].get("nifti_uncompressed_sha256")
            and scan["voxel_to_world_affine"] == receipt["scan"].get("voxel_to_world_affine_ras_mm"),
            "output scan geometry or NIfTI identity differs from intake")
    label_ids = sorted({int(row["label_id"]) for row in receipt["meshes"]})
    require(label_ids == plan.get("label_ids"), "plan label identities differ from output meshes")
    source_counts, nifti_sha, info = _audit_source_counts(
        archive_path,
        member_name=scan["nifti_member"],
        scan=scan,
        expected_nifti_sha256=scan["nifti_uncompressed_sha256"],
        label_ids=label_ids,
        np=np,
    )
    require(nifti_sha == receipt["source"].get("nifti_uncompressed_sha256"),
            "independent NIfTI hash differs from compiler receipt")
    output_directory = Path(compiler_receipt_path).parent
    checksum_sha = _verify_output_manifest(output_directory, receipt)
    geometry_by_label = scan.get("label_spatial_geometry_candidates", {})
    seen: set[tuple[int, str]] = set()
    mesh_results = []
    for record in receipt["meshes"]:
        label_id = int(record["label_id"])
        side = record.get("ras_x_side")
        require(side in ("left", "right") and (label_id, side) not in seen,
                "duplicate or invalid label-side mesh identity")
        seen.add((label_id, side))
        require(record.get("source_semantic_id")
                == f"{source['source_id']}:segmentation-label:{label_id}",
                "source semantic identity differs")
        expected = source_counts[label_id][side]
        mesh_path = output_directory / record["mesh_file"]
        require(mesh_path.parent == output_directory and mesh_path.name == record["mesh_file"],
                "mesh path escapes its output directory")
        mesh_results.append(_audit_mesh(
            mesh_path,
            record,
            expected_voxel_count=expected,
            voxel_volume_mm3=float(info["voxel_volume_in_source_units_cubed"]),
            voxel_center_bounds_ijk=geometry_by_label[str(label_id)]["voxel_center_bounds_ijk"],
            affine=scan["voxel_to_world_affine"],
            side=side,
            np=np,
        ))
    require(seen == {(label, side) for label in label_ids for side in ("left", "right")},
            "every selected label must have exactly one left and one right candidate")
    labels = {int(row["label_id"]): row for row in intake["labels"]}
    partition_results = []
    for label_id in label_ids:
        total = source_counts[label_id]["left"] + source_counts[label_id]["right"]
        expected_total = int(scan["label_voxel_counts"].get(str(label_id), 0))
        require(total == expected_total,
                f"label {label_id} RAS-X sides do not conserve source voxel occupancy")
        partition_results.append({
            "label_id": label_id,
            "label_name": labels[label_id]["name"],
            "source_voxel_count": expected_total,
            "left_voxel_count": source_counts[label_id]["left"],
            "right_voxel_count": source_counts[label_id]["right"],
            "voxel_centers_on_plane": source_counts[label_id]["on_plane"],
            "exact_partition": total == expected_total and source_counts[label_id]["on_plane"] == 0,
        })
    return {
        "schema": AUDIT_SCHEMA,
        "status": "independent_ras_x_source_partition_and_mesh_audit_passed",
        "intake_receipt_sha256": intake_sha,
        "source_archive_sha256": archive_sha,
        "nifti_uncompressed_sha256": nifti_sha,
        "trial_plan_sha256": plan_sha,
        "compiler_receipt_sha256": _sha256_bytes(receipt_raw),
        "checksum_manifest_sha256": checksum_sha,
        "scan_id": scan_id,
        "source_ras_x_plane_index_coordinate": _ras_x_plane(scan["voxel_to_world_affine"], np)[2],
        "independently_streamed_partition_counts": partition_results,
        "independently_reparsed_meshes": mesh_results,
        "qualification": {
            "source_archive_and_nifti_identity_reverified": True,
            "source_voxel_values_streamed_and_recounted": True,
            "left_and_right_voxel_counts_reproduce_source": True,
            "candidate_surfaces_reparsed_without_surface_compiler": True,
            "mesh_topology_recomputed": True,
            "ras_x_halfspace_verified": True,
            "source_voxel_grid_and_envelope_verified": True,
            "automatic_segmentation_accuracy": False,
            "numi_human_subject_binding": False,
            "mechanics_or_physiology": False,
            "clinical_anatomy": False,
        },
        "boundary": (
        "The audit independently re-counts source labels by RAS-X voxel-center sign and checks each PLY's "
        "side half-space, source voxel grid/envelope, closed topology, and occupancy volume. It does not validate automatic segmentation "
            "accuracy, clinical anatomy, cross-scan registration, Human subject binding, or mechanics."
        ),
    }


def add_compile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-surface-receipt", type=Path, required=True)
    parser.add_argument("--source-surface-plan", type=Path, required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--label-ids", type=int, nargs="+", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new immutable candidate directory")
    parser.set_defaults(handler=run_compile)


def add_audit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new independent audit JSON path")
    parser.set_defaults(handler=run_audit)


def run_compile(arguments: argparse.Namespace) -> int:
    receipt = compile_ras_x_candidates(
        intake_path=arguments.intake,
        archive_path=arguments.archive,
        source_surface_receipt_path=arguments.source_surface_receipt,
        source_surface_plan_path=arguments.source_surface_plan,
        scan_id=arguments.scan_id,
        label_ids=list(arguments.label_ids),
        trial_plan_path=arguments.plan,
        output_directory=arguments.output,
    )
    print(json.dumps({
        "schema": receipt["schema"],
        "status": receipt["status"],
        "output": str(Path(arguments.output).resolve()),
        "receipt_sha256": _sha256_file(Path(arguments.output) / "receipt.json"),
        "mesh_count": len(receipt["meshes"]),
        "closed_two_manifold_mesh_count": sum(
            bool(row["mesh_metrics"]["closed_two_manifold"]) for row in receipt["meshes"]
        ),
        "source_voxel_partition_counts": [
            {
                "label_id": label_id,
                "left": next(row["source_voxel_count"] for row in receipt["meshes"]
                             if row["label_id"] == label_id and row["ras_x_side"] == "left"),
                "right": next(row["source_voxel_count"] for row in receipt["meshes"]
                              if row["label_id"] == label_id and row["ras_x_side"] == "right"),
            }
            for label_id in sorted(set(row["label_id"] for row in receipt["meshes"]))
        ],
        "qualification": receipt["qualification"],
    }, sort_keys=True))
    return 0


def run_audit(arguments: argparse.Namespace) -> int:
    output = Path(arguments.output)
    require(not output.exists() and not output.is_symlink(),
            "audit output exists; choose a new immutable path")
    result = audit_ras_x_candidates(
        intake_path=arguments.intake,
        archive_path=arguments.archive,
        trial_plan_path=arguments.plan,
        compiler_receipt_path=arguments.receipt,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical(result) + b"\n")
    print(json.dumps({
        "schema": result["schema"],
        "status": result["status"],
        "output": str(output.resolve()),
        "audit_sha256": _sha256_file(output),
        "mesh_count": len(result["independently_reparsed_meshes"]),
        "source_partition_count": len(result["independently_streamed_partition_counts"]),
    }, sort_keys=True))
    return 0
