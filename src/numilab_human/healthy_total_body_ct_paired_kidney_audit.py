"""Measure paired-kidney source-mask topology and geometry across CT scans.

The measurements remain within each scan's own NIfTI RAS+ frame. They describe
automatic source labels, not verified clinical anatomy or a Numi Human binding.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import zipfile

from . import model as human
from .healthy_total_body_ct_source import (
    MAX_MASK_BYTES,
    NIFTI_HEADER_BYTES,
    _load_numpy,
    parse_nifti_header,
)
from .whole_body_embeddedness import atomic_json


SCHEMA = "numi.healthy-total-body-ct-paired-kidney-audit.v1"
PLAN_SCHEMA = "numi.healthy-total-body-ct-paired-kidney-plan.v1"
INTAKE_SCHEMA = "HumanPack.external-segmentation-source-ingest.v2"
CODE_FILES = ("healthy_total_body_ct_paired_kidney_audit.py",
              "healthy_total_body_ct_source.py")
LABEL_NAME = "Kidneys"
CONNECTIVITY = "full26"
EXPECTED_LABEL_ID = 6


def require(condition: bool, message: str) -> None:
    if not condition:
        raise human.ImportError("healthy CT paired-kidney audit: " + message)


def _affine_close(left: Any, right: Any) -> bool:
    return (isinstance(left, list) and isinstance(right, list)
            and len(left) == len(right) == 4
            and all(isinstance(a, list) and isinstance(b, list)
                    and len(a) == len(b) == 4
                    and all(type(x) in (int, float) and type(y) in (int, float)
                            and math.isfinite(x) and math.isfinite(y)
                            and abs(x - y) <= 1.0e-6
                            for x, y in zip(a, b, strict=True))
                    for a, b in zip(left, right, strict=True)))


def _determinant3(affine: list[list[float]]) -> float:
    a, b, c = (row[:3] for row in affine[:3])
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def _stream_label_crop(image: Any, info: dict[str, Any], lower: list[int],
                       upper: list[int], label_id: int, *, np: Any,
                       digest: Any) -> Any:
    """Hash a full NIfTI member while retaining only a bounded label crop."""
    size_i, size_j, size_k = info["shape_ijk"]
    plane_voxels = size_i * size_j
    plane_bytes = plane_voxels * (info["bitpix"] // 8)
    chunk_planes = max(1, (16 * 1024 * 1024) // plane_bytes)
    chunk_bytes = plane_bytes * chunk_planes
    expected_bytes = info["voxel_count"] * (info["bitpix"] // 8)
    require(expected_bytes <= MAX_MASK_BYTES,
            "uncompressed source mask exceeds the registered size bound")
    crop_shape = tuple(upper[axis] - lower[axis] + 1 for axis in range(3))
    crop = np.zeros(crop_shape, dtype=np.bool_)
    consumed = 0
    first_voxel = 0
    dtype = "<f4" if info["byte_order"] == "little" else ">f4"
    while consumed < expected_bytes:
        raw = image.read(min(chunk_bytes, expected_bytes - consumed))
        require(raw and len(raw) % (4 * plane_voxels) == 0,
                "NIfTI voxel stream ends inside an axial slice")
        digest.update(raw)
        consumed += len(raw)
        values = np.frombuffer(raw, dtype=dtype)
        if info["label_scaling_slope"] != 1.0 or info["label_scaling_intercept"] != 0.0:
            values = values * info["label_scaling_slope"] + info["label_scaling_intercept"]
        require(bool(np.isfinite(values).all())
                and bool(np.equal(values, np.rint(values)).all()),
                "NIfTI source contains non-finite or non-integer labels")
        block_planes = values.size // plane_voxels
        block = values.reshape((size_i, size_j, block_planes), order="F")
        first_plane = first_voxel // plane_voxels
        z_start = max(lower[2], first_plane)
        z_end = min(upper[2] + 1, first_plane + block_planes)
        if z_start < z_end:
            crop[:, :, z_start - lower[2]:z_end - lower[2]] = (
                block[lower[0]:upper[0] + 1,
                      lower[1]:upper[1] + 1,
                      z_start - first_plane:z_end - first_plane] == label_id
            )
        first_voxel += values.size
    require(consumed == expected_bytes and image.read(1) == b"",
            "NIfTI payload length differs from header")
    return crop


def _summarize_components(mask: Any, lower_ijk: list[int], affine: list[list[float]],
                          voxel_volume_mm3: float, *, np: Any, ndimage: Any
                          ) -> dict[str, Any]:
    require(mask.ndim == 3 and bool(mask.any()), "kidney mask must be nonempty and three-dimensional")
    require(len(lower_ijk) == 3 and all(type(value) is int and value >= 0
                                        for value in lower_ijk),
            "crop lower bound is malformed")
    structure = np.ones((3, 3, 3), dtype=np.bool_)
    component_map, component_count = ndimage.label(mask, structure=structure)
    require(component_count == 2,
            f"expected two 26-connected kidney components; observed {component_count}")
    counts = np.bincount(component_map.reshape(-1), minlength=component_count + 1)[1:]
    centers_local = ndimage.center_of_mass(mask, component_map, [1, 2])
    components = []
    for component_id, (voxel_count, center_local) in enumerate(
            zip(counts.tolist(), centers_local, strict=True), start=1):
        ijk = [float(center_local[axis] + lower_ijk[axis]) for axis in range(3)]
        ras = [sum(affine[row][column] * point
                   for column, point in enumerate((*ijk, 1.0)))
               for row in range(3)]
        components.append({
            "component_id": component_id,
            "voxel_count": int(voxel_count),
            "source_mask_occupancy_ml": float(voxel_count * voxel_volume_mm3 / 1000.0),
            "centroid_voxel_center_ijk": ijk,
            "centroid_ras_mm": ras,
        })
    left, right = sorted(components, key=lambda row: row["centroid_ras_mm"][0])
    lateral_separation = right["centroid_ras_mm"][0] - left["centroid_ras_mm"][0]
    require(lateral_separation > 0.0, "paired component centroids have no RAS+ lateral separation")
    total_voxels = int(sum(counts))
    left_ml, right_ml = (left["source_mask_occupancy_ml"],
                         right["source_mask_occupancy_ml"])
    return {
        "connectivity": CONNECTIVITY,
        "component_count": int(component_count),
        "total_voxel_count": total_voxels,
        "source_mask_occupancy_ml": float(total_voxels * voxel_volume_mm3 / 1000.0),
        "left_by_ras_x": left,
        "right_by_ras_x": right,
        "right_minus_left_ras_x_mm": float(lateral_separation),
        "right_to_left_occupancy_ratio": float(right_ml / left_ml),
        "smaller_to_larger_occupancy_ratio": float(min(left_ml, right_ml) / max(left_ml, right_ml)),
    }


def _read_scan(archive: zipfile.ZipFile, scan: dict[str, Any], label_id: int,
               shape_ijk: list[int], *, np: Any, ndimage: Any) -> dict[str, Any]:
    scan_id, member = scan.get("scan_id"), scan.get("nifti_member")
    require(isinstance(scan_id, str) and scan_id
            and isinstance(member, str) and member in archive.namelist(),
            "source scan does not name a retained NIfTI member")
    require(not member.startswith("/") and "/../" not in f"/{member}/",
            f"scan {scan_id} has an unsafe archive member path")
    info_record = archive.getinfo(member)
    require(0 < info_record.file_size <= MAX_MASK_BYTES,
            f"scan {scan_id} exceeds the registered compressed-mask bound")
    digest = hashlib.sha256()
    with archive.open(member) as zipped_member:
        with gzip.GzipFile(fileobj=zipped_member, mode="rb") as image:
            header = image.read(NIFTI_HEADER_BYTES)
            digest.update(header)
            info = parse_nifti_header(header)
            require(info["shape_ijk"] == shape_ijk
                    and info["affine_source"] == scan.get("affine_source")
                    and _affine_close(info["voxel_to_world_affine"],
                                      scan.get("voxel_to_world_affine")),
                    f"scan {scan_id} NIfTI geometry differs from intake")
            source_counts = scan.get("label_voxel_counts")
            geometry = scan.get("label_spatial_geometry_candidates")
            require(isinstance(source_counts, dict) and isinstance(geometry, dict),
                    f"scan {scan_id} lacks source mask counts or geometry")
            expected_count = source_counts.get(str(label_id))
            candidate = geometry.get(str(label_id))
            require(type(expected_count) is int and expected_count > 0
                    and isinstance(candidate, dict)
                    and candidate.get("voxel_count") == expected_count,
                    f"scan {scan_id} kidney mask is absent or disagrees with intake")
            bounds = candidate.get("voxel_center_bounds_ijk")
            require(isinstance(bounds, dict), f"scan {scan_id} kidney crop bounds are absent")
            lower, upper = bounds.get("minimum_inclusive"), bounds.get("maximum_inclusive")
            require(isinstance(lower, list) and isinstance(upper, list)
                    and len(lower) == len(upper) == 3
                    and all(type(lo) is int and type(hi) is int and 0 <= lo <= hi < shape_ijk[i]
                            for i, (lo, hi) in enumerate(zip(lower, upper, strict=True))),
                    f"scan {scan_id} kidney crop bounds are invalid")
            mask = _stream_label_crop(image, info, lower, upper, label_id,
                                      np=np, digest=digest)
    require(digest.hexdigest() == scan.get("nifti_uncompressed_sha256"),
            f"scan {scan_id} uncompressed NIfTI hash differs from intake")
    require(info["spatial_units"] == "mm", f"scan {scan_id} is not expressed in millimetres")

    measured_count = int(mask.sum(dtype=np.int64))
    require(measured_count == expected_count,
            f"scan {scan_id} kidney voxel count differs from intake")
    affine = info["voxel_to_world_affine"]
    voxel_volume = abs(_determinant3(affine))
    require(math.isclose(voxel_volume, float(scan.get("voxel_volume_mm3", math.nan)),
                         rel_tol=1.0e-8, abs_tol=1.0e-10),
            f"scan {scan_id} affine voxel volume differs from intake")
    geometry_out = _summarize_components(mask, lower, affine, voxel_volume,
                                         np=np, ndimage=ndimage)
    require(geometry_out["total_voxel_count"] == expected_count,
            f"scan {scan_id} component voxel counts do not sum to the source label")
    return {
        "scan_id": scan_id,
        "nifti_member": member,
        "nifti_uncompressed_sha256": digest.hexdigest(),
        "affine_source": info["affine_source"],
        "voxel_to_world_affine": affine,
        "affine_sha256": hashlib.sha256(json.dumps(
            affine, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()).hexdigest(),
        "label_id": label_id,
        "label_name": LABEL_NAME,
        "source_label_voxel_count": expected_count,
        **geometry_out,
    }


def _validate_plan(plan_path: Path, *, intake_sha256: str, archive_sha256: str,
                   scan_count: int, label_id: int, code_hashes: dict[str, str]) -> str:
    require(plan_path.is_file() and not plan_path.is_symlink(),
            "preregistered plan is not a retained regular file")
    plan = human.read_json(plan_path)
    require(plan.get("schema") == PLAN_SCHEMA
            and plan.get("input_intake_sha256") == intake_sha256
            and plan.get("source_archive_sha256") == archive_sha256
            and plan.get("scan_count") == scan_count
            and plan.get("label_name") == LABEL_NAME
            and plan.get("label_id") == label_id
            and plan.get("connectivity") == CONNECTIVITY
            and plan.get("predicate_source_sha256") == code_hashes,
            "preregistered plan does not bind this source and predicate")
    return human.sha256(plan_path)


def compile_audit(intake_path: Path, archive_path: Path, plan_path: Path) -> dict[str, Any]:
    intake_path, archive_path, plan_path = (Path(p).resolve()
                                            for p in (intake_path, archive_path, plan_path))
    require(all(path.is_file() and not path.is_symlink()
                for path in (intake_path, archive_path)),
            "source intake or archive is not a retained regular file")
    intake = human.read_json(intake_path)
    require(intake.get("schema") == INTAKE_SCHEMA, "unsupported source intake schema")
    source = intake.get("source")
    scans = intake.get("scans")
    labels = intake.get("labels")
    require(isinstance(source, dict) and isinstance(scans, list) and scans
            and isinstance(labels, list), "intake lacks source, scan, or label records")
    archive_sha = human.sha256(archive_path)
    require(archive_sha == source.get("archive_sha256"),
            "source archive hash differs from intake")
    kidney_labels = [row for row in labels if isinstance(row, dict)
                     and row.get("name") == LABEL_NAME]
    require(len(kidney_labels) == 1
            and kidney_labels[0].get("label_id") == EXPECTED_LABEL_ID,
            "source label dictionary kidney identity differs")
    label_id = kidney_labels[0]["label_id"]
    code_hashes = {name: human.sha256(Path(__file__).with_name(name)) for name in CODE_FILES}
    plan_sha = _validate_plan(plan_path, intake_sha256=human.sha256(intake_path),
                              archive_sha256=archive_sha, scan_count=len(scans),
                              label_id=label_id, code_hashes=code_hashes)
    shape_ijk = source.get("voxel_array_shape_ijk")
    require(isinstance(shape_ijk, list) and len(shape_ijk) == 3
            and all(type(size) is int and size > 0 for size in shape_ijk),
            "intake source array shape is invalid")
    np = _load_numpy()
    try:
        from scipy import ndimage
    except ImportError as error:
        raise human.ImportError("healthy CT paired-kidney audit requires SciPy") from error
    rows = []
    seen = set()
    with zipfile.ZipFile(archive_path) as archive:
        for scan in scans:
            require(isinstance(scan, dict), "malformed scan intake row")
            scan_id = scan.get("scan_id")
            require(isinstance(scan_id, str) and scan_id and scan_id not in seen,
                    "scan IDs must be unique")
            seen.add(scan_id)
            rows.append(_read_scan(archive, scan, label_id, shape_ijk,
                                   np=np, ndimage=ndimage))
    require({name: human.sha256(Path(__file__).with_name(name)) for name in CODE_FILES}
            == code_hashes, "predicate source changed during audit")
    separation = sorted(row["right_minus_left_ras_x_mm"] for row in rows)
    ratios = sorted(row["smaller_to_larger_occupancy_ratio"] for row in rows)
    volumes = sorted(row["source_mask_occupancy_ml"] for row in rows)

    def distribution(values: list[float]) -> dict[str, float]:
        middle = len(values) // 2
        median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0
        return {"minimum": float(values[0]), "median": float(median),
                "maximum": float(values[-1])}

    result = {
        "schema": SCHEMA,
        "status": "measured_external_source_kidney_pair_geometry",
        "source": {
            "intake_path": intake_path.name,
            "intake_sha256": human.sha256(intake_path),
            "archive_path": archive_path.name,
            "archive_sha256": archive_sha,
            "source_id": source.get("source_id"),
            "source_release": source.get("release"),
            "coordinate_convention": source.get("coordinate_convention"),
            "scan_frames_kept_separate": True,
        },
        "trial_plan_sha256": plan_sha,
        "predicate_source_sha256": code_hashes,
        "method": {
            "label_name": LABEL_NAME,
            "label_id": label_id,
            "connectivity": CONNECTIVITY,
            "component_centroid": "arithmetic mean of occupied voxel centres transformed through each scan's NIfTI affine",
            "side_assignment": "lower/higher RAS+ X centroids are reported as left/right ordering only",
            "occupancy": "source label voxel count multiplied by affine determinant, expressed in mL",
            "no_cross_participant_coordinate_pooling": True,
        },
        "counts": {
            "scan_count": len(rows),
            "measured_scan_count": len(rows),
            "two_component_scan_count": sum(row["component_count"] == 2 for row in rows),
            "ordered_pair_scan_count": sum(row["right_minus_left_ras_x_mm"] > 0 for row in rows),
        },
        "distributions": {
            "right_minus_left_ras_x_mm": distribution(separation),
            "smaller_to_larger_source_mask_occupancy_ratio": distribution(ratios),
            "total_kidney_source_mask_occupancy_ml": distribution(volumes),
        },
        "qualification": {
            "source_label_pair_geometry_measured": True,
            "segmentation_accuracy": False,
            "numi_subject_binding": False,
            "physical_kidney_volume": False,
            "organ_mechanics": False,
            "physiology": False,
            "clinical_anatomy": False,
        },
        "rows": rows,
        "boundary": (
            "Automatic external CT kidney source-mask topology and scan-local RAS+ centroids only. "
            "The two components are described by lateral centroid order; this does not independently "
            "verify segmentation accuracy, clinical left/right labels, physical organ volume, Numi "
            "subject binding, mechanics, or physiology."
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compile_audit(args.intake, args.archive, args.plan)
    output = args.output.resolve()
    require(not output.exists() and not output.is_symlink(),
            "refusing to replace an existing output")
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "distributions": result["distributions"], "output": str(output)},
                     sort_keys=True))
    return 0 if result["counts"]["two_component_scan_count"] == len(result["rows"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
