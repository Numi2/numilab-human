"""Resolve the scan-local component geometry of TCIA's Psoas mask."""
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
from .healthy_total_body_ct_paired_kidney_audit import (
    _affine_close,
    _determinant3,
    _stream_label_crop,
)
from .healthy_total_body_ct_source import (
    MAX_MASK_BYTES,
    NIFTI_HEADER_BYTES,
    _load_numpy,
    parse_nifti_header,
)
from .whole_body_embeddedness import atomic_json


SCHEMA = "numi.healthy-total-body-ct-psoas-component-audit.v1"
PLAN_SCHEMA = "numi.healthy-total-body-ct-psoas-component-plan.v1"
INTAKE_SCHEMA = "HumanPack.external-segmentation-source-ingest.v2"
CODE_FILES = ("healthy_total_body_ct_psoas_component_audit.py",
              "healthy_total_body_ct_paired_kidney_audit.py",
              "healthy_total_body_ct_source.py")
LABEL_NAME = "Psoas"
EXPECTED_LABEL_ID = 36
CONNECTIVITY = "full26"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise human.ImportError("healthy CT Psoas component audit: " + message)


def _distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    require(bool(ordered), "cannot summarize an empty cohort distribution")
    middle = len(ordered) // 2
    median = (ordered[middle] if len(ordered) % 2
              else (ordered[middle - 1] + ordered[middle]) / 2.0)
    return {"minimum": float(ordered[0]), "median": float(median),
            "maximum": float(ordered[-1])}


def _summarize_mask(mask: Any, lower_ijk: list[int], affine: list[list[float]],
                    voxel_volume_mm3: float, *, np: Any, ndimage: Any) -> dict[str, Any]:
    require(mask.ndim == 3 and bool(mask.any()), "Psoas crop must be nonempty and three-dimensional")
    measured_count = int(mask.sum(dtype=np.int64))
    component_map, component_count = ndimage.label(
        mask, structure=np.ones((3, 3, 3), dtype=np.bool_))
    require(component_count > 0, "Psoas has no 26-connected component")
    counts = np.bincount(component_map.reshape(-1), minlength=component_count + 1)[1:]
    centers = ndimage.center_of_mass(mask, component_map,
                                     list(range(1, component_count + 1)))
    components = []
    for component_id, (count, local_center) in enumerate(
            zip(counts.tolist(), centers, strict=True), start=1):
        ijk = [float(local_center[axis] + lower_ijk[axis]) for axis in range(3)]
        ras = [sum(affine[row][column] * point
                   for column, point in enumerate((*ijk, 1.0)))
               for row in range(3)]
        components.append({
            "component_id": component_id,
            "voxel_count": int(count),
            "source_mask_occupancy_ml": float(count * voxel_volume_mm3 / 1000.0),
            "fraction_of_label_voxels": float(count / measured_count),
            "centroid_voxel_center_ijk": ijk,
            "centroid_ras_mm": ras,
        })
    components.sort(key=lambda component: component["centroid_ras_mm"][0])
    for rank, component in enumerate(components, start=1):
        component["ras_x_order"] = rank
    by_size = sorted(components, key=lambda component: component["voxel_count"], reverse=True)
    two_largest = by_size[:2]
    balance = min(two_largest[0]["voxel_count"], two_largest[1]["voxel_count"]) / max(
        two_largest[0]["voxel_count"], two_largest[1]["voxel_count"])
    smallest_component = min(components, key=lambda component: component["voxel_count"])
    return {
        "component_count": int(component_count),
        "total_source_mask_occupancy_ml": float(measured_count * voxel_volume_mm3 / 1000.0),
        "smallest_component_fraction_of_label_voxels": smallest_component["fraction_of_label_voxels"],
        "two_largest_component_balance_ratio": float(balance),
        "components_ordered_by_ras_x": components,
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
                    f"scan {scan_id} Psoas mask is absent or disagrees with intake")
            bounds = candidate.get("voxel_center_bounds_ijk")
            require(isinstance(bounds, dict), f"scan {scan_id} Psoas crop bounds are absent")
            lower, upper = bounds.get("minimum_inclusive"), bounds.get("maximum_inclusive")
            require(isinstance(lower, list) and isinstance(upper, list)
                    and len(lower) == len(upper) == 3
                    and all(type(lo) is int and type(hi) is int and 0 <= lo <= hi < shape_ijk[i]
                            for i, (lo, hi) in enumerate(zip(lower, upper, strict=True))),
                    f"scan {scan_id} Psoas crop bounds are invalid")
            mask = _stream_label_crop(image, info, lower, upper, label_id,
                                      np=np, digest=digest)
    require(digest.hexdigest() == scan.get("nifti_uncompressed_sha256"),
            f"scan {scan_id} uncompressed NIfTI hash differs from intake")
    require(info["spatial_units"] == "mm", f"scan {scan_id} is not expressed in millimetres")
    measured_count = int(mask.sum(dtype=np.int64))
    require(measured_count == expected_count,
            f"scan {scan_id} Psoas voxel count differs from intake")
    affine = info["voxel_to_world_affine"]
    voxel_volume = abs(_determinant3(affine))
    require(math.isclose(voxel_volume, float(scan.get("voxel_volume_mm3", math.nan)),
                         rel_tol=1.0e-8, abs_tol=1.0e-10),
            f"scan {scan_id} affine voxel volume differs from intake")

    component_summary = _summarize_mask(mask, lower, affine, voxel_volume,
                                        np=np, ndimage=ndimage)
    return {
        "scan_id": scan_id,
        "nifti_member": member,
        "nifti_uncompressed_sha256": digest.hexdigest(),
        "affine_source": info["affine_source"],
        "affine_sha256": hashlib.sha256(json.dumps(
            affine, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()).hexdigest(),
        "label_id": label_id,
        "label_name": LABEL_NAME,
        "source_label_voxel_count": expected_count,
        "voxel_to_world_affine": affine,
        "connectivity": CONNECTIVITY,
        **component_summary,
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
    source, scans, labels = intake.get("source"), intake.get("scans"), intake.get("labels")
    require(isinstance(source, dict) and isinstance(scans, list) and scans
            and isinstance(labels, list), "intake lacks source, scan, or label records")
    archive_sha = human.sha256(archive_path)
    require(archive_sha == source.get("archive_sha256"),
            "source archive hash differs from intake")
    matches = [row for row in labels if isinstance(row, dict)
               and row.get("name") == LABEL_NAME]
    require(len(matches) == 1 and matches[0].get("label_id") == EXPECTED_LABEL_ID,
            "source label dictionary Psoas identity differs")
    label_id = matches[0]["label_id"]
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
        raise human.ImportError("healthy CT Psoas audit requires SciPy") from error
    rows, seen = [], set()
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
    component_counts = [row["component_count"] for row in rows]
    extra = [row for row in rows if row["component_count"] > 2]
    extra_fractions = sorted(row["smallest_component_fraction_of_label_voxels"]
                             for row in extra)
    balances = sorted(row["two_largest_component_balance_ratio"] for row in rows)

    result = {
        "schema": SCHEMA,
        "status": "measured_external_source_psoas_component_geometry",
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
            "component_centroids": "occupied voxel-centre means transformed through each scan's NIfTI affine",
            "component_order": "components are ranked by RAS+ X for reporting; this does not confer named muscle or clinical laterality",
            "occupancy": "source label voxel count multiplied by affine determinant, expressed in mL",
            "no_cross_participant_coordinate_pooling": True,
        },
        "counts": {
            "scan_count": len(rows),
            "measured_scan_count": len(rows),
            "two_component_scan_count": sum(count == 2 for count in component_counts),
            "three_component_scan_count": sum(count == 3 for count in component_counts),
            "more_than_three_component_scan_count": sum(count > 3 for count in component_counts),
        },
        "distributions": {
            "full26_component_count": _distribution([float(value) for value in component_counts]),
            "two_largest_component_balance_ratio": _distribution(balances),
            "extra_component_smallest_fraction_among_exception_scans": (
                _distribution(extra_fractions) if extra_fractions else None),
        },
        "exception_scans": [{
            "scan_id": row["scan_id"],
            "component_count": row["component_count"],
            "smallest_component_fraction_of_label_voxels": row["smallest_component_fraction_of_label_voxels"],
            "components_ordered_by_ras_x": row["components_ordered_by_ras_x"],
        } for row in extra],
        "qualification": {
            "source_label_component_geometry_measured": True,
            "individual_muscle_identity": False,
            "clinical_segmentation_accuracy": False,
            "numi_subject_binding": False,
            "physical_muscle_volume": False,
            "muscle_mechanics": False,
        },
        "rows": rows,
        "boundary": (
            "Automatic external CT Psoas source-mask component geometry only. "
            "RAS+ centroid ordering and voxel occupancy do not establish named "
            "muscle identity, clinical segmentation accuracy, physical volume, "
            "Numi subject binding, or mechanics."
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
                      "distributions": result["distributions"],
                      "exception_scans": result["exception_scans"],
                      "output": str(output)}, sort_keys=True))
    return 0 if result["counts"]["more_than_three_component_scan_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
