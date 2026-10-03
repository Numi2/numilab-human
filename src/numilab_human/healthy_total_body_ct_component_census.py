"""Recheck connected components in source-labelled CT masks across a cohort.

This reports voxel-grid connectivity for external segmentation labels only.
It does not infer vessel lumens or admit scan-local masks as Numi anatomy.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import zipfile
from typing import Any

from . import model as human
from .healthy_total_body_ct_source import (
    MAX_MASK_BYTES,
    NIFTI_HEADER_BYTES,
    _load_numpy,
    parse_nifti_header,
)


SCHEMA = "numi.healthy-total-body-ct-component-census.v1"
PLAN_SCHEMA = "numi.healthy-total-body-ct-component-census-plan.v1"
INTAKE_SCHEMA = "HumanPack.external-segmentation-source-ingest.v2"
CODE_FILES = (
    "healthy_total_body_ct_component_census.py",
    "healthy_total_body_ct_source.py",
)
TARGET_LABELS = (
    "Aorta",
    "VCI",
    "Bladder",
    "Brain",
    "Heart",
    "Kidneys",
    "Liver",
    "Lung",
    "Pancreas",
    "Spleen",
    "Thyroid",
    "Psoas",
    "Skeletal-muscle",
    "Subcutaneous-fat",
    "Torso-fat",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise human.ImportError("healthy CT component census: " + message)


def _component_stats(mask: Any, *, np: Any, ndimage: Any) -> dict[str, Any]:
    require(getattr(mask, "ndim", None) == 3 and bool(mask.any()),
            "component input must be a nonempty three-dimensional mask")
    voxel_count = int(mask.sum(dtype=np.int64))
    structures = {
        "face6": ndimage.generate_binary_structure(3, 1),
        "full26": np.ones((3, 3, 3), dtype=np.bool_),
    }
    result: dict[str, Any] = {"voxel_count": voxel_count}
    for name, structure in structures.items():
        component_map, component_count = ndimage.label(mask, structure=structure)
        require(component_count > 0, f"{name} connectivity lost a nonempty mask")
        sizes = np.bincount(component_map.reshape(-1), minlength=component_count + 1)[1:]
        largest = int(sizes.max())
        result[name] = {
            "component_count": int(component_count),
            "largest_component_voxel_count": largest,
            "largest_component_fraction": largest / voxel_count,
        }
    return result


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


def _read_scan(
    archive: zipfile.ZipFile,
    scan: dict[str, Any],
    labels_by_name: dict[str, int],
    expected_shape: list[int],
    *,
    np: Any,
    ndimage: Any,
) -> list[dict[str, Any]]:
    scan_id = scan.get("scan_id")
    member = scan.get("nifti_member")
    require(isinstance(scan_id, str) and scan_id
            and isinstance(member, str) and member in archive.namelist(),
            "scan intake does not name a retained archive member")
    require("/../" not in f"/{member}/" and not member.startswith("/"),
            f"scan {scan_id} has an unsafe archive member path")
    info_record = archive.getinfo(member)
    require(0 < info_record.file_size <= MAX_MASK_BYTES,
            f"scan {scan_id} compressed mask exceeds the registered size bound")

    digest = hashlib.sha256()
    with archive.open(member) as zipped_member:
        with gzip.GzipFile(fileobj=zipped_member, mode="rb") as image:
            header = image.read(NIFTI_HEADER_BYTES)
            digest.update(header)
            info = parse_nifti_header(header)
            require(info["shape_ijk"] == expected_shape
                    and info["affine_source"] == scan.get("affine_source")
                    and _affine_close(info["voxel_to_world_affine"],
                                      scan.get("voxel_to_world_affine")),
                    f"scan {scan_id} current NIfTI header differs from intake")
            expected_bytes = info["voxel_count"] * (info["bitpix"] // 8)
            raw = image.read(expected_bytes + 1)
            require(len(raw) == expected_bytes and image.read(1) == b"",
                    f"scan {scan_id} NIfTI voxel payload length differs from its header")
            digest.update(raw)
    require(digest.hexdigest() == scan.get("nifti_uncompressed_sha256"),
            f"scan {scan_id} uncompressed NIfTI hash differs from intake")

    counts = scan.get("label_voxel_counts")
    geometry = scan.get("label_spatial_geometry_candidates")
    require(isinstance(counts, dict) and isinstance(geometry, dict),
            f"scan {scan_id} intake is missing source label counts or geometry")
    dtype = "<f4" if info["byte_order"] == "little" else ">f4"
    values = np.frombuffer(raw, dtype=dtype)
    require(values.size == info["voxel_count"],
            f"scan {scan_id} voxel array size differs from its header")
    voxels = values.reshape(tuple(info["shape_ijk"]), order="F")
    scaled = voxels
    if info["label_scaling_slope"] != 1.0 or info["label_scaling_intercept"] != 0.0:
        scaled = voxels * info["label_scaling_slope"] + info["label_scaling_intercept"]

    rows: list[dict[str, Any]] = []
    for name in TARGET_LABELS:
        label_id = labels_by_name[name]
        key = str(label_id)
        expected_count = counts.get(key, 0)
        if expected_count == 0:
            rows.append({
                "scan_id": scan_id,
                "label_id": label_id,
                "label": name,
                "status": "missing_label",
            })
            continue
        candidate = geometry.get(key)
        require(isinstance(candidate, dict)
                and type(candidate.get("voxel_count")) is int
                and candidate["voxel_count"] == expected_count,
                f"scan {scan_id}/{name} spatial candidate count differs from intake")
        bounds = candidate.get("voxel_center_bounds_ijk")
        require(isinstance(bounds, dict), f"scan {scan_id}/{name} has no voxel bounds")
        lower = bounds.get("minimum_inclusive")
        upper = bounds.get("maximum_inclusive")
        require(isinstance(lower, list) and isinstance(upper, list)
                and len(lower) == len(upper) == 3
                and all(type(lo) is int and type(hi) is int
                        and 0 <= lo <= hi < info["shape_ijk"][axis]
                        for axis, (lo, hi) in enumerate(zip(lower, upper, strict=True))),
                f"scan {scan_id}/{name} has invalid source voxel bounds")
        crop = scaled[tuple(slice(lo, hi + 1) for lo, hi in zip(lower, upper, strict=True))]
        mask = crop == label_id
        measured_count = int(mask.sum(dtype=np.int64))
        require(measured_count == expected_count,
                f"scan {scan_id}/{name} voxel count differs from source intake")
        stats = _component_stats(mask, np=np, ndimage=ndimage)
        rows.append({
            "scan_id": scan_id,
            "label_id": label_id,
            "label": name,
            "status": "measured",
            **stats,
        })
        del crop, mask
    return rows


def _distribution(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    selected = [row for row in rows
                if row.get("label") == label and row.get("status") == "measured"]
    require(selected, f"no scan has the registered {label} label")
    for key in ("face6", "full26"):
        values = sorted(row[key]["component_count"] for row in selected)
        largest_fractions = sorted(row[key]["largest_component_fraction"] for row in selected)
        distribution = {
            "scan_count": len(values),
            "minimum": values[0],
            "median": (values[(len(values) - 1) // 2]
                       + values[len(values) // 2]) / 2.0,
            "maximum": values[-1],
            "single_component_scan_count": sum(value == 1 for value in values),
            "largest_component_fraction_minimum": largest_fractions[0],
            "largest_component_fraction_median": (
                largest_fractions[(len(values) - 1) // 2]
                + largest_fractions[len(values) // 2]
            ) / 2.0,
            "largest_component_fraction_maximum": largest_fractions[-1],
        }
        if key == "face6":
            result = {"face6": distribution}
        else:
            result["full26"] = distribution
    return result


def _validate_trial_plan(
    plan_path: Path,
    *,
    intake_sha: str,
    archive_sha: str,
    scan_count: int,
    source_id: str,
    source_release: str,
    label_ids: dict[str, int],
    code_hashes: dict[str, str],
) -> str:
    plan_path = Path(plan_path).resolve()
    require(plan_path.is_file() and not plan_path.is_symlink(),
            "preregistered trial plan is not a retained regular file")
    plan = human.read_json(plan_path)
    require(plan.get("schema") == PLAN_SCHEMA
            and plan.get("input_intake_sha256") == intake_sha
            and plan.get("source_archive_sha256") == archive_sha
            and plan.get("scan_count") == scan_count
            and plan.get("source_id") == source_id
            and plan.get("source_release") == source_release
            and plan.get("labels") == list(TARGET_LABELS)
            and plan.get("label_ids") == label_ids
            and plan.get("connectivity_modes") == ["face6", "full26"]
            and plan.get("predicate_source_sha256") == code_hashes
            and isinstance(plan.get("question"), str)
            and isinstance(plan.get("predictions"), list)
            and isinstance(plan.get("validity_checks"), list)
            and isinstance(plan.get("analysis"), list)
            and isinstance(plan.get("stop_rule"), str)
            and isinstance(plan.get("evidence_boundary"), str),
            "preregistered plan does not bind this source, code, and census design")
    return human.sha256(plan_path)


def compile_census(
    intake_path: Path,
    archive_path: Path,
    trial_plan_path: Path,
) -> dict[str, Any]:
    intake_path, archive_path = Path(intake_path).resolve(), Path(archive_path).resolve()
    require(intake_path.is_file() and not intake_path.is_symlink(),
            "source intake is not a retained regular file")
    require(archive_path.is_file() and not archive_path.is_symlink(),
            "source archive is not a retained regular file")
    intake_sha = human.sha256(intake_path)
    intake = human.read_json(intake_path)
    require(intake.get("schema") == INTAKE_SCHEMA,
            "unsupported source intake schema")
    source = intake.get("source")
    scans = intake.get("scans")
    labels = intake.get("labels")
    intake_counts = intake.get("counts")
    require(isinstance(source, dict) and isinstance(scans, list) and scans
            and isinstance(labels, list) and labels
            and isinstance(intake_counts, dict),
            "source intake is missing source, scan, or label records")
    archive_sha = source.get("archive_sha256")
    archive_bytes = source.get("archive_bytes")
    require(isinstance(archive_sha, str) and type(archive_bytes) is int
            and archive_path.stat().st_size == archive_bytes
            and human.sha256(archive_path) == archive_sha,
            "source archive byte count or SHA-256 differs from intake")
    expected_shape = source.get("voxel_array_shape_ijk")
    require(isinstance(expected_shape, list) and len(expected_shape) == 3
            and all(type(value) is int and value > 0 for value in expected_shape)
            and len(scans) == intake_counts.get("scan_count"),
            "source array shape or scan count is invalid")

    labels_by_name: dict[str, int] = {}
    labels_by_id: set[int] = set()
    for row in labels:
        require(isinstance(row, dict)
                and isinstance(row.get("name"), str)
                and type(row.get("label_id")) is int
                and row["label_id"] > 0
                and row["label_id"] not in labels_by_id
                and row["name"] not in labels_by_name,
                "source label dictionary has malformed or duplicate identities")
        labels_by_id.add(row["label_id"])
        labels_by_name[row["name"]] = row["label_id"]
    require(all(name in labels_by_name for name in TARGET_LABELS),
            "source label dictionary omits a selected anatomical label")

    try:
        import scipy.ndimage as ndimage
    except ImportError as error:
        raise human.ImportError(
            "healthy CT component census: install SciPy for 3-D connected components"
        ) from error
    np = _load_numpy()
    code_hashes = {name: human.sha256(Path(__file__).with_name(name))
                   for name in CODE_FILES}
    plan_sha = _validate_trial_plan(
        trial_plan_path,
        intake_sha=intake_sha,
        archive_sha=archive_sha,
        scan_count=len(scans),
        source_id=source.get("source_id"),
        source_release=source.get("release"),
        label_ids={name: labels_by_name[name] for name in TARGET_LABELS},
        code_hashes=code_hashes,
    )
    scan_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for scan in scans:
            require(isinstance(scan, dict), "source scan row is malformed")
            scan_id = scan.get("scan_id")
            require(isinstance(scan_id, str) and scan_id and scan_id not in scan_ids,
                    "source scan IDs must be nonempty and unique")
            scan_ids.add(scan_id)
            rows.extend(_read_scan(archive, scan, labels_by_name, expected_shape,
                                   np=np, ndimage=ndimage))
    require(human.sha256(archive_path) == archive_sha
            and human.sha256(intake_path) == intake_sha,
            "source archive or intake changed during census")
    expected_pairs = len(scans) * len(TARGET_LABELS)
    measured = [row for row in rows if row["status"] == "measured"]
    missing = [row for row in rows if row["status"] == "missing_label"]
    require(len(rows) == expected_pairs,
            "component-census row count differs from scan-label design")
    distributions = {name: _distribution(rows, name) for name in TARGET_LABELS
                     if any(row.get("label") == name and row.get("status") == "measured"
                            for row in rows)}
    result = {
        "schema": SCHEMA,
        "status": "completed_source_mask_connectivity_census",
        "source": {
            "source_id": source.get("source_id"),
            "release": source.get("release"),
            "doi": source.get("doi"),
            "license": source.get("license"),
            "intake_path": intake_path.name,
            "intake_sha256": intake_sha,
            "archive_sha256": archive_sha,
            "archive_bytes": archive_bytes,
            "scan_count": len(scans),
            "participant_frames_kept_separate": True,
        },
        "trial_plan_sha256": plan_sha,
        "method": {
            "experimental_unit": "one source scan and one registered label ID",
            "connectivity": {
                "face6": "voxels joined through shared faces",
                "full26": "voxels joined through faces, edges, or corners",
            },
            "component_count_is_voxel_mask_topology_only": True,
            "nifti_uncompressed_member_hashes_rechecked": True,
            "selected_voxel_counts_redecoded_and_matched_to_intake": True,
        },
        "labels": list(TARGET_LABELS),
        "counts": {
            "scan_count": len(scans),
            "selected_label_count": len(TARGET_LABELS),
            "expected_scan_label_pairs": expected_pairs,
            "measured_scan_label_pairs": len(measured),
            "missing_scan_label_pairs": len(missing),
        },
        "distributions": distributions,
        "qualification": {
            "source_mask_connectivity_measured": True,
            "source_vessel_lumen_connectivity": False,
            "current_numi_subject_binding": False,
            "physical_tissue_volume_owner": False,
            "clinical_anatomy": False,
            "mechanics_or_physiology": False,
        },
        "predicate_source_sha256": code_hashes,
        "rows": rows,
        "boundary": (
            "Connected-component counts describe source-label voxel adjacency in each "
            "scan's own array. They do not establish segmentation accuracy, anatomical "
            "surface quality, a vascular lumen or branch graph, subject binding, "
            "physical volume ownership, mechanics, physiology, or clinical anatomy."
        ),
    }
    require({name: human.sha256(Path(__file__).with_name(name)) for name in CODE_FILES}
            == code_hashes, "predicate source changed during compilation")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compile_census(args.intake, args.archive, args.plan)
    output = args.output.resolve()
    require(not output.exists() and not output.is_symlink(),
            "refusing to replace an existing output")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(payload)
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "output": str(output)}, sort_keys=True))
    return 0 if result["counts"]["missing_scan_label_pairs"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
