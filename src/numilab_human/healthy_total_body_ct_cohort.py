"""Summarize named TCIA label geometry across the registered CT cohort.

This consumes the immutable mask intake receipt. It does not reread images,
register participants, or promote raster occupancy to tissue volume or mass.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical


SCHEMA = "HumanPack.external-segmentation-source-cohort-summary.v1"
COMPILER = "numilab-human.healthy-total-body-ct-cohort.2"
INPUT_SCHEMA = "HumanPack.external-segmentation-source-ingest.v2"
MAX_RECEIPT_BYTES = 64 * 1024 * 1024
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("healthy total-body CT cohort: " + message)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: list[float], units: str) -> dict[str, Any]:
    if not values:
        return {
            "observed_scan_count": 0,
            "units": units,
            "minimum": None,
            "q25_linear": None,
            "median": None,
            "q75_linear": None,
            "maximum": None,
        }
    return {
        "observed_scan_count": len(values),
        "units": units,
        "minimum": min(values),
        "q25_linear": _quantile(values, 0.25),
        "median": _quantile(values, 0.5),
        "q75_linear": _quantile(values, 0.75),
        "maximum": max(values),
    }


def summarize_intake(receipt: dict[str, Any], *, input_receipt_sha256: str) -> dict[str, Any]:
    require(SHA256.fullmatch(input_receipt_sha256) is not None,
            "input receipt SHA-256 is malformed")
    require(receipt.get("schema") == INPUT_SCHEMA,
            "input is not a supported Healthy Total Body CT intake receipt")
    source = receipt.get("source")
    counts = receipt.get("counts")
    labels = receipt.get("labels")
    scans = receipt.get("scans")
    require(isinstance(source, dict) and isinstance(counts, dict) and
            isinstance(labels, list) and isinstance(scans, list),
            "intake receipt has incomplete source, count, label, or scan records")
    source_id = source.get("source_id")
    archive_sha = source.get("archive_sha256")
    dictionary_sha = source.get("label_dictionary_sha256")
    require(source_id == "healthy_total_body_cts_v3" and
            isinstance(archive_sha, str) and SHA256.fullmatch(archive_sha) and
            isinstance(dictionary_sha, str) and SHA256.fullmatch(dictionary_sha),
            "source or label dictionary identity is incomplete")
    shape = source.get("voxel_array_shape_ijk")
    require(isinstance(shape, list) and len(shape) == 3 and
            all(type(value) is int and value > 0 for value in shape),
            "source voxel-array shape is incomplete")
    require(scans and len(scans) == counts.get("scan_count"),
            "scan count differs from the intake summary")
    scan_count = len(scans)

    label_map: dict[int, dict[str, Any]] = {}
    names: set[str] = set()
    for row in labels:
        require(isinstance(row, dict), "label dictionary contains a non-object row")
        label_id = row.get("label_id")
        name = row.get("name")
        coverage = row.get("scan_coverage_count")
        require(type(label_id) is int and label_id > 0 and isinstance(name, str) and name.strip(),
                "label dictionary contains an invalid source identity")
        require(label_id not in label_map and name not in names,
                "label dictionary contains duplicate identities")
        require(type(coverage) is int and 0 <= coverage <= scan_count,
                f"label {label_id} has invalid scan coverage")
        label_map[label_id] = row
        names.add(name)
    require(labels and len(labels) == counts.get("data_dictionary_label_count"),
            "label dictionary count differs from the intake summary")

    scan_by_id: dict[str, dict[str, Any]] = {}
    candidates_by_label: dict[int, list[dict[str, Any]]] = {
        label_id: [] for label_id in label_map
    }
    coverage_by_label = {label_id: 0 for label_id in label_map}
    scan_frames = []
    for scan in scans:
        require(isinstance(scan, dict), "scan inventory contains a non-object row")
        scan_id = scan.get("scan_id")
        image_sha = scan.get("nifti_uncompressed_sha256")
        voxel_volume = scan.get("voxel_volume_mm3")
        voxel_spacing = scan.get("voxel_spacing_mm")
        affine = scan.get("voxel_to_world_affine")
        require(isinstance(scan_id, str) and re.fullmatch(r"\d{3}", scan_id) and
                scan_id not in scan_by_id, "scan inventory has a malformed or duplicate ID")
        require(isinstance(image_sha, str) and SHA256.fullmatch(image_sha),
                f"scan {scan_id} has a malformed NIfTI identity")
        require(isinstance(voxel_volume, (int, float)) and math.isfinite(voxel_volume) and
                voxel_volume > 0.0, f"scan {scan_id} has invalid voxel volume")
        require(isinstance(voxel_spacing, list) and len(voxel_spacing) == 3 and
                all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0.0
                    for v in voxel_spacing), f"scan {scan_id} has invalid voxel spacing")
        require(isinstance(affine, list) and len(affine) == 4 and
                all(isinstance(row, list) and len(row) == 4 and
                    all(isinstance(v, (int, float)) and math.isfinite(v) for v in row)
                    for row in affine), f"scan {scan_id} has invalid affine")
        require(scan.get("affine_source") in ("qform", "sform") and
                affine[3] == [0.0, 0.0, 0.0, 1.0],
                f"scan {scan_id} has an invalid affine source or homogeneous row")
        linear = [row[:3] for row in affine[:3]]
        determinant = (
            linear[0][0] * (linear[1][1] * linear[2][2] - linear[1][2] * linear[2][1])
            - linear[0][1] * (linear[1][0] * linear[2][2] - linear[1][2] * linear[2][0])
            + linear[0][2] * (linear[1][0] * linear[2][1] - linear[1][1] * linear[2][0])
        )
        require(math.isclose(abs(determinant), float(voxel_volume),
                             rel_tol=1.0e-6, abs_tol=1.0e-8),
                f"scan {scan_id} voxel volume disagrees with its affine")
        raw_counts = scan.get("label_voxel_counts")
        raw_volumes = scan.get("label_raster_volume_candidate_ml")
        raw_geometry = scan.get("label_spatial_geometry_candidates")
        observed_ids = scan.get("observed_nonbackground_label_ids")
        require(all(isinstance(v, dict) for v in (raw_counts, raw_volumes, raw_geometry)) and
                isinstance(observed_ids, list), f"scan {scan_id} has incomplete label geometry")
        require(all(type(value) is int for value in observed_ids) and
                len(observed_ids) == len(set(observed_ids)),
                f"scan {scan_id} has duplicate or invalid observed label IDs")
        observed = set()
        for raw_id, value in raw_counts.items():
            try:
                label_id = int(raw_id)
            except (TypeError, ValueError) as error:
                raise HumanImportError(
                    f"healthy total-body CT cohort: scan {scan_id} has a noninteger label key"
                ) from error
            require(str(label_id) == raw_id and type(value) is int and value >= 0,
                    f"scan {scan_id} has an invalid voxel count")
            if label_id == 0:
                continue
            require(label_id in label_map and value > 0,
                    f"scan {scan_id} has an unknown or empty nonbackground label")
            observed.add(label_id)
            expected_volume = value * float(voxel_volume) / 1000.0
            volume = raw_volumes.get(raw_id)
            geometry = raw_geometry.get(raw_id)
            require(isinstance(volume, (int, float)) and math.isfinite(volume) and volume > 0.0 and
                    math.isclose(float(volume), expected_volume, rel_tol=1.0e-12, abs_tol=1.0e-12),
                    f"scan {scan_id} label {label_id} raster volume disagrees with voxel count")
            require(isinstance(geometry, dict) and
                    type(geometry.get("voxel_count")) is int and geometry["voxel_count"] == value,
                    f"scan {scan_id} label {label_id} spatial count disagrees with voxel count")
            centroid_ijk = geometry.get("centroid_voxel_center_ijk")
            centroid_ras = geometry.get("centroid_ras_mm")
            envelope = geometry.get("voxel_envelope_aabb_ras_mm")
            bounds = geometry.get("voxel_center_bounds_ijk")
            require(isinstance(centroid_ijk, list) and len(centroid_ijk) == 3 and
                    isinstance(centroid_ras, list) and len(centroid_ras) == 3 and
                    isinstance(envelope, dict) and isinstance(bounds, dict),
                    f"scan {scan_id} label {label_id} spatial candidate is incomplete")
            require(all(isinstance(v, (int, float)) and math.isfinite(v)
                        for v in centroid_ijk + centroid_ras),
                    f"scan {scan_id} label {label_id} centroid is non-finite")
            lower = bounds.get("minimum_inclusive")
            upper = bounds.get("maximum_inclusive")
            envelope_low = envelope.get("minimum")
            envelope_high = envelope.get("maximum")
            require(isinstance(lower, list) and isinstance(upper, list) and
                    len(lower) == len(upper) == 3 and
                    all(type(lo) is int and type(hi) is int and
                        0 <= lo <= hi < shape[axis]
                        for axis, (lo, hi) in enumerate(zip(lower, upper, strict=True))),
                    f"scan {scan_id} label {label_id} voxel bounds are invalid")
            require(isinstance(envelope_low, list) and isinstance(envelope_high, list) and
                    len(envelope_low) == len(envelope_high) == 3 and
                    all(isinstance(v, (int, float)) and math.isfinite(v)
                        for v in envelope_low + envelope_high) and
                    all(lo <= hi for lo, hi in zip(envelope_low, envelope_high, strict=True)),
                    f"scan {scan_id} label {label_id} world envelope is invalid")
            require(all(lower[axis] - 1.0e-9 <= centroid_ijk[axis] <= upper[axis] + 1.0e-9
                        for axis in range(3)),
                    f"scan {scan_id} label {label_id} voxel centroid leaves its bounds")
            transformed = [
                sum(float(affine[axis][j]) * float(centroid_ijk[j]) for j in range(3))
                + float(affine[axis][3])
                for axis in range(3)
            ]
            require(all(math.isclose(transformed[axis], float(centroid_ras[axis]),
                                     rel_tol=1.0e-7, abs_tol=1.0e-4)
                        for axis in range(3)),
                    f"scan {scan_id} label {label_id} RAS centroid disagrees with its affine")
            corners = itertools.product(
                (lower[0] - 0.5, upper[0] + 0.5),
                (lower[1] - 0.5, upper[1] + 0.5),
                (lower[2] - 0.5, upper[2] + 0.5),
            )
            transformed_corners = [
                [sum(float(affine[axis][j]) * point[j] for j in range(3))
                 + float(affine[axis][3]) for axis in range(3)]
                for point in corners
            ]
            expected_envelope = [
                [min(point[axis] for point in transformed_corners) for axis in range(3)],
                [max(point[axis] for point in transformed_corners) for axis in range(3)],
            ]
            require(all(math.isclose(float(envelope_bound[axis]),
                                     expected_envelope[bound_index][axis],
                                     rel_tol=1.0e-7, abs_tol=2.0e-4)
                        for bound_index, envelope_bound in enumerate((envelope_low, envelope_high))
                        for axis in range(3)),
                    f"scan {scan_id} label {label_id} RAS envelope disagrees with affine")
            candidates_by_label[label_id].append({
                "scan_id": scan_id,
                "voxel_count": value,
                "voxel_occupancy_volume_candidate_ml": float(volume),
                "centroid_voxel_center_ijk": centroid_ijk,
                "voxel_center_bounds_ijk": bounds,
                "centroid_ras_mm": centroid_ras,
                "voxel_envelope_aabb_ras_mm": envelope,
            })
        require(observed == set(observed_ids),
                f"scan {scan_id} observed label list disagrees with voxel counts")
        require(set(raw_volumes) == {str(i) for i in observed} and
                set(raw_geometry) == {str(i) for i in observed},
                f"scan {scan_id} volume/spatial candidates disagree with observed labels")
        for label_id in observed:
            coverage_by_label[label_id] += 1
        scan_by_id[scan_id] = scan
        scan_frames.append({
            "scan_id": scan_id,
            "nifti_uncompressed_sha256": image_sha,
            "affine_source": scan.get("affine_source"),
            "voxel_spacing_mm": voxel_spacing,
            "voxel_volume_mm3": float(voxel_volume),
            "voxel_to_world_affine": affine,
        })

    require(len(scan_by_id) == scan_count, "scan IDs are not unique")
    scan_ids = sorted(scan_by_id)
    require(sum(coverage_by_label.values()) == sum(
        len(candidates) for candidates in candidates_by_label.values()
    ), "label coverage accounting is inconsistent")
    label_rows = []
    for label_id, dictionary_row in sorted(label_map.items()):
        candidates = sorted(candidates_by_label[label_id], key=lambda row: row["scan_id"])
        require(coverage_by_label[label_id] == dictionary_row["scan_coverage_count"],
                f"label {label_id} scan coverage disagrees with intake dictionary")
        volumes = [row["voxel_occupancy_volume_candidate_ml"] for row in candidates]
        voxel_counts = [float(row["voxel_count"]) for row in candidates]
        present = {row["scan_id"] for row in candidates}
        label_rows.append({
            "label_id": label_id,
            "source_semantic_id": f"{source_id}:segmentation-label:{label_id}",
            "name": dictionary_row["name"],
            "scan_coverage_count": len(candidates),
            "scan_coverage_fraction": len(candidates) / scan_count,
            "missing_scan_ids": [scan_id for scan_id in scan_ids if scan_id not in present],
            "voxel_count_distribution": _distribution(voxel_counts, "voxels"),
            "voxel_occupancy_volume_candidate_ml_distribution": _distribution(volumes, "mL"),
            "per_scan_spatial_candidates": candidates,
        })

    observed_label_count = sum(coverage > 0 for coverage in coverage_by_label.values())
    require(observed_label_count == counts.get("observed_label_count") and
            len(label_map) - observed_label_count == counts.get("unobserved_dictionary_label_count"),
            "label coverage totals disagree with intake summary")
    qualification = receipt.get("qualification")
    require(isinstance(qualification, dict) and
            qualification.get("release_archive_identity_verified") is True and
            qualification.get("zip_member_crc_verified") is True and
            qualification.get("all_registered_segmentation_masks_scanned") is True and
            qualification.get("nifti_geometry_and_units_verified") is True and
            qualification.get("integer_labels_joined_to_source_dictionary") is True,
            "source intake lacks its archive, CRC, or full-scan verification gates")
    return {
        "schema": SCHEMA,
        "compiler": COMPILER,
        "status": "source_cohort_geometry_summary",
        "source": {
            "source_id": source_id,
            "release": source.get("release"),
            "doi": source.get("doi"),
            "license": source.get("license"),
            "archive_sha256": archive_sha,
            "label_dictionary_sha256": dictionary_sha,
            "intake_receipt_sha256": input_receipt_sha256,
            "intake_schema": INPUT_SCHEMA,
            "coordinate_convention": source.get("coordinate_convention"),
            "segmentation_generation": source.get("segmentation_generation"),
        },
        "cohort": {
            "scan_count": scan_count,
            "label_dictionary_count": len(label_map),
            "observed_label_count": observed_label_count,
            "unobserved_label_count": len(label_map) - observed_label_count,
            "distribution_quantile_method": "linear interpolation at (n - 1) * probability; observed scans only",
            "participant_frames_registered_to_one_another": False,
        },
        "scan_frames": scan_frames,
        "labels": label_rows,
        "qualification": {
            "input_intake_schema_and_hash_recorded": True,
            "per_scan_counts_volumes_and_spatial_candidates_reconciled": True,
            "label_names_preserved_from_source_dictionary": True,
            "source_intake_archive_crc_nifti_and_label_gates_verified": True,
            "source_masks_redecoded": False,
            "cross_scan_registration": False,
            "source_subjects_bound_to_current_numi_subject": False,
            "physical_tissue_volume_owner": False,
            "mechanical_or_physiological_admission": False,
            "clinical_qualification": False,
        },
        "boundary": (
            "Descriptive statistics over source segmentation labels, voxel counts, "
            "affine-scaled raster occupancy candidates, and independently framed "
            "spatial candidates. Source label identities are not cross-suite ontology "
            "mappings; cohort coordinates are not pooled; no physical tissue volume, "
            "mass, mechanics, calibration, or clinical claim is made."
        ),
    }


def compile_from_receipt(path: Path) -> dict[str, Any]:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "input receipt is not a regular file")
    require(path.stat().st_size <= MAX_RECEIPT_BYTES, "input receipt exceeds the size limit")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise HumanImportError("healthy total-body CT cohort: input receipt is malformed JSON") from error
    require(isinstance(value, dict), "input receipt is not an object")
    return summarize_intake(value, input_receipt_sha256=hashlib.sha256(raw).hexdigest())


def _immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    require(not path.is_symlink(), "output is redirected")
    if path.exists():
        require(path.read_bytes() == payload, "output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--receipt", type=Path, required=True,
                        help="validated healthy-total-body-ct-source intake receipt")
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    result = compile_from_receipt(arguments.receipt)
    output = arguments.output.resolve()
    digest = _immutable_write(output, result)
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(output),
        "sha256": digest,
        **result["cohort"],
        **result["qualification"],
    }, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    arguments = parser.parse_args(argv)
    try:
        return run(arguments)
    except (HumanImportError, OSError, TypeError, ValueError) as error:
        print(f"healthy total-body CT cohort: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
