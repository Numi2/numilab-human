"""Inventory TCIA's CC BY whole-body segmentation masks without admitting mechanics.

The archive contains 3-D label maps rather than a Numi-ready deformable Human.
This importer preserves each scan's NIfTI frame and per-mask voxel counts while
rejecting unknown labels, malformed geometry, and archive drift.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct
from typing import Any, BinaryIO
import xml.etree.ElementTree as ET
import zipfile

from .model import ImportError as HumanImportError
from .physiology import canonical, read_json


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.external-segmentation-source-ingest.v1"
SOURCE_CONFIG = ROOT / "config/healthy-total-body-cts-source.v1.json"
DEFAULT_ARCHIVE = ROOT / "Build/healthy-total-body-ct-20260923/healthy-total-body-ct-segmentations-v3.zip"
NIFTI_HEADER_BYTES = 352
NIFTI_CHUNK_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_MASK_BYTES = 2 * 1024 * 1024 * 1024
LABEL_NAMESPACE = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
CASE_FILE = re.compile(r"Healthy-Total-Body-CTs-(\d{3})\.nii\.gz\Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("healthy total-body CT source: " + message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_vector(values: tuple[float, ...], label: str) -> list[float]:
    require(all(math.isfinite(value) for value in values), f"{label} contains a non-finite value")
    return [float(value) for value in values]


def _rotation_from_quaternion(b: float, c: float, d: float) -> list[list[float]]:
    squared = b * b + c * c + d * d
    if squared > 1.0:
        magnitude = math.sqrt(squared)
        b, c, d = b / magnitude, c / magnitude, d / magnitude
        a = 0.0
    else:
        a = math.sqrt(max(0.0, 1.0 - squared))
    return [
        [a * a + b * b - c * c - d * d, 2.0 * (b * c - a * d), 2.0 * (b * d + a * c)],
        [2.0 * (b * c + a * d), a * a + c * c - b * b - d * d, 2.0 * (c * d - a * b)],
        [2.0 * (b * d - a * c), 2.0 * (c * d + a * b), a * a + d * d - c * c - b * b],
    ]


def parse_nifti_header(header: bytes) -> dict[str, Any]:
    require(len(header) == NIFTI_HEADER_BYTES, "NIfTI header is truncated")
    little = struct.unpack_from("<i", header, 0)[0]
    big = struct.unpack_from(">i", header, 0)[0]
    require(little == 348 or big == 348, "unsupported NIfTI header size or byte order")
    endian = "<" if little == 348 else ">"
    require(header[344:348] == b"n+1\x00", "only single-file NIfTI-1 images are supported")
    dimensions = struct.unpack_from(endian + "8h", header, 40)
    require(dimensions[0] == 3 and all(value == 1 for value in dimensions[4:]),
            "segmentation is not a single 3-D NIfTI volume")
    shape = [int(value) for value in dimensions[1:4]]
    require(all(value > 0 for value in shape), "NIfTI dimensions are not positive")
    datatype = struct.unpack_from(endian + "h", header, 70)[0]
    bitpix = struct.unpack_from(endian + "h", header, 72)[0]
    require(datatype == 16 and bitpix == 32,
            "segmentation voxel encoding changed; expected float32 labels")
    pixdim = struct.unpack_from(endian + "8f", header, 76)
    spacing = _finite_vector(tuple(abs(float(value)) for value in pixdim[1:4]), "voxel spacing")
    require(all(value > 0.0 for value in pixdim[1:4]), "NIfTI voxel spacing is not positive")
    offset = struct.unpack_from(endian + "f", header, 108)[0]
    require(math.isfinite(offset) and offset == NIFTI_HEADER_BYTES,
            "unexpected NIfTI data offset or extension")
    spatial_units = header[123] & 0x07
    units_by_code = {1: "m", 2: "mm", 3: "micron"}
    require(spatial_units in units_by_code, "NIfTI spatial units are missing or unsupported")
    qform_code = struct.unpack_from(endian + "h", header, 252)[0]
    sform_code = struct.unpack_from(endian + "h", header, 254)[0]
    slope, intercept = struct.unpack_from(endian + "2f", header, 112)
    require(math.isfinite(slope) and math.isfinite(intercept), "NIfTI intensity scaling is non-finite")
    if slope == 0.0:
        slope, intercept = 1.0, 0.0
    if sform_code > 0:
        affine = [list(struct.unpack_from(endian + "4f", header, offset)) for offset in (280, 296, 312)]
        affine_source = "sform"
    elif qform_code > 0:
        b, c, d = struct.unpack_from(endian + "3f", header, 256)
        rotation = _rotation_from_quaternion(float(b), float(c), float(d))
        qfac = -1.0 if pixdim[0] < 0 else 1.0
        scales = [float(pixdim[1]), float(pixdim[2]), qfac * float(pixdim[3])]
        offsets = struct.unpack_from(endian + "3f", header, 268)
        affine = [
            [rotation[row][column] * scales[column] for column in range(3)] + [float(offsets[row])]
            for row in range(3)
        ]
        affine_source = "qform"
    else:
        raise HumanImportError("healthy total-body CT source: NIfTI has neither qform nor sform")
    affine = [_finite_vector(tuple(row), "voxel-to-world affine") for row in affine]
    affine.append([0.0, 0.0, 0.0, 1.0])
    linear = [row[:3] for row in affine[:3]]
    determinant = (
        linear[0][0] * (linear[1][1] * linear[2][2] - linear[1][2] * linear[2][1])
        - linear[0][1] * (linear[1][0] * linear[2][2] - linear[1][2] * linear[2][0])
        + linear[0][2] * (linear[1][0] * linear[2][1] - linear[1][1] * linear[2][0])
    )
    require(math.isfinite(determinant) and abs(determinant) > 0.0,
            "voxel-to-world affine is singular")
    return {
        "shape_ijk": shape,
        "datatype": "float32",
        "bitpix": int(bitpix),
        "voxel_spacing": spacing,
        "spatial_units": units_by_code[spatial_units],
        "label_scaling_slope": float(slope),
        "label_scaling_intercept": float(intercept),
        "voxel_to_world_affine": affine,
        "affine_source": affine_source,
        "voxel_volume_in_source_units_cubed": abs(float(determinant)),
        "voxel_count": math.prod(shape),
        "data_offset_bytes": int(offset),
        "byte_order": "little" if endian == "<" else "big",
    }


def _workbook_shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    name = "xl/sharedStrings.xml"
    require(name in workbook.namelist(), "label workbook has no shared-string table")
    try:
        root = ET.fromstring(workbook.read(name))
    except (ET.ParseError, KeyError) as error:
        raise HumanImportError("healthy total-body CT source: label workbook shared strings are malformed") from error
    return ["".join(part.text or "" for part in entry.findall(".//m:t", LABEL_NAMESPACE))
            for entry in root.findall("m:si", LABEL_NAMESPACE)]


def read_label_dictionary(workbook_bytes: bytes) -> list[dict[str, Any]]:
    try:
        with zipfile.ZipFile(io.BytesIO(workbook_bytes)) as workbook:
            strings = _workbook_shared_strings(workbook)
            sheet = "xl/worksheets/sheet2.xml"
            require(sheet in workbook.namelist(), "label workbook tissue sheet is missing")
            root = ET.fromstring(workbook.read(sheet))
    except (zipfile.BadZipFile, ET.ParseError, KeyError) as error:
        raise HumanImportError("healthy total-body CT source: label workbook is malformed") from error

    rows: list[list[str]] = []
    for row in root.findall(".//m:sheetData/m:row", LABEL_NAMESPACE):
        values = []
        for cell in row.findall("m:c", LABEL_NAMESPACE):
            raw = cell.find("m:v", LABEL_NAMESPACE)
            value = raw.text if raw is not None else ""
            if cell.attrib.get("t") == "s" and value:
                index = int(value)
                require(0 <= index < len(strings), "label workbook shared-string index is invalid")
                value = strings[index]
            values.append(value)
        rows.append(values)
    require(rows and rows[0][:2] == ["Index", "Label"],
            "label workbook tissue sheet has unexpected columns")
    result = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for row in rows[1:]:
        if len(row) < 2 or not row[0] and not row[1]:
            continue
        try:
            label_id = int(row[0])
        except (TypeError, ValueError) as error:
            raise HumanImportError("healthy total-body CT source: label workbook contains a noninteger index") from error
        label = row[1].strip()
        require(label_id > 0 and label and label_id not in seen_ids and label not in seen_names,
                "label workbook has an empty or duplicate tissue identity")
        seen_ids.add(label_id)
        seen_names.add(label)
        result.append({"label_id": label_id, "name": label})
    require(result, "label workbook contains no tissue values")
    return sorted(result, key=lambda item: item["label_id"])


def _load_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise HumanImportError(
            "whole-body mask voxel audit requires NumPy; install numilab-human[volume-ingest]"
        ) from error
    return np


def _scan_mask(
    stream: BinaryIO,
    *,
    member_name: str,
    allowed_labels: set[int],
    np: Any,
) -> tuple[dict[str, Any], list[int]]:
    digest = hashlib.sha256()
    with gzip.GzipFile(fileobj=stream, mode="rb") as image:
        header = image.read(NIFTI_HEADER_BYTES)
        digest.update(header)
        info = parse_nifti_header(header)
        data_bytes_expected = info["voxel_count"] * (info["bitpix"] // 8)
        counts = [0] * (max(allowed_labels) + 1)
        data_bytes = 0
        while True:
            block = image.read(NIFTI_CHUNK_BYTES)
            if not block:
                break
            require(len(block) % 4 == 0, f"{member_name} ends within a float32 voxel")
            data_bytes += len(block)
            require(data_bytes <= data_bytes_expected, f"{member_name} has excess voxel data")
            digest.update(block)
            values = np.frombuffer(block, dtype="<f4" if info["byte_order"] == "little" else ">f4")
            if info["label_scaling_slope"] != 1.0 or info["label_scaling_intercept"] != 0.0:
                values = values * info["label_scaling_slope"] + info["label_scaling_intercept"]
            require(bool(np.isfinite(values).all()), f"{member_name} has non-finite labels")
            rounded = np.rint(values)
            require(bool(np.equal(values, rounded).all()), f"{member_name} has non-integer segmentation labels")
            if values.size:
                low = float(rounded.min())
                high = float(rounded.max())
                require(low >= 0 and high <= max(allowed_labels),
                        f"{member_name} contains a label outside the source dictionary")
                chunk_counts = np.bincount(rounded.astype(np.uint8, copy=False), minlength=len(counts))
                for label_id, count in enumerate(chunk_counts):
                    counts[label_id] += int(count)
        require(data_bytes == data_bytes_expected,
                f"{member_name} voxel payload length disagrees with NIfTI dimensions")
    observed = {label_id for label_id, count in enumerate(counts) if count > 0}
    require(observed <= allowed_labels | {0}, f"{member_name} contains an unmapped tissue label")
    info.update({
        "source_member": member_name,
        "nifti_uncompressed_sha256": digest.hexdigest(),
        "label_voxel_counts": {str(label_id): count for label_id, count in enumerate(counts) if count},
        "observed_nonbackground_label_ids": sorted(observed - {0}),
        "nonbackground_voxel_count": sum(counts) - counts[0],
    })
    return info, counts


def compile_source(archive: Path, *, source_config: Path = SOURCE_CONFIG) -> dict[str, Any]:
    archive = Path(archive)
    source_config = Path(source_config)
    require(archive.is_file() and not archive.is_symlink(), "source archive is not a regular file")
    require(archive.stat().st_size <= MAX_ARCHIVE_BYTES, "source archive exceeds the pinned size bound")
    config = read_json(source_config)
    require(isinstance(config, dict), "source registration is not an object")
    require(config.get("schema") == "numi.human.external-segmentation-source.v1" and
            config.get("source_id") == "healthy_total_body_cts_v3",
            "unsupported external source registration")
    raw_sha = _file_sha256(archive)
    require(archive.stat().st_size == config.get("archive_bytes") and
            raw_sha == config.get("archive_sha256"),
            "source archive bytes or SHA-256 differ from the registered release")

    try:
        source_zip = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as error:
        raise HumanImportError("healthy total-body CT source: source archive is not a valid ZIP") from error
    with source_zip:
        corrupt = source_zip.testzip()
        require(corrupt is None, f"source archive CRC failed at {corrupt}")
        names = source_zip.namelist()
        prefix = config["nifti_member_directory"]
        scan_members = sorted(
            name for name in names
            if name.startswith(prefix) and name.endswith(".nii.gz") and "/._" not in name
        )
        expected_count = config.get("expected_scan_count")
        require(type(expected_count) is int and len(scan_members) == expected_count,
                "segmentation-mask count differs from the registered cohort")
        case_ids = []
        for name in scan_members:
            basename = name.rsplit("/", 1)[-1]
            match = CASE_FILE.fullmatch(basename)
            require(match is not None, f"unrecognized segmentation filename: {basename}")
            case_ids.append(match.group(1))
            entry = source_zip.getinfo(name)
            require(0 < entry.file_size <= MAX_MASK_BYTES,
                    f"source member has an invalid expanded size: {basename}")
        require(len(case_ids) == len(set(case_ids)), "cohort has duplicate scan identities")
        workbook_name = config["label_dictionary_member"]
        require(workbook_name in names, "source archive has no registered label workbook")
        label_bytes = source_zip.read(workbook_name)
        labels = read_label_dictionary(label_bytes)
        label_map = {row["label_id"]: row["name"] for row in labels}
        require(0 not in label_map, "label dictionary uses the reserved background value")

        np = _load_numpy()
        cohort_scans = []
        subject_coverage = {label_id: 0 for label_id in label_map}
        voxel_volume_values = []
        common_shape = None
        for member in scan_members:
            with source_zip.open(member) as compressed_nifti:
                scan, _ = _scan_mask(
                    compressed_nifti,
                    member_name=member,
                    allowed_labels=set(label_map),
                    np=np,
                )
            require(scan["spatial_units"] == config.get("expected_spatial_units"),
                    f"{member} has unexpected spatial units")
            if common_shape is None:
                common_shape = scan["shape_ijk"]
            require(scan["shape_ijk"] == common_shape,
                    "cohort scans use inconsistent voxel-array dimensions")
            require(scan["shape_ijk"] == config.get("expected_shape_ijk"),
                    "voxel-array dimensions differ from the registered release")
            accepted_spacings = config.get("accepted_voxel_spacings_mm")
            require(isinstance(accepted_spacings, list) and any(
                isinstance(expected, list) and len(expected) == 3 and
                all(abs(actual - reference) <= 1.0e-6
                    for actual, reference in zip(scan["voxel_spacing"], expected, strict=True))
                for expected in accepted_spacings
            ), f"{member} voxel spacing differs from registered source observations")
            voxel_volume_values.append(scan["voxel_volume_in_source_units_cubed"])
            for label_id in scan["observed_nonbackground_label_ids"]:
                require(label_id in subject_coverage, f"{member} uses unmapped tissue label {label_id}")
                subject_coverage[label_id] += 1
            cohort_scans.append({
                "scan_id": CASE_FILE.fullmatch(member.rsplit("/", 1)[-1]).group(1),
                "nifti_member": member,
                "nifti_uncompressed_sha256": scan["nifti_uncompressed_sha256"],
                "label_scaling_slope": scan["label_scaling_slope"],
                "label_scaling_intercept": scan["label_scaling_intercept"],
                "voxel_spacing_mm": scan["voxel_spacing"],
                "voxel_volume_mm3": scan["voxel_volume_in_source_units_cubed"],
                "voxel_to_world_affine": scan["voxel_to_world_affine"],
                "affine_source": scan["affine_source"],
                "label_voxel_counts": scan["label_voxel_counts"],
                "label_raster_volume_candidate_ml": {
                    label_id: count * scan["voxel_volume_in_source_units_cubed"] / 1000.0
                    for label_id, count in scan["label_voxel_counts"].items()
                    if label_id != "0"
                },
                "observed_nonbackground_label_ids": scan["observed_nonbackground_label_ids"],
                "nonbackground_voxel_count": scan["nonbackground_voxel_count"],
            })

    observed_spacings = sorted({
        tuple(scan["voxel_spacing_mm"]) for scan in cohort_scans
    })
    observed_voxel_volumes = sorted({round(value, 9) for value in voxel_volume_values})
    require(archive.stat().st_size == config.get("archive_bytes") and
            _file_sha256(archive) == raw_sha,
            "source archive changed during the scan")
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.healthy-total-body-ct-source.3",
        "status": "partial_source_inventory",
        "source": {
            "source_id": config["source_id"],
            "title": config["title"],
            "release": config["release"],
            "doi": config["doi"],
            "dataset_page": config["dataset_page"],
            "archive_url": config["archive_url"],
            "archive_sha256": raw_sha,
            "archive_bytes": archive.stat().st_size,
            "license": config["license"],
            "license_url": config["license_url"],
            "citation": config["citation"],
            "provenance": config["provenance"],
            "source_config_sha256": _sha256(canonical(config)),
            "label_dictionary_member": config["label_dictionary_member"],
            "label_dictionary_sha256": _sha256(label_bytes),
            "segmentation_generation": "MOOSE automatic segmentation at the 90-minute timepoint, as described by TCIA",
            "excluded_assets": "DICOM CT images were not downloaded or used",
            "voxel_array_shape_ijk": common_shape,
            "documented_nominal_voxel_spacing_mm": config["expected_voxel_spacing_mm"],
            "observed_voxel_spacings_mm": [list(value) for value in observed_spacings],
            "observed_voxel_volumes_mm3": observed_voxel_volumes,
            "coordinate_convention": "NIfTI RAS+ in millimetres; source scans retain independent voxel-to-world affines",
        },
        "counts": {
            "scan_count": len(cohort_scans),
            "data_dictionary_label_count": len(labels),
            "observed_label_count": sum(count > 0 for count in subject_coverage.values()),
            "unobserved_dictionary_label_count": sum(count == 0 for count in subject_coverage.values()),
        },
        "labels": [
            {**row, "scan_coverage_count": subject_coverage[row["label_id"]]}
            for row in labels
        ],
        "scans": cohort_scans,
        "qualification": {
            "release_archive_identity_verified": True,
            "zip_member_crc_verified": True,
            "all_registered_segmentation_masks_scanned": True,
            "nifti_geometry_and_units_verified": True,
            "integer_labels_joined_to_source_dictionary": True,
            "source_subjects_bound_to_current_numi_subject": False,
            "expert_segmentation_accuracy_qualified": False,
            "skin_layer_registered": False,
            "vascular_lumen_registered": False,
            "physical_tissue_volume_owner": False,
            "material_calibration": False,
            "mechanical_or_physiological_admission": False,
            "clinical_qualification": False,
        },
        "boundary": config["evidence_boundary"],
    }


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
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE,
                        help="TCIA CC BY segmentation ZIP; DICOM CT images are not needed")
    parser.add_argument("--source-config", type=Path, default=SOURCE_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    result = compile_source(arguments.archive, source_config=arguments.source_config)
    output = arguments.output.resolve()
    digest = _immutable_write(output, result)
    print(json.dumps({"schema": SCHEMA, "output": str(output), "sha256": digest,
                      **result["counts"], **result["qualification"]}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    arguments = parser.parse_args(argv)
    try:
        return run(arguments)
    except (HumanImportError, OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as error:
        print(f"healthy total-body CT source: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
