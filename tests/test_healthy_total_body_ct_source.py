from __future__ import annotations

import gzip
import hashlib
import io
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import zipfile

import pytest

from numilab_human.healthy_total_body_ct_source import (
    compile_source,
    parse_nifti_header,
)
from numilab_human.model import ImportError
from numilab_human.physiology import canonical


NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def nifti_mask(values: list[float], *, shape: tuple[int, int, int] = (2, 2, 1)) -> bytes:
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, *shape, 1, 1, 1, 1)
    struct.pack_into("<h", header, 70, 16)
    struct.pack_into("<h", header, 72, 32)
    struct.pack_into("<8f", header, 76, -1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0)
    struct.pack_into("<f", header, 108, 352.0)
    header[123] = 2
    struct.pack_into("<h", header, 252, 1)
    struct.pack_into("<h", header, 254, 0)
    struct.pack_into("<3f", header, 256, 0.0, 0.0, 0.0)
    struct.pack_into("<3f", header, 268, 10.0, 20.0, 30.0)
    header[344:348] = b"n+1\x00"
    payload = struct.pack("<" + "f" * len(values), *values)
    return gzip.compress(bytes(header) + payload, mtime=0)


def label_workbook() -> bytes:
    shared = ["Index", "Label", "Heart", "Liver"]
    strings = "".join(f"<si><t>{value}</t></si>" for value in shared)
    shared_xml = f'<sst xmlns="{NS}" count="4" uniqueCount="4">{strings}</sst>'
    rows = [
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>',
        '<row r="2"><c r="A2"><v>1</v></c><c r="B2" t="s"><v>2</v></c></row>',
        '<row r="3"><c r="A3"><v>2</v></c><c r="B3" t="s"><v>3</v></c></row>',
    ]
    sheet_xml = f'<worksheet xmlns="{NS}"><sheetData>{"".join(rows)}</sheetData></worksheet>'
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr("xl/sharedStrings.xml", shared_xml)
        workbook.writestr("xl/worksheets/sheet2.xml", sheet_xml)
    return output.getvalue()


def make_source(tmp_path: Path, second_values: list[float] | None = None) -> tuple[Path, Path]:
    if second_values is None:
        second_values = [0.0, 0.0, 2.0, 2.0]
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as source:
        source.writestr(
            "fixture/Healthy-Total-Body-CTs-001.nii.gz",
            nifti_mask([0.0, 1.0, 2.0, 1.0]),
        )
        source.writestr(
            "fixture/Healthy-Total-Body-CTs-002.nii.gz",
            nifti_mask(second_values),
        )
        source.writestr("fixture/labels.xlsx", label_workbook())
    config = {
        "schema": "numi.human.external-segmentation-source.v1",
        "source_id": "healthy_total_body_cts_v3",
        "title": "Synthetic source fixture",
        "release": "fixture",
        "doi": "fixture-doi",
        "dataset_page": "https://example.invalid/dataset",
        "archive_url": "https://example.invalid/archive.zip",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "archive_bytes": archive.stat().st_size,
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "citation": "Synthetic fixture",
        "provenance": "Synthetic test source",
        "expected_scan_count": 2,
        "nifti_member_directory": "fixture/",
        "label_dictionary_member": "fixture/labels.xlsx",
        "expected_shape_ijk": [2, 2, 1],
        "expected_spatial_units": "mm",
        "expected_voxel_spacing_mm": [1.0, 1.0, 2.0],
        "accepted_voxel_spacings_mm": [[1.0, 1.0, 2.0]],
        "evidence_boundary": "synthetic test only",
    }
    config_path = tmp_path / "source.json"
    config_path.write_bytes(canonical(config) + b"\n")
    return archive, config_path


def test_nifti_qform_keeps_source_frame_and_physical_voxel_scale() -> None:
    raw = nifti_mask([0.0, 1.0, 2.0, 1.0])
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as image:
        header = image.read(352)
    result = parse_nifti_header(header)
    assert result["shape_ijk"] == [2, 2, 1]
    assert result["spatial_units"] == "mm"
    assert result["affine_source"] == "qform"
    assert result["voxel_volume_in_source_units_cubed"] == pytest.approx(2.0)
    assert result["voxel_to_world_affine"] == [
        [1.0, 0.0, 0.0, 10.0],
        [0.0, 1.0, 0.0, 20.0],
        [0.0, 0.0, -2.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_source_compiles_label_inventory_without_promoting_physical_ownership(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    archive, config = make_source(tmp_path)
    result = compile_source(archive, source_config=config)
    assert result["status"] == "partial_source_inventory"
    assert result["counts"] == {
        "scan_count": 2,
        "data_dictionary_label_count": 2,
        "observed_label_count": 2,
        "unobserved_dictionary_label_count": 0,
    }
    assert result["labels"] == [
        {"label_id": 1, "name": "Heart", "scan_coverage_count": 1},
        {"label_id": 2, "name": "Liver", "scan_coverage_count": 2},
    ]
    assert result["scans"][0]["label_voxel_counts"] == {"0": 1, "1": 2, "2": 1}
    assert result["scans"][0]["label_raster_volume_candidate_ml"] == {
        "1": pytest.approx(0.004), "2": pytest.approx(0.002),
    }
    assert result["source"]["observed_voxel_volumes_mm3"] == [2.0]
    assert result["scans"][0]["label_scaling_slope"] == 1.0
    assert result["scans"][0]["label_scaling_intercept"] == 0.0
    assert result["qualification"]["all_registered_segmentation_masks_scanned"]
    assert result["qualification"]["physical_tissue_volume_owner"] is False
    assert result["qualification"]["mechanical_or_physiological_admission"] is False


def test_wrong_archive_identity_and_fractional_tissue_values_fail_closed(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    archive, config = make_source(tmp_path)
    config_value = __import__("json").loads(config.read_text())
    config_value["archive_sha256"] = "0" * 64
    config.write_bytes(canonical(config_value) + b"\n")
    with pytest.raises(ImportError, match="archive bytes or SHA-256"):
        compile_source(archive, source_config=config)

    archive, config = make_source(tmp_path, [0.0, 0.0, 1.5, 2.0])
    with pytest.raises(ImportError, match="non-integer segmentation labels"):
        compile_source(archive, source_config=config)


def test_owner_cli_exposes_the_source_ingest_command(tmp_path: Path) -> None:
    from numilab_human.cli import parser

    parsed = parser().parse_args([
        "healthy-total-body-ct-source",
        "--archive", str(tmp_path / "masks.zip"),
        "--output", str(tmp_path / "receipt.json"),
    ])
    assert parsed.command == "healthy-total-body-ct-source"
