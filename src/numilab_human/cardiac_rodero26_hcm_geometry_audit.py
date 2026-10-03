"""Audit the pinned HCM1 tetra mesh against its imported activation field.

This source-level audit binds geometry, element tags, and activation-map
coverage. It does not register the patient mesh to Numi's healthy subject or
admit a mechanical or electrical runtime state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical
from .cardiac_rodero26_hcm_activation_import import EP_ARCHIVE, MESH_ARCHIVE, POINT_COUNT


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.rodero26-hcm-source-geometry-audit.v2"
TAG_NAMES = {
    1: "left_ventricle", 2: "right_ventricle", 3: "left_atrium", 4: "right_atrium",
    5: "aorta", 6: "pulmonary_artery", 7: "mitral_valve", 8: "tricuspid_valve",
    9: "aortic_valve", 10: "pulmonary_valve", 11: "left_anterior_pulmonary_vein_plane",
    12: "left_posterior_pulmonary_vein_plane", 13: "right_anterior_pulmonary_vein_plane",
    14: "right_posterior_pulmonary_vein_plane", 15: "left_atrial_appendage_plane",
    16: "superior_vena_cava_plane", 17: "inferior_vena_cava_plane",
    18: "left_anterior_pulmonary_vein_ring", 19: "left_posterior_pulmonary_vein_ring",
    20: "right_anterior_pulmonary_vein_ring", 21: "right_posterior_pulmonary_vein_ring",
    22: "left_atrial_appendage_ring", 23: "superior_vena_cava_ring",
    24: "inferior_vena_cava_ring", 25: "left_ventricular_fast_endocardial_layer",
    26: "bachmann_bundle", 27: "atrioventricular_isolating_plane",
    28: "right_ventricular_fast_endocardial_layer", 29: "septal_fast_conduction_layer",
}
HCM1_EP_TAG_DEFINITIONS = {
    "LV": 1, "RV": 2, "LA": 3, "RA": 4, "atria": [3, 4],
    "FEC_LV": 25, "FEC_RV": 28, "FEC_SV": 29,
    "fast_endo": [25, 28, 29], "BB": 26, "AV_plane": [27],
    "aorta": [5], "pulmonary_artery": [6],
    "vein_rings": [18, 19, 20, 21, 22, 23, 24],
    "valve_planes": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
}
HCM1_EP_TAG_MEMBER = {
    "path": "HCM1_EP/inputs/json_files/tags_EP.json",
    "bytes": 335,
    "sha256": "bc9d786e289fe36eb4df8e62fb7f5967b90947a79513f42a89396a9e77fa9f99",
}
HCM1_CONDUCTION_TAG_IDS = (25, 26, 27, 28, 29)
ACTIVATION_FIELD = "hcm1-sample53-activation-time-ms.f64le"
ACTIVATION_MEMBER_SHA256 = "0987c560ba3fca8134a2f0d676ec5b5e8a5fe152506e8e0705aa519bddafdbdf"
ACTIVATION_FIELD_SHA256 = "e0f42c5f5b0a64c6ca0c164ff5938982be8b9f9c0c03cadf8ae5b821f2a9da35"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("Rodero-26 HCM geometry audit: " + message)


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _read_nonblank_line(stream, label: str) -> str:
    while True:
        line = stream.readline(1024)
        require(bool(line), f"truncated VTK before {label}")
        if line.strip():
            try:
                return line.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise HumanImportError(f"Rodero-26 HCM geometry audit: malformed {label}") from error


def _read_exact(stream, byte_count: int, label: str) -> bytes:
    data = stream.read(byte_count)
    require(len(data) == byte_count, f"truncated VTK {label} payload")
    return data


def _validate_tag_schema(tag_source: Any) -> dict[str, Any]:
    require(isinstance(tag_source, dict), "publisher EP tag member is absent")
    require(
        all(tag_source.get(key) == value for key, value in HCM1_EP_TAG_MEMBER.items())
        and tag_source.get("definitions") == HCM1_EP_TAG_DEFINITIONS,
        "publisher EP tag member or conduction-label definitions changed",
    )
    return tag_source["definitions"]


def parse_vtk_tetra_mesh(path: Path, *, expected_points: int = POINT_COUNT) -> dict[str, Any]:
    """Read reviewed binary VTK points, tetrahedra, tags, and cell fibers."""
    import numpy as np

    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "mesh is not a regular file")
    try:
        stream = path.open("rb")
    except OSError as error:
        raise HumanImportError("Rodero-26 HCM geometry audit: cannot read mesh") from error
    with stream:
        header = [stream.readline(1024) for _ in range(6)]
        require(header[0].startswith(b"# vtk DataFile Version ")
                and header[2].strip().lower() == b"binary"
                and header[3].strip() == b"DATASET UNSTRUCTURED_GRID",
                "mesh is not a legacy binary unstructured grid")
        try:
            point_fields = header[5].decode("ascii").split()
            require(len(point_fields) == 3 and point_fields[0] == "POINTS"
                    and point_fields[2] == "float", "unsupported VTK point declaration")
            point_count = int(point_fields[1])
        except (UnicodeDecodeError, ValueError) as error:
            raise HumanImportError("Rodero-26 HCM geometry audit: malformed VTK POINTS declaration") from error
        require(point_count == expected_points, "mesh point count differs from the pinned source")
        point_bytes = _read_exact(stream, point_count * 3 * 4, "POINTS")
        points_um = np.frombuffer(point_bytes, dtype=">f4").astype(np.float64).reshape(point_count, 3)
        require(bool(np.isfinite(points_um).all()), "mesh coordinates contain non-finite values")

        cell_types_header = _read_nonblank_line(stream, "CELL_TYPES")
        parts = cell_types_header.split()
        require(len(parts) == 2 and parts[0] == "CELL_TYPES", "VTK CELL_TYPES declaration is malformed")
        try:
            cell_count = int(parts[1])
        except ValueError as error:
            raise HumanImportError("Rodero-26 HCM geometry audit: invalid CELL_TYPES count") from error
        require(cell_count > 0, "mesh has no cells")
        types_raw = _read_exact(stream, cell_count * 4, "CELL_TYPES")
        cell_types = np.frombuffer(types_raw, dtype=">i4")
        require(bool(np.equal(cell_types, 10).all()), "pinned HCM1 mesh contains non-tetrahedron cells")

        cells_header = _read_nonblank_line(stream, "CELLS")
        parts = cells_header.split()
        require(len(parts) == 3 and parts[0] == "CELLS", "VTK CELLS declaration is malformed")
        try:
            declared_cells, total_integers = int(parts[1]), int(parts[2])
        except ValueError as error:
            raise HumanImportError("Rodero-26 HCM geometry audit: invalid CELLS dimensions") from error
        require(declared_cells == cell_count and total_integers == 5 * cell_count,
                "VTK tetrahedron connectivity dimensions disagree")
        cells_raw = _read_exact(stream, total_integers * 4, "CELLS")
        rows = np.frombuffer(cells_raw, dtype=">i4").reshape(cell_count, 5)
        require(bool(np.equal(rows[:, 0], 4).all()), "VTK cell list contains a non-tetrahedron record")
        connectivity = rows[:, 1:].astype(np.int32, copy=False)
        require(int(connectivity.min()) >= 0 and int(connectivity.max()) < point_count,
                "tetrahedron connectivity references an invalid point")

        cell_data_header = _read_nonblank_line(stream, "CELL_DATA")
        parts = cell_data_header.split()
        require(len(parts) == 2 and parts[0] == "CELL_DATA", "VTK CELL_DATA declaration is malformed")
        try:
            require(int(parts[1]) == cell_count, "VTK cell-data count differs from the tetrahedron count")
        except ValueError as error:
            raise HumanImportError("Rodero-26 HCM geometry audit: invalid CELL_DATA count") from error
        scalar_header = _read_nonblank_line(stream, "elemTag SCALARS")
        require(scalar_header.split() == ["SCALARS", "elemTag", "int", "1"],
                "VTK source element-tag array is missing or has changed")
        lookup_header = _read_nonblank_line(stream, "elemTag LOOKUP_TABLE")
        require(lookup_header == "LOOKUP_TABLE default", "VTK elemTag lookup table changed")
        tags_raw = _read_exact(stream, cell_count * 4, "elemTag")
        tags = np.frombuffer(tags_raw, dtype=">i4").astype(np.int32, copy=False)
        require(int(tags.min()) >= min(TAG_NAMES) and int(tags.max()) <= max(TAG_NAMES),
                "VTK elemTag contains an unregistered anatomy identity")

        fiber_header = _read_nonblank_line(stream, "fiber vectors")
        require(fiber_header.split() == ["VECTORS", "fiber", "float"],
                "VTK source cell-fiber vector array is missing or changed")
        fiber_raw = _read_exact(stream, cell_count * 3 * 4, "fiber vectors")
        fibers = np.frombuffer(fiber_raw, dtype=">f4").reshape(cell_count, 3)
        require(bool(np.isfinite(fibers).all()), "VTK cell-fiber vectors contain non-finite values")
        fiber_norms = np.linalg.norm(fibers.astype(np.float64), axis=1)
        require(bool((np.abs(fiber_norms - 1.0) <= 1.0e-6).all()),
                "VTK source cell-fiber vectors are not unit directions")
        require(stream.read() == b"\n", "VTK fiber payload has truncation or trailing data")

    return {
        "points_um": points_um,
        "connectivity": connectivity,
        "tags": tags,
        "fiber_vectors": fibers,
        "fiber_bytes": fiber_raw,
        "fiber_norms": fiber_norms,
        "cell_type_counts": {"vtk_tetra": int(cell_count)},
        "point_count": int(point_count),
        "cell_count": int(cell_count),
    }


def _activation_field(path: Path, receipt_path: Path, expected_points: int):
    import numpy as np

    field_path, receipt_path = Path(path), Path(receipt_path)
    require(field_path.is_file() and not field_path.is_symlink(), "activation field is not a regular file")
    require(receipt_path.is_file() and not receipt_path.is_symlink(), "activation receipt is not a regular file")
    try:
        receipt = json.loads(receipt_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise HumanImportError("Rodero-26 HCM geometry audit: activation receipt is malformed") from error
    receipt_digest = receipt.pop("report_sha256", None)
    require(
        isinstance(receipt_digest, str)
        and hashlib.sha256(canonical(receipt)).hexdigest() == receipt_digest,
        "activation receipt report digest is invalid",
    )
    require(receipt.get("schema") == "HumanPack.rodero26-hcm-source-activation-import.v1",
            "activation receipt schema changed")
    mesh = receipt.get("source_files", {}).get("mesh", {})
    ep = receipt.get("source_files", {}).get("electrophysiology", {})
    require(mesh.get("sha256") == MESH_ARCHIVE["sha256"]
            and mesh.get("bytes") == MESH_ARCHIVE["bytes"]
            and ep.get("sha256") == EP_ARCHIVE["sha256"]
            and ep.get("bytes") == EP_ARCHIVE["bytes"],
            "activation receipt is bound to different HCM1 source archives")
    _validate_tag_schema(receipt.get("source_members", {}).get("tags"))
    activation = receipt.get("source_members", {}).get("activation", {})
    output = receipt.get("output", {}).get(field_path.name, {})
    require(activation.get("sample_id") == 53 and activation.get("unit") == "ms"
            and activation.get("value_count") == expected_points
            and activation.get("source_member") == "HCM1_EP/activation_maps/53.dat"
            and activation.get("source_member_sha256") == ACTIVATION_MEMBER_SHA256
            and activation.get("field_sha256") == ACTIVATION_FIELD_SHA256
            and output.get("dtype") == "little-endian float64"
            and output.get("order") == "HCM1.vtk POINTS order; one value per point",
            "activation receipt does not match the HCM1 point-order contract")
    size, digest = _sha256_file(field_path)
    require(size == expected_points * 8 and digest == output.get("sha256")
            and digest == activation.get("field_sha256")
            and digest == ACTIVATION_FIELD_SHA256,
            "activation field differs from the immutable source receipt")
    values = np.fromfile(field_path, dtype="<f8")
    require(values.size == expected_points and bool(np.isfinite(values).all())
            and bool(np.all((values >= 0.0) | (values == -1.0))),
            "activation field contains an invalid time or inactive sentinel")
    return values, receipt, {
        "bytes": size,
        "sha256": digest,
        "import_report_sha256": receipt_digest,
    }


def audit_geometry(
    mesh_path: Path,
    activation_field_path: Path,
    activation_receipt_path: Path,
    *,
    fiber_output_path: Path | None = None,
) -> dict[str, Any]:
    import numpy as np

    mesh_path = Path(mesh_path)
    mesh_bytes, mesh_sha = _sha256_file(mesh_path)
    require(mesh_bytes == MESH_ARCHIVE["bytes"] and mesh_sha == MESH_ARCHIVE["sha256"],
            "mesh differs from the pinned HCM1 source file")
    mesh = parse_vtk_tetra_mesh(mesh_path)
    activation, activation_receipt, activation_identity = _activation_field(
        activation_field_path, activation_receipt_path, mesh["point_count"]
    )
    require(_sha256_file(mesh_path) == (mesh_bytes, mesh_sha), "mesh changed during geometry audit")

    points = mesh["points_um"]
    connectivity = mesh["connectivity"]
    tags = mesh["tags"].astype(np.int64, copy=False)
    tag_count = max(TAG_NAMES) + 1
    tetra_count_by_tag = np.bincount(tags, minlength=tag_count)
    volume_um3_by_tag = np.zeros(tag_count, dtype=np.float64)
    orientation_counts = {"positive": 0, "negative": 0, "degenerate": 0}
    activation_incidence = np.zeros((tag_count, 5), dtype=np.int64)
    minimum_nonzero_det_um3 = math.inf
    maximum_abs_det_um3 = 0.0
    chunk_cells = 100_000
    for start in range(0, len(connectivity), chunk_cells):
        end = min(start + chunk_cells, len(connectivity))
        nodes = connectivity[start:end]
        tetra = points[nodes]
        a = tetra[:, 1] - tetra[:, 0]
        b = tetra[:, 2] - tetra[:, 0]
        c = tetra[:, 3] - tetra[:, 0]
        signed_six_volume = np.einsum("ij,ij->i", a, np.cross(b, c))
        require(bool(np.isfinite(signed_six_volume).all()), "tetrahedron volume calculation is non-finite")
        orientation_counts["positive"] += int(np.count_nonzero(signed_six_volume > 0.0))
        orientation_counts["negative"] += int(np.count_nonzero(signed_six_volume < 0.0))
        orientation_counts["degenerate"] += int(np.count_nonzero(signed_six_volume == 0.0))
        nonzero = np.abs(signed_six_volume[signed_six_volume != 0.0])
        if nonzero.size:
            minimum_nonzero_det_um3 = min(minimum_nonzero_det_um3, float(nonzero.min()))
            maximum_abs_det_um3 = max(maximum_abs_det_um3, float(nonzero.max()))
        volume_um3 = np.abs(signed_six_volume) / 6.0
        volume_um3_by_tag += np.bincount(tags[start:end], weights=volume_um3, minlength=tag_count)
        active_incidence = np.count_nonzero(activation[nodes] >= 0.0, axis=1)
        flat_histogram = np.bincount(
            tags[start:end] * 5 + active_incidence, minlength=tag_count * 5
        )
        activation_incidence += flat_histogram.reshape(tag_count, 5)

    require(int(tetra_count_by_tag.sum()) == mesh["cell_count"], "element-tag tally lost tetrahedra")
    bounds_min_um = points.min(axis=0)
    bounds_max_um = points.max(axis=0)
    tags_report = []
    for tag_id, name in TAG_NAMES.items():
        tags_report.append({
            "tag_id": tag_id,
            "name": name,
            "tetrahedron_count": int(tetra_count_by_tag[tag_id]),
            "tetrahedral_volume_candidate_ml": float(volume_um3_by_tag[tag_id] / 1.0e12),
            "tetra_node_activation_incidence_count_by_active_nodes": [
                int(value) for value in activation_incidence[tag_id]
            ],
        })
    fiber_bytes = mesh["fiber_bytes"]
    fiber_norms = mesh["fiber_norms"]
    fiber_output = None
    if fiber_output_path is not None:
        fiber_output_path = Path(fiber_output_path)
        fiber_digest = _write_immutable_bytes(fiber_output_path, fiber_bytes)
        fiber_output = {
            "path": str(Path(os.path.abspath(fiber_output_path))),
            "bytes": len(fiber_bytes),
            "sha256": fiber_digest,
            "dtype": "big-endian float32",
            "shape": [mesh["cell_count"], 3],
            "order": "HCM1.vtk CELL_DATA order; one vector per tetrahedron",
        }
    conduction_counts = {
        str(tag_id): int(tetra_count_by_tag[tag_id])
        for tag_id in HCM1_CONDUCTION_TAG_IDS
    }
    active = activation[activation >= 0.0]
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.rodero26-hcm-geometry-audit.2",
        "status": "source_geometry_and_activation_audited_candidate",
        "source": {
            "mesh_record": MESH_ARCHIVE["record"],
            "mesh_file": mesh_path.name,
            "mesh_bytes": mesh_bytes,
            "mesh_sha256": mesh_sha,
            "mesh_units": "micron, as declared by the publisher",
            "mesh_to_m_scale": 1.0e-6,
            "point_count": mesh["point_count"],
            "cell_count": mesh["cell_count"],
            "cell_encoding": "VTK_TETRA, four big-endian int32 point indices per cell",
            "cell_type_counts": mesh["cell_type_counts"],
            "cell_data_array": "elemTag, big-endian int32",
            "coordinate_bounds_um": {
                "minimum": [float(value) for value in bounds_min_um],
                "maximum": [float(value) for value in bounds_max_um],
            },
        },
        "activation": {
            "source_receipt": str(Path(activation_receipt_path)),
            "source_receipt_schema": activation_receipt["schema"],
            "source_receipt_report_sha256": activation_identity["import_report_sha256"],
            "sample_id": 53,
            "field_file": Path(activation_field_path).name,
            "field_bytes": activation_identity["bytes"],
            "field_sha256": activation_identity["sha256"],
            "point_order_matches_mesh": True,
            "unit": "ms",
            "active_point_count": int(active.size),
            "inactive_sentinel_count": int(np.count_nonzero(activation == -1.0)),
            "minimum_active_ms": float(active.min()),
            "maximum_active_ms": float(active.max()),
        },
        "element_tags": tags_report,
        "cell_fiber_field": {
            "array": "CELL_DATA VECTORS fiber float",
            "vector_count": mesh["cell_count"],
            "bytes": len(fiber_bytes),
            "sha256": hashlib.sha256(fiber_bytes).hexdigest(),
            "finite_vector_count": int(len(fiber_norms)),
            "unit_vector_tolerance": 1.0e-6,
            "unit_vector_count": int(len(fiber_norms)),
            "norm_min": float(fiber_norms.min()),
            "norm_median": float(np.median(fiber_norms)),
            "norm_max": float(fiber_norms.max()),
            "maximum_absolute_unit_norm_error": float(
                np.max(np.abs(fiber_norms - 1.0))
            ),
            "extracted_sidecar": fiber_output,
        },
        "fast_conduction_geometry": {
            "publisher_tag_schema_member": HCM1_EP_TAG_MEMBER,
            "publisher_tag_ids": list(HCM1_CONDUCTION_TAG_IDS),
            "base_mesh_cell_count_by_conduction_tag": conduction_counts,
            "base_mesh_contains_conduction_cells": any(conduction_counts.values()),
            "separate_electromechanics_mesh_imported": False,
        },
        "mesh_quality": {
            "signed_orientation_cell_counts": orientation_counts,
            "minimum_nonzero_signed_six_volume_um3": (
                None if math.isinf(minimum_nonzero_det_um3) else minimum_nonzero_det_um3
            ),
            "maximum_absolute_signed_six_volume_um3": maximum_abs_det_um3,
            "total_tetrahedral_volume_candidate_ml": float(volume_um3_by_tag.sum() / 1.0e12),
            "all_cells_are_tetrahedra": True,
            "zero_volume_tetrahedron_count": orientation_counts["degenerate"],
        },
        "qualification": {
            "pinned_mesh_identity_verified": True,
            "vtk_connectivity_and_cell_types_verified": True,
            "element_tags_bound_to_publisher_dictionary": True,
            "publisher_ep_tag_member_hash_verified": True,
            "source_cell_fiber_field_and_unit_norms_verified": True,
            "fast_conduction_geometry_present": any(conduction_counts.values()),
            "point_activation_map_bound_to_same_mesh": True,
            "source_units_verified_from_publisher_record": True,
            "mesh_registered_to_current_numi_healthy_subject": False,
            "myocardial_material_calibration": False,
            "electrophysiology_executed_by_numi": False,
            "electromechanical_coupling": False,
            "accepted_native_heartbeat": False,
            "clinical_qualification": False,
        },
        "boundary": (
            "The HCM1 patient-variant tetra mesh, publisher element tags, and sample 53 "
            "activation-time field are geometrically cross-checked in source order. "
            "Tetrahedral volumes are source geometry candidates. This is not a Numi "
            "electrical solve, healthy-subject anatomy registration, myocardial mechanics, "
            "native accepted heartbeat, or clinical qualification."
        ),
    }


def _write_immutable(path: Path, value: dict[str, Any]) -> str:
    raw = canonical(value) + b"\n"
    return _write_immutable_bytes(path, raw)


def _write_immutable_bytes(path: Path, raw: bytes) -> str:
    path = Path(path)
    require(not path.is_symlink(), "output path is redirected by a symlink")
    if path.exists():
        require(path.read_bytes() == raw, "output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{os.getpid()}.pending")
        try:
            with temporary.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return hashlib.sha256(raw).hexdigest()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mesh-vtk", type=Path, required=True)
    parser.add_argument("--activation-field", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument(
        "--fiber-output", type=Path,
        help="optional immutable source-order CELL_DATA fiber vector sidecar",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    result = audit_geometry(
        arguments.mesh_vtk,
        arguments.activation_field,
        arguments.activation_receipt,
        fiber_output_path=arguments.fiber_output,
    )
    output = Path(os.path.abspath(arguments.output))
    digest = _write_immutable(output, result)
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(output),
        "sha256": digest,
        "point_count": result["source"]["point_count"],
        "cell_count": result["source"]["cell_count"],
        "active_point_count": result["activation"]["active_point_count"],
        "inactive_point_count": result["activation"]["inactive_sentinel_count"],
        "tag_count": sum(row["tetrahedron_count"] > 0 for row in result["element_tags"]),
        "degenerate_tetrahedron_count": result["mesh_quality"]["zero_volume_tetrahedron_count"],
    }, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    arguments = parser.parse_args(argv)
    try:
        return run(arguments)
    except (HumanImportError, OSError, ValueError, TypeError, KeyError) as error:
        print(f"rodero HCM1 geometry audit: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
