"""Independent audit of source-NIfTI-bound organ tetrahedral candidates."""
from __future__ import annotations

from array import array
from collections import Counter
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
PLAN_PATH = ARTIFACT / "plan.json"
RECEIPT_PATH = ARTIFACT / "meshes/receipt.json"
FACE_CORNERS = (
    (0, 1, ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1))),
    (0, -1, ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1))),
    (1, 1, ((-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1))),
    (1, -1, ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1))),
    (2, 1, ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))),
    (2, -1, ((-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1))),
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def json_file(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), f"not a regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON is not an object: {path}")
    return value


def read_mask_indices(archive_path: Path, member_name: str, expected_sha: str,
                      selected_labels: set[int], allowed_labels: set[int]) -> tuple[dict[int, list[np.ndarray]], list[int]]:
    chunks: dict[int, list[np.ndarray]] = {label: [] for label in selected_labels}
    digest = hashlib.sha256()
    data_bytes = 0
    with zipfile.ZipFile(archive_path) as archive:
        require(member_name in archive.namelist(), "registered NIfTI member is absent")
        with archive.open(member_name, "r") as member, gzip.GzipFile(fileobj=member, mode="rb") as image:
            header = image.read(352)
            require(len(header) == 352 and header[344:348] == b"n+1\x00", "unsupported NIfTI header")
            little = struct.unpack_from("<i", header, 0)[0]
            big = struct.unpack_from(">i", header, 0)[0]
            require(little == 348 or big == 348, "invalid NIfTI header size")
            endian = "<" if little == 348 else ">"
            dims = struct.unpack_from(endian + "8h", header, 40)
            require(dims[0] == 3 and all(x == 1 for x in dims[4:]), "NIfTI is not one 3-D mask")
            dimensions = [int(x) for x in dims[1:4]]
            datatype = struct.unpack_from(endian + "h", header, 70)[0]
            bitpix = struct.unpack_from(endian + "h", header, 72)[0]
            offset = struct.unpack_from(endian + "f", header, 108)[0]
            require(datatype == 16 and bitpix == 32 and offset == 352.0,
                    "NIfTI encoding differs from the registered float32 mask")
            slope, intercept = struct.unpack_from(endian + "2f", header, 112)
            if slope == 0.0:
                slope, intercept = 1.0, 0.0
            digest.update(header)
            size_i, _, _ = dimensions
            expected_data_bytes = math.prod(dimensions) * 4
            while True:
                block = image.read(16 * 1024 * 1024)
                if not block:
                    break
                require(len(block) % 4 == 0, "NIfTI ends within a float32 voxel")
                first_voxel = data_bytes // 4
                data_bytes += len(block)
                require(data_bytes <= expected_data_bytes, "NIfTI has excess voxel data")
                digest.update(block)
                values = np.frombuffer(block, dtype=f"{endian}f4")
                if slope != 1.0 or intercept != 0.0:
                    values = values * slope + intercept
                require(bool(np.isfinite(values).all()), "NIfTI has nonfinite label values")
                rounded = np.rint(values)
                require(bool(np.equal(values, rounded).all()), "NIfTI has noninteger label values")
                observed = {int(x) for x in np.unique(rounded)}
                require(observed <= allowed_labels | {0}, "NIfTI has labels outside its intake dictionary")
                for label in (observed - {0}) & selected_labels:
                    local = np.flatnonzero(rounded == label)
                    if local.size:
                        indices = local.astype(np.uint64, copy=False) + first_voxel
                        chunks[label].append(indices.astype(np.uint32, copy=False))
            require(data_bytes == expected_data_bytes, "NIfTI payload length does not match its header")
    require(digest.hexdigest() == expected_sha, "NIfTI uncompressed hash differs from intake")
    return chunks, dimensions


def indices_to_voxels(chunks: list[np.ndarray], dimensions: list[int]) -> set[tuple[int, int, int]]:
    size_i, size_j, size_k = dimensions
    require(bool(chunks), "source label has no voxel indices")
    linear = np.concatenate(chunks)
    require(len(linear) < size_i * size_j * size_k, "source label exceeds the mask volume")
    return {
        (int(index % size_i), int((index // size_i) % size_j), int(index // (size_i * size_j)))
        for index in linear
    }


def voxel_boundary(voxels: set[tuple[int, int, int]]) -> Counter:
    result = Counter()
    for cell in voxels:
        for axis, direction, corners in FACE_CORNERS:
            neighbor = list(cell)
            neighbor[axis] += direction
            if tuple(neighbor) in voxels:
                continue
            quad = [tuple(2 * cell[axis] + corner[axis] for axis in range(3)) for corner in corners]
            result[tuple(sorted((quad[0], quad[1], quad[2])))] += 1
            result[tuple(sorted((quad[0], quad[2], quad[3])))] += 1
    return result


def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def audit_mesh(mesh: dict, scan: dict, expected_voxels: set[tuple[int, int, int]]) -> dict:
    path = ROOT / mesh["path"]
    require(path.is_file() and not path.is_symlink(), f"missing mesh: {path}")
    require(sha(path) == mesh["sha256"], f"mesh hash mismatch: {path}")
    matrix = [[float(value) for value in row] for row in scan["voxel_to_world_affine"]]
    voxel_volume_m3 = abs(matrix[0][0] * matrix[1][1] * matrix[2][2]) * 1e-9
    expected_volume = len(expected_voxels) * voxel_volume_m3
    require(mesh["source_voxel_count"] == len(expected_voxels), "receipt source voxel count differs from NIfTI")

    with gzip.open(path, "rt", encoding="ascii") as stream:
        require(stream.readline().strip() == "# vtk DataFile Version 3.0", "unsupported VTK header")
        stream.readline()
        require(stream.readline().strip() == "ASCII", "VTK is not ASCII")
        require(stream.readline().strip() == "DATASET UNSTRUCTURED_GRID", "VTK dataset is not unstructured grid")
        point_header = stream.readline().split()
        require(len(point_header) == 3 and point_header[0] == "POINTS" and point_header[2] == "double",
                "VTK point header is malformed")
        point_count = int(point_header[1])
        points = array("d")
        doubled_points = []
        for _ in range(point_count):
            values = tuple(float(value) for value in stream.readline().split())
            require(len(values) == 3 and all(math.isfinite(value) for value in values),
                    "VTK point is malformed or nonfinite")
            points.extend(values)
            ijk = tuple((values[axis] * 1000.0 - matrix[axis][3]) / matrix[axis][axis]
                        for axis in range(3))
            doubled = tuple(round(value * 2.0) for value in ijk)
            require(max(abs(2.0 * ijk[axis] - doubled[axis]) for axis in range(3)) < 5e-5,
                    "VTK point is off the source voxel half-grid")
            doubled_points.append(doubled)

        cell_header = stream.readline().split()
        require(len(cell_header) == 3 and cell_header[0] == "CELLS", "VTK cell header is malformed")
        cell_count, packed_count = int(cell_header[1]), int(cell_header[2])
        require(packed_count == 5 * cell_count and cell_count == 6 * len(expected_voxels),
                "VTK tetrahedron count is not six per source voxel")
        bits = max(1, (point_count - 1).bit_length())
        mask = (1 << bits) - 1
        face_incidence: dict[int, int] = {}
        volumes = []
        minimum_volume = math.inf
        for _ in range(cell_count):
            row = [int(value) for value in stream.readline().split()]
            require(len(row) == 5 and row[0] == 4, "VTK cell is not a four-node tetrahedron")
            ids = row[1:]
            require(min(ids) >= 0 and max(ids) < point_count and len(set(ids)) == 4,
                    "VTK tetrahedron point IDs are invalid")
            p = [tuple(points[3 * index + axis] for axis in range(3)) for index in ids]
            a = tuple(p[1][axis] - p[0][axis] for axis in range(3))
            b = tuple(p[2][axis] - p[0][axis] for axis in range(3))
            c = tuple(p[3][axis] - p[0][axis] for axis in range(3))
            volume = sum(a[axis] * cross(b, c)[axis] for axis in range(3)) / 6.0
            require(math.isfinite(volume) and volume > 0.0, "VTK contains a nonpositive tetrahedron")
            volumes.append(volume)
            minimum_volume = min(minimum_volume, volume)
            for face in itertools.combinations(ids, 3):
                a_id, b_id, c_id = sorted(face)
                key = (a_id << (2 * bits)) | (b_id << bits) | c_id
                face_incidence[key] = face_incidence.get(key, 0) + 1

        type_header = stream.readline().split()
        require(len(type_header) == 2 and type_header[0] == "CELL_TYPES" and
                int(type_header[1]) == cell_count, "VTK cell-type header is malformed")
        require(all(stream.readline().strip() == "10" for _ in range(cell_count)),
                "VTK contains a non-tetrahedron cell type")
        require(not any(line.strip() for line in stream), "VTK contains trailing content")

    require(point_count == mesh["point_count"], "serialized point count differs from compiler receipt")
    require(all(incidence in (1, 2) for incidence in face_incidence.values()),
            "tetrahedron faces have invalid incidence")
    boundary = Counter()
    for key, incidence in face_incidence.items():
        if incidence != 1:
            continue
        a_id = key >> (2 * bits)
        b_id = (key >> bits) & mask
        c_id = key & mask
        triangle = tuple(sorted((doubled_points[a_id], doubled_points[b_id], doubled_points[c_id])))
        boundary[triangle] += 1
    expected_boundary = voxel_boundary(expected_voxels)
    require(boundary == expected_boundary, "serialized mesh boundary differs from exact source NIfTI mask")
    volume_sum = math.fsum(volumes)
    relative_error = abs(volume_sum - expected_volume) / expected_volume
    require(relative_error <= 1e-12, "serialized tetrahedral volume does not close to source NIfTI occupancy")
    return {
        "label_id": mesh["label_id"],
        "source_label_name": mesh["source_label_name"],
        "mesh_path": mesh["path"],
        "mesh_sha256": mesh["sha256"],
        "source_voxel_count": len(expected_voxels),
        "point_count": point_count,
        "tetrahedron_count": cell_count,
        "positive_tetrahedron_count": cell_count,
        "minimum_tetrahedron_volume_m3": minimum_volume,
        "tetrahedral_volume_sum_m3": volume_sum,
        "source_voxel_occupancy_volume_m3": expected_volume,
        "volume_relative_error": relative_error,
        "mask_boundary_triangle_count": sum(expected_boundary.values()),
        "serialized_boundary_triangle_count": sum(boundary.values()),
        "serialized_boundary_matches_exact_source_mask": True,
        "tetrahedron_face_incidence_is_valid": True,
    }


def main() -> None:
    plan = json_file(PLAN_PATH)
    receipt = json_file(RECEIPT_PATH)
    require(plan["schema"] == "numi.healthy-total-body-ct-organ-voxel-tets-plan.v2" and
            receipt["schema"] == "HumanPack.external-segmentation-organ-voxel-tet-volume-candidates.v2",
            "unsupported organ-volume evidence schema")
    require(sha(PLAN_PATH) == receipt["plan_sha256"], "receipt plan hash mismatch")
    for relative, expected_sha in plan["instrument_sources"].items():
        current_path = ROOT / relative
        current_sha = sha(current_path) if current_path.is_file() else None
        if current_sha != expected_sha:
            require(relative == "src/numilab_human/healthy_total_body_ct_organ_voxel_tets.py" and
                    sha(ARTIFACT / "healthy_total_body_ct_organ_voxel_tets_at_build.py") == expected_sha,
                    f"instrument source drifted without an exact build snapshot: {relative}")
    require(receipt["compiler_source_sha256"] ==
            plan["instrument_sources"]["src/numilab_human/healthy_total_body_ct_organ_voxel_tets.py"],
            "receipt does not bind the preregistered compiler source")
    intake_path = ROOT / plan["intake_receipt_path"]
    intake = json_file(intake_path)
    scan = next(row for row in intake["scans"] if row["scan_id"] == plan["scan_id"])
    archive_path = ROOT / plan["source_archive_path"]
    require(sha(archive_path) == plan["source_archive_sha256"] == receipt["source_archive_sha256"],
            "registered source archive hash mismatch")
    require(sha(intake_path) == receipt["source_intake_sha256"], "intake receipt hash mismatch")
    labels = {int(row["label_id"]): row for row in plan["labels"]}
    allowed = {int(label_id) for label_id in scan["label_voxel_counts"]}
    chunks, dimensions = read_mask_indices(
        archive_path, scan["nifti_member"], scan["nifti_uncompressed_sha256"], set(labels), allowed)
    require(dimensions == [512, 512, 828], "source mask dimensions differ from the registered release")
    meshes = receipt["meshes"]
    require({int(mesh["label_id"]) for mesh in meshes} == set(labels) and len(meshes) == len(labels),
            "receipt does not contain one mesh per registered label")
    results = []
    for mesh in meshes:
        label_id = int(mesh["label_id"])
        label = labels[label_id]
        require(mesh["source_label_name"] == label["name"] and
                mesh["source_nifti_uncompressed_sha256"] == scan["nifti_uncompressed_sha256"],
                "mesh label identity or source NIfTI hash differs")
        voxels = indices_to_voxels(chunks.pop(label_id), dimensions)
        require(len(voxels) == int(scan["label_voxel_counts"][str(label_id)]) == label["source_voxel_count"],
                "source NIfTI voxel count differs from plan and intake")
        results.append(audit_mesh(mesh, scan, voxels))
        del voxels
    require(len(results) == len(labels), "independent verifier omitted a label")
    verification = {
        "schema": "numi.human.scan-organ-voxel-tet-independent-verification.v1",
        "status": "passed",
        "plan_sha256": sha(PLAN_PATH),
        "compiler_receipt_sha256": sha(RECEIPT_PATH),
        "verifier_source_sha256": sha(Path(__file__)),
        "inputs": {
            "intake_sha256": sha(intake_path),
            "source_archive_sha256": sha(archive_path),
            "source_nifti_uncompressed_sha256": scan["nifti_uncompressed_sha256"],
        },
        "meshes": results,
        "qualification": {
            "all_source_voxel_counts_match_plan_and_intake": True,
            "all_serialized_tetrahedra_positive": True,
            "all_tetrahedral_faces_have_valid_incidence": True,
            "all_serialized_boundaries_match_exact_source_masks": True,
            "all_tetrahedral_volumes_close_to_source_voxel_occupancy": True,
            "numi_subject_binding": False,
            "mechanics_admitted": False,
            "tissue_material_calibrated": False,
            "physiology_qualified": False,
        },
        "boundary": "Independent geometry verification only; automatic source labels do not establish expert segmentation quality or Numi Human subject anatomy.",
    }
    output = ARTIFACT / "independent-verification.json"
    output.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = sha(output)
    (ARTIFACT / "independent-verification.json.sha256").write_text(f"{digest}  {output.name}\n")
    print(json.dumps({"status": "passed", "labels": len(results),
                      "source_voxels": sum(row["source_voxel_count"] for row in results),
                      "tetrahedra": sum(row["tetrahedron_count"] for row in results),
                      "verification_sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
