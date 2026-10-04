"""Independent standard-library audit of serialized CT kidney tetrahedra."""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "Docs/media/bilateral-kidney-voxel-tet-candidate-20261004"
RECEIPT_PATH = ARTIFACT / "meshes/receipt.json"
INTAKE_PATH = ROOT / "Docs/media/healthy-total-body-ct-source-20261003/intake-v4.json"
PLY_VERTEX = struct.Struct("<ddd")
PLY_FACE = struct.Struct("<Biii")


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


def read_ply_triangles(path: Path, matrix: list[list[float]]) -> Counter:
    points = []
    triangles = Counter()
    with gzip.open(path, "rb") as stream:
        header = []
        while True:
            row = stream.readline()
            require(bool(row), f"truncated PLY header: {path}")
            header.append(row)
            if row == b"end_header\n":
                break
        require(header[1] == b"format binary_little_endian 1.0\n", "unexpected PLY format")
        vertex_count = next(int(row.split()[2]) for row in header if row.startswith(b"element vertex "))
        face_count = next(int(row.split()[2]) for row in header if row.startswith(b"element face "))
        for _ in range(vertex_count):
            raw = stream.read(PLY_VERTEX.size)
            require(len(raw) == PLY_VERTEX.size, "truncated PLY vertices")
            points.append(PLY_VERTEX.unpack(raw))
        for _ in range(face_count):
            raw = stream.read(PLY_FACE.size)
            require(len(raw) == PLY_FACE.size, "truncated PLY faces")
            count, a, b, c = PLY_FACE.unpack(raw)
            require(count == 3 and max(a, b, c) < vertex_count, "invalid PLY triangle")
            key = []
            for point in (points[a], points[b], points[c]):
                ijk = tuple((point[d] - matrix[d][3]) / matrix[d][d] for d in range(3))
                doubled = tuple(round(value * 2.0) for value in ijk)
                require(max(abs(2.0 * ijk[d] - doubled[d]) for d in range(3)) < 2e-5,
                        "PLY vertex is off the source voxel half-grid")
                key.append(doubled)
            triangles[tuple(sorted(key))] += 1
        require(not stream.read(1), "PLY contains trailing payload")
    return triangles


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def audit_mesh(mesh: dict, scan: dict) -> dict:
    path = ROOT / mesh["path"]
    require(path.is_file() and not path.is_symlink(), f"missing mesh: {path}")
    require(sha(path) == mesh["sha256"], f"mesh hash mismatch: {path}")
    affine = scan["voxel_to_world_affine"]
    matrix = [[float(x) for x in row] for row in affine]
    voxel_volume_m3 = abs(matrix[0][0] * matrix[1][1] * matrix[2][2]) * 1e-9
    require(math.isclose(mesh["source_voxel_volume_m3"],
                         mesh["source_voxel_count"] * voxel_volume_m3, rel_tol=1e-12),
            "receipt voxel count and source affine do not agree")
    surface_path = ROOT / mesh["source_surface_path"]
    require(sha(surface_path) == mesh["source_surface_sha256"], "surface hash mismatch")
    expected_boundary = read_ply_triangles(surface_path, matrix)

    with gzip.open(path, "rt", encoding="ascii") as stream:
        require(stream.readline().strip() == "# vtk DataFile Version 3.0", "unsupported VTK header")
        stream.readline()
        require(stream.readline().strip() == "ASCII", "VTK is not ASCII")
        require(stream.readline().strip() == "DATASET UNSTRUCTURED_GRID", "VTK dataset is not unstructured grid")
        point_header = stream.readline().split()
        require(len(point_header) == 3 and point_header[0] == "POINTS" and point_header[2] == "double",
                "VTK point header is malformed")
        point_count = int(point_header[1])
        points = []
        for _ in range(point_count):
            values = tuple(float(value) for value in stream.readline().split())
            require(len(values) == 3 and all(math.isfinite(value) for value in values),
                    "VTK point is malformed or nonfinite")
            points.append(values)
        cell_header = stream.readline().split()
        require(len(cell_header) == 3 and cell_header[0] == "CELLS", "VTK cell header is malformed")
        cell_count, packed_count = int(cell_header[1]), int(cell_header[2])
        require(packed_count == 5 * cell_count, "VTK cell connectivity size is wrong")
        face_incidence: dict[tuple[int, int, int], int] = {}
        volumes = []
        nonpositive = 0
        for _ in range(cell_count):
            row = [int(value) for value in stream.readline().split()]
            require(len(row) == 5 and row[0] == 4, "VTK cell is not a four-node tetrahedron")
            ids = row[1:]
            require(min(ids) >= 0 and max(ids) < point_count and len(set(ids)) == 4,
                    "VTK tetrahedron point IDs are invalid")
            p = [points[index] for index in ids]
            a = tuple(p[1][d] - p[0][d] for d in range(3))
            b = tuple(p[2][d] - p[0][d] for d in range(3))
            c = tuple(p[3][d] - p[0][d] for d in range(3))
            volume = sum(a[d] * cross(b, c)[d] for d in range(3)) / 6.0
            if not math.isfinite(volume) or volume <= 0.0:
                nonpositive += 1
            volumes.append(volume)
            for face in itertools.combinations(ids, 3):
                key = tuple(sorted(face))
                face_incidence[key] = face_incidence.get(key, 0) + 1
        cell_type_header = stream.readline().split()
        require(len(cell_type_header) == 2 and cell_type_header[0] == "CELL_TYPES" and
                int(cell_type_header[1]) == cell_count, "VTK cell-type header is malformed")
        bad_cell_types = sum(stream.readline().strip() != "10" for _ in range(cell_count))
        require(not any(line.strip() for line in stream), "VTK contains trailing content")

    require(point_count == mesh["point_count"] and cell_count == mesh["tetrahedron_count"],
            "serialized mesh counts differ from compiler receipt")
    require(bad_cell_types == 0, "VTK contains non-tetrahedron cell types")
    require(nonpositive == 0, "VTK contains a nonpositive or nonfinite tetrahedron")
    require(all(incidence in (1, 2) for incidence in face_incidence.values()),
            "tetrahedral faces have invalid incidence")
    boundary = Counter()
    for face, incidence in face_incidence.items():
        if incidence != 1:
            continue
        triangle = []
        for point_id in face:
            p = points[point_id]
            world_mm = tuple(value * 1000.0 for value in p)
            ijk = tuple((world_mm[d] - matrix[d][3]) / matrix[d][d] for d in range(3))
            doubled = tuple(round(value * 2.0) for value in ijk)
            require(max(abs(2.0 * ijk[d] - doubled[d]) for d in range(3)) < 5e-5,
                    "VTK boundary point is off source voxel half-grid")
            triangle.append(doubled)
        boundary[tuple(sorted(triangle))] += 1
    require(boundary == expected_boundary, "serialized tetrahedral boundary differs from the source PLY")
    volume_sum = math.fsum(volumes)
    expected_volume = float(mesh["source_voxel_volume_m3"])
    relative_error = abs(volume_sum - expected_volume) / expected_volume
    require(relative_error <= 1e-12, "serialized tetrahedral volume does not close to voxel occupancy")
    require(cell_count == 6 * mesh["source_voxel_count"], "tetrahedron count is not six per source voxel")
    return {
        "scan_id": mesh["scan_id"],
        "side": mesh["side"],
        "mesh_path": mesh["path"],
        "mesh_sha256": mesh["sha256"],
        "point_count": point_count,
        "tetrahedron_count": cell_count,
        "source_voxel_count": mesh["source_voxel_count"],
        "positive_tetrahedron_count": cell_count,
        "minimum_tetrahedron_volume_m3": min(volumes),
        "maximum_tetrahedron_volume_m3": max(volumes),
        "tetrahedral_volume_sum_m3": volume_sum,
        "source_voxel_occupancy_volume_m3": expected_volume,
        "volume_relative_error": relative_error,
        "source_boundary_triangle_count": sum(expected_boundary.values()),
        "serialized_boundary_triangle_count": sum(boundary.values()),
        "serialized_boundary_matches_source_ply": True,
        "tetrahedron_face_incidence_is_valid": True,
    }


def main() -> None:
    receipt = json_file(RECEIPT_PATH)
    intake = json_file(INTAKE_PATH)
    scans = {row["scan_id"]: row for row in intake["scans"]}
    require(receipt["schema"] == "HumanPack.external-segmentation-voxel-tet-volume-candidates.v1",
            "unsupported volume candidate receipt")
    require(sha(Path(receipt["plan_path"]) if Path(receipt["plan_path"]).is_absolute()
                else ROOT / receipt["plan_path"]) == receipt["plan_sha256"], "plan hash mismatch")
    results = [audit_mesh(mesh, scans[mesh["scan_id"]]) for mesh in receipt["meshes"]]
    require(len(results) == 4 and {(row["scan_id"], row["side"]) for row in results} ==
            {(scan, side) for scan in ("001", "002") for side in ("left", "right")},
            "expected four scan-specific left/right kidney meshes")
    result = {
        "schema": "numi.human.scan-specific-kidney-voxel-tet-independent-verification.v1",
        "status": "passed",
        "plan_sha256": receipt["plan_sha256"],
        "compiler_receipt_sha256": sha(RECEIPT_PATH),
        "compiler_source_sha256": receipt["compiler_source_sha256"],
        "verifier_source_sha256": sha(Path(__file__)),
        "inputs": {
            "intake_sha256": sha(INTAKE_PATH),
            "surface_export_receipt_sha256": receipt["surface_export_receipt_sha256"],
            "surface_verification_sha256": receipt["current_independent_surface_verification_sha256"],
        },
        "meshes": results,
        "qualification": {
            "all_serialized_tetrahedra_positive": True,
            "all_tetrahedral_faces_have_valid_incidence": True,
            "all_serialized_boundaries_equal_source_surfaces": True,
            "all_tetrahedral_volumes_close_to_source_voxel_occupancy": True,
            "numi_subject_binding": False,
            "mechanics_admitted": False,
            "material_calibrated": False,
            "perfusion_or_physiology": False,
        },
        "upstream_receipt_digest_mismatch_preserved": (
            receipt["upstream_export_receipt_verification_digest_consistent"] is False
        ),
    }
    output = ARTIFACT / "verification.json"
    output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    (ARTIFACT / "verification.json.sha256").write_text(f"{sha(output)}  verification.json\n")
    print(json.dumps({"status": result["status"], "meshes": len(results),
                      "tetrahedra": sum(row["tetrahedron_count"] for row in results),
                      "boundary_matches": result["qualification"]["all_serialized_boundaries_equal_source_surfaces"],
                      "volume_closure": result["qualification"]["all_tetrahedral_volumes_close_to_source_voxel_occupancy"],
                      "verification_sha256": sha(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
