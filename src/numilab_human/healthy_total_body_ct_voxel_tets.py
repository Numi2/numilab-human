"""Build voxel-exact tetrahedral geometry candidates from registered CT surfaces.

The voxel occupancy is reconstructed from the exact closed surface, then every
occupied source voxel is divided into six conforming tetrahedra. This creates
an organ-shaped volume mesh; it does not assign tissue material, mass ownership,
subject identity, perfusion, or physiological behavior.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, BinaryIO, TextIO

from .model import ImportError as HumanImportError
from .physiology import canonical

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.external-segmentation-voxel-tet-volume-candidates.v1"
PLAN_SCHEMA = "numi.healthy-total-body-ct-voxel-tet-volume-plan.v1"
PLY_VERTEX = struct.Struct("<ddd")
PLY_FACE = struct.Struct("<Biii")
FACE_CORNERS = (
    (0, 1, ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1))),
    (0, -1, ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1))),
    (1, 1, ((-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1))),
    (1, -1, ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1))),
    (2, 1, ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))),
    (2, -1, ((-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1))),
)
PERMUTATIONS = tuple(itertools.permutations((0, 1, 2)))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("healthy total-body CT voxel tetrahedra: " + message)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"input is not a JSON object: {path}")
    return value


def _root_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _matrix(scan: dict[str, Any]) -> tuple[list[list[float]], float]:
    affine = scan.get("voxel_to_world_affine")
    require(isinstance(affine, list) and len(affine) == 4 and
            all(isinstance(row, list) and len(row) == 4 for row in affine),
            "source affine is malformed")
    matrix = [[float(value) for value in row] for row in affine]
    require(all(math.isfinite(value) for row in matrix for value in row),
            "source affine contains nonfinite coordinates")
    require(matrix[3] == [0.0, 0.0, 0.0, 1.0], "source affine is not homogeneous")
    for row in range(3):
        for column in range(3):
            if row != column:
                require(abs(matrix[row][column]) <= 1e-12,
                        "voxel tetrahedra currently require the pinned axis-aligned CT affine")
    scales = [matrix[index][index] for index in range(3)]
    require(all(value != 0.0 for value in scales), "source voxel spacing contains zero")
    voxel_volume_mm3 = abs(scales[0] * scales[1] * scales[2])
    require(voxel_volume_mm3 > 0.0, "source voxel volume is not positive")
    return matrix, voxel_volume_mm3


def _ijk(point_mm: tuple[float, float, float], matrix: list[list[float]]) -> tuple[float, float, float]:
    return tuple((point_mm[index] - matrix[index][3]) / matrix[index][index]
                 for index in range(3))


def _ply(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    require(path.is_file() and not path.is_symlink(), f"candidate PLY is not a regular file: {path}")
    vertices_count = faces_count = None
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with gzip.open(path, "rb") as stream:
        header: list[bytes] = []
        while True:
            line = stream.readline()
            require(bool(line), f"candidate PLY header is truncated: {path}")
            header.append(line)
            if line == b"end_header\n":
                break
        require(header[:2] == [b"ply\n", b"format binary_little_endian 1.0\n"],
                "candidate PLY is not binary-little-endian PLY 1.0")
        for line in header:
            fields = line.decode("ascii").strip().split()
            if len(fields) == 3 and fields[0] == "element" and fields[1] == "vertex":
                vertices_count = int(fields[2])
            if len(fields) == 3 and fields[0] == "element" and fields[1] == "face":
                faces_count = int(fields[2])
        require(isinstance(vertices_count, int) and vertices_count > 0 and
                isinstance(faces_count, int) and faces_count > 0,
                "candidate PLY counts are missing")
        for _ in range(vertices_count):
            raw = stream.read(PLY_VERTEX.size)
            require(len(raw) == PLY_VERTEX.size, "candidate PLY vertex payload is truncated")
            vertex = PLY_VERTEX.unpack(raw)
            require(all(math.isfinite(value) for value in vertex), "candidate PLY has nonfinite vertex")
            vertices.append(vertex)
        for _ in range(faces_count):
            raw = stream.read(PLY_FACE.size)
            require(len(raw) == PLY_FACE.size, "candidate PLY face payload is truncated")
            count, a, b, c = PLY_FACE.unpack(raw)
            require(count == 3 and min(a, b, c) >= 0 and max(a, b, c) < vertices_count,
                    "candidate PLY has an invalid triangle")
            faces.append((a, b, c))
        require(not stream.read(1), "candidate PLY has unexpected trailing payload")
    return vertices, faces


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _signed_surface_volume(vertices: list[tuple[float, float, float]],
                           faces: list[tuple[int, int, int]]) -> float:
    total = 0.0
    for a, b, c in faces:
        va, vb, vc = vertices[a], vertices[b], vertices[c]
        total += sum(va[index] * _cross(vb, vc)[index] for index in range(3)) / 6.0
    return total


def _surface_boundary_cells(vertices: list[tuple[float, float, float]],
                            faces: list[tuple[int, int, int]],
                            matrix: list[list[float]]) -> tuple[set[tuple[int, int, int]], Counter]:
    boundary: set[tuple[int, int, int]] = set()
    source_triangles: Counter = Counter()
    for face in faces:
        world = [vertices[index] for index in face]
        fractional = [_ijk(point, matrix) for point in world]
        doubled = []
        for point in fractional:
            scaled = tuple(round(value * 2.0) for value in point)
            require(max(abs(point[index] * 2.0 - scaled[index]) for index in range(3)) < 2e-5,
                    "candidate surface vertex is off the source voxel half-grid")
            doubled.append(scaled)
        source_triangles[tuple(sorted(doubled))] += 1

        ranges = [max(point[axis] for point in fractional) - min(point[axis] for point in fractional)
                  for axis in range(3)]
        plane_axes = [axis for axis, width in enumerate(ranges) if width < 2e-5]
        require(len(plane_axes) == 1, "candidate triangle is not a voxel-aligned boundary face")
        axis = plane_axes[0]
        normal = _cross(tuple(world[1][i] - world[0][i] for i in range(3)),
                        tuple(world[2][i] - world[0][i] for i in range(3)))
        outward = sum(normal[i] * matrix[i][axis] for i in range(3))
        require(math.isfinite(outward) and abs(outward) > 1e-12,
                "candidate triangle has no outward voxel-axis direction")
        direction = 1 if outward > 0.0 else -1
        center = [sum(point[i] for point in fractional) / 3.0 for i in range(3)]
        cell = [round(center[i]) for i in range(3)]
        plane = sum(point[axis] for point in fractional) / 3.0
        cell[axis] = round(plane - 0.5 * direction)
        boundary.add(tuple(cell))
    return boundary, source_triangles


def _fill_interior(boundary: set[tuple[int, int, int]]) -> tuple[set[tuple[int, int, int]], tuple[int, int, int]]:
    require(bool(boundary), "candidate surface has no boundary voxels")
    lows = [min(cell[axis] for cell in boundary) - 1 for axis in range(3)]
    highs = [max(cell[axis] for cell in boundary) + 1 for axis in range(3)]
    nx, ny, nz = (highs[axis] - lows[axis] + 1 for axis in range(3))
    count = nx * ny * nz
    require(count <= 8_000_000, "candidate voxel envelope exceeds the bounded flood-fill limit")
    plane = nx * ny
    occupied_boundary = bytearray(count)
    for i, j, k in boundary:
        ix, iy, iz = i - lows[0], j - lows[1], k - lows[2]
        occupied_boundary[iz * plane + iy * nx + ix] = 1
    outside = bytearray(count)
    pending: deque[int] = deque()

    def seed(index: int) -> None:
        if not occupied_boundary[index] and not outside[index]:
            outside[index] = 1
            pending.append(index)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if i in (0, nx - 1) or j in (0, ny - 1) or k in (0, nz - 1):
                    seed(k * plane + j * nx + i)

    while pending:
        flat = pending.popleft()
        iz, rem = divmod(flat, plane)
        iy, ix = divmod(rem, nx)
        if ix:
            seed(flat - 1)
        if ix + 1 < nx:
            seed(flat + 1)
        if iy:
            seed(flat - nx)
        if iy + 1 < ny:
            seed(flat + nx)
        if iz:
            seed(flat - plane)
        if iz + 1 < nz:
            seed(flat + plane)

    voxels: set[tuple[int, int, int]] = set()
    for k in range(1, nz - 1):
        for j in range(1, ny - 1):
            start = k * plane + j * nx
            for i in range(1, nx - 1):
                flat = start + i
                if not outside[flat]:
                    voxels.add((i + lows[0], j + lows[1], k + lows[2]))
    require(boundary <= voxels, "surface-adjacent source voxels were lost during interior fill")
    return voxels, (nx, ny, nz)


def _voxel_boundary_triangles(voxels: set[tuple[int, int, int]]) -> Counter:
    result: Counter = Counter()
    for i, j, k in voxels:
        for axis, sign, corners in FACE_CORNERS:
            neighbor = [i, j, k]
            neighbor[axis] += sign
            if tuple(neighbor) in voxels:
                continue
            quad = [tuple(2 * (i, j, k)[d] + corner[d] for d in range(3)) for corner in corners]
            result[tuple(sorted((quad[0], quad[1], quad[2])))] += 1
            result[tuple(sorted((quad[0], quad[2], quad[3])))] += 1
    return result


def _point_index_map(voxels: set[tuple[int, int, int]]) -> dict[tuple[int, int, int], int]:
    points: dict[tuple[int, int, int], int] = {}
    for i, j, k in sorted(voxels):
        for di, dj, dk in itertools.product((0, 1), repeat=3):
            key = (2 * i - 1 + 2 * di, 2 * j - 1 + 2 * dj, 2 * k - 1 + 2 * dk)
            points.setdefault(key, len(points))
    return points


def _world_m(point_ijk2: tuple[int, int, int], matrix: list[list[float]]) -> tuple[float, float, float]:
    return tuple((matrix[axis][axis] * (point_ijk2[axis] * 0.5) + matrix[axis][3]) * 1e-3
                 for axis in range(3))


def _tetrahedra_for_voxel(cell: tuple[int, int, int], point_ids: dict[tuple[int, int, int], int],
                          matrix: list[list[float]]) -> list[tuple[int, int, int, int]]:
    result = []
    i, j, k = cell
    for permutation in PERMUTATIONS:
        a, b, _c = permutation
        corners = [(0, 0, 0)]
        first = [0, 0, 0]
        first[a] = 1
        corners.append(tuple(first))
        second = first.copy()
        second[b] = 1
        corners.append(tuple(second))
        corners.append((1, 1, 1))
        coordinates = [tuple(2 * base + (2 * bit - 1) for base, bit in zip(cell, corner, strict=True))
                       for corner in corners]
        ids = [point_ids[coordinate] for coordinate in coordinates]
        p = [_world_m(coordinate, matrix) for coordinate in coordinates]
        u = tuple(p[1][d] - p[0][d] for d in range(3))
        v = tuple(p[2][d] - p[0][d] for d in range(3))
        w = tuple(p[3][d] - p[0][d] for d in range(3))
        signed_volume = sum(u[d] * _cross(v, w)[d] for d in range(3)) / 6.0
        require(math.isfinite(signed_volume) and abs(signed_volume) > 0.0,
                "generated tetrahedron is degenerate")
        if signed_volume < 0.0:
            ids[2], ids[3] = ids[3], ids[2]
            signed_volume = -signed_volume
        result.append((ids[0], ids[1], ids[2], ids[3]))
    return result


def _write_vtk(path: Path, voxels: set[tuple[int, int, int]], matrix: list[list[float]]) -> dict[str, Any]:
    point_ids = _point_index_map(voxels)
    tet_count = 6 * len(voxels)
    require(tet_count > 0, "candidate has no source voxels")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink() and not path.exists(), f"mesh output already exists: {path}")
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    total_volume = 0.0
    volume_compensation = 0.0
    min_volume = math.inf
    max_volume = 0.0
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
                import io
                with io.TextIOWrapper(compressed, encoding="ascii", newline="\n") as stream:
                    stream.write("# vtk DataFile Version 3.0\n")
                    stream.write("Numi scan-specific kidney voxel-tet geometry candidate; metres; NIfTI RAS+\n")
                    stream.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
                    stream.write(f"POINTS {len(point_ids)} double\n")
                    points_by_id = [None] * len(point_ids)
                    for coordinate, index in point_ids.items():
                        points_by_id[index] = _world_m(coordinate, matrix)
                    for point in points_by_id:
                        stream.write(f"{point[0]:.17g} {point[1]:.17g} {point[2]:.17g}\n")
                    stream.write(f"CELLS {tet_count} {5 * tet_count}\n")
                    for cell in sorted(voxels):
                        for tet in _tetrahedra_for_voxel(cell, point_ids, matrix):
                            stream.write(f"4 {tet[0]} {tet[1]} {tet[2]} {tet[3]}\n")
                            coords = [points_by_id[index] for index in tet]
                            u = tuple(coords[1][d] - coords[0][d] for d in range(3))
                            v = tuple(coords[2][d] - coords[0][d] for d in range(3))
                            w = tuple(coords[3][d] - coords[0][d] for d in range(3))
                            volume = sum(u[d] * _cross(v, w)[d] for d in range(3)) / 6.0
                            require(volume > 0.0 and math.isfinite(volume),
                                    "serialized tetrahedron is not positive and finite")
                            adjusted_volume = volume - volume_compensation
                            updated_total = total_volume + adjusted_volume
                            volume_compensation = (updated_total - total_volume) - adjusted_volume
                            total_volume = updated_total
                            min_volume = min(min_volume, volume)
                            max_volume = max(max_volume, volume)
                    stream.write(f"CELL_TYPES {tet_count}\n")
                    for _ in range(tet_count):
                        stream.write("10\n")
                    stream.flush()
        digest = _sha_file(temporary)
        os.replace(temporary, path)
        path.chmod(0o644)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "point_count": len(point_ids),
        "tetrahedron_count": tet_count,
        "tetrahedron_signed_volume_sum_m3": total_volume,
        "tetrahedron_volume_min_m3": min_volume,
        "tetrahedron_volume_max_m3": max_volume,
    }


def _immutable_json(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    require(not path.is_symlink(), "JSON output is redirected")
    if path.exists():
        require(path.read_bytes() == payload, "output is immutable; choose a new output path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return _sha(payload)


def compile_candidate(*, plan_path: Path, output_dir: Path) -> dict[str, Any]:
    plan_path, output_dir = Path(plan_path), Path(output_dir)
    plan = _read_json(plan_path)
    require(plan.get("schema") == PLAN_SCHEMA and plan.get("status") == "preregistered",
            "unsupported or already-used candidate plan")
    require(plan.get("tetrahedra_per_source_voxel") == 6 and
            plan.get("voxel_fill") == "closed_surface_boundary_voxels_plus_6-connected_exterior_flood_fill",
            "candidate plan uses an unsupported reconstruction method")
    instrument_sources = plan.get("instrument_sources")
    require(isinstance(instrument_sources, dict) and instrument_sources,
            "candidate plan does not bind its compiler sources")
    for relative, expected_hash in instrument_sources.items():
        source_path = _root_path(relative)
        require(_sha_file(source_path) == expected_hash,
                f"compiler source hash drifted: {relative}")
    require(instrument_sources.get("src/numilab_human/healthy_total_body_ct_voxel_tets.py") ==
            _sha_file(Path(__file__)), "candidate plan does not bind this compiler source")
    intake_path = _root_path(plan["intake_receipt_path"])
    export_path = _root_path(plan["component_receipt_path"])
    require(_sha_file(intake_path) == plan["intake_receipt_sha256"], "source intake receipt hash drifted")
    require(_sha_file(export_path) == plan["component_receipt_sha256"], "component candidate receipt hash drifted")
    intake = _read_json(intake_path)
    export = _read_json(export_path)
    require(intake.get("schema") == "HumanPack.external-segmentation-source-ingest.v2" and
            export.get("schema") == "numi.human.scan-specific-bilateral-kidney-candidate-receipt.v1" and
            export.get("status") == "independent_verification_passed",
            "pinned source intake or surface candidate is not admitted")
    verification_path = _root_path(plan["surface_verification_path"])
    verification = _read_json(verification_path)
    verification_hash = _sha_file(verification_path)
    export_result_path = _root_path(plan["surface_export_result_path"])
    require(_sha_file(export_result_path) == plan["surface_export_result_sha256"] and
            export.get("export_result_sha256") == plan["surface_export_result_sha256"],
            "source surface export result hash drifted")
    checksum_path = _root_path(plan["surface_verification_checksum_path"])
    require(checksum_path.is_file() and not checksum_path.is_symlink(),
            "independent surface verification checksum is not a regular file")
    checksum_text = checksum_path.read_text(encoding="ascii").split()
    require(verification_hash == plan["surface_verification_sha256"] and
            len(checksum_text) == 2 and checksum_text[0] == verification_hash and
            checksum_text[1] == "verification.json",
            "independent surface verification file and checksum disagree")
    require(verification.get("schema") ==
            "numi.human.scan-specific-bilateral-kidney-candidate-independent-verification.v1" and
            verification.get("status") == "passed" and
            verification.get("export_receipt_sha256") == export.get("export_result_sha256"),
            "independent surface verification is not bound to the exact export result")
    require(intake.get("source", {}).get("archive_sha256") == plan.get("source_archive_sha256"),
            "source archive identity disagrees with the plan")
    scans = {row["scan_id"]: row for row in intake.get("scans", [])}
    candidates = export.get("candidates")
    require(isinstance(candidates, list) and len(candidates) == 4,
            "the bilateral surface export must contain four candidates")
    candidate_sources: dict[str, dict[str, Any]] = {}
    meshes: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: (row["path"], row["side"])):
        scan_id = candidate["path"].split("/scan-")[1][:3]
        require(scan_id in scans and candidate.get("side") in {"left", "right"},
                "surface candidate scan or laterality is invalid")
        scan = scans[scan_id]
        source = next((row for row in plan["source_scans"] if row["scan_id"] == scan_id), None)
        require(source is not None and candidate.get("path") == source["candidates"][candidate["side"]]["path"],
                "candidate source path differs from the preregistered plan")
        candidate_plan = source["candidates"][candidate["side"]]
        require(source.get("source_nifti_uncompressed_sha256") == scan.get("nifti_uncompressed_sha256"),
                "source NIfTI hash differs between candidate plan and intake")
        path = _root_path(candidate["path"])
        require(_sha_file(path) == candidate["sha256"], f"source surface hash drifted: {candidate['path']}")
        require(candidate.get("vertex_count") > 0 and candidate.get("triangle_count") > 0,
                "source surface contains no triangles")
        matrix, voxel_volume_mm3 = _matrix(scan)
        vertices, faces = _ply(path)
        require(len(vertices) == candidate["vertex_count"] and len(faces) == candidate["triangle_count"],
                "source surface row counts differ from its receipt")
        signed_volume_mm3 = _signed_surface_volume(vertices, faces)
        require(signed_volume_mm3 > 0.0 and math.isfinite(signed_volume_mm3),
                "source candidate winding is not outward")
        independent_rows = verification.get("scans", {}).get(scan_id, {}).get("candidates", [])
        independent = next((row for row in independent_rows if row.get("side") == candidate["side"]), None)
        require(independent is not None and independent.get("path") == candidate["path"] and
                independent.get("sha256") == candidate["sha256"],
                "current independent surface verification does not bind this candidate")
        centroid = independent.get("volume_centroid_ras_mm")
        centroid_x = candidate_plan.get("centroid_ras_x_mm")
        require(isinstance(centroid_x, (int, float)) and math.isfinite(centroid_x) and
                isinstance(centroid, list) and len(centroid) == 3 and
                math.isclose(float(centroid[0]), float(centroid_x), rel_tol=0.0, abs_tol=1e-8),
                "source candidate laterality differs from independent volume-centroid audit")
        require(centroid_x < 0 if candidate["side"] == "left" else centroid_x > 0,
                "candidate laterality disagrees with its RAS-X centroid")
        boundary, source_triangles = _surface_boundary_cells(vertices, faces, matrix)
        voxels, lattice_bounds = _fill_interior(boundary)
        generated_triangles = _voxel_boundary_triangles(voxels)
        require(generated_triangles == source_triangles,
                "reconstructed voxel-domain boundary differs from exact source triangles")
        raster_volume_mm3 = len(voxels) * voxel_volume_mm3
        relative_volume_error = abs(raster_volume_mm3 - signed_volume_mm3) / signed_volume_mm3
        require(relative_volume_error <= float(plan["maximum_source_surface_volume_relative_error"]),
                "reconstructed voxel occupancy volume disagrees with the exact source surface")
        output_name = f"scan-{scan_id}-kidney-{candidate['side']}.vtk.gz"
        output_path = output_dir / output_name
        mesh = _write_vtk(output_path, voxels, matrix)
        expected_volume_m3 = len(voxels) * voxel_volume_mm3 * 1e-9
        tet_volume_error = abs(mesh["tetrahedron_signed_volume_sum_m3"] - expected_volume_m3) / expected_volume_m3
        require(tet_volume_error <= float(plan["maximum_tetrahedral_volume_relative_error"]),
                "serialized tetrahedron volume does not close to source voxel occupancy")
        require(mesh["tetrahedron_count"] == 6 * len(voxels),
                "tetrahedron count differs from the fixed six-per-voxel partition")
        mesh.update({
            "scan_id": scan_id,
            "side": candidate["side"],
            "source_surface_path": candidate["path"],
            "source_surface_sha256": candidate["sha256"],
            "source_nifti_uncompressed_sha256": source["source_nifti_uncompressed_sha256"],
            "source_voxel_count": len(voxels),
            "flood_fill_lattice_dimensions_with_padding": list(lattice_bounds),
            "source_voxel_volume_m3": expected_volume_m3,
            "source_surface_volume_mm3": signed_volume_mm3,
            "source_raster_volume_mm3": raster_volume_mm3,
            "source_surface_volume_relative_error": relative_volume_error,
            "tetrahedral_volume_relative_error": tet_volume_error,
            "source_boundary_triangle_count": sum(source_triangles.values()),
            "exact_source_boundary_triangle_multiset": True,
            "all_tetrahedra_positive": mesh["tetrahedron_volume_min_m3"] > 0.0,
        })
        meshes.append(mesh)
        candidate_sources[f"{scan_id}:{candidate['side']}"] = {
            "surface_sha256": candidate["sha256"],
            "voxel_count": len(voxels),
            "volume_m3": expected_volume_m3,
        }
    result = {
        "schema": SCHEMA,
        "status": ("geometry_candidate_passed" if verification_hash == export.get("verification_sha256")
                   else "geometry_candidate_passed_with_upstream_digest_mismatch"),
        "compiler": "numilab-human.healthy-total-body-ct-voxel-tets.1",
        "compiler_source_sha256": _sha(Path(__file__).read_bytes()),
        "plan_path": str(plan_path.relative_to(ROOT)) if plan_path.is_relative_to(ROOT) else str(plan_path),
        "plan_sha256": _sha_file(plan_path),
        "source_intake_path": str(intake_path.relative_to(ROOT)) if intake_path.is_relative_to(ROOT) else str(intake_path),
        "source_intake_sha256": _sha_file(intake_path),
        "surface_export_receipt_path": str(export_path.relative_to(ROOT)) if export_path.is_relative_to(ROOT) else str(export_path),
        "surface_export_receipt_sha256": _sha_file(export_path),
        "upstream_export_receipt_verification_sha256": export.get("verification_sha256"),
        "current_independent_surface_verification_sha256": verification_hash,
        "upstream_export_receipt_verification_digest_consistent": (
            export.get("verification_sha256") == verification_hash
        ),
        "surface_candidates": candidate_sources,
        "meshes": meshes,
        "qualification": {
            "source_intake_and_surface_hashes_bound": True,
            "current_independent_surface_verification_bound": True,
            "upstream_export_receipt_verification_digest_consistent": (
                export.get("verification_sha256") == verification_hash
            ),
            "voxel_occupancy_reconstructed_from_closed_surface": True,
            "exact_source_boundary_triangle_multiset": True,
            "six_positive_tetrahedra_per_voxel": True,
            "source_voxel_occupancy_volume_closed": True,
            "expert_segmentation_review": False,
            "numi_subject_binding": False,
            "organ_physical_volume_owner": False,
            "tissue_material_calibrated": False,
            "organ_mass_owner": False,
            "mechanics_admitted": False,
            "lumen_or_perfusion": False,
            "physiological_or_clinical_validation": False,
        },
        "boundary": (
            "Four scan-specific kidney voxel unions were reconstructed from their "
            "source-bound closed surfaces and decomposed into conforming tetrahedral "
            "geometry in each source scan's own RAS+ frame. This is a mesh candidate "
            "from automatic segmentation; it has no expert review, Numi subject, "
            "mechanical mass, tissue material, perfusion, or physiological owner. "
            "The upstream component receipt contains a stale embedded verification "
            "digest; this run binds the current separately checksummed verification "
            "file and preserves that receipt inconsistency."
        ),
    }
    receipt_path = output_dir / "receipt.json"
    result["receipt_sha256"] = _immutable_json(receipt_path, result)
    return result


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    try:
        result = compile_candidate(plan_path=arguments.plan, output_dir=arguments.output)
        print(json.dumps({"schema": SCHEMA, "receipt_sha256": result["receipt_sha256"],
                          "meshes": len(result["meshes"]),
                          "tetrahedra": sum(row["tetrahedron_count"] for row in result["meshes"]),
                          "all_source_boundaries_exact": result["qualification"]["exact_source_boundary_triangle_multiset"],
                          "upstream_verification_digest_consistent": result["upstream_export_receipt_verification_digest_consistent"],
                          "mechanics_admitted": False}, sort_keys=True))
        return 0
    except (HumanImportError, OSError, KeyError, TypeError, ValueError, struct.error) as error:
        print(f"healthy total-body CT voxel tetrahedra: {error}")
        return 2
