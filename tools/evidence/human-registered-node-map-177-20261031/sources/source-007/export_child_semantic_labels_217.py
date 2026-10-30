#!/usr/bin/env python3
"""Export existing native five-column labels for TetGen child cells.

The default uses exact Float32 parent-cell geometry or strict centroid
containment. An explicitly bounded reference-registration option can map
outside-parent centroids to a closest parent tetrahedron and records its
barycentric point plus world residual. Such inferred semantics are not
geometric ancestry, mass transport, or physical admission.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

import p1_reference_anchor_217 as p1_anchor

INVALID = 0xFFFFFFFF
MATERIALS = {15, 16}
LEGACY_MAX_NODES = 300_000
LEGACY_MAX_CELLS = 200_000
FULL_REFERENCE_MAX_NODES = 500_000
FULL_REFERENCE_MAX_CELLS = 2_000_000
MAX_NODES = LEGACY_MAX_NODES
MAX_CELLS = LEGACY_MAX_CELLS
CAPACITY_PROFILE = "legacy"

def configure_capacity_profile(name: str) -> dict[str, int | str]:
    """Apply an explicit child-output limit profile; source-parent limits stay fixed."""
    global MAX_NODES, MAX_CELLS, CAPACITY_PROFILE
    if name == "legacy":
        MAX_NODES, MAX_CELLS = LEGACY_MAX_NODES, LEGACY_MAX_CELLS
    elif name == "full-reference-500k-nodes-2m-cells":
        MAX_NODES, MAX_CELLS = FULL_REFERENCE_MAX_NODES, FULL_REFERENCE_MAX_CELLS
    else:
        raise ValueError(f"unknown capacity profile: {name}")
    CAPACITY_PROFILE = name
    return capacity_profile_receipt()

def capacity_profile_receipt() -> dict[str, int | str]:
    return {"name": CAPACITY_PROFILE, "max_nodes_per_child": MAX_NODES,
            "max_cells_per_child": MAX_CELLS, "max_nodes_combined": MAX_NODES,
            "max_cells_combined": MAX_CELLS}

MAX_PARENT_NODES = 300_000
MAX_PARENT_CELLS = 200_000
BARY_TOL = 1.0e-10
QUERY_PAD_M = 1.0e-12
LABEL_HEADER = [
    "child_tet_row", "material_index", "geometric_donor_body",
    "anatomical_region_id", "stable_source_face_id",
]
SOURCE_HEADER = [
    "child_tet_row", "selected_parent_tet_row", "label_match_kind",
    "matching_parent_tet_count", "semantic_tie_count",
    "centroid_barycentric0", "centroid_barycentric1",
    "centroid_barycentric2", "centroid_barycentric3",
    "centroid_registered_reference_x_m", "centroid_registered_reference_y_m",
    "centroid_registered_reference_z_m", "centroid_reference_residual_x_m",
    "centroid_reference_residual_y_m", "centroid_reference_residual_z_m",
    "centroid_reference_residual_norm_m", "reference_registration_distance_m",
    "reference_registration_max_distance_m",
    "reference_registration_ambiguity_tolerance_m",
    "material_index", "geometric_donor_body", "anatomical_region_id",
    "stable_source_face_id",
]

SAME_MATERIAL_230_OWNER_PATH = Path(
    "/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/"
    "fine-physical-volume-706/current-semantic-atlas-088/export_authoring_registered_surface_sidecars_230_semantic_088.py")
SAME_MATERIAL_230_OWNER_SHA256 = "121e3bf44d90ba3d7b82ccaa40a26a92d1675c6d277a4409ae95f8c05ad2485d"


class LabelMappingRejected(ValueError):
    def __init__(self, message: str, witness: dict[str, Any]):
        super().__init__(message)
        self.witness = witness


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_same_material_230_owner(native_module) -> tuple[Any, dict[str, Any]]:
    path = SAME_MATERIAL_230_OWNER_PATH
    actual = sha256_file(path)
    if actual != SAME_MATERIAL_230_OWNER_SHA256:
        raise ValueError(f"pinned current 230 same-material owner SHA mismatch: {actual}")
    alias = "export_child_semantic_labels_217"
    previous = sys.modules.get(alias)
    sys.modules[alias] = native_module
    name = "_numi_current_semantic_atlas_088_owner"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load pinned current 230 same-material owner")
    owner = importlib.util.module_from_spec(spec)
    sys.modules[name] = owner
    try:
        spec.loader.exec_module(owner)
    finally:
        sys.modules.pop(name, None)
        if previous is None:
            sys.modules.pop(alias, None)
        else:
            sys.modules[alias] = previous
    if getattr(owner, "native", None) is not native_module:
        raise ValueError("230 current-reference owner did not bind this 217 exporter")
    return owner, {"path": str(path), "sha256": actual,
                   "method": "current-semantic-atlas-088 closest_same_material"}


def make_same_material_current_reference(parent_base: Path, native_module):
    owner, owner_pin = load_same_material_230_owner(native_module)
    parent, index, material_indices, parent_meta = owner.load_parent_current(parent_base)
    if not material_indices or any(material not in material_indices for material in MATERIALS):
        raise ValueError("current source330 reference lacks a required same-material 15/16 domain")
    context = {"owner": owner, "index": index, "material_indices": material_indices,
               "parent_meta": parent_meta, "owner_pin": owner_pin}
    return parent, context


def write_rejection_witness(path: Path, exc: Exception, *, parent_base: Path,
                            child_node: Path, child_ele: Path,
                            maximum_distance_m: float | None,
                            ambiguity_tolerance_m: float | None) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite rejection witness {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    witness = exc.witness if isinstance(exc, LabelMappingRejected) else {}
    inputs = {"parent_mass": parent_base,
              "parent_nodes": Path(str(parent_base) + ".nodes.csv"),
              "parent_source": Path(str(parent_base) + ".source.csv"),
              "parent_tets": Path(str(parent_base) + ".tets.csv"),
              "parent_labels": Path(str(parent_base) + ".labels.csv"),
              "child_node": child_node, "child_elements": child_ele,
              "current_230_owner": SAME_MATERIAL_230_OWNER_PATH}
    payload = {
        "schema": "numi.native-domain-217.same-material-reference-rejection.v1",
        "status": "REJECTED_NO_LABELS_EMITTED",
        "reason": str(exc),
        "witness": witness,
        "registration": {"selection": "same-material current-world source330 closest parent tet",
                         "maximum_distance_m": maximum_distance_m,
                         "ambiguity_tolerance_m": ambiguity_tolerance_m,
                         "labels_are_direct_geometric_ancestry": False,
                         "mass_allocation_performed": False},
        "inputs": {name: {"path": str(item), "sha256": sha256_file(item)}
                   for name, item in inputs.items()},
        "labels_output_emitted": False,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def records(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.partition("#")[0].strip()
            if line:
                yield line_number, line.split()


def read_csv(path: Path, header: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"required native parent CSV is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != header:
            raise ValueError(f"unexpected header in {path}: {reader.fieldnames}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"malformed row width in {path}")
    return rows


def uint(text: str, context: str) -> int:
    try:
        value = int(text, 10)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} is not an unsigned integer: {text!r}") from exc
    if value < 0 or value > INVALID:
        raise ValueError(f"{context} is outside uint32: {text!r}")
    return value


def finite_float(text: str, context: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} is not numeric: {text!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context} is not finite")
    return value


def tet_volume(points: np.ndarray) -> float:
    return float(np.dot(points[1] - points[0],
                        np.cross(points[2] - points[0], points[3] - points[0])) / 6.0)


def canonical_f32_tet_key(points: np.ndarray, tet: np.ndarray):
    vertices = np.asarray(points[tet], dtype="<f4").copy()
    vertices[vertices == 0.0] = 0.0
    bits = vertices.view("<u4").reshape((4, 3))
    return tuple(sorted(tuple(int(v) for v in row) for row in bits))


def read_parent(base: Path) -> dict[str, Any]:
    node_path = Path(str(base) + ".nodes.csv")
    source_path = Path(str(base) + ".source.csv")
    tet_path = Path(str(base) + ".tets.csv")
    label_path = Path(str(base) + ".labels.csv")
    node_rows = read_csv(node_path, [
        "parent_node_index", "compiled_global_node", "current_x_m", "current_y_m",
        "current_z_m", "rest_x_m", "rest_y_m", "rest_z_m",
        "compiled_lumped_mass_kg", "node_frame_owner_body", "donor_local_x_m",
        "donor_local_y_m", "donor_local_z_m", "mass_bits",
    ])
    if not 4 <= len(node_rows) <= MAX_PARENT_NODES:
        raise ValueError("native parent node count is outside bounds")
    points = np.empty((len(node_rows), 3), dtype=np.float64)
    for row_id, row in enumerate(node_rows):
        if uint(row["parent_node_index"], f"parent node row {row_id}") != row_id:
            raise ValueError("native parent node rows are not contiguous")
        for axis, key in enumerate(("rest_x_m", "rest_y_m", "rest_z_m")):
            points[row_id, axis] = finite_float(row[key], f"parent node {row_id} {key}")

    mass_rows = read_csv(base, ["source_tet_row", "material_index", "mass_kg"])
    tet_rows = read_csv(tet_path, [
        "parent_tet_row", "material_index", "donor_body", "parent_node0",
        "parent_node1", "parent_node2", "parent_node3",
        "parent_reference_volume_m3", "parent_mass_kg",
    ])
    label_rows = read_csv(label_path, [
        "parent_tet_row", "material_index", "geometric_donor_body",
        "anatomical_region_id", "stable_source_face_id",
    ])
    if not 1 <= len(tet_rows) <= MAX_PARENT_CELLS or not (
        len(mass_rows) == len(label_rows) == len(tet_rows)
    ):
        raise ValueError("native parent cell, mass, and label row counts differ or exceed bounds")
    # The native parent packet owner performs donor-source mass closure. Here we
    # pin and validate the required companion's established table header.
    source_header = [
        "object_local_node", "material_index", "donor_body", "mass_kg",
        "local_x_m", "local_y_m", "local_z_m",
    ]
    with source_path.open("r", encoding="utf-8", newline="") as stream:
        if next(csv.reader(stream), None) != source_header:
            raise ValueError(f"unexpected header in {source_path}")

    tets = np.empty((len(tet_rows), 4), dtype=np.int64)
    materials = np.empty(len(tet_rows), dtype=np.uint32)
    donors = np.empty(len(tet_rows), dtype=np.uint32)
    regions = np.empty(len(tet_rows), dtype=np.uint32)
    faces = np.empty(len(tet_rows), dtype=np.uint32)
    for row_id, (tetrow, massrow, labelrow) in enumerate(zip(tet_rows, mass_rows, label_rows)):
        if uint(tetrow["parent_tet_row"], f"parent tet row {row_id}") != row_id or \
           uint(massrow["source_tet_row"], f"mass tet row {row_id}") != row_id or \
           uint(labelrow["parent_tet_row"], f"label tet row {row_id}") != row_id:
            raise ValueError(f"native parent cell/mass/label rows are misaligned at {row_id}")
        material = uint(tetrow["material_index"], f"parent material row {row_id}")
        donor = uint(tetrow["donor_body"], f"parent donor row {row_id}")
        label_material = uint(labelrow["material_index"], f"label material row {row_id}")
        label_donor = uint(labelrow["geometric_donor_body"], f"label donor row {row_id}")
        region = uint(labelrow["anatomical_region_id"], f"label region row {row_id}")
        face = uint(labelrow["stable_source_face_id"], f"label source face row {row_id}")
        mass_material = uint(massrow["material_index"], f"mass material row {row_id}")
        mass = finite_float(massrow["mass_kg"], f"mass row {row_id}")
        tet_mass = finite_float(tetrow["parent_mass_kg"], f"tet mass row {row_id}")
        if material not in MATERIALS or label_material != material or mass_material != material:
            raise ValueError(f"conflicting native parent material provenance at row {row_id}")
        if donor == INVALID or label_donor != donor:
            raise ValueError(f"conflicting native parent geometric donor at row {row_id}")
        if region == INVALID or face == INVALID:
            raise ValueError(f"missing native parent region or stable source face at row {row_id}")
        if mass <= 0.0 or mass != tet_mass:
            raise ValueError(f"native parent mass differs between cell ledgers at row {row_id}")
        tet = np.asarray([
            uint(tetrow[f"parent_node{corner}"], f"parent tet {row_id} node {corner}")
            for corner in range(4)
        ], dtype=np.int64)
        if np.any(tet >= len(points)) or len(set(tet.tolist())) != 4:
            raise ValueError(f"native parent tetrahedron {row_id} references invalid nodes")
        volume = tet_volume(points[tet])
        recorded_volume = finite_float(tetrow["parent_reference_volume_m3"], f"parent volume row {row_id}")
        if volume <= 0.0 or abs(volume - recorded_volume) > max(1.0e-15, 1.0e-8 * recorded_volume):
            raise ValueError(f"native parent reference tetrahedron invalid at row {row_id}")
        tets[row_id] = tet
        materials[row_id] = material
        donors[row_id] = donor
        regions[row_id] = region
        faces[row_id] = face
    return {
        "points": points, "tets": tets, "materials": materials,
        "donors": donors, "regions": regions, "faces": faces,
        "paths": {"mass": base, "nodes": node_path, "source": source_path,
                  "tets": tet_path, "labels": label_path},
    }


def read_tetgen_vectorized(node_path: Path, ele_path: Path) -> dict[str, np.ndarray]:
    """Validated vectorized reader for the authenticated full-domain P1 handoff."""
    def next_header(path: Path, width: int, name: str):
        iterator = iter(records(path))
        try:
            line_no, fields = next(iterator)
        except StopIteration as exc:
            raise ValueError(f"{name} file is empty") from exc
        if len(fields) != width:
            raise ValueError(f"{name} header needs {width} fields")
        return line_no, fields

    def read_body(path: Path, skiprows: int, rows: int, columns: int, name: str):
        try:
            values = np.loadtxt(path, dtype=np.float64, comments="#", skiprows=skiprows,
                                ndmin=2, encoding="utf-8")
        except (OSError, ValueError) as exc:
            raise ValueError(f"{name} table is malformed: {exc}") from exc
        if values.shape != (rows, columns):
            raise ValueError(f"{name} table shape {values.shape} differs from {(rows, columns)}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} table contains nonfinite values")
        return values

    def ids(values: np.ndarray, name: str):
        if (np.any(values < 0.0) or np.any(values > INVALID) or
                np.any(values != np.floor(values))):
            raise ValueError(f"{name} contains an invalid unsigned integer")
        return values.astype(np.uint64)

    node_header_line, node_header = next_header(node_path, 4, "child TetGen node")
    try:
        node_count, dim, attrs, markers = map(int, node_header)
    except ValueError as exc:
        raise ValueError("child TetGen node header is not integral") from exc
    if not 4 <= node_count <= MAX_NODES or dim != 3 or attrs != 0 or markers not in (0, 1):
        raise ValueError("child TetGen node header invalid or out of bounds")
    node_data = read_body(node_path, node_header_line, node_count, 4 + markers,
                          "child TetGen node")
    node_ids = ids(node_data[:, 0], "child node IDs")
    id_order = np.argsort(node_ids, kind="stable")
    sorted_ids = node_ids[id_order]
    if np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise ValueError(f"duplicate child node ID {int(sorted_ids[np.flatnonzero(sorted_ids[1:] == sorted_ids[:-1])[0]])}")
    if markers and np.any(node_data[:, 4] != np.floor(node_data[:, 4])):
        raise ValueError("child TetGen node table has a nonintegral boundary marker")
    with np.errstate(over="ignore", invalid="ignore"):
        points = node_data[:, 1:4].astype(np.float32)
    if not np.isfinite(points).all():
        raise ValueError("child TetGen node coordinates overflow Float32")

    ele_header_line, ele_header = next_header(ele_path, 3, "child TetGen element")
    try:
        count, corners, nattrs = map(int, ele_header)
    except ValueError as exc:
        raise ValueError("child TetGen element header is not integral") from exc
    if not 1 <= count <= MAX_CELLS or corners != 4 or nattrs != 1:
        raise ValueError("child TetGen elements must be bounded linear tets with one material attribute")
    data = read_body(ele_path, ele_header_line, count, 6, "child TetGen element")
    element_ids = ids(data[:, 0], "child element IDs")
    if len(np.unique(element_ids)) != count:
        raise ValueError("duplicate child element ID")
    ref_ids = ids(data[:, 1:5].reshape(-1), "child element node IDs")
    positions = np.searchsorted(sorted_ids, ref_ids)
    missing = positions >= node_count
    if np.any(missing):
        raise ValueError(f"child element references missing node ID {int(ref_ids[np.flatnonzero(missing)[0]])}")
    mismatched = sorted_ids[positions] != ref_ids
    if np.any(mismatched):
        raise ValueError(f"child element references missing node ID {int(ref_ids[np.flatnonzero(mismatched)[0]])}")
    tets = id_order[positions].reshape(count, 4).astype(np.int64, copy=False)
    material_values = data[:, 5]
    valid_material = np.isin(material_values, tuple(MATERIALS))
    integral_material = material_values == np.floor(material_values)
    if np.any(~valid_material | ~integral_material):
        bad = int(np.flatnonzero(~valid_material | ~integral_material)[0])
        raise ValueError(f"child cell {bad} has unsupported material {material_values[bad]}")
    materials = material_values.astype(np.uint32)
    ordered_tets = np.sort(tets, axis=1)
    repeated = np.any(ordered_tets[:, 1:] == ordered_tets[:, :-1], axis=1)
    if np.any(repeated):
        raise ValueError(f"child cell {int(np.flatnonzero(repeated)[0])} repeats nodes")
    if len(np.unique(ordered_tets, axis=0)) != count:
        raise ValueError("child element table duplicates a tetrahedron")

    # Evaluate the exact same F32-coordinate determinant in F64, vectorized.
    p0 = points[tets[:, 0]].astype(np.float64)
    a = points[tets[:, 1]].astype(np.float64) - p0
    b = points[tets[:, 2]].astype(np.float64) - p0
    c = points[tets[:, 3]].astype(np.float64) - p0
    det6 = (a[:, 0] * (b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1]) -
            a[:, 1] * (b[:, 0] * c[:, 2] - b[:, 2] * c[:, 0]) +
            a[:, 2] * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]))
    invalid = ~np.isfinite(det6) | (det6 <= 0.0)
    if np.any(invalid):
        raise ValueError(f"child cell {int(np.flatnonzero(invalid)[0])} is nonpositive after Float32 emission")
    return {"points": points, "tets": tets, "materials": materials}


def read_tetgen(node_path: Path, ele_path: Path) -> dict[str, np.ndarray]:
    node_iter = iter(records(node_path))
    try:
        _, header = next(node_iter)
    except StopIteration as exc:
        raise ValueError("child TetGen node file is empty") from exc
    if len(header) != 4:
        raise ValueError("child TetGen node header needs four fields")
    node_count, dim, attrs, markers = map(int, header)
    if not 4 <= node_count <= MAX_NODES or dim != 3 or attrs != 0 or markers not in (0, 1):
        raise ValueError("child TetGen node header invalid or out of bounds")
    points = np.empty((node_count, 3), dtype=np.float32)
    node_lookup: dict[int, int] = {}
    for row_id in range(node_count):
        try:
            line_no, fields = next(node_iter)
        except StopIteration as exc:
            raise ValueError("child TetGen node table truncated") from exc
        if len(fields) != 4 + markers:
            raise ValueError(f"child TetGen node row {line_no} has invalid width")
        node_id = uint(fields[0], f"child node ID at line {line_no}")
        if node_id in node_lookup:
            raise ValueError(f"duplicate child node ID {node_id}")
        node_lookup[node_id] = row_id
        for axis in range(3):
            points[row_id, axis] = np.float32(finite_float(fields[axis + 1], f"child node {row_id}"))
        if markers and int(fields[4]) != int(fields[4]):
            raise ValueError(f"invalid boundary marker on child node row {row_id}")
    try:
        next(node_iter)
    except StopIteration:
        pass
    else:
        raise ValueError("child TetGen node table has extra rows")

    ele_iter = iter(records(ele_path))
    try:
        _, header = next(ele_iter)
    except StopIteration as exc:
        raise ValueError("child TetGen element file is empty") from exc
    if len(header) != 3:
        raise ValueError("child TetGen element header needs three fields")
    count, corners, nattrs = map(int, header)
    if not 1 <= count <= MAX_CELLS or corners != 4 or nattrs != 1:
        raise ValueError("child TetGen elements must be bounded linear tets with one material attribute")
    tets = np.empty((count, 4), dtype=np.int64)
    materials = np.empty(count, dtype=np.uint32)
    element_ids: set[int] = set()
    canonical: set[tuple[int, int, int, int]] = set()
    for row_id in range(count):
        try:
            line_no, fields = next(ele_iter)
        except StopIteration as exc:
            raise ValueError("child TetGen element table truncated") from exc
        if len(fields) != 6:
            raise ValueError(f"child TetGen element row {line_no} has invalid width")
        element_id = uint(fields[0], f"child element ID at line {line_no}")
        if element_id in element_ids:
            raise ValueError(f"duplicate child element ID {element_id}")
        element_ids.add(element_id)
        tet = []
        for corner in range(4):
            node_id = uint(fields[corner + 1], f"child element {element_id} node {corner}")
            if node_id not in node_lookup:
                raise ValueError(f"child element {element_id} references missing node ID {node_id}")
            tet.append(node_lookup[node_id])
        material = finite_float(fields[5], f"child material row {row_id}")
        if material not in MATERIALS or int(material) != material:
            raise ValueError(f"child cell {row_id} has unsupported material {material}")
        key = tuple(sorted(tet))
        if len(set(key)) != 4 or key in canonical:
            raise ValueError(f"child cell {row_id} repeats nodes or duplicates a tetrahedron")
        volume = tet_volume(points[np.asarray(tet)].astype(np.float64))
        if volume <= 0.0 or not math.isfinite(volume):
            raise ValueError(f"child cell {row_id} is nonpositive after Float32 emission")
        canonical.add(key)
        tets[row_id] = tet
        materials[row_id] = int(material)
    try:
        next(ele_iter)
    except StopIteration:
        pass
    else:
        raise ValueError("child TetGen element table has extra rows")
    return {"points": points, "tets": tets, "materials": materials}


def semantic(parent: dict[str, Any], row: int) -> tuple[int, int, int, int]:
    return (int(parent["materials"][row]), int(parent["donors"][row]),
            int(parent["regions"][row]), int(parent["faces"][row]))


def build_index(parent: dict[str, Any]) -> dict[str, Any]:
    points, tets = parent["points"], parent["tets"]
    centroids = points[tets].mean(axis=1)
    radii = np.linalg.norm(points[tets] - centroids[:, None, :], axis=2).max(axis=1)
    matrices = np.transpose(points[tets[:, 1:]] - points[tets[:, :1]], (0, 2, 1))
    try:
        inverses = np.linalg.inv(matrices)
    except np.linalg.LinAlgError as exc:
        raise ValueError("native parent contains a singular tetrahedron") from exc
    if not np.isfinite(inverses).all():
        raise ValueError("native parent has nonfinite tetrahedral inverse")
    f32 = points.astype("<f4")
    f32[f32 == 0.0] = 0.0
    bits = f32.view("<u4").reshape((-1, 3))
    exact: dict[tuple[tuple[int, int, int], ...], list[int]] = {}
    for row_id, tet in enumerate(tets):
        key = tuple(sorted(tuple(int(v) for v in bits[node]) for node in tet))
        exact.setdefault(key, []).append(row_id)
    return {"centroids": centroids, "max_radius": float(radii.max()) + QUERY_PAD_M,
            "inverses": inverses, "tree": cKDTree(centroids), "exact": exact}


def barycentric(parent: dict[str, Any], index: dict[str, Any],
                parent_row: int, point: np.ndarray) -> np.ndarray:
    tet = parent["tets"][parent_row]
    q = index["inverses"][parent_row] @ (point - parent["points"][tet[0]])
    return np.asarray([1.0 - q.sum(), q[0], q[1], q[2]], dtype=np.float64)


def closest_point_on_triangle(point: np.ndarray, triangle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b, c = triangle
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    normal2 = float(np.dot(normal, normal))
    if not math.isfinite(normal2) or normal2 <= 0.0:
        raise ValueError("parent tetrahedron has a degenerate face during reference registration")
    projected = point - normal * (float(np.dot(point - a, normal)) / normal2)
    v2 = projected - a
    d00, d01, d11 = float(np.dot(ab, ab)), float(np.dot(ab, ac)), float(np.dot(ac, ac))
    d20, d21 = float(np.dot(v2, ab)), float(np.dot(v2, ac))
    denominator = d00 * d11 - d01 * d01
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("parent tetrahedron has a numerically degenerate face during reference registration")
    wb = (d11 * d20 - d01 * d21) / denominator
    wc = (d00 * d21 - d01 * d20) / denominator
    inside = np.asarray([1.0 - wb - wc, wb, wc], dtype=np.float64)
    if float(np.min(inside)) >= -BARY_TOL:
        inside = np.maximum(inside, 0.0)
        inside /= float(inside.sum())
        return inside @ triangle, inside

    best = None
    for edge_row, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
        start, delta = triangle[i], triangle[j] - triangle[i]
        delta2 = float(np.dot(delta, delta))
        if delta2 <= 0.0:
            raise ValueError("parent tetrahedron has a degenerate edge during reference registration")
        t = min(1.0, max(0.0, float(np.dot(point - start, delta)) / delta2))
        edge_bary = np.zeros(3, dtype=np.float64)
        edge_bary[i], edge_bary[j] = 1.0 - t, t
        candidate = edge_bary @ triangle
        distance2 = float(np.dot(candidate - point, candidate - point))
        key = (distance2, edge_row)
        if best is None or key < best[0]:
            best = (key, candidate, edge_bary)
    assert best is not None
    return best[1], best[2]


def closest_point_on_parent_tet(parent: dict[str, Any], index: dict[str, Any],
                                parent_row: int, point: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray, float]:
    tet = parent["tets"][parent_row]
    tet_points = parent["points"][tet]
    bary = barycentric(parent, index, parent_row, point)
    if float(np.min(bary)) >= -BARY_TOL:
        bary = np.maximum(bary, 0.0)
        bary /= float(bary.sum())
        registered = bary @ tet_points
        residual = float(np.linalg.norm(point - registered))
        return registered, bary, residual

    best = None
    face_rows = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    for face_id, face in enumerate(face_rows):
        registered, face_bary = closest_point_on_triangle(point, tet_points[list(face)])
        full_bary = np.zeros(4, dtype=np.float64)
        full_bary[list(face)] = face_bary
        residual = float(np.linalg.norm(point - registered))
        key = (residual * residual, face_id)
        if best is None or key < best[0]:
            best = (key, registered, full_bary, residual)
    assert best is not None
    return best[1], best[2], best[3]


def locate_by_bounded_reference_registration(
        parent: dict[str, Any], index: dict[str, Any], child_row: int,
        point: np.ndarray, material: int, maximum_distance_m: float,
        ambiguity_tolerance_m: float) -> dict[str, Any]:
    center_distance, seed_row = index["tree"].query(point, k=1)
    if not math.isfinite(float(center_distance)):
        raise ValueError(f"child row {child_row} cannot query the parent reference cells")
    seed_row = int(seed_row)
    _, _, seed_distance = closest_point_on_parent_tet(parent, index, seed_row, point)
    query_radius = seed_distance + index["max_radius"] + ambiguity_tolerance_m + QUERY_PAD_M
    candidate_rows = index["tree"].query_ball_point(point, query_radius)
    measured = []
    for parent_row in candidate_rows:
        registered, bary, distance = closest_point_on_parent_tet(parent, index, int(parent_row), point)
        measured.append((distance, int(parent_row), registered, bary))
    if not measured:
        raise ValueError(f"child row {child_row} has no closest parent reference cell")
    best_distance = min(item[0] for item in measured)
    if best_distance > maximum_distance_m:
        raise ValueError(
            f"child row {child_row} nearest parent reference residual {best_distance:.17g} m "
            f"exceeds declared bound {maximum_distance_m:.17g} m")
    tied = [item for item in measured
            if abs(item[0] - best_distance) <= ambiguity_tolerance_m]
    tied.sort(key=lambda item: item[1])
    tied_rows = [item[1] for item in tied]
    materials = {int(parent["materials"][row]) for row in tied_rows}
    if materials != {material}:
        raise ValueError(
            f"child row {child_row} closest-parent material conflict: child={material} "
            f"parents={sorted(materials)} rows={tied_rows[:16]}")
    tuples = {semantic(parent, row) for row in tied_rows}
    if len(tuples) != 1:
        details = sorted((row, semantic(parent, row)) for row in tied_rows)
        raise ValueError(
            f"child row {child_row} has ambiguous closest-parent semantic labels {details[:16]}")
    distance, selected, registered, bary = tied[0]
    return {
        "parent_row": selected,
        "kind": "bounded_closest_parent_tet_reference_registration",
        "candidate_count": len(tied),
        "tie_count": len(tied) - 1,
        "bary": bary,
        "registered_point": registered,
        "registration_distance_m": distance,
    }


def locate_same_material_current_reference(
        parent: dict[str, Any], child: dict[str, Any], child_row: int,
        context: dict[str, Any], maximum_distance_m: float,
        ambiguity_tolerance_m: float) -> dict[str, Any]:
    tet = child["tets"][child_row]
    material = int(child["materials"][child_row])
    centroid = child["points"][tet].astype(np.float64).mean(axis=0)
    base_witness = {"child_tet_row": int(child_row), "child_material_index": material,
                    "child_centroid_world_m": [float(x) for x in centroid],
                    "maximum_distance_m": float(maximum_distance_m),
                    "ambiguity_tolerance_m": float(ambiguity_tolerance_m),
                    "parent_row_domain": "source330 global parent_tet_row"}
    material_indices = context["material_indices"]
    if material not in material_indices:
        raise LabelMappingRejected(
            f"child row {child_row} has no source330 same-material-{material} reference domain",
            {**base_witness, "tied_parent_rows": [], "tied_semantic_tuples": []})
    try:
        registration = context["owner"].closest_same_material(
            parent, context["index"], material_indices, material, centroid,
            maximum_distance_m, ambiguity_tolerance_m)
    except Exception as exc:
        raise LabelMappingRejected(
            f"child row {child_row} same-material current-reference registration rejected: {exc}",
            {**base_witness, "owner_rejection": str(exc)}) from exc
    ties = sorted(set(map(int, registration.get("ties", [registration["tet"]]))))
    if not ties or any(row < 0 or row >= len(parent["tets"]) for row in ties):
        raise LabelMappingRejected(
            f"child row {child_row} current-reference owner returned an invalid global parent row tie set",
            {**base_witness, "tied_parent_rows": ties})
    tied_semantics = [{"parent_tet_row": row,
                       "semantic_tuple": list(semantic(parent, row))} for row in ties]
    tuples = {tuple(item["semantic_tuple"]) for item in tied_semantics}
    if len(tuples) != 1:
        raise LabelMappingRejected(
            f"child row {child_row} closest same-material parent tie has conflicting semantic tuples",
            {**base_witness, "tied_parent_rows": ties,
             "tied_semantic_tuples": tied_semantics,
             "nearest_distance_m": float(registration["distance"])})
    selected = int(registration["tet"])
    if selected not in ties or int(parent["materials"][selected]) != material:
        raise LabelMappingRejected(
            f"child row {child_row} current-reference owner selected a row outside the same-material tie set",
            {**base_witness, "selected_parent_tet_row": selected,
             "tied_parent_rows": ties, "tied_semantic_tuples": tied_semantics})
    distance = float(registration["distance"])
    if not math.isfinite(distance) or distance > maximum_distance_m:
        raise LabelMappingRejected(
            f"child row {child_row} same-material reference distance exceeds declared bound",
            {**base_witness, "selected_parent_tet_row": selected,
             "tied_parent_rows": ties, "distance_m": distance})
    registered = np.asarray(registration["projected"], dtype=np.float64)
    bary = np.asarray(registration["bary"], dtype=np.float64)
    if registered.shape != (3,) or bary.shape != (4,) or not np.isfinite(registered).all() or not np.isfinite(bary).all():
        raise LabelMappingRejected(
            f"child row {child_row} current-reference owner returned invalid projection data",
            {**base_witness, "selected_parent_tet_row": selected,
             "tied_parent_rows": ties, "registered_reference_point": registered.tolist(),
             "barycentric": bary.tolist()})
    return {"parent_row": selected,
            "kind": "same_material_bounded_current_reference_registration",
            "candidate_count": len(ties), "tie_count": len(ties) - 1,
            "bary": bary, "registered_point": registered,
            "registration_distance_m": distance,
            "same_material_tied_parent_rows": ties,
            "exact_near_tie_fallback": registration.get("exactTieFallback")}


def locate(parent: dict[str, Any], index: dict[str, Any], child: dict[str, Any],
           child_row: int, reference_registration_max_distance_m: float | None = None,
           reference_registration_ambiguity_tolerance_m: float | None = None,
           same_material_current_reference: dict[str, Any] | None = None) -> dict[str, Any]:
    if (reference_registration_max_distance_m is None) != (reference_registration_ambiguity_tolerance_m is None):
        raise ValueError("bounded reference registration requires both a maximum distance and ambiguity tolerance")
    if reference_registration_max_distance_m is not None:
        if not math.isfinite(reference_registration_max_distance_m) or reference_registration_max_distance_m <= 0.0:
            raise ValueError("bounded reference-registration maximum distance must be finite and positive")
        if not math.isfinite(reference_registration_ambiguity_tolerance_m) or reference_registration_ambiguity_tolerance_m < 0.0:
            raise ValueError("reference-registration ambiguity tolerance must be finite and nonnegative")
    tet = child["tets"][child_row]
    material = int(child["materials"][child_row])
    centroid = child["points"][tet].astype(np.float64).mean(axis=0)
    if same_material_current_reference is not None:
        if reference_registration_max_distance_m is None or reference_registration_ambiguity_tolerance_m is None:
            raise ValueError("same-material current-reference mode requires explicit distance and ambiguity bounds")
        return locate_same_material_current_reference(
            parent, child, child_row, same_material_current_reference,
            float(reference_registration_max_distance_m),
            float(reference_registration_ambiguity_tolerance_m))
    key = canonical_f32_tet_key(child["points"], tet)
    exact_rows = index["exact"].get(key, [])
    if exact_rows:
        exact_materials = {int(parent["materials"][row]) for row in exact_rows}
        if exact_materials != {material}:
            raise ValueError(f"child row {child_row} exact parent geometry has conflicting material {sorted(exact_materials)}")
        tuples = {semantic(parent, row) for row in exact_rows}
        if len(tuples) != 1:
            raise ValueError(f"child row {child_row} exact parent geometry has conflicting provenance {sorted(tuples)}")
        selected = min(exact_rows)
        return {"parent_row": selected, "kind": "exact_f32_parent_tet_geometry",
                "candidate_count": len(exact_rows), "tie_count": len(exact_rows) - 1,
                "bary": barycentric(parent, index, selected, centroid)}

    candidates = index["tree"].query_ball_point(centroid, index["max_radius"])
    contained: list[tuple[int, np.ndarray]] = []
    for parent_row in candidates:
        b = barycentric(parent, index, int(parent_row), centroid)
        if np.min(b) >= -BARY_TOL:
            contained.append((int(parent_row), b))
    if not contained:
        if reference_registration_max_distance_m is None:
            raise ValueError(f"child row {child_row} centroid is outside all retained parent cells")
        return locate_by_bounded_reference_registration(
            parent, index, child_row, centroid, material,
            reference_registration_max_distance_m,
            float(reference_registration_ambiguity_tolerance_m))
    found_materials = {int(parent["materials"][row]) for row, _ in contained}
    if found_materials != {material}:
        raise ValueError(f"child row {child_row} material {material} conflicts with containing parent materials {sorted(found_materials)}")
    tuples = {semantic(parent, row) for row, _ in contained}
    if len(tuples) != 1:
        detail = sorted((row, semantic(parent, row)) for row, _ in contained)
        raise ValueError(f"child row {child_row} has conflicting parent semantic boundary labels {detail[:16]}")
    selected, b = min(contained, key=lambda item: item[0])
    return {"parent_row": selected, "kind": "parent_centroid_containment",
            "candidate_count": len(contained), "tie_count": len(contained) - 1,
            "bary": b}


def assign_labels(parent: dict[str, Any], child: dict[str, Any],
                  reference_registration_max_distance_m: float | None = None,
                  reference_registration_ambiguity_tolerance_m: float | None = None,
                  same_material_current_reference: dict[str, Any] | None = None,
                  child_prefix_count: int | None = None):
    index = (same_material_current_reference["index"]
             if same_material_current_reference is not None else build_index(parent))
    if child_prefix_count is not None and (child_prefix_count <= 0 or child_prefix_count > len(child["tets"])):
        raise ValueError("diagnostic child prefix count must be positive and no larger than the child cell count")
    row_count = len(child["tets"]) if child_prefix_count is None else child_prefix_count
    labels = np.empty((row_count, 5), dtype=np.uint32)
    ancestry = []
    for child_row in range(row_count):
        found = locate(parent, index, child, child_row,
                       reference_registration_max_distance_m,
                       reference_registration_ambiguity_tolerance_m,
                       same_material_current_reference)
        parent_row = int(found["parent_row"])
        values = semantic(parent, parent_row)
        if values[0] != int(child["materials"][child_row]):
            raise ValueError(f"child row {child_row} material conflicts with selected parent")
        labels[child_row] = (child_row, *values)
        tet = child["tets"][child_row]
        centroid = child["points"][tet].astype(np.float64).mean(axis=0)
        registered = found.get("registered_point")
        if registered is None:
            registered = found["bary"] @ parent["points"][parent["tets"][parent_row]]
        residual = centroid - registered
        residual_norm = float(np.linalg.norm(residual))
        if not np.isfinite(residual).all() or not math.isfinite(residual_norm):
            raise ValueError(f"child row {child_row} registration reconstruction is nonfinite")
        ancestry.append({
            "child_tet_row": child_row,
            "selected_parent_tet_row": parent_row,
            "label_match_kind": found["kind"],
            "matching_parent_tet_count": found["candidate_count"],
            "semantic_tie_count": found["tie_count"],
            "centroid_barycentric": found["bary"].tolist(),
            "registered_reference_point": registered.tolist(),
            "reference_residual": residual.tolist(),
            "reference_residual_norm_m": residual_norm,
            "reference_registration_distance_m": found.get("registration_distance_m"),
            "same_material_tied_parent_rows": found.get("same_material_tied_parent_rows"),
            "exact_near_tie_fallback": found.get("exact_near_tie_fallback"),
            "material_index": values[0],
            "geometric_donor_body": values[1],
            "anatomical_region_id": values[2],
            "stable_source_face_id": values[3],
        })
    return labels, ancestry


def export(parent_base: Path, node_path: Path, ele_path: Path,
           labels_path: Path, source_map_path: Path | None = None,
           receipt_path: Path | None = None,
           reference_registration_max_distance_m: float | None = None,
           reference_registration_ambiguity_tolerance_m: float | None = None,
           same_material_current_reference_registration: bool = False) -> dict[str, Any]:
    if (reference_registration_max_distance_m is None) != (reference_registration_ambiguity_tolerance_m is None):
        raise ValueError("bounded reference registration requires both a maximum distance and ambiguity tolerance")
    if same_material_current_reference_registration and reference_registration_max_distance_m is None:
        raise ValueError("same-material current-reference mode requires explicit distance and ambiguity bounds")
    context = None
    if same_material_current_reference_registration:
        context_owner = load_same_material_230_owner(sys.modules[__name__])
        owner, owner_pin = context_owner
        parent, parent_index, material_indices, parent_meta = owner.load_parent_current(parent_base)
        if any(material not in material_indices for material in MATERIALS):
            raise ValueError("current source330 reference lacks required same-material 15/16 cells")
        context = {"owner": owner, "index": parent_index,
                   "material_indices": material_indices,
                   "parent_meta": parent_meta, "owner_pin": owner_pin}
    else:
        parent = read_parent(parent_base)
    child = read_tetgen(node_path, ele_path)
    labels, ancestry = assign_labels(parent, child,
                                     reference_registration_max_distance_m,
                                     reference_registration_ambiguity_tolerance_m,
                                     context)
    source_map_path = source_map_path or Path(str(labels_path) + ".source.csv")
    receipt_path = receipt_path or Path(str(labels_path) + ".receipt.json")
    outputs = [labels_path, source_map_path, receipt_path]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite an existing child-label output")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="child-labels-217-", dir=labels_path.parent))
    try:
        temp_labels = temp_root / labels_path.name
        temp_source = temp_root / source_map_path.name
        temp_receipt = temp_root / receipt_path.name
        with temp_labels.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(LABEL_HEADER)
            writer.writerows(labels.tolist())
        with temp_source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SOURCE_HEADER, lineterminator="\n")
            writer.writeheader()
            for row in ancestry:
                b = row["centroid_barycentric"]
                writer.writerow({
                    "child_tet_row": row["child_tet_row"],
                    "selected_parent_tet_row": row["selected_parent_tet_row"],
                    "label_match_kind": row["label_match_kind"],
                    "matching_parent_tet_count": row["matching_parent_tet_count"],
                    "semantic_tie_count": row["semantic_tie_count"],
                    **{f"centroid_barycentric{i}": format(b[i], ".17g") for i in range(4)},
                    **{f"centroid_registered_reference_{axis}_m": format(row["registered_reference_point"][i], ".17g")
                       for i, axis in enumerate(("x", "y", "z"))},
                    **{f"centroid_reference_residual_{axis}_m": format(row["reference_residual"][i], ".17g")
                       for i, axis in enumerate(("x", "y", "z"))},
                    "centroid_reference_residual_norm_m": format(row["reference_residual_norm_m"], ".17g"),
                    "reference_registration_distance_m": (
                        "" if row["reference_registration_distance_m"] is None
                        else format(row["reference_registration_distance_m"], ".17g")),
                    "reference_registration_max_distance_m": (
                        "" if reference_registration_max_distance_m is None
                        else format(reference_registration_max_distance_m, ".17g")),
                    "reference_registration_ambiguity_tolerance_m": (
                        "" if reference_registration_ambiguity_tolerance_m is None
                        else format(reference_registration_ambiguity_tolerance_m, ".17g")),
                    "material_index": row["material_index"],
                    "geometric_donor_body": row["geometric_donor_body"],
                    "anatomical_region_id": row["anatomical_region_id"],
                    "stable_source_face_id": row["stable_source_face_id"],
                })
        same_material_mode = same_material_current_reference_registration
        inputs = {"parent_mass": parent_base, **parent["paths"],
                  "child_node": node_path, "child_elements": ele_path}
        if same_material_mode:
            inputs["current_same_material_230_owner"] = SAME_MATERIAL_230_OWNER_PATH
        registration_enabled = reference_registration_max_distance_m is not None
        same_material_mode = same_material_current_reference_registration
        receipt = {
            "status": "CHILD_SEMANTIC_LABELS_EXPORTED_NOT_PHYSICAL_ADMISSION",
            "capacity_profile": capacity_profile_receipt(),
            "parent_source_limits": {"max_nodes": MAX_PARENT_NODES, "max_cells": MAX_PARENT_CELLS},
            "admission": False,
            "method": (
                "same_material_current_world_source330_closest_parent_tet_reference_registration"
                if same_material_mode else
                "exact_f32_parent_tet_geometry_then_parent_tet_centroid_containment_then_bounded_closest_parent_tet_reference_registration"
                if registration_enabled else
                "exact_f32_parent_tet_geometry_then_parent_tet_centroid_containment"),
            "label_fields": LABEL_HEADER,
            "labels_are_direct_geometric_ancestry": False,
            "bounded_reference_registration": {
                "enabled": registration_enabled,
                "maximum_distance_m": reference_registration_max_distance_m,
                "ambiguity_tolerance_m": reference_registration_ambiguity_tolerance_m,
                "selection_mode": ("same-material-only over current source330 parent rows" if same_material_mode
                                   else "exact geometry/containment first, bounded closest reference only outside all parent cells"),
                "semantics": "Inferred semantic reference registration only; not geometric ancestry, overlap, or mass transport.",
            },
            "same_material_current_reference": ({
                "enabled": True,
                "owner": context["owner_pin"],
                "parent_coordinate_source": "compiled-parent-cell-mass.csv.nodes.csv current_x_m/current_y_m/current_z_m",
                "parent_coordinate_sha256": context["parent_meta"]["nodeSHA256"],
                "parent_global_tet_row_count": context["parent_meta"]["tetCount"],
                "semantic_source": "immutable source330 parent labels indexed by global parent_tet_row",
                "labels_are_direct_geometric_ancestry": False,
                "mass_allocation_performed": False,
            } if same_material_mode else {"enabled": False}),
            "scope": (
                "Every child row uses the closest current-world source330 parent tetrahedron among the complete global row domain with the same material as the child. Material, donor, region, and stable face remain separate source330 semantic fields; registration is an inferred reference anchor only."
                if same_material_mode else
                "Each row copies separate material, geometric donor, anatomical region, and stable source face from an exact-F32 match, a centroid-containing native parent tet, or an explicitly bounded closest-parent-tet reference registration. The latter records parent-cell barycentrics and world residual."
                if registration_enabled else
                "Each row copies separate material, geometric donor, anatomical region, and stable source face from an exact-F32 or centroid-containing native parent tet. No nearest-cell extrapolation is used."),
            "tie_policy": ("The current 230 same-material closest-tet owner supplies global parent rows. All tied rows within the declared ambiguity tolerance must share the complete (material, donor, region, stable face) tuple; conflicting tuples reject with a saved witness."
                           if same_material_mode else
                           "Containment and closest-parent ties are accepted only when every tied parent has the same complete semantic tuple; the lowest parent_tet_row wins and the tie count is recorded. Conflicting material or labels fail closed."),
            "coordinate_space": ("source330 current_x_m/current_y_m/current_z_m and Float32 child TetGen coordinates in one registered current reference frame"
                                 if same_material_mode else
                                 "native parent rest_x_m and Float32 child TetGen coordinates in one registered reference frame"),
            "counts": {
                "child_nodes": int(len(child["points"])),
                "child_tetrahedra": int(len(child["tets"])),
                "parent_nodes": int(len(parent["points"])),
                "parent_tetrahedra": int(len(parent["tets"])),
                "exact_geometry_matches": sum(r["label_match_kind"] == "exact_f32_parent_tet_geometry" for r in ancestry),
                "centroid_containment_matches": sum(r["label_match_kind"] == "parent_centroid_containment" for r in ancestry),
                "bounded_reference_registration_matches": sum(
                    r["label_match_kind"] == "bounded_closest_parent_tet_reference_registration"
                    for r in ancestry),
                "same_material_current_reference_registration_matches": sum(
                    r["label_match_kind"] == "same_material_bounded_current_reference_registration"
                    for r in ancestry),
                "same_material_exact_near_tie_fallbacks": sum(
                    r.get("exact_near_tie_fallback") is not None for r in ancestry),
                "boundary_ties": sum(r["semantic_tie_count"] > 0 for r in ancestry),
            },
            "inputs": {name: {"path": str(path), "sha256": sha256_file(path)}
                       for name, path in inputs.items()},
            "outputs": {
                "labels": {"path": str(labels_path), "sha256": sha256_file(temp_labels)},
                "parent_row_map": {"path": str(source_map_path), "sha256": sha256_file(temp_source)},
            },
        }
        temp_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        if any(path.exists() for path in outputs):
            raise FileExistsError("an output appeared during export; refusing to replace it")
        committed: list[Path] = []
        try:
            for temp, dest in zip((temp_labels, temp_source, temp_receipt), outputs):
                os.replace(temp, dest)
                committed.append(dest)
        except Exception:
            for path in committed:
                path.unlink(missing_ok=True)
            raise
        return receipt
    finally:
        try:
            temp_root.rmdir()
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-mass-csv", required=True, type=Path)
    parser.add_argument("--child-node", required=True, type=Path)
    parser.add_argument("--child-ele", required=True, type=Path)
    parser.add_argument("--output-labels-csv", required=True, type=Path)
    parser.add_argument("--source-map-csv", type=Path)
    parser.add_argument("--receipt-json", type=Path)
    parser.add_argument("--reference-registration-max-distance-m", type=float,
                        help="opt into inferred semantic registration for outside-parent centroids with this hard residual bound")
    parser.add_argument("--capacity-profile", choices=("legacy", "full-reference-500k-nodes-2m-cells"),
                        default="legacy", help="explicit child mesh capacity; source-parent limits remain 300k/200k")
    parser.add_argument("--reference-registration-ambiguity-tolerance-m", type=float,
                        help="required with bounded reference registration; tied closest cells with conflicting semantic tuples are rejected")
    parser.add_argument("--same-material-current-reference-registration", action="store_true",
                        help="explicitly register every child centroid to the closest current source330 tet of the same material; inferred semantic reference only")
    parser.add_argument("--p1-supported-reference-anchor", action="store_true",
                        help="explicitly choose an inferred typed reference anchor only from each cell's authenticated P1 parent-row union")
    parser.add_argument("--p1-skin-support-npz", type=Path)
    parser.add_argument("--p1-skin-support-receipt-json", type=Path)
    parser.add_argument("--p1-fat-support-npz", type=Path)
    parser.add_argument("--p1-fat-support-receipt-json", type=Path)
    parser.add_argument("--combined-element-map-csv-gz", type=Path)
    parser.add_argument("--combined-node-map-csv-gz", type=Path)
    parser.add_argument("--combined-handoff-receipt-json", type=Path)
    parser.add_argument("--parent-full86-weights-csv", type=Path)
    parser.add_argument("--anchor-csr-npz", type=Path)
    parser.add_argument("--anchor-projection-csv", type=Path,
                        help="native-readable row crosswalk bound to the complete P1 anchor CSR")
    parser.add_argument("--projection-helper-path", type=Path,
                        default=p1_anchor.DEFAULT_PROJECTION_HELPER_PATH)
    parser.add_argument("--projection-helper-sha256", default=p1_anchor.DEFAULT_PROJECTION_HELPER_SHA256)
    parser.add_argument("--diagnostic-child-prefix-count", type=int)
    parser.add_argument("--collect-rejection-census", action="store_true",
                        help="scan the requested complete child domain and retain every row-level anchor rejection; emit no labels if any fail")
    parser.add_argument("--rejection-witness-json", type=Path,
                        help="required for inferred registration modes; preserves a structured witness if a row/bound/tie is rejected")
    args = parser.parse_args()
    if args.same_material_current_reference_registration and args.p1_supported_reference_anchor:
        parser.error("same-material global registration and P1-supported anchor modes are mutually exclusive")
    if args.p1_supported_reference_anchor:
        required = {
            "--p1-skin-support-npz": args.p1_skin_support_npz,
            "--p1-skin-support-receipt-json": args.p1_skin_support_receipt_json,
            "--p1-fat-support-npz": args.p1_fat_support_npz,
            "--p1-fat-support-receipt-json": args.p1_fat_support_receipt_json,
            "--combined-element-map-csv-gz": args.combined_element_map_csv_gz,
            "--combined-node-map-csv-gz": args.combined_node_map_csv_gz,
            "--combined-handoff-receipt-json": args.combined_handoff_receipt_json,
            "--parent-full86-weights-csv": args.parent_full86_weights_csv,
            "--anchor-csr-npz": args.anchor_csr_npz,
            "--anchor-projection-csv": args.anchor_projection_csv,
            "--rejection-witness-json": args.rejection_witness_json,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("P1-supported reference-anchor mode requires " + ", ".join(missing))
        if args.reference_registration_max_distance_m is None or args.reference_registration_ambiguity_tolerance_m is None:
            parser.error("P1-supported reference-anchor mode requires explicit distance and ambiguity bounds")
        if args.diagnostic_child_prefix_count is not None and args.diagnostic_child_prefix_count <= 0:
            parser.error("diagnostic prefix count must be positive")
    if args.same_material_current_reference_registration:
        if args.reference_registration_max_distance_m is None or args.reference_registration_ambiguity_tolerance_m is None:
            parser.error("same-material current-reference mode requires explicit distance and ambiguity bounds")
        if args.rejection_witness_json is None:
            parser.error("same-material current-reference mode requires --rejection-witness-json")
    configure_capacity_profile(args.capacity_profile)
    if args.p1_supported_reference_anchor:
        try:
            result = p1_anchor.export_p1_supported_reference_anchor(
                sys.modules[__name__], parent_base=args.parent_mass_csv,
                parent_full86_path=args.parent_full86_weights_csv,
                child_node_path=args.child_node, child_ele_path=args.child_ele,
                skin_support_path=args.p1_skin_support_npz,
                skin_support_receipt_path=args.p1_skin_support_receipt_json,
                fat_support_path=args.p1_fat_support_npz,
                fat_support_receipt_path=args.p1_fat_support_receipt_json,
                combined_element_map_path=args.combined_element_map_csv_gz,
                combined_node_map_path=args.combined_node_map_csv_gz,
                combined_handoff_receipt_path=args.combined_handoff_receipt_json,
                output_labels_path=args.output_labels_csv,
                output_source_map_path=(args.source_map_csv or Path(str(args.output_labels_csv) + ".source.csv")),
                output_anchor_csr_path=args.anchor_csr_npz,
                output_anchor_projection_path=args.anchor_projection_csv,
                output_receipt_path=(args.receipt_json or Path(str(args.output_labels_csv) + ".receipt.json")),
                rejection_witness_path=args.rejection_witness_json,
                progress_path=Path(str(args.receipt_json or Path(str(args.output_labels_csv) + ".receipt.json")) + ".progress.json"),
                projection_helper_path=args.projection_helper_path,
                projection_helper_sha256=args.projection_helper_sha256,
                max_distance_m=args.reference_registration_max_distance_m,
                ambiguity_tolerance_m=args.reference_registration_ambiguity_tolerance_m,
                prefix_count=args.diagnostic_child_prefix_count,
                collect_rejections=args.collect_rejection_census)
        except Exception as exc:
            p1_anchor.write_preflight_rejection(
                args.rejection_witness_json, str(exc),
                {"parent_mass": args.parent_mass_csv,
                 "parent_full86_weights": args.parent_full86_weights_csv,
                 "child_node": args.child_node, "child_elements": args.child_ele,
                 "skin_p1_support": args.p1_skin_support_npz,
                 "skin_p1_support_receipt": args.p1_skin_support_receipt_json,
                 "fat_p1_support": args.p1_fat_support_npz,
                 "fat_p1_support_receipt": args.p1_fat_support_receipt_json,
                 "combined_element_map": args.combined_element_map_csv_gz,
                 "combined_node_map": args.combined_node_map_csv_gz,
                 "combined_handoff_receipt": args.combined_handoff_receipt_json,
                 "projection_helper": args.projection_helper_path},
                args.reference_registration_max_distance_m,
                args.reference_registration_ambiguity_tolerance_m)
            parser.exit(3, f"child_semantic_labels=REJECTED reason={exc}\n")
        print(json.dumps({"status": result["status"],
                          "exported_child_rows": result["counts"]["exported_child_rows"],
                          "child_tetrahedra": result["counts"]["child_tetrahedra"],
                          "p1_pool_rows": result["counts"]["p1_pool_rows"],
                          "reference_anchor_tie_rows": result["counts"]["reference_anchor_tie_rows"],
                          "labels": result["outputs"]["labels"],
                          "parent_row_map": result["outputs"]["parent_row_map"],
                          "p1_anchor_csr": result["outputs"]["p1_anchor_csr"],
                          "p1_anchor_projection_csv": result["outputs"]["p1_anchor_projection_csv"]}, separators=(",", ":")))
        return 0
    try:
        result = export(args.parent_mass_csv, args.child_node, args.child_ele,
                        args.output_labels_csv, args.source_map_csv, args.receipt_json,
                        args.reference_registration_max_distance_m,
                        args.reference_registration_ambiguity_tolerance_m,
                        args.same_material_current_reference_registration)
    except Exception as exc:
        if args.same_material_current_reference_registration and args.rejection_witness_json is not None:
            try:
                write_rejection_witness(
                    args.rejection_witness_json, exc,
                    parent_base=args.parent_mass_csv, child_node=args.child_node,
                    child_ele=args.child_ele,
                    maximum_distance_m=args.reference_registration_max_distance_m,
                    ambiguity_tolerance_m=args.reference_registration_ambiguity_tolerance_m)
            except Exception as witness_error:
                parser.exit(3, f"child_semantic_labels=REJECTED reason={exc}; rejection_witness_write_failed={witness_error}\n")
        parser.exit(3, f"child_semantic_labels=REJECTED reason={exc}\n")
    print(json.dumps({
        "status": result["status"],
        "child_tetrahedra": result["counts"]["child_tetrahedra"],
        "exact_geometry_matches": result["counts"]["exact_geometry_matches"],
        "centroid_containment_matches": result["counts"]["centroid_containment_matches"],
        "bounded_reference_registration_matches": result["counts"]["bounded_reference_registration_matches"],
        "same_material_current_reference_registration_matches": result["counts"]["same_material_current_reference_registration_matches"],
        "boundary_ties": result["counts"]["boundary_ties"],
        "labels": result["outputs"]["labels"],
        "parent_row_map": result["outputs"]["parent_row_map"],
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
