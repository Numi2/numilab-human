"""Import a pinned anatomical tetrahedral heart without stepping physics.

The output is an offline source asset, not a Matter package. Source cap cells,
regional labels, per-element fibres/sheets, and ventricular coordinates survive
the import. No registration, material fit, unloaded reference, or blood mass is
invented. The only geometry changes are mm -> m and a recorded tetrahedron
vertex permutation for positive orientation.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tarfile

from .model import ImportError as HumanImportError

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/cardiac-wall-rodero18.v1.json"
ARCHIVE_BYTES = 53661258
ARCHIVE_SHA256 = "b50919a711dc3914cc0a4dd9cabc19679a9f36be9ac6eb26a23ca6799f501720"
MEMBER_BYTES = 154064076
MEMBER_SHA256 = "f0d4f3fc21ea229fa1b334888d374ee606a2915b414fff047764aad44f1c0a41"
CONFIG_SHA256 = "23c931fef53edced85e0e0a36c73d8490dddb87db6bd988482a2cdfc5a1442cc"
SCHEMA = "HumanPack.cardiac-wall-source-asset.v1"
FACE_NODES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
CHAMBERS = {1: "left_ventricle", 2: "right_ventricle", 3: "left_atrium", 4: "right_atrium"}
CAP_LABELS = frozenset(range(7, 18))


def require(value: bool, message: str) -> None:
    if not value:
        raise HumanImportError("cardiac wall source: " + message)


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1 << 20):
            h.update(data)
    return h.hexdigest()


@dataclass
class Mesh:
    points: array                 # Float64 metres, source node order
    cells: array                  # uint32, source cell order
    labels: array                 # uint32, source regional IDs
    fibres: array                 # Float64, unchanged source components
    sheets: array
    uvc: dict[str, array]
    member_sha256: str

    @property
    def node_count(self) -> int:
        return len(self.points) // 3

    @property
    def cell_count(self) -> int:
        return len(self.cells) // 4


class Rows:
    """Strict parser for this source's line-oriented ASCII VTK subset."""

    def __init__(self, stream: io.BufferedIOBase):
        self.stream = stream
        self.hasher = hashlib.sha256()
        self.line = 0

    def next(self) -> list[str]:
        while raw := self.stream.readline():
            self.hasher.update(raw)
            self.line += 1
            require(len(raw) <= 4096, f"oversized VTK row {self.line}")
            try:
                fields = raw.decode("ascii").split()
            except UnicodeError as error:
                raise HumanImportError("cardiac wall source: non-ASCII VTK") from error
            if fields:
                return fields
        return []

    def expect(self, fields: list[str]) -> None:
        require(self.next() == fields, f"expected {' '.join(fields)} at VTK row {self.line}")

    def numbers(self, count: int, width: int, integer: bool = False) -> array:
        output = array("I" if integer else "d")
        for _ in range(count):
            row = self.next()
            require(len(row) == width, f"wrong numeric row width at {self.line}")
            try:
                values = [int(x) for x in row] if integer else [float(x) for x in row]
            except ValueError as error:
                raise HumanImportError(f"cardiac wall source: invalid number at {self.line}") from error
            require(all(0 <= x <= 0xffffffff for x in values) if integer
                    else all(math.isfinite(x) for x in values), f"invalid numeric range at {self.line}")
            output.extend(values)
        return output


def parse_vtk(stream: io.BufferedIOBase) -> Mesh:
    rows = Rows(stream)
    rows.expect(["#", "vtk", "DataFile", "Version", "3.0"])
    require(bool(rows.next()), "missing VTK title")
    rows.expect(["ASCII"])
    rows.expect(["DATASET", "UNSTRUCTURED_GRID"])
    header = rows.next()
    require(len(header) == 3 and header[0] == "POINTS" and header[2] == "float", "unsupported points header")
    try:
        node_count = int(header[1])
    except ValueError as error:
        raise HumanImportError("cardiac wall source: invalid point count") from error
    require(4 <= node_count <= 1_000_000, "point count outside source importer bound")
    points = rows.numbers(node_count, 3)
    for i, value in enumerate(points):
        metres = value * 0.001
        require(math.isfinite(metres) and (value == 0 or metres != 0), "metre conversion range")
        points[i] = metres
    header = rows.next()
    require(len(header) == 2 and header[0] == "CELL_TYPES", "missing cell types")
    try:
        cell_count = int(header[1])
    except ValueError as error:
        raise HumanImportError("cardiac wall source: invalid cell count") from error
    require(1 <= cell_count <= 5_000_000, "cell count outside source importer bound")
    types = rows.numbers(cell_count, 1, True)
    require(all(x == 10 for x in types), "only authored tetrahedra are supported")
    rows.expect(["CELLS", str(cell_count), str(cell_count * 5)])
    cells = array("I")
    for _ in range(cell_count):
        values = rows.numbers(1, 5, True)
        require(values[0] == 4 and max(values[1:]) < node_count and len(set(values[1:])) == 4,
                f"invalid tetrahedron at VTK row {rows.line}")
        cells.extend(values[1:])
    rows.expect(["CELL_DATA", str(cell_count)])
    rows.expect(["SCALARS", "ID", "int", "1"])
    rows.expect(["LOOKUP_TABLE", "default"])
    labels = rows.numbers(cell_count, 1, True)
    require(all(1 <= x <= 24 for x in labels), "unknown anatomical label")
    rows.expect(["VECTORS", "fibres", "float"])
    fibres = rows.numbers(cell_count, 3)
    rows.expect(["VECTORS", "sheets", "float"])
    sheets = rows.numbers(cell_count, 3)
    rows.expect(["POINT_DATA", str(node_count)])
    uvc = {}
    for name in ("RHO.dat", "PHI.dat", "Z.dat", "V.dat"):
        rows.expect(["SCALARS", name, "float", "1"])
        rows.expect(["LOOKUP_TABLE", "default"])
        uvc[name] = rows.numbers(node_count, 1)
    require(not rows.next(), "unexpected trailing VTK content")
    return Mesh(points, cells, labels, fibres, sheets, uvc, rows.hasher.hexdigest())


def integer_points(points: array) -> tuple[list[tuple[int, int, int]], int]:
    """Exact integer representation of the emitted binary64 SI coordinates."""
    denominator = max(x.as_integer_ratio()[1] for x in points)
    output = []
    for offset in range(0, len(points), 3):
        output.append(tuple(n * (denominator // d) for n, d in
                            (x.as_integer_ratio() for x in points[offset:offset + 3])))
    return output, denominator


def determinant(a: tuple, b: tuple, c: tuple, d: tuple) -> int:
    u = tuple(b[i] - a[i] for i in range(3))
    v = tuple(c[i] - a[i] for i in range(3))
    w = tuple(d[i] - a[i] for i in range(3))
    return (u[0] * (v[1] * w[2] - v[2] * w[1])
            - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0]))


def face_at(cells: array, encoded: int) -> tuple[int, int, int]:
    offset, face = (encoded >> 2) * 4, encoded & 3
    return tuple(cells[offset + local] for local in FACE_NODES[face])


def cyclic(face: tuple) -> tuple:
    position = face.index(min(face))
    return face[position:] + face[:position]


def prepare_topology(mesh: Mesh) -> tuple[dict, array, array, array, array]:
    """Exact orientation and face incidence; no meshing or coordinate repair.

    Sorting packed face records bounds memory without a multi-million-entry
    dictionary. The source cell index survives sorting and orientation changes.
    A positive determinant and manifold incidence do not prove global embedding.
    """
    require(array("I").itemsize == 4 and array("d").itemsize == 8, "unsupported host array widths")
    points, denominator = integer_points(mesh.points)
    require(len(set(points)) == mesh.node_count, "duplicate SI node coordinates")
    used = bytearray(mesh.node_count)
    cells = array("I", mesh.cells)
    reversed_ids = array("I")
    volumes = Counter()
    volumes_exact = Counter()
    minimum, maximum = None, 0
    node_bits = mesh.node_count.bit_length()
    record_bits = (mesh.cell_count * 4).bit_length()
    mask = (1 << record_bits) - 1
    records = []
    cell_keys = set()
    for cell in range(mesh.cell_count):
        offset = cell * 4
        indices = tuple(cells[offset:offset + 4])
        key = tuple(sorted(indices))
        require(key not in cell_keys, f"duplicate tetrahedron {cell}")
        cell_keys.add(key)
        det = determinant(*(points[i] for i in indices))
        require(det != 0, f"degenerate tetrahedron {cell}")
        if det < 0:
            cells[offset], cells[offset + 1] = cells[offset + 1], cells[offset]
            reversed_ids.append(cell)
            det = -det
        minimum = det if minimum is None else min(minimum, det)
        maximum = max(maximum, det)
        volumes_exact[int(mesh.labels[cell])] += det
        for i in indices:
            used[i] = 1
        for face in range(4):
            encoded = cell * 4 + face
            a, b, c = sorted(face_at(cells, encoded))
            key = (a << (2 * node_bits)) | (b << node_bits) | c
            records.append((key << record_bits) | encoded)
    require(all(used), "unused source nodes")
    del cell_keys, used
    records.sort()
    boundary, owners = array("I"), array("I")
    interior_count = 0
    i = 0
    while i < len(records):
        key, encoded = records[i] >> record_bits, records[i] & mask
        j = i + 1
        while j < len(records) and records[j] >> record_bits == key:
            j += 1
        require(j - i <= 2, f"nonmanifold tetrahedral face at source cell {encoded >> 2}")
        if j - i == 2:
            other = records[i + 1] & mask
            require(cyclic(face_at(cells, encoded)) != cyclic(face_at(cells, other)),
                    f"same-side adjacent tetrahedra {encoded >> 2}/{other >> 2}")
            interior_count += 1
        else:
            boundary.extend(face_at(cells, encoded))
            owners.append(encoded >> 2)
        i = j
    divisor = 6 * denominator ** 3
    for label, value in sorted(volumes_exact.items()):
        volumes[str(label)] = value / divisor
    report = {
        "predicate": "exact integer determinant of emitted binary64 metre coordinates",
        "positive_oriented_tetrahedra": mesh.cell_count,
        "source_negative_orientation_count": len(reversed_ids),
        "source_vertex_permutation": "swap local vertices 0 and 1 only for recorded negative source cells",
        "minimum_tetrahedron_volume_m3": minimum / divisor,
        "maximum_tetrahedron_volume_m3": maximum / divisor,
        "regional_cell_counts": dict(sorted(Counter(str(x) for x in mesh.labels).items())),
        "regional_geometric_volume_m3": dict(volumes),
        "boundary_face_count": len(owners), "interior_face_count": interior_count,
        "global_tetrahedral_embedding": "not_checked",
        "source_surface_edits": False,
        "blood_volume_or_mass_assigned": False,
    }
    return report, cells, reversed_ids, boundary, owners


def audit_frames(mesh: Mesh) -> dict:
    """Retain unnormalized authored frames and expose their actual defects."""
    result = {}
    for label in sorted(set(mesh.labels)):
        result[str(label)] = {"count": 0, "zero_fibres": 0, "zero_sheets": 0,
                              "parallel_frames": 0, "max_fibre_norm_error": 0.0,
                              "max_sheet_norm_error": 0.0, "max_abs_normalized_dot": 0.0}
    for i, label in enumerate(mesh.labels):
        row = result[str(label)]
        f, s = mesh.fibres[3*i:3*i+3], mesh.sheets[3*i:3*i+3]
        fn, sn = math.sqrt(sum(x*x for x in f)), math.sqrt(sum(x*x for x in s))
        require(math.isfinite(fn) and math.isfinite(sn), f"frame norm range at cell {i}")
        row["count"] += 1
        row["zero_fibres"] += fn == 0
        row["zero_sheets"] += sn == 0
        row["max_fibre_norm_error"] = max(row["max_fibre_norm_error"], abs(fn - 1))
        row["max_sheet_norm_error"] = max(row["max_sheet_norm_error"], abs(sn - 1))
        if fn and sn:
            dot = sum(f[j] * s[j] for j in range(3)) / fn / sn
            row["max_abs_normalized_dot"] = max(row["max_abs_normalized_dot"], abs(dot))
            cross = [f[(j+1)%3]*s[(j+2)%3] - f[(j+2)%3]*s[(j+1)%3] for j in range(3)]
            row["parallel_frames"] += not any(cross)
    return {"coordinate_edits_or_orthonormalization": False, "regions": result,
            "native_element_frame_binding": "not_implemented",
            "fibre_origin": "source rule-based field, not measured myocardial microstructure"}


def audit_boundary(mesh: Mesh, boundary: array, owners: array) -> tuple[dict, array]:
    """Components of the extracted material boundary, including authored caps."""
    count = len(owners)
    require(count > 0, "empty material boundary")
    parent = list(range(count))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = defaultdict(list)
    faces = [tuple(boundary[3*i:3*i+3]) for i in range(count)]
    for i, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            edges[min(u, v), max(u, v)].append((i, u < v))
    defects = []
    for key, rows in edges.items():
        if len(rows) != 2 or rows[0][1] == rows[1][1]:
            defects.append(list(key))
        first = root(rows[0][0])
        for row in rows[1:]:
            parent[root(row[0])] = first
    groups = defaultdict(list)
    for i in range(count):
        groups[root(i)].append(i)
    points, denominator = integer_points(mesh.points)
    components, face_components = [], array("I", [0]) * count
    for component, ids in enumerate(sorted(groups.values(), key=lambda ids: min(ids))):
        for i in ids:
            face_components[i] = component
        nodes = {v for i in ids for v in faces[i]}
        # Common exact origin makes the result independent of global translation.
        origin = points[min(nodes)]
        volume = sum(determinant(origin, *(points[v] for v in faces[i])) for i in ids)
        labels = Counter(str(mesh.labels[owners[i]]) for i in ids)
        chamber_labels = set(int(x) for x in labels) & CHAMBERS.keys()
        cap_count = sum(labels.get(str(label), 0) for label in CAP_LABELS)
        component_defects = [key for key in defects if key[0] in nodes and key[1] in nodes]
        components.append({
            "id": component, "face_count": len(ids), "node_count": len(nodes),
            "owner_label_face_counts": dict(sorted(labels.items())),
            "signed_material_outward_volume_m3": volume / (6 * denominator**3),
            "candidate_chamber": CHAMBERS[next(iter(chamber_labels))]
                                 if len(chamber_labels) == 1 and volume < 0 else None,
            "numerical_closure_face_count": cap_count,
            "edge_manifold_orientation_defect_count": len(component_defects),
            "native_cavity_admitted": False,
        })
    return {"orientation": "outward from tetrahedral material",
            "edge_manifold_orientation_defect_count": len(defects),
            "edge_defect_witnesses": defects[:16], "components": components,
            "vertex_manifold_and_embeddedness": "not_checked",
            "native_pressure_boundary": "not_admitted: numerical caps require explicit port ownership; embedding and source mechanics remain unqualified"}, face_components


def array_bytes(values: array) -> bytes:
    copy = array(values.typecode, values)
    if sys.byteorder != "little":
        copy.byteswap()
    return copy.tobytes()


def write_immutable(path: Path, data: bytes) -> None:
    require(not path.is_symlink(), f"symlink output {path}")
    if path.exists():
        require(path.is_file() and path.read_bytes() == data, f"refusing to replace changed output {path}")
        return
    with path.open("xb") as stream:
        stream.write(data)


def load_config(path: Path) -> tuple[dict, str]:
    # Pin the entire semantic/units/license/calibration contract, and hash the
    # same byte snapshot we parse. A late file edit cannot rebind provenance.
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    require(sha256 == CONFIG_SHA256, "source configuration hash mismatch")
    return json.loads(data), sha256


def import_source(archive: Path, output: Path, *, config: Path = CONFIG) -> dict:
    require(archive.is_file() and archive.stat().st_size == ARCHIVE_BYTES, "source archive byte count mismatch")
    require(digest(archive) == ARCHIVE_SHA256, "source archive hash mismatch")
    configuration, config_sha256 = load_config(config)
    source = configuration["source"]
    require(source["archive"]["sha256"] == ARCHIVE_SHA256
            and source["member"]["sha256"] == MEMBER_SHA256
            and source["original_length_unit"] == "mm", "source configuration identity mismatch")
    with tarfile.open(archive, "r:gz") as container:
        members = container.getmembers()
        require(len(members) == 1 and members[0].name == "18.vtk" and members[0].isfile()
                and members[0].size == MEMBER_BYTES, "unexpected archive member")
        with container.extractfile(members[0]) as stream:
            mesh = parse_vtk(stream)
    require(mesh.member_sha256 == MEMBER_SHA256, "VTK member hash mismatch")
    require(mesh.node_count == 300965 and mesh.cell_count == 1470083, "pinned mesh counts mismatch")
    topology, cells, reversed_ids, boundary, owners = prepare_topology(mesh)
    boundary_audit, component_ids = audit_boundary(mesh, boundary, owners)
    frame_audit = audit_frames(mesh)
    buffers = {
        "nodes.f64le": (mesh.points, [mesh.node_count, 3], "metres; original source node order"),
        "tetrahedra.u32le": (cells, [mesh.cell_count, 4], "positive orientation; original source cell order"),
        "source_reversed_cells.u32le": (reversed_ids, [len(reversed_ids)], "swap local 0/1 to recover source connectivity"),
        "labels.u32le": (mesh.labels, [mesh.cell_count], "unmodified regional source IDs"),
        "fibres.f64le": (mesh.fibres, [mesh.cell_count, 3], "unmodified dimensionless source vectors"),
        "sheets.f64le": (mesh.sheets, [mesh.cell_count, 3], "unmodified dimensionless source vectors"),
        "boundary.u32le": (boundary, [len(owners), 3], "outward material boundary; all source cap faces retained"),
        "boundary_owners.u32le": (owners, [len(owners)], "original source tetrahedron index"),
        "boundary_components.u32le": (component_ids, [len(owners)], "deterministic boundary component ID"),
    }
    for name, values in mesh.uvc.items():
        buffers[f"uvc_{name[:-4].lower()}.f64le"] = (values, [mesh.node_count], "unmodified source coordinate including -10 sentinel")
    require(not output.is_symlink(), "symlink output directory")
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, (values, shape, role) in buffers.items():
        data = array_bytes(values)
        records[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                         "shape": shape, "role": role}
        write_immutable(output / name, data)
    result = {"schema": SCHEMA, "source_config": configuration,
              "source_config_sha256": config_sha256, "buffers": records,
              "topology": topology, "boundary": boundary_audit, "frames": frame_audit,
              "qualification": {"status": "source_imported_runtime_unqualified", "physical_steps": 0,
                                "native_matter_package": False, "material_fit": False,
                                "body_registration": False, "mechanical_blood_mass": False,
                                "anatomical_wall_simulation": False}}
    write_immutable(output / "manifest.json", canonical(result))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="Pinned Rodero CT case18 archive")
    parser.add_argument("--output", required=True, type=Path, help="Immutable offline source asset directory")
    arguments = parser.parse_args(argv)
    try:
        result = import_source(arguments.archive, arguments.output)
        print(json.dumps({"status": result["qualification"]["status"], "output": str(arguments.output),
                          "nodes": result["buffers"]["nodes.f64le"]["shape"][0],
                          "tetrahedra": result["topology"]["positive_oriented_tetrahedra"],
                          "boundary_components": len(result["boundary"]["components"])}))
        return 0
    except (HumanImportError, OSError, ValueError, KeyError, tarfile.TarError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
