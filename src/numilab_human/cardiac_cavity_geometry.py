"""Extract pinned anatomical cavity surfaces without evolving physical state.

Coincident OBJ seam vertices may be identified only when their authored decimal
coordinates are exactly equal. This records a quotient of source topology; it
never fills openings, moves vertices, or assigns blood volume or mechanical mass.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

from .model import ImportError as HumanImportError
from .physiology import canonical, load_anatomy, read_json

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.cardiac-cavity-surfaces.v1"
ARCHIVE = "partof_BP3D_4.0_obj_99.zip"
ARCHIVE_SHA256 = "9fbc713fffeee924a5a657d9813d84d7eb957bded63adb854931dd5e3eb61c97"
ARCHIVE_BYTES = 64888505
TABLE_HASHES = {
    "partof_element_parts.txt": "3f5f6df1028eb122b30de77c711597b6bb8e5541658e5985859fd228adbf88ea",
    "isa_element_parts.txt": "a3de74423f943b0d724ae8f59b3a817f87c423a544f8db98113b1980817cbeaf",
}
CAVITIES = (
    ("right_atrium", "FMA11359", "cavity of right atrium", "FJ2424", "8bd340059ff697cfe9ce57f2427e8c5a982e248a1d2230d477a4fe094adc2a2a"),
    ("right_ventricle", "FMA9291", "cavity of right ventricle", "FJ2423", "390747e238ba9239961d02d1e02516d7af7aacc86d2bc0c29fc835381032ceeb"),
    ("left_atrium", "FMA9465", "cavity of left atrium", "FJ2425", "223233fef054fb67b81d27881f10a1acce47c9ade147546b6859fa90e4e37dc3"),
    ("left_ventricle", "FMA9466", "cavity of left ventricle", "FJ2422", "88654bd73fa155daf0b44fd67bccaec81c9e37c105e046076993f2684c463d95"),
)
_DECIMAL = re.compile(r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
_INDEX = re.compile(r"^-?[1-9][0-9]*$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("cardiac cavity geometry: " + message)


def parse_obj(data: bytes, source_name: str = "surface.obj") -> dict:
    """Parse the source's triangular v/vn/f subset; reject silent triangulation.

    Exact rational coordinate keys avoid equating distinct decimal coordinates
    merely because they round to the same binary floating-point number.
    """
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise HumanImportError("cardiac cavity geometry: invalid OBJ encoding") from error
    vertices, keys, normals, triangles, comments = [], [], [], [], []
    for line_number, raw in enumerate(lines, 1):
        if raw.lstrip().startswith("#"):
            comments.append(raw)
        fields = raw.split("#", 1)[0].split()
        if not fields:
            continue
        kind = fields[0]
        context = f"{source_name}:{line_number}"
        if kind in {"v", "vn"}:
            require(len(fields) == 4 and all(_DECIMAL.fullmatch(x) and len(x) <= 128 for x in fields[1:]), f"invalid {kind} row at {context}")
            require(all(abs(int(x.lower().split("e")[1])) <= 1000 if "e" in x.lower() else True for x in fields[1:]), f"unbounded decimal exponent at {context}")
            numbers = [float(x) for x in fields[1:]]
            require(all(math.isfinite(x) for x in numbers), f"nonfinite coordinate at {context}")
            if kind == "v":
                rational = tuple(Fraction(x) for x in fields[1:])
                require(all(x == 0 or value != 0 for x, value in zip(rational, numbers)), f"coordinate underflow at {context}")
                vertices.append(numbers)
                keys.append(rational)
            else:
                require(any(numbers), f"zero normal at {context}")
                normals.append(numbers)
        elif kind == "f":
            require(len(fields) == 4, f"face must be an authored triangle at {context}")
            face = []
            for token in fields[1:]:
                parts = token.split("/")
                require(len(parts) == 1 or (len(parts) == 3 and parts[1] == ""), f"unsupported face reference at {context}")
                require(bool(_INDEX.fullmatch(parts[0])), f"invalid vertex index at {context}")
                index = int(parts[0]); index = index - 1 if index > 0 else len(vertices) + index
                require(0 <= index < len(vertices), f"vertex index outside current source table at {context}")
                if len(parts) == 3:
                    require(bool(_INDEX.fullmatch(parts[2])), f"invalid normal index at {context}")
                    normal = int(parts[2]); normal = normal - 1 if normal > 0 else len(normals) + normal
                    require(0 <= normal < len(normals), f"normal index outside current source table at {context}")
                face.append(index)
            require(len(set(face)) == 3, f"repeated triangle vertex at {context}")
            triangles.append(face)
        elif kind in {"g", "usemtl"}:
            require(len(fields) == 2, f"invalid metadata row at {context}")
        else:
            require(False, f"unsupported OBJ directive {kind} at {context}")
    require(bool(vertices) and bool(triangles), "empty source surface")
    require(len({tuple(sorted(t)) for t in triangles}) == len(triangles), "duplicate source triangle")
    used = {i for face in triangles for i in face}
    require(used == set(range(len(vertices))), "unused source vertices")
    # Distinct authored decimals must remain distinct in the published FP64 geometry.
    binary_keys, metre_keys = {}, {}
    for vertex, key in zip(vertices, keys):
        previous = binary_keys.setdefault(tuple(vertex), key)
        require(previous == key, "distinct source coordinates collapse in Float64")
        metres = tuple(x * 0.001 for x in vertex)
        require(all(x == 0 or y != 0 for x, y in zip(vertex, metres)), "metre conversion underflows")
        require(metre_keys.setdefault(metres, key) == key, "distinct source coordinates collapse during metre conversion")
    return {"vertices_mm": vertices, "triangles": triangles, "coordinate_keys": keys, "comments": comments}


def analyze_topology(vertices: list, triangles: list) -> dict:
    """Report edge, orientation, face-component and vertex-link defects."""
    require(bool(vertices) and bool(triangles), "empty topology")
    require(all(isinstance(v, (list, tuple)) and len(v) == 3 and all(type(x) in (float, int) and math.isfinite(x) for x in v) for v in vertices), "invalid topology vertices")
    edges, incident = defaultdict(list), defaultdict(list)
    degenerate, repeated, duplicates = [], [], []
    face_keys = set()
    for i, face in enumerate(triangles):
        require(isinstance(face, (list, tuple)) and len(face) == 3 and all(type(x) is int and 0 <= x < len(vertices) for x in face), "invalid topology triangle")
        a, b, c = face
        if len(set(face)) != 3:
            repeated.append(i)
        key = tuple(sorted(face))
        if key in face_keys:
            duplicates.append(i)
        face_keys.add(key)
        u = [vertices[b][k] - vertices[a][k] for k in range(3)]
        v = [vertices[c][k] - vertices[a][k] for k in range(3)]
        cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        require(all(math.isfinite(x) for x in cross), "nonfinite triangle arithmetic")
        if not any(cross):
            degenerate.append(i)
        for x, y in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted((x, y)))].append((x, y, i))
        for vertex in set(face):
            incident[vertex].append(i)
    boundary = [rows[0] for _, rows in sorted(edges.items()) if len(rows) == 1]
    nonmanifold = [list(e) for e, rows in sorted(edges.items()) if len(rows) > 2]
    orientation = [list(e) for e, rows in sorted(edges.items()) if len(rows) == 2 and rows[0][:2] == rows[1][:2]]
    adjacency = defaultdict(set)
    for rows in edges.values():
        for a in rows:
            adjacency[a[2]].update(b[2] for b in rows if b[2] != a[2])
    unseen = set(range(len(triangles))); components = []
    while unseen:
        todo = [min(unseen)]; component = []
        while todo:
            i = todo.pop()
            if i not in unseen:
                continue
            unseen.remove(i); component.append(i); todo.extend(adjacency[i] & unseen)
        components.append(sorted(component))
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for a, b, _ in boundary:
        outgoing[a].append(b); incoming[b].append(a)
    boundary_vertices = set(incoming) | set(outgoing)
    loop_defects = sorted(i for i in boundary_vertices if len(incoming[i]) != 1 or len(outgoing[i]) != 1)
    loops = []
    if not loop_defects:
        unseen = set(boundary_vertices)
        while unseen:
            start = min(unseen); loop = []; i = start
            while i in unseen:
                unseen.remove(i); loop.append(i); i = outgoing[i][0]
            require(i == start, "boundary traversal failed")
            loops.append(loop)
    vertex_defects = []
    for vertex, faces in sorted(incident.items()):
        link = defaultdict(set); link_edges = Counter()
        for f in faces:
            others = [v for v in triangles[f] if v != vertex]
            if len(others) != 2:
                continue
            a, b = others; link[a].add(b); link[b].add(a); link_edges[tuple(sorted((a, b)))] += 1
        reached = set(); todo = [next(iter(link))] if link else []
        while todo:
            i = todo.pop()
            if i in reached:
                continue
            reached.add(i); todo.extend(link[i] - reached)
        degrees = sorted(len(neighbors) for neighbors in link.values())
        expected = ([1, 1] + [2] * (len(degrees) - 2)) if vertex in boundary_vertices else [2] * len(degrees)
        if not degrees or degrees != expected or len(reached) != len(link) or any(n != 1 for n in link_edges.values()):
            vertex_defects.append(vertex)
    unused = sorted(set(range(len(vertices))) - set(incident))
    defects = bool(nonmanifold or orientation or degenerate or repeated or duplicates or vertex_defects or loop_defects or unused)
    return {"vertex_count": len(vertices), "face_count": len(triangles), "edge_count": len(edges),
            "unused_vertex_ids": unused,
            "boundary_edges": [[a, b] for a, b, _ in boundary], "boundary_edge_count": len(boundary),
            "boundary_loops": loops, "boundary_loop_count": len(loops), "boundary_branch_vertex_ids": loop_defects,
            "nonmanifold_edges": nonmanifold, "orientation_defect_edges": orientation,
            "degenerate_face_ids": degenerate, "repeated_vertex_face_ids": repeated, "duplicate_face_ids": duplicates,
            "vertex_manifold_defect_ids": vertex_defects, "face_components": components,
            "face_component_count": len(components), "euler_characteristic": len(incident) - len(edges) + len(triangles),
            "closed_oriented_manifold_candidate": not boundary and not defects,
            "self_intersection_status": "not_checked", "interdomain_overlap_status": "not_checked"}


def exact_coordinate_quotient(parsed: dict) -> dict:
    unique, source_map, vertices = {}, [], []
    for coordinates, key in zip(parsed["vertices_mm"], parsed["coordinate_keys"]):
        if key not in unique:
            unique[key] = len(vertices); vertices.append(coordinates)
        source_map.append(unique[key])
    triangles = [[source_map[i] for i in face] for face in parsed["triangles"]]
    return {"method": "exact_authored_decimal_coordinate_identification_no_tolerance",
            "source_vertex_to_vertex": source_map, "identified_vertex_count": len(source_map) - len(vertices),
            "vertices_m": [[x * 0.001 for x in vertex] for vertex in vertices], "triangles": triangles,
            "topology": analyze_topology(vertices, triangles)}


def extract_cavity_surfaces(*, sources: Path = ROOT / "Sources", source_lock: Path = ROOT / "sources.lock.json") -> dict:
    """Return four exact cavity surfaces with source and quotient topology."""
    try:
        lock = read_json(source_lock)
        metadata = lock["sources"]["bodyparts3d_4"]
        require(metadata["version"] == "4.0" and metadata["license"] == "CC-BY-4.0", "unexpected source identity")
        for name, expected in TABLE_HASHES.items():
            require(metadata["files"][name]["sha256"] == expected, "unrecognized anatomy table identity")
        anatomy = load_anatomy(sources, source_lock)
        require(metadata["files"][ARCHIVE]["sha256"] == ARCHIVE_SHA256 and metadata["files"][ARCHIVE]["bytes"] == ARCHIVE_BYTES, "unrecognized archive lock")
        archive_path = sources / ARCHIVE
        require(archive_path.is_file() and not archive_path.is_symlink(), "missing or redirected source archive")
        require(archive_path.stat().st_size == ARCHIVE_BYTES, "archive byte count differs")
        hasher = hashlib.sha256()
        with archive_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                hasher.update(chunk)
        require(hasher.hexdigest() == ARCHIVE_SHA256, "archive SHA256 differs")
        chambers = []
        with zipfile.ZipFile(archive_path) as archive:
            for source_id, concept, name, member_id, expected in CAVITIES:
                for table in anatomy["tables"].values():
                    require(table.get((concept, name)) == {member_id}, "cavity concept membership differs")
                member = f"partof_BP3D_4.0_obj_99/{member_id}.obj"
                require(sum(info.filename == member for info in archive.infolist()) == 1, "missing or duplicate archive member")
                data = archive.read(member)
                require(hashlib.sha256(data).hexdigest() == expected, "source member SHA256 differs")
                parsed = parse_obj(data, member)
                require(any(line.startswith("# Bounds(mm):") for line in parsed["comments"]), "missing source millimetre convention")
                raw = analyze_topology(parsed["vertices_mm"], parsed["triangles"])
                quotient = exact_coordinate_quotient(parsed)
                require(quotient["topology"]["closed_oriented_manifold_candidate"], "cavity quotient has source topology defects")
                chambers.append({"source_id": source_id, "semantic_id": "FMA:" + concept[3:], "concept_id": concept,
                                 "source_name": name, "member_id": member_id,
                                 "source": {"archive": ARCHIVE, "member": member, "sha256": expected, "bytes": len(data),
                                            "header_comments": parsed["comments"]},
                                 "vertices_m": [[x * 0.001 for x in v] for v in parsed["vertices_mm"]],
                                 "triangles": parsed["triangles"], "raw_topology": raw,
                                 "exact_coordinate_quotient": quotient,
                                 "physical_volume_m3": None, "mechanical_mass_kg": None})
        return {"schema": SCHEMA, "anatomy_source": anatomy["source"],
                "archive": {"file": ARCHIVE, "sha256": ARCHIVE_SHA256, "bytes": ARCHIVE_BYTES},
                "coordinate_frame": "BodyParts3D 4.0 source frame", "source_units": "mm", "output_units": "m",
                "length_scale": 0.001, "source_coordinate_edits": False, "added_faces": 0,
                "license_provenance": {"current_database_license": "CC-BY-4.0", "url": metadata["license_url"],
                                       "source_header_license": "CC-BY-SA-2.1-Japan", "header_preserved": True},
                "chambers": chambers,
                "physical_volume_status": "not_admitted_requires_embeddedness_interdomain_and_model_registration",
                "mechanical_mass_status": "unassigned_requires_nonduplicated_blood_tissue_partition",
                "boundary": "Exact source cavity surfaces and coincident-seam quotient only; no self-intersection, interdomain disjointness, CVSim calibration, mechanical mass, or physical evolution qualification."}
    except HumanImportError:
        raise
    except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as error:
        raise HumanImportError(f"cardiac cavity geometry: invalid pinned source: {error}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=ROOT / "Sources")
    parser.add_argument("--source-lock", type=Path, default=ROOT / "sources.lock.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = extract_cavity_surfaces(sources=args.sources, source_lock=args.source_lock)
        encoded = canonical(result) + b"\n"
        require(not args.output.is_symlink(), "redirected output")
        require(not args.output.exists() or args.output.read_bytes() == encoded, "output is immutable")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not args.output.exists():
            with args.output.open("xb") as stream:
                stream.write(encoded)
        print(json.dumps({"status": "extracted", "output": str(args.output), "sha256": hashlib.sha256(encoded).hexdigest()}))
        return 0
    except (HumanImportError, OSError) as error:
        print(json.dumps({"status": "rejected", "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
