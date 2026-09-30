"""Audit exact source PCL/ACL contact geometry against the failed native step."""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

from numilab_human.open_knee import (
    EXPECTED_HASHES, FACE_STRUCT, HEADER_STRUCT, NODE_SET_STRUCT, NODE_STRUCT,
    REGION_STRUCT, SURFACE_PAIR_STRUCT, SURFACE_STRUCT, TETRAHEDRON_STRUCT,
)
from tools.audit_patellofemoral_tetrahedral_separation import (
    aabb_candidates, cross, difference, dot, exact_lattice,
    tetrahedron_relation,
)


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "Build/patellofemoral-surface-20260930/left/open-knee-oks003-left.nhknee"
MANIFEST = PAYLOAD.with_suffix(".manifest.json")
FEBIO = ROOT / "Sources/open-knee-oks003/FeBio_custom.feb"
EVIDENCE = ROOT / "Docs/media/patellar-motion-authority-20260930"
OUTPUT = ROOT / "Docs/media/cruciate-source-contact-20260930/receipt.json"
NAMES = ("PCL_@_ACL_ContactFaces", "ACL_@_PCL_ContactFaces")
SELECTED = ("QAT", "PCL", "ACL", "MCL", "PTL", "LCL")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("cruciate source contact: " + message)


def name(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("ascii")


def strictly_crosses(a: tuple[tuple[int, int, int], ...],
                     b: tuple[tuple[int, int, int], ...]) -> bool:
    """Exact integer edge-through-triangle predicate on retained Float32 nodes."""
    for edge_triangle, target in ((a, b), (b, a)):
        origin = target[0]
        normal = cross(difference(target[1], origin),
                       difference(target[2], origin))
        if normal == (0, 0, 0):
            continue
        for i in range(3):
            first = difference(edge_triangle[i], origin)
            second = difference(edge_triangle[(i + 1) % 3], origin)
            d0, d1 = dot(normal, first), dot(normal, second)
            if d0 * d1 >= 0:
                continue
            numerator = tuple(d1 * x - d0 * y
                              for x, y in zip(first, second, strict=True))
            denominator = d1 - d0
            sign = 1 if denominator > 0 else -1
            inside = True
            for edge in range(3):
                start = difference(target[edge], origin)
                direction = difference(target[(edge + 1) % 3], target[edge])
                relative = tuple(numerator[k] - denominator * start[k]
                                 for k in range(3))
                inside &= sign * dot(normal, cross(direction, relative)) > 0
            if inside:
                return True
    return False


def self_check() -> None:
    flat = ((0, 0, 0), (4, 0, 0), (0, 4, 0))
    crossing = ((1, 1, -2), (1, 1, 2), (2, 1, 2))
    separated = ((1, 1, 3), (1, 1, 4), (2, 1, 4))
    boundary_only = ((4, 4, -2), (4, 4, 2), (5, 4, 2))
    require(strictly_crosses(flat, crossing) and
            strictly_crosses(crossing, flat) and
            not strictly_crosses(flat, separated) and
            not strictly_crosses(flat, boundary_only),
            "exact triangle predicate self-check failed")


def decode():
    raw = PAYLOAD.read_bytes()
    manifest = json.loads(MANIFEST.read_text())
    require(sha(raw) == manifest["payload"]["sha256"],
            "payload does not match its pinned manifest")
    h = HEADER_STRUCT.unpack_from(raw)
    require(h[0] == b"NHKNEE1\0" and h[1] == 3 and h[11] == 0,
            "expected left NHKNEE1 v3 payload")
    offset = HEADER_STRUCT.size
    regions = []
    for i in range(h[3]):
        r = REGION_STRUCT.unpack_from(raw, offset + i * REGION_STRUCT.size)
        regions.append((name(r[0]), r[3], r[4]))
    offset += h[3] * REGION_STRUCT.size
    surfaces = {}
    ordered_surfaces = []
    for i in range(h[6]):
        row = SURFACE_STRUCT.unpack_from(raw, offset + i * SURFACE_STRUCT.size)
        surface_name = name(row[0])
        surfaces[surface_name] = (row[2], row[3])
        ordered_surfaces.append(surface_name)
    offset += h[6] * SURFACE_STRUCT.size + h[8] * NODE_SET_STRUCT.size
    surface_pairs = []
    for i in range(h[10]):
        row = SURFACE_PAIR_STRUCT.unpack_from(
            raw, offset + i * SURFACE_PAIR_STRUCT.size)
        surface_pairs.append((name(row[0]), ordered_surfaces[row[1]],
                              ordered_surfaces[row[2]]))
    offset += h[10] * SURFACE_PAIR_STRUCT.size
    nodes = np.ndarray((h[4], 3), dtype="<f4", buffer=raw,
                       offset=offset, strides=(NODE_STRUCT.size, 4))
    offset += h[4] * NODE_STRUCT.size + h[5] * TETRAHEDRON_STRUCT.size
    faces = np.frombuffer(raw, dtype="<u4", count=h[7] * 3,
                          offset=offset).reshape(-1, 3)
    return raw, regions, surfaces, surface_pairs, exact_lattice(nodes), faces


def run() -> dict:
    self_check()
    raw, regions, surfaces, surface_pairs, integer, faces = decode()
    febio = FEBIO.read_bytes()
    require(sha(febio) == EXPECTED_HASHES["FeBio_custom.feb"],
            "pinned FEBio file changed")
    root = ET.fromstring(febio)
    source_pair = root.find(".//contact[@surface_pair='PCL_To_ACL']")
    require(source_pair is not None and
            source_pair.get("type") == "sliding-elastic" and
            all(n in surfaces for n in NAMES),
            "source PCL/ACL contact pair changed")
    a, b = (faces[first:first + count] for first, count in
            (surfaces[n] for n in NAMES))
    require(len(a) == 4091 and len(b) == 11744,
            "source contact face counts changed")
    a_points, b_points = integer[a], integer[b]
    a_min, a_max = a_points.min(axis=1), a_points.max(axis=1)
    b_min, b_max = b_points.min(axis=1), b_points.max(axis=1)

    @lru_cache(maxsize=8192)
    def triangle(side: int, index: int):
        row = a_points[index] if side == 0 else b_points[index]
        return tuple(tuple(int(x) for x in vertex) for vertex in row)

    candidate_count = 0
    crossings = []
    first_crossings = []
    for i, j in aabb_candidates(a_min, a_max, b_min, b_max):
        candidate_count += 1
        if strictly_crosses(triangle(0, i), triangle(1, j)):
            crossings.append((i, j))
            if len(first_crossings) < 12:
                first_crossings.append({"pcl_face": int(i), "acl_face": int(j)})

    h = HEADER_STRUCT.unpack_from(raw)
    node_offset = (HEADER_STRUCT.size + h[3] * REGION_STRUCT.size +
                   h[6] * SURFACE_STRUCT.size +
                   h[8] * NODE_SET_STRUCT.size +
                   h[10] * SURFACE_PAIR_STRUCT.size)
    tet_offset = node_offset + h[4] * NODE_STRUCT.size
    tetrahedra = np.frombuffer(raw, dtype="<u4", count=h[5] * 4,
                               offset=tet_offset).reshape(-1, 4)
    pcl_nodes = {int(v) for i, _ in crossings for v in a[i]}
    acl_nodes = {int(v) for _, j in crossings for v in b[j]}
    anchor_counts = {}
    for tissue, nodes in (("PCL", pcl_nodes), ("ACL", acl_nodes)):
        anchor_counts[tissue] = sum(
            NODE_STRUCT.unpack_from(raw, node_offset + index * NODE_STRUCT.size)[-1] & 1
            for index in nodes)
    needed_faces = {tuple(sorted(int(v) for v in a[i])) for i, _ in crossings}
    needed_faces.update(tuple(sorted(int(v) for v in b[j]))
                        for _, j in crossings)
    adjacent_tetrahedra = {}
    for index, cell in enumerate(tetrahedra):
        for excluded in range(4):
            face = tuple(sorted(int(v) for corner, v in enumerate(cell)
                                if corner != excluded))
            if face in needed_faces:
                adjacent_tetrahedra.setdefault(face, []).append(index)
    require(len(adjacent_tetrahedra) == len(needed_faces) and
            all(len(indices) == 1 for indices in adjacent_tetrahedra.values()),
            "crossed source contact face does not bound one tetrahedron")

    @lru_cache(maxsize=256)
    def tetrahedron(index: int):
        return tuple(tuple(int(x) for x in integer[v])
                     for v in tetrahedra[index])

    solid_overlap_count = 0
    for i, j in crossings:
        pcl = adjacent_tetrahedra[tuple(sorted(int(v) for v in a[i]))][0]
        acl = adjacent_tetrahedra[tuple(sorted(int(v) for v in b[j]))][0]
        solid_overlap_count += tetrahedron_relation(
            tetrahedron(pcl), tetrahedron(acl)) == "interior_overlap"
    require(solid_overlap_count == len(crossings),
            "crossed source contact faces changed solid-domain relation")

    all_pair_rows = []
    for pair_name, first_name, second_name in surface_pairs:
        first_start, first_count = surfaces[first_name]
        second_start, second_count = surfaces[second_name]
        first_faces = faces[first_start:first_start + first_count]
        second_faces = faces[second_start:second_start + second_count]
        if pair_name == "PCL_To_ACL":
            pair_candidates, pair_crossings = candidate_count, len(crossings)
        else:
            first_points = integer[first_faces]
            second_points = integer[second_faces]
            pair_candidates = pair_crossings = 0

            @lru_cache(maxsize=8192)
            def first_triangle(index: int):
                return tuple(tuple(int(x) for x in vertex)
                             for vertex in first_points[index])

            for first_index, second_index in aabb_candidates(
                    first_points.min(axis=1), first_points.max(axis=1),
                    second_points.min(axis=1), second_points.max(axis=1)):
                pair_candidates += 1
                other = tuple(tuple(int(x) for x in vertex)
                              for vertex in second_points[second_index])
                pair_crossings += strictly_crosses(
                    first_triangle(first_index), other)
        all_pair_rows.append({
            "pair": pair_name,
            "first_surface": first_name,
            "second_surface": second_name,
            "first_faces": len(first_faces),
            "second_faces": len(second_faces),
            "aabb_candidate_face_pairs": pair_candidates,
            "exact_strict_crossing_face_pairs": pair_crossings,
        })

    selected = [r for r in regions if r[0] in SELECTED]
    require(tuple(r[0] for r in selected) == SELECTED,
            "native selected object order changed")
    logs = {}
    for variant in ("baseline", "unprescribed"):
        path = EVIDENCE / f"native-{variant}.log"
        text = path.read_text()
        matches = re.findall(
            r"contact_primitive side=(\d+) object=(\d+).*?"
            r"contact_vertex side=\1 corner=0 node=(\d+).*?"
            r"contact_vertex side=\1 corner=1 node=(\d+).*?"
            r"contact_vertex side=\1 corner=2 node=(\d+)",
            text, re.S)
        require(len(matches) == 2, f"{variant} native contact witness missing")
        witness = {}
        for _, object_id, *node_ids in matches:
            object_index = int(object_id)
            require(object_index in (1, 2), "native cruciate object changed")
            local = []
            for node in map(int, node_ids):
                for region_name, first, count in selected:
                    if node < count:
                        require(region_name == selected[object_index][0],
                                "native node has a different region owner")
                        local.append(first + node)
                        break
                    node -= count
                else:
                    raise RuntimeError("native node escapes selected regions")
            target = set(local)
            source_faces = a if object_index == 1 else b
            matching = np.flatnonzero(np.all(np.isin(source_faces, list(target)),
                                            axis=1))
            require(len(matching) == 1,
                    "native crossing face is outside source-authored contact")
            witness[selected[object_index][0]] = int(matching[0])
        require(strictly_crosses(triangle(0, witness["PCL"]),
                                 triangle(1, witness["ACL"])),
                f"{variant} native witness does not cross in source geometry")
        logs[variant] = {"log_sha256": sha(path.read_bytes()),
                         "source_contact_faces": witness,
                         "exact_source_triangle_crossing": True}
    return {
        "schema": "numi.human.cruciate-source-contact.v1",
        "status": "source_authored_pcl_acl_contact_initially_intersects",
        "payload_sha256": sha(raw),
        "febio_sha256": sha(febio),
        "source_contact_pair": "PCL_To_ACL",
        "source_contact_mode": "sliding-elastic",
        "pcl_contact_faces": len(a),
        "acl_contact_faces": len(b),
        "aabb_candidate_face_pairs": candidate_count,
        "exact_strict_crossing_face_pairs": len(crossings),
        "crossing_source_nodes": {"PCL": len(pcl_nodes), "ACL": len(acl_nodes)},
        "crossing_rigid_anchor_nodes": anchor_counts,
        "crossing_adjacent_tetrahedra_with_exact_interior_overlap":
            solid_overlap_count,
        "all_source_contact_pairs": all_pair_rows,
        "source_contact_pair_count": len(all_pair_rows),
        "source_contact_pairs_with_initial_crossings": sum(
            row["exact_strict_crossing_face_pairs"] > 0
            for row in all_pair_rows),
        "source_contact_pair_strict_crossings_total": sum(
            row["exact_strict_crossing_face_pairs"]
            for row in all_pair_rows),
        "first_exact_crossings": first_crossings,
        "native_witnesses": logs,
        "loaded_knee_qualified": False,
        "boundary": "The pinned source contains strict initial crossings in multiple source-authored contact pairs, including PCL/ACL crossing triangles with overlapping adjacent solid tetrahedra. Removing these pairs would discard source-authored contact; the native barrier correctly rejects the intersecting initialization. This audit is not a resolved contact state or loaded knee step.",
    }


if __name__ == "__main__":
    result = run()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("status", "aabb_candidate_face_pairs",
                       "exact_strict_crossing_face_pairs")}, sort_keys=True))
