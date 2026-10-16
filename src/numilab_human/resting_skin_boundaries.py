"""Repair small atlas shell openings without changing its registered skinning.

Exact coincident vertices may share topology only when every ABI5 weight is
identical. Small non-ocular boundary rings receive explicitly derived faces.
The two eye openings remain anatomical interfaces, not an enclosed-volume
certificate. All original vertex IDs, coordinates, normals and weights remain.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from .resting_scene import load_rigid, load_skin, source_shell_world, _require
from .resting_anatomy import _cap_loop


def boundary_loops(faces: np.ndarray) -> tuple[list[list[int]], int]:
    edges = defaultdict(list)
    for triangle in faces:
        for a, b in zip(triangle, np.roll(triangle, -1)):
            edges[min(int(a), int(b)), max(int(a), int(b))].append((int(a), int(b)))
    _require(all(len(v) <= 2 for v in edges.values()), "skin has a nonmanifold edge")
    _require(all(len(v) == 1 or v[0] == v[1][::-1] for v in edges.values()),
             "skin has inconsistent triangle winding")
    remaining = {rows[0] for rows in edges.values() if len(rows) == 1}
    count = len(remaining)
    loops = []
    while remaining:
        start = min(remaining)[0]
        route = [start]
        seen = {start: 0}
        while True:
            outgoing = sorted(b for a, b in remaining if a == route[-1])
            _require(bool(outgoing), "skin boundary has an open directed chain")
            nxt = outgoing[0]
            remaining.remove((route[-1], nxt))
            if nxt in seen:
                # A touching pair of rings may have a degree-four boundary
                # vertex. Split at the repeated vertex, preserving each edge.
                at = seen[nxt]
                loops.append(route[at:])
                route = route[:at+1]
                seen = {v: i for i, v in enumerate(route)}
                if nxt == start:
                    break
            else:
                seen[nxt] = len(route)
                route.append(nxt)
    _require(sum(map(len, loops)) == count and all(len(x) >= 3 for x in loops),
             "skin boundary decomposition lost an edge")
    return loops, count



def audit_retained_capture_boundaries(source_faces, source_vertex_count,
                                     captured_surface, pack_first_vertex,
                                     retained_interfaces) -> dict:
    """Classify declared skin apertures after a complete source/capture ID join.

    The caller authenticates the payload, accepted capture, and original boundary
    receipt. This function proves that their *directed* source rings survive
    unchanged; counts, nearby coordinates, or a guessed ID offset do not suffice.
    It neither caps the apertures nor grants self/solid/enclosure admission.
    """
    from . import resting_anatomy_verify as verify

    _require(type(source_vertex_count) is int and source_vertex_count > 0,
             "invalid source skin vertex count")
    _require(type(pack_first_vertex) is int and pack_first_vertex >= 0,
             "invalid explicit skin capture offset")
    faces = np.asarray(source_faces)
    _require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
             and faces.dtype.kind in "iu" and np.all(faces >= 0)
             and np.all(faces < source_vertex_count), "invalid source skin faces")
    _require(isinstance(retained_interfaces, list) and len(retained_interfaces) == 2,
             "expected the two declared source ocular interfaces")

    def cycle(values):
        _require(isinstance(values, list) and len(values) >= 3
                 and all(type(x) is int and 0 <= x < source_vertex_count
                         for x in values) and len(set(values)) == len(values),
                 "invalid directed source boundary ring")
        start = values.index(min(values))
        return tuple(values[start:] + values[:start])

    declared = []
    for row in retained_interfaces:
        _require(isinstance(row, dict) and row.get("interface") ==
                 "ocular surface opening; passive eyeball interface",
                 "unrecognized retained skin interface")
        declared.append(cycle(row.get("source_vertex_ids")))
    _require(len(set(declared)) == len(declared), "duplicate source boundary declaration")
    source_loops, source_edges = boundary_loops(faces)
    _require(sorted(map(cycle, source_loops)) == sorted(declared),
             "current source boundaries differ from declared directed ocular rings")

    positions, capture_faces, normals, gids = captured_surface
    positions, capture_faces, gids = map(np.asarray, (positions, capture_faces, gids))
    _require(positions.dtype == np.dtype("float32") and positions.ndim == 2
             and positions.shape[1] == 3 and np.isfinite(positions).all(),
             "invalid captured skin coordinates")
    _require(gids.ndim == 1 and len(gids) == len(positions)
             and gids.dtype.kind in "iu" and len(np.unique(gids)) == len(gids),
             "invalid captured skin source identities")
    _require(capture_faces.ndim == 2 and capture_faces.shape[1] == 3
             and capture_faces.dtype.kind in "iu" and np.all(capture_faces >= 0)
             and np.all(capture_faces < len(positions)), "invalid captured skin faces")
    _require(np.all(gids >= pack_first_vertex)
             and np.all(gids - pack_first_vertex < source_vertex_count),
             "captured skin identities are outside the explicit source interval")
    _, join = verify.retain_bound_source_faces(
        (51007, 1), captured_surface, faces, 0,
        np.arange(len(faces), dtype=np.int64))
    _require(join["source_capture_vertex_namespace_offset"] == pack_first_vertex,
             "captured skin offset differs from the accepted source mapping")
    capture_loops, capture_edges = boundary_loops(capture_faces)
    source_ids = (gids - pack_first_vertex).astype(np.int64)
    actual = [source_ids[loop].tolist() for loop in capture_loops]
    _require(sorted(map(cycle, actual)) == sorted(declared)
             and capture_edges == source_edges,
             "accepted capture boundaries differ from declared directed ocular rings")
    return {
        "identity": [51007, 1], "passed": True,
        "boundary_edge_count": capture_edges, "boundary_loop_count": len(actual),
        "all_ordered_source_facets_match": True,
        "source_capture_vertex_namespace_offset": pack_first_vertex,
        "directed_source_boundary_rings_unchanged": True,
        "retained_anatomical_interfaces": retained_interfaces,
        "unexplained_boundary_edge_count": 0,
        "self_intersection_checked": False,
        "closed_volume_or_enclosure_admitted": False,
        "geometry_bytes_changed": False,
        "qualification": "Source-bound ocular aperture classification only; "
                         "self-intersections, eye interfaces, and other anatomy gates remain required.",
    }

def repair(rigid_path: Path, skin_path: Path, output: Path) -> dict:
    rigid = load_rigid(rigid_path)
    skin = load_skin(skin_path, rigid)
    raw = skin["raw"]
    _, first, inverse = np.unique(skin["vertices"], axis=0, return_index=True, return_inverse=True)
    representative = first[inverse]
    _require(np.array_equal(skin["weights"], skin["weights"][representative]),
             "coincident skin vertices have different physical skinning weights")
    offset = 60 + 36 * skin["binding_count"] + 56 * skin["vertex_count"]
    source_faces = np.frombuffer(raw, "<u4", skin["index_count"], offset).reshape(-1, 3)
    faces = representative[source_faces]
    _require(np.all(faces[:, 0] != faces[:, 1]) and np.all(faces[:, 1] != faces[:, 2]) and
             np.all(faces[:, 0] != faces[:, 2]), "topology welding collapsed a source face")
    loops, before = boundary_loops(faces)
    world, owners, _ = source_shell_world(rigid, skin)
    caps, retained, repairs = [], [], []
    for loop in loops:
        points = world[loop]
        extent = np.ptp(points, axis=0)
        if extent.max() > .012:
            # These source apertures lie in the two orbital regions. Admission
            # also checks their owner/count/extent to reject a new large tear.
            _require(np.all(owners[loop] == 23) and .02 < extent.max() < .035 and
                     45 <= len(loop) <= 80, "unexplained large skin boundary")
            retained.append({"source_vertex_ids": list(map(int, loop)),
                             "interface": "ocular surface opening; passive eyeball interface",
                             "bounds_world_m": [points.min(axis=0).tolist(), points.max(axis=0).tolist()]})
        else:
            triangles, receipt = _cap_loop(world, loop)
            caps.extend(triangles)
            receipt["extent_m"] = extent.tolist()
            repairs.append(receipt)
    _require(len(retained) == 2, "expected the two source ocular interfaces")
    result = np.concatenate((faces, np.asarray(caps, dtype=np.int64)), axis=0).astype("<u4")
    after_loops, after = boundary_loops(result)
    _require(after == sum(len(x["source_vertex_ids"]) for x in retained) and len(after_loops) == 2,
             "small skin openings remain after repair")
    header = bytearray(raw[:60])
    struct.pack_into("<I", header, 20, int(result.size))
    payload = bytes(header) + raw[60:offset] + result.tobytes() + raw[offset + 4 * skin["index_count"]:]
    _require(not output.exists(), "derived skin output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    derived = load_skin(output, rigid)
    _require(np.array_equal(derived["vertices"], skin["vertices"]) and
             np.array_equal(derived["weights"], skin["weights"]), "skin repair changed registered coordinates or weights")
    receipt = {"source_skin": {"path": str(skin_path.resolve()), "sha256": skin["sha256"]},
               "derived_skin": {"path": str(output.resolve()), "sha256": hashlib.sha256(payload).hexdigest()},
               "registration_fingerprint32": skin["registration_fingerprint"],
               "exact_coordinate_weight_identical_seam_vertices": int(len(representative) - len(first)),
               "boundary_edges_after_exact_weld": before, "boundary_edges_after_small_caps": after,
               "retained_anatomical_interfaces": retained, "derived_cap_repairs": repairs,
               "added_triangle_count": len(caps), "source_triangle_count": len(source_faces),
               "coordinates_normals_weights_vertex_ids_unchanged": True,
               "qualification": "topology repair only; ocular openings retained; not a closed body-volume or self-intersection certificate"}
    output.with_suffix(".boundary-repair.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rigid", type=Path, required=True)
    p.add_argument("--skin", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = repair(a.rigid, a.skin, a.output)
    print(json.dumps({k: r[k] for k in ("derived_skin", "boundary_edges_after_small_caps", "added_triangle_count")}))
