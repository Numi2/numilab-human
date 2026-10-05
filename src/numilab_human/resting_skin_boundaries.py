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
