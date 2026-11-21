"""Find short shared skin/fat–periosteum routes in the registered dorsal sacrum."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

import numpy as np

from audit import BONE, accepted_pose, contacts, digest, fem_boundaries, muscle_geometry, records, rows
from audit_bed_layer_surfaces import role_surfaces
from screen_separated_posterior import MARGIN


def face_geometry(points, cells, faces):
    opposite = {}
    for tet in cells:
        for omitted in range(4):
            face = tuple(sorted(tet[:omitted] + tet[omitted + 1:]))
            if face in faces:
                assert face not in opposite
                opposite[face] = tet[omitted]
    face_ids = np.array(sorted(faces))
    tri = points[face_ids]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    inside = points[[opposite[tuple(f)] for f in face_ids]] - tri[:, 0]
    normal[np.einsum("ij,ij->i", normal, inside) > 0.] *= -1.
    lengths = np.linalg.norm(normal, axis=1)
    assert np.all(lengths > 0.)
    area = np.bincount(face_ids.ravel(), weights=np.repeat(lengths / 6., 3),
                       minlength=len(points))
    unit = np.stack([np.bincount(face_ids.ravel(), weights=np.repeat(x / 6., 3),
                                 minlength=len(points)) for x in normal.T], axis=1)
    return area, unit


def main():
    p = ArgumentParser(description=__doc__)
    for key in ("run", "bone", "reader", "fit", "output"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    p_nodes = a.run / "shell-mass.csv.nodes.csv"
    p_cells = a.run / "shell-mass.csv.tets.csv"
    log = a.run / "stdout.log"
    node_rows, cell_rows = rows(p_nodes), rows(p_cells)
    cells = [[int(r[f"parent_node{i}"]) for i in range(4)] for r in cell_rows]
    layers = role_surfaces(cell_rows)
    reference = np.array([[r[f"current_{k}_m"] for k in "xyz"] for r in node_rows], dtype=np.float32)
    with np.load(a.fit) as fit:
        assert np.array_equal(fit["cells"], cells)
        x = fit["nodes_current"].astype(np.float32)
    n = len(x) // 3
    assert np.array_equal(reference[:2*n], x[:2*n])
    x = x.astype(float)
    area, normal = face_geometry(x, cells, set(layers["deep"]))
    bone = np.memmap(a.bone, mode="r", dtype=BONE)
    bt = bone["vertices"].astype(float)
    sacral = (bone["body"] == 1) & (bone["bone"] == 1)
    lo, hi = bt[sacral].min(axis=(0, 1)), bt[sacral].max(axis=(0, 1))
    keep = (area > 0.) & (normal[:, 2] > .6 * area)
    nodes = np.flatnonzero(keep & np.all((x[:, :2] > lo[:2]) & (x[:, :2] < hi[:2]), axis=1))
    b_norm = np.cross(bt[:, 1] - bt[:, 0], bt[:, 2] - bt[:, 0])
    starts, anchors, objects, evidence = records(log)
    reg = json.loads(evidence["resting_initial_anatomical_registration"].partition("=")[2])
    pose, pose_cert = accepted_pose(a.run / "native/resting-body-motion-partition.csv", reg)
    payload = json.loads((a.run / "launch.json").read_text())
    mt, muscle_face, mhash, _, _ = muscle_geometry(payload, pose, a.reader)
    fv, _ = fem_boundaries(starts, objects, anchors)
    hits = contacts(x[nodes, :2], bt)
    muscles = contacts(x[nodes, :2], mt)
    fem = contacts(x[nodes, :2], fv)
    shell = contacts(x[nodes, :2], x[np.concatenate([np.asarray(v) for v in layers.values()])])
    out, reject = [], Counter()
    audited = []
    for i, node in enumerate(nodes):
        z = x[node, 2]
        above = [(j, y) for j, y in hits[i] if y > z + MARGIN]
        if not above or not sacral[above[0][0]]:
            reject["first_bone_not_sacrum"] += 1
            continue
        b, h = above[0]
        if b_norm[b, 2] >= -.3 * np.linalg.norm(b_norm[b]):
            reject["not_dorsal_sacral_entry"] += 1
            continue
        # No interior triangle crossings may occur along the proposed
        # periosteal route. Reject a path whose start already touches bone.
        if any(abs(y - z) <= MARGIN for _, y in hits[i]) or any(
                z + MARGIN < y < h - MARGIN for _, y in shell[i]):
            reject["other_shell_or_bone_interface"] += 1
            continue
        obstruction = [(j, y) for j, y in muscles[i] if z-MARGIN < y < h+MARGIN]
        if obstruction:
            reject["muscle_crosses_segment"] += 1
            audited.append({"node": int(node), "gap_m": h-z, "obstacle": muscle_face(obstruction[0][0])})
            continue
        if any(z-MARGIN < y < h+MARGIN for _, y in fem[i]):
            reject["registered_fem_crosses_segment"] += 1
            continue
        gap = h - z
        outer_gap = x[node % n, 2] - x[:n, 2].min()
        out.append(dict(deep_shell_row=int(node), source_sacral_face=int(bone[b]["triangle"]),
                        deep_point_m=x[node].tolist(), endpoint_m=[float(x[node,0]), float(x[node,1]), h],
                        interface_gap_m=float(gap), tributary_area_m2=float(area[node]),
                        outer_relative_height_m=float(outer_gap), source_body_owner=int(node_rows[node]["node_frame_owner_body"])))
    assert len(out) + sum(reject.values()) == len(nodes)
    summary = [{
        "distance_limit_m": radius,
        "route_count": sum(1 for r in out if r["interface_gap_m"] <= radius),
        "area_m2": sum(r["tributary_area_m2"] for r in out if r["interface_gap_m"] <= radius),
    } for radius in (.001, .002, .004, .006, .008, .012, .02, .03)]
    result = {
        "scope": "Bed-normal geometric segments from the emitted deep-fat surface to the first registered dorsal sacral face. Segment crossing and orientation tests only. No elastic runtime rows installed.",
        "inputs_sha256": {f.name: digest(f) for f in (p_nodes,p_cells,a.bone,a.fit,log)},
        "muscle_payload_sha256": mhash, "pose_comparison": pose_cert,
        "tested": len(nodes), "rejected": dict(reject), "gaps": summary,
        "direct_routes": out, "sample_obstructions": audited[:20],
    }
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: result[k] for k in ("tested","rejected","gaps")}))


if __name__ == "__main__":
    main()
