"""Screen real posterior fat–glute–bone load paths in the posed source.

Each quadrature row requires a nonpenetrating fat-to-muscle gap, a complete
gluteal chord and a hit on one of that muscle's registered skeletal donors.
The upper bound ignores bulk shell compliance and muscle/rigid joint motion.
It cannot establish support balance.
"""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

import numpy as np

from audit import BONE, contacts, digest, rows
from audit_bed_layer_surfaces import role_surfaces


def first_chord(hits, normals, xyz_z, precision):
    """Keep the first oriented entry and exit, rejecting an inside start."""
    entries = [(i, z) for i, z in hits if abs(normals[i, 2]) > 1.e-10]
    if len(entries) < 2:
        return None
    first = entries[0][1]
    if first < xyz_z + precision:
        return None
    gap = [h for h in entries if h[1] <= first + precision]
    if any(normals[i, 2] >= 0. for i, _ in gap):
        return None
    exits = [h for h in entries if h[1] > first + precision]
    if not exits or normals[exits[0][0], 2] <= 0.:
        return None
    return gap[0], exits[0]


def paths(native, bone_path, muscle_dir, crossings, ply_gap=0.05):
    tets = rows(native / "shell-mass.csv.tets.csv")
    node_path = native / "shell-mass.csv.nodes.csv"
    mesh = np.array([[r[f"current_{axis}_m"] for axis in "xyz"]
                     for r in rows(node_path)], dtype=np.float32).astype(float)
    assert len(mesh) % 3 == 0
    n = len(mesh) // 3
    boundaries = role_surfaces(tets)
    faces = np.asarray(boundaries["deep"], dtype=int)
    deep = mesh[faces]
    source_columns = mesh[faces % n]
    crosses = set(v[1] for v in crossings["initial"]["witnesses"])
    # Keep correspondence with the full boundary auditor's global face IDs.
    roles = list(boundaries)
    deep_offset = sum(len(boundaries[k]) for k in roles[:roles.index("deep")])
    normals = np.cross(deep[:, 1] - deep[:, 0], deep[:, 2] - deep[:, 0])
    axis_area = .5 * abs(normals[:, 2])
    radius = .5 * np.linalg.norm(deep.max(axis=1) - deep.min(axis=1), axis=1)
    mesh_height = source_columns[:, :, 2].min()
    foot = (source_columns[:, :, 2].max(axis=1) < mesh_height + ply_gap)
    # Use each polygon only once and decline sideways and degenerate faces.
    eligible = np.flatnonzero(foot & (axis_area > 0.) & (radius > 0.))
    xyz = deep[eligible].mean(axis=1)
    bone = np.memmap(bone_path, dtype=BONE, mode="r")
    audits = []
    quote = 1.e-7
    for muscle_id, shaft in ((29, 131), (30, 145)):
        file = muscle_dir / f"gluteal-{muscle_id}.npz"
        with np.load(file) as data:
            tri = data["source_nodes"][data["source_faces"]]
        muscle_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        candidates = np.flatnonzero(np.all(xyz[:, :2] >= tri.min(axis=(0, 1))[:2], axis=1)
                                  & np.all(xyz[:, :2] <= tri.max(axis=(0, 1))[:2], axis=1))
        points = xyz[candidates]
        available = [128, shaft]
        banks = bone[np.isin(bone["body"], available)]
        muscle_hits = contacts(points[:, :2], tri)
        donor_hits = contacts(points[:, :2], banks["vertices"])
        full_hits = contacts(points[:, :2], bone["vertices"])
        excluded = Counter()
        traces = []
        for local, (origin, mh, dh, fh) in enumerate(zip(points, muscle_hits, donor_hits, full_hits)):
            face = int(eligible[candidates[local]])
            chord = first_chord(mh, muscle_normals, origin[2], quote)
            if chord is None:
                excluded["source_chord_absent_or_invalid_at_fat_boundary"] += 1
                continue
            (enter_face, enter_z), (exit_face, exit_z) = chord
            hit = next(((j, z) for j, z in dh if enter_z + quote < z < exit_z - quote), None)
            if hit is None:
                excluded["no_registered_donor_inside_chord"] += 1
                continue
            bone_row, z = hit
            donor = banks[bone_row]
            if any(origin[2] - quote <= h < z - quote for _, h in fh):
                excluded["another_physical_bone_in_front_of_donor"] += 1
                continue
            if face + deep_offset in crosses:
                excluded["fat_face_intersects_a_registered_bone"] += 1
                continue
            # A rigid-body column sensitivity bound. Assume all downward
            # travel can close this projected gap before compressing the layer.
            trace = {"source_muscle": muscle_id, "entry_face": int(enter_face),
                     "exit_face": int(exit_face), "fat_face": face + deep_offset,
                     "fat_face_nodes": faces[face].tolist(),
                     "quadrature_point_m": origin.tolist(),
                     "bone_body": int(donor["body"]), "bone": int(donor["bone"]),
                     "bone_triangle": int(donor["triangle"]), "initial_gap_m": float(enter_z-origin[2]),
                     "gluteal_donor_depth_m": float(z-enter_z),
                     "area_m2": float(axis_area[face])}
            assert trace["initial_gap_m"] > 0. and trace["gluteal_donor_depth_m"] > 0.
            traces.append(trace)
        audits.append({
            "muscle": muscle_id, "attached_body_ids": available, "source_hash": digest(file),
            "projected_candidates": len(candidates), "rejected": dict(excluded), "rows": traces,
        })
    # Arc surfaces can share the same horizontal projection. Filter rows by
    # depth before carrying them into a physical integration, even if this
    # geometric screen finds positive load capacity.
    return {"source_bones": digest(bone_path), "shell": digest(node_path), "tested_surface_faces": len(eligible),
            "seating_height_envelope_m": ply_gap, "banks": audits}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("native", "bone_path", "muscle_dir", "crossings", "output"):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    result = paths(a.native, a.bone_path, a.muscle_dir, json.loads(a.crossings.read_text()))
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps([{"muscle": r["muscle"], "candidate": r["projected_candidates"],
                       "admitted": len(r["rows"]), "rejected": r["rejected"]}
                      for r in result["banks"]]))
