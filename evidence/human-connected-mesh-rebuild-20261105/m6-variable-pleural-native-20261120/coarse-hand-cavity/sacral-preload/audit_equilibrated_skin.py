"""Verify Float32 cells and shell crossings in a saved preload trial."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

import numpy as np

from audit import BONE, digest, rows
from audit_bed_layer_surfaces import audit, install_for_float32, owner, role_surfaces
from extension import determinant, quality



def changed_bone_pairs(base, candidate, groups, bone_path):
    face_ids = np.array([face for faces in groups.values() for face in faces])
    roles = [role for role, faces in groups.items() for _ in faces]
    moved = np.any(candidate.astype(np.float32) != base.astype(np.float32), axis=1)
    selected = np.flatnonzero(moved[face_ids].any(axis=1))
    faces = face_ids[selected]
    bounds = np.r_[base[faces].reshape(-1, 3), candidate[faces].reshape(-1, 3)]
    lo, hi = bounds.min(axis=0), bounds.max(axis=0)
    bank = np.memmap(bone_path, mode="r", dtype=BONE)
    t = bank["vertices"]
    trimmed = bank[np.all(t.min(axis=1) <= hi, axis=1) & np.all(t.max(axis=1) >= lo, axis=1)]
    points = trimmed["vertices"].reshape(-1, 3)
    indices = np.arange(len(points)).reshape(-1, 3)
    target = owner.float32_surface_records(points, indices)[0]
    crossings = []
    for x in (base, candidate):
        surface = owner._prepare_surface_aabb(owner.float32_surface_records(x, faces)[0])
        found = owner._audit_pair_prepared_first(surface, target)["triangle_pairs"]
        crossings.append({(int(selected[i]), int(trimmed[j]["body"]), int(trimmed[j]["bone"]),
                           int(trimmed[j]["triangle"])) for i, j in found})
    old, new = crossings
    additions = sorted(new - old)
    removals = sorted(old - new)
    return {
        "changed_faces": len(faces), "tested_bone_faces": len(trimmed),
        "before": len(old), "after": len(new),
        "added": dict(Counter(roles[i] for i, *_ in additions)),
        "removed": dict(Counter(roles[i] for i, *_ in removals)),
        "added_witnesses": additions, "removed_witnesses": removals,
    }

def main():
    p = ArgumentParser(description=__doc__)
    for name in ("run", "fit", "trial", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--bone", type=Path)
    a = p.parse_args()
    support = json.loads((a.trial / "result.json").read_text())
    evidence = a.trial / "last-equilibrated.npz"
    cells_path = a.run / "shell-mass.csv.tets.csv"
    with np.load(a.fit) as source:
        nodes = source["nodes_current"].astype(float)
        rest = source["nodes_reference"].astype(float)
        cells = source["cells"]
    assert digest(a.fit) == support["hashes"][a.fit.name]
    assert digest(cells_path) == support["hashes"][cells_path.name]
    rows_of_cells = rows(cells_path)
    assert np.array_equal(cells, [[int(r[f"parent_node{i}"]) for i in range(4)] for r in rows_of_cells])
    n = len(nodes) // 3
    assert n * 3 == len(nodes)
    with np.load(evidence) as state:
        v, j, fixed_fraction, force, residual = (state[k] for k in (
            "coordinates", "determinant", "load_fraction", "contact_force_n", "residual"))
    assert np.max(np.abs(residual)) < 1.e-4
    assert abs(force.sum() - float(fixed_fraction) * support["physical_weight_n"]) < 1.e-4
    assert float(fixed_fraction) == support["last_converged_fraction"]
    u = np.zeros_like(nodes)
    u[:2*n, 2] = v[:2*n]
    plane_path = Path("support-hull-01.json")
    assert digest(plane_path) == support["hashes"][plane_path.name]
    bed = json.loads(plane_path.read_text())["bed_offset_m"]
    lateral = np.flatnonzero(nodes[:2*n, 2] - bed <= .045)
    if support.get("deep_tangential", False):
        with (a.run / "stdout.log").open() as stream:
            anchors = {2*n + int(line.partition(" local_vertex=")[2].split()[0])
                       for line in stream if line.startswith("whole_body_skin_attachment ")}
        assert len(anchors) == 14
        deep = [i for i in range(2*n, len(nodes)) if nodes[i, 2] - bed <= .045 and i not in anchors]
        lateral = np.r_[lateral, deep]
    if support["lateral_coordinates"]:
        assert len(lateral) * 2 == support["lateral_coordinates"]
        u[lateral, :2] = v[2*n:-3].reshape(-1, 2)
    x = nodes + u
    expected = determinant(x, cells) / determinant(rest, cells)
    assert np.allclose(j, expected, rtol=1.e-9, atol=1.e-9)
    emitted = x.astype(np.float32).astype(float)
    d = determinant(emitted, cells)
    f32j = d / determinant(rest, cells)
    q = quality(emitted, cells, d)
    install_for_float32()
    groups = role_surfaces(rows_of_cells)
    pairs, _ = audit(emitted, groups, np.ones(len(nodes), dtype=bool))
    report = {
        "scope": "Exact Float32 preloaded shell in the initial skeletal reference, before applying the global rigid seed. A common rigid transform preserves intersection pairs. Separate bone, fascia, and bed audits remain required.",
        "state_sha256": digest(evidence), "fit_sha256": digest(a.fit),
        "maximum_residual": float(np.max(np.abs(residual))), "load_fraction": float(fixed_fraction),
        "total_force_n": float(force.sum()), "active_nodes": int((force > 0.).sum()),
        "max_deformation_m": float(np.linalg.norm(u, axis=1).max()),
        "recomputed_min_J": float(f32j.min()), "recomputed_max_J": float(f32j.max()),
        "nonpositive_cells": int((d <= 0.).sum()),
        "cells_below_quality_floor": int((q < 1.e-5).sum()),
        "shell_crossing_pair_counts": {role: len(ids) for role, ids in pairs.items()},
        "shell_crossing_pairs": pairs,
    }
    if a.bone is not None:
        report["bone_sha256"] = digest(a.bone)
        report["changed_face_bone_pairs"] = changed_bone_pairs(nodes, emitted, groups, a.bone)
    with a.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print({key: val for key, val in report.items() if key not in ("scope", "shell_crossing_pairs", "changed_face_bone_pairs")})


if __name__ == "__main__":
    main()
