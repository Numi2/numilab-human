"""Check every changed posterior boundary face with the existing exact predicates."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np

from audit import BONE, accepted_pose, digest, fem_boundaries, muscle_geometry, records, rows


def targets(points, bounds):
    lo, hi = bounds
    return np.flatnonzero(np.all(points.max(axis=1) >= lo, axis=1)
                          & np.all(points.min(axis=1) <= hi, axis=1))


def check(run, bone_path, candidate, reader, shared_tools):
    sys.path.insert(0, str(shared_tools))
    from audit_bed_layer_surfaces import audit, install_for_float32, owner, role_surfaces

    install_for_float32()
    node_rows = rows(run / "shell-mass.csv.nodes.csv")
    base = np.array([[float(v[f"current_{a}_m"]) for a in "xyz"] for v in node_rows],
                    dtype=np.float32)
    n = len(base) // 3
    assert len(base) == 3*n
    with np.load(candidate) as fitted:
        current = fitted["nodes_current"]
        cells = fitted["cells"]
    changed = np.any(base != current, axis=1)
    assert changed.any() and not changed[:2*n].any()
    tets = rows(run / "shell-mass.csv.tets.csv")
    base_cells = np.array([[int(v[f"parent_node{i}"]) for i in range(4)] for v in tets])
    assert np.array_equal(cells, base_cells)
    roles = role_surfaces(tets)
    all_faces = np.array([face for group in roles.values() for face in group])
    marked = np.flatnonzero(changed[all_faces].any(axis=1))
    before, touch_count = audit(base, roles, changed)
    after, after_touch_count = audit(current, roles, changed)
    assert touch_count == after_touch_count
    bounds = (np.minimum(base[all_faces[marked]].min(axis=(0, 1)), current[all_faces[marked]].min(axis=(0, 1))),
              np.maximum(base[all_faces[marked]].max(axis=(0, 1)), current[all_faces[marked]].max(axis=(0, 1))))

    bones = np.memmap(bone_path, dtype=BONE, mode="r")
    fem_nodes, anchors, fem_cells, evidence = records(run / "stdout.log")
    world, descriptor = fem_boundaries(fem_nodes, fem_cells, anchors)
    reg = json.loads(evidence["resting_initial_anatomical_registration"].partition("=")[2])
    pose, metrics = accepted_pose(run / "native/resting-body-motion-partition.csv", reg)
    gm, muscle_identity, muscle_hash, _, _ = muscle_geometry(json.loads((run / "launch.json").read_text()), pose, reader)
    def bone_identity(i):
        return {s: int(bones[s][i]) for s in ("body", "bone", "triangle")}

    banks = {}
    groups = [("bone", bones["vertices"], bone_identity), ("muscle_source", gm, muscle_identity),
              ("other_fem_surfaces", world, descriptor.__getitem__)]
    for label, points, identity in groups:
        use = targets(points, bounds)
        tri = points[use]
        target, _ = owner.float32_surface_records(tri.reshape(-1, 3), np.arange(tri.size // 3).reshape(-1, 3))
        result = []
        for position in (base, current):
            r, _ = owner.float32_surface_records(position, all_faces[marked])
            data = owner._audit_pair_prepared_first(owner._prepare_surface_aabb(r), target)
            result.append(set(map(tuple, data["triangle_pairs"])))
        a, b = result
        banks[label] = {"whole_target_triangles": len(points), "certified_aabb_subset": len(use),
                        "prior_pairs": len(a), "candidate_pairs": len(b), "added_pairs": len(b - a),
                        "preserved_pairs": len(a & b),
                        "preserved_pair_witnesses": [
                            {"skin_fat_face": all_faces[marked[i]].tolist(), "target": identity(use[j])}
                            for i, j in sorted(a & b)],
                        "added_pair_witnesses": [
                            {"skin_fat_face": all_faces[marked[i]].tolist(), "target": identity(use[j])}
                            for i, j in sorted(b - a)]}
    return {"candidate_sha256": digest(candidate), "shell_startup_inputs_sha256": digest(run / "launch.json"),
            "bone_sha256": digest(bone_path), "muscle_sha256": muscle_hash, "muscle_pose_check": metrics,
            "scope": "Exact triangle predicates on changed shell faces at one Float32 state. Excludes containment and intervening poses. Other shell geometry is identical. Muscle source uses first accepted rigid pose.",
            "changed_node_count": int(changed.sum()),
            "changed_face_counts_by_role": touch_count, "prior_shell_pairs": before,
            "candidate_shell_pairs": after, "cross_structure": banks}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("bone", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reader", type=Path)
    parser.add_argument("shared_tools", type=Path)
    parser.add_argument("output", type=Path)
    a = parser.parse_args()
    result = check(a.run, a.bone, a.candidate, a.reader, a.shared_tools)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print({s: (v["prior_pairs"], v["candidate_pairs"], v["added_pairs"])
           for s, v in result["cross_structure"].items()})
    print({"before": sum(map(len, result["prior_shell_pairs"].values())),
           "after": sum(map(len, result["candidate_shell_pairs"].values()))})


if __name__ == "__main__":
    main()
