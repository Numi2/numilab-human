"""Compare the changed physical skin/fat shell faces on the exact native Float32 lattice."""
from argparse import ArgumentParser
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

from fit_bank_group_partitions import ROLE_COUNT
from fit_hand_cavity import PARENT, table
from compare_bank_trials import emitted, witnesses
from build_hand_trial import HERE, digest

sys.path.insert(0, str(HERE.parent / "resting-surface-clearance-001" / "src"))
sys.path.insert(0, str(HERE.parent / "resting-clearance-and-atrium-005"))
from fast_skin_intersection_audit import install_for_float32, owner


def role_surfaces(rows):
    owners = defaultdict(list)
    for row in rows:
        tet = tuple(int(row[f"parent_node{i}"]) for i in range(4))
        for face in combinations(tet, 3):
            owners[tuple(sorted(face))].append(int(row["material_index"]))
    groups = {role: [] for role in ("outer", "interface", "deep", "side")}
    for face, materials in sorted(owners.items()):
        assert len(materials) in (1, 2)
        columns = {i // ROLE_COUNT for i in face}
        if len(materials) == 1:
            role = "outer" if columns == {0} else "deep" if columns == {2} else "side"
            groups[role].append(face)
        elif materials[0] != materials[1]:
            groups["interface"].append(face)
    assert all(groups[role] for role in ("outer", "deep", "interface"))
    return groups


def audit(points, groups, changed):
    faces = [face for group in groups.values() for face in group]
    role_by_id = [role for role, group in groups.items() for _ in group]
    records, _ = owner.float32_surface_records(points, faces)
    selected = {i for i, face in enumerate(faces) if changed[list(face)].any()}
    result = owner._audit_self_pairs_touching_faces(records, selected)
    pairs = defaultdict(list)
    for a, b in result["triangle_pairs"]:
        class_pair = "/".join(sorted({role_by_id[a], role_by_id[b]}))
        pairs[class_pair].append([a, b])
    return dict(pairs), {role: sum(role_by_id[i] == role for i in selected)
                         for role in groups}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    install_for_float32()
    cell_path = PARENT.with_name(PARENT.name + ".tets.csv")
    before, before_mass = emitted(args.before)
    after, after_mass = emitted(args.after)
    changed = np.any(before != after, axis=1)
    assert np.any(changed) and np.min(after[:ROLE_COUNT, 2]) >= 0.0
    groups = role_surfaces(table(cell_path))
    a, changed_count = audit(before, groups, changed)
    b, _ = audit(after, groups, changed)
    a_points, b_points = witnesses(args.before), witnesses(args.after)
    result = {
        "topology_sha256": digest(cell_path),
        "before_nodes_sha256": digest(args.before / "initial-skin-fem.csv"),
        "after_nodes_sha256": digest(args.after / "initial-skin-fem.csv"),
        "scope": "Initial emitted paired physical shell. Triangle pairs include at least one changed face.",
        "changed_nodes": int(changed.sum()),
        "changed_faces": changed_count,
        "all_faces_by_layer": {role: len(faces) for role, faces in groups.items()},
        "before_pairs": a,
        "after_pairs": b,
        "new_pairs": {region: sorted(set(map(tuple,b[region]))-set(map(tuple,a.get(region, [])))) for region in b},
        "new_skin_bone_point_pairs": sorted(b_points.keys() - a_points.keys()),
        "emitted_nodal_mass_change_kg": float(np.sum(after_mass) - np.sum(before_mass)),
        "outer_rows_within_one_um": [{"row": int(i), "xyz_m": after[i].tolist()}
                                     for i in np.flatnonzero(after[:ROLE_COUNT, 2] <= 1.e-6)],
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in
                      ("changed_nodes","changed_faces","new_pairs","outer_rows_within_one_um")}))


if __name__ == "__main__":
    main()
