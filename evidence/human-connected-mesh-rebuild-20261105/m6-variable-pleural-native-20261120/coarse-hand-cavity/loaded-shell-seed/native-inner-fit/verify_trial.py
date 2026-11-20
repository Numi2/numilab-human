"""Compare exact compiled tissue geometry, mass and contact of a native trial."""
from argparse import ArgumentParser
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

from build_trial import COARSE, REFERENCE, digest

sys.path.insert(0, str(COARSE))
sys.path.insert(0, str(COARSE / "exact-shell-bone-audit-032"))
from audit import TRIANGLE_ROW, records
from audit_bed_layer_surfaces import audit, install_for_float32, owner, role_surfaces
from audit_attachment_path import event


def table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def points(rows, name):
    return np.array([[float(row[f"{name}_{axis}_m"]) for axis in "xyz"] for row in rows])


def hsh(path):
    return str(path), digest(path)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--baseline", type=Path, default=REFERENCE)
    args = parser.parse_args()
    paths = (args.baseline, args.trial)
    node_sets = [table(path / "shell-mass.csv.nodes.csv") for path in paths]
    tet_sets = [table(path / "shell-mass.csv.tets.csv") for path in paths]
    label_sets = [table(path / "shell-mass.csv.labels.csv") for path in paths]
    assert label_sets[0] == label_sets[1]
    indices = [*[f"parent_node{i}" for i in range(4)], "material_index", "donor_body"]
    topology = [np.array([[int(row[name]) for name in indices] for row in src]) for src in tet_sets]
    np.testing.assert_array_equal(*topology)
    tets = topology[0][:, :4]
    owner_columns = ("compiled_global_node", "node_frame_owner_body", "parent_node_index")
    IDs = [[tuple(row[name] for name in owner_columns) for row in source] for source in node_sets]
    assert IDs[0] == IDs[1]
    masses = [np.array([float(row["compiled_lumped_mass_kg"]) for row in src]) for src in node_sets]
    typed_delta = defaultdict(float)
    for label, a, b in zip(label_sets[0], *tet_sets, strict=True):
        bank = tuple(int(label[name]) for name in
                     ("material_index", "geometric_donor_body", "anatomical_region_id"))
        typed_delta[bank] += float(b["parent_mass_kg"]) - float(a["parent_mass_kg"])
    ref = [points(src, "rest") for src in node_sets]
    current = [points(src, "current") for src in node_sets]
    changed = np.any(current[0] != current[1], axis=1)
    assert np.flatnonzero(changed).tolist() == [3703, 4034]
    assert np.flatnonzero(np.any(ref[0] != ref[1], axis=1)).tolist() == [3703, 4034]
    fields = {}
    for label, coordinates in (("rest", ref), ("current", current)):
        quality = []
        for xyz in coordinates:
            p = xyz[tets]
            det = np.linalg.det(p[:, 1:] - p[:, :1])
            assert np.all(det > 0.)
            edges = p[:, :, None, :] - p[:, None, :, :]
            diameter = np.linalg.norm(edges, axis=3).max(axis=(1, 2))
            quality.append(float(np.min(det / diameter**3)))
        assert quality[1] >= quality[0]
        pairs = ((3703, 1292, 6114), (4034, 1623, 6445))
        thickness = np.array([[np.linalg.norm(coordinates[1][center] - coordinates[1][neighbor])
                               for neighbor in (outer, deep)] for center, outer, deep in pairs])
        assert np.all(thickness[:, 0] >= .0005) and np.all(thickness[:, 0] <= .0025)
        assert np.all(thickness[:, 1] >= .0005) and np.all(thickness[:, 1] <= .01)
        fields[label] = {"positive_cells": len(tets), "minimum_shape_before_after": quality,
                         "affected_skin_fat_thickness_m": thickness.tolist()}

    groups = role_surfaces(tet_sets[0])
    install_for_float32()
    before_self, count = audit(current[0], groups, changed)
    after_self, _ = audit(current[1], groups, changed)
    new_self = {key: sorted(set(map(tuple, after)) - set(map(tuple, before_self.get(key, []))))
                for key, after in after_self.items()}
    faces = np.array([face for group in groups.values() for face in group if changed[list(face)].any()])
    all_changed_triangles = np.vstack([p[faces].reshape(-1, 3) for p in current])
    mins, maxs = all_changed_triangles.min(axis=0), all_changed_triangles.max(axis=0)
    bone_path = COARSE / "thickness-rebuild-034/inputs/physical-bone-initial.bin"
    bones = np.memmap(bone_path, dtype=TRIANGLE_ROW, mode="r")
    tri_xyz = bones["vertices"]
    intersects = np.all(tri_xyz.max(axis=1) >= mins, axis=1) & np.all(tri_xyz.min(axis=1) <= maxs, axis=1)
    rows = np.flatnonzero(intersects)
    bone_records = tuple(records(tri_xyz[rows]))

    def exact_hits(xyz):
        skin, _ = owner.float32_surface_records(xyz, faces)
        tree = owner._prepare_surface_aabb(skin)
        pairs = owner._audit_pair_prepared_first(tree, bone_records)["triangle_pairs"]
        return {(a, int(rows[b])) for a, b in pairs}

    before_bone, after_bone = map(exact_hits, current)
    report = {
        "input_sha256": dict(hsh(path / name) for path in paths for name in
                             ("shell-mass.csv.nodes.csv", "shell-mass.csv.tets.csv",
                              "shell-mass.csv.labels.csv", "stdout.log")),
        "native_row_changes": np.flatnonzero(changed).tolist(),
        "same_compiled_tet_topology_and_typed_donor_ancestry": True,
        "reference_and_emitted_cells": fields,
        "native_skin_fat_mass_before_after_kg": [float(m.sum()) for m in masses],
        "maximum_per_typed_bank_mass_delta_kg": float(max(map(abs, typed_delta.values()))),
        "node_mass_delta_maximum_kg": float(np.max(abs(masses[1] - masses[0]))),
        "mass_bank_count": len(typed_delta),
        "changed_boundary_faces_by_role": count,
        "new_exact_shell_face_pairs": new_self,
        "bone_asset_sha256": digest(bone_path),
        "local_bone_crossing_pairs_before_after": [len(before_bone), len(after_bone)],
        "new_local_bone_pairs": sorted(after_bone - before_bone),
    }
    if json.loads((args.trial / "execution.json").read_text())["returncode"] == 0:
        rows = table(args.trial / "native/resting-coupled.csv")
        pressure = np.array([float(row["pleural_pa"]) for row in rows])
        terminal = event(args.trial / "stdout.log", "resting_fascia_fem_terminal_audit=passed")
        bed = event(args.trial / "stdout.log", "resting_fem_skin_bed_terminal_history")
        report["native_terminal"] = {"ligaments": terminal, "bed_impulse": bed,
                                     "last_coupled_time_s": float(rows[-1]["time_s"]),
                                     "pleural_pressure_range_pa": [float(pressure.min()), float(pressure.max())]}
    with args.receipt.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: val for name, val in report.items() if name != "input_sha256"}))


if __name__ == "__main__":
    main()
