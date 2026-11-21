"""Independently compare the emitted native skin/fat deck with the mapped cavity."""
from argparse import ArgumentParser
import csv
from pathlib import Path
import json

import numpy as np

from cavity import LABEL, oriented_boundary
from map_cavity import digest


def records(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def report(run, fit, source):
    src_nodes, nodes, tets, labels = (run / name for name in
                                        ("initial-skin-fem.csv", "shell-mass.csv.nodes.csv",
                                         "shell-mass.csv.tets.csv", "shell-mass.csv.labels.csv"))
    actual = records(tets)
    actual_cells = np.array([[int(row[f"parent_node{i}"]) for i in range(4)] for row in actual])
    def current_pose(path):
        return np.array([[float(row[f"current_{axis}_m"]) for axis in "xyz"]
                         for row in records(path)], dtype=np.float32)
    initial_pose = current_pose(nodes)
    control_pose = current_pose(source / nodes.name)
    np.testing.assert_array_equal(initial_pose, control_pose)
    with np.load(fit / "posterior-interface-fit.npz") as f:
        discrepancy = float(np.abs(initial_pose - f["nodes_current"]).max())
        np.testing.assert_allclose(initial_pose, f["nodes_current"], rtol=0., atol=4.e-9)
        np.testing.assert_array_equal(np.sort(actual_cells, axis=1), np.sort(f["cells"], axis=1))
        changed = f["changed_rows"]
        canonical = f["original_cells"][changed]
        assert oriented_boundary(f["cells"][changed]) == oriented_boundary(canonical)
    old_labels = records(source / labels.name)
    for before, after in zip(old_labels, records(labels), strict=True):
        assert tuple(before[k] for k in LABEL) == tuple(after[k] for k in LABEL)
    native_tet_mass = np.array([float(row["parent_mass_kg"]) for row in actual])
    native_node_mass = np.array([float(row["compiled_lumped_mass_kg"]) for row in records(nodes)])
    with np.load(fit / "balanced-mass.npz") as mapping:
        expected_tet, expected_node = (mapping[k].astype(np.float32).astype(float)
                                      for k in ("cell_mass_kg", "node_mass_kg"))
    np.testing.assert_allclose(native_tet_mass, expected_tet, rtol=2.e-6, atol=0)
    np.testing.assert_allclose(native_node_mass, expected_node, rtol=2.e-6, atol=0)
    return {
        "changed_rows": changed.tolist(),
        "positions_equal_to_native_control_float32": True,
        "maximum_offline_authoring_pose_roundoff_m": discrepancy,
        "typed_material_anatomical_donor_banks_equal": True,
        "actual_tet_mass_kg": float(native_tet_mass.sum()),
        "actual_node_mass_kg": float(native_node_mass.sum()),
        "maximum_tet_mass_error_kg": float(np.abs(native_tet_mass - expected_tet).max()),
        "maximum_node_mass_error_kg": float(np.abs(native_node_mass - expected_node).max()),
        "hashes": {p.name: digest(p) for p in (src_nodes, nodes, tets, labels)},
    }


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    for n in ("run", "fit", "source", "out"):
        p.add_argument(n, type=Path)
    a = p.parse_args()
    result = report(a.run, a.fit, a.source)
    with a.out.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(result)
