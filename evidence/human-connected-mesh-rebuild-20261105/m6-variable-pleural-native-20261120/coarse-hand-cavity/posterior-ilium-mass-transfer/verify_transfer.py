"""Compare native cooked cells and masses to the prepared registration ledger."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from audit import digest, rows
from extension import determinant, quality
from map_mass import regional_context


def verify(reference, fitted, shared, native=None):
    old, typed, reference_nodes, cells = regional_context(reference, shared)
    with np.load(fitted / "posterior-interface-fit.npz") as fit:
        positions = fit["nodes_reference"]
        assert np.array_equal(fit["cells"], cells)
    with np.load(fitted / "balanced-mass.npz") as mapping:
        src, dst, w = (mapping[key] for key in ("parent_cell", "target_cell", "mass_kg"))
        expected_cell, expected_node = (mapping[key] for key in ("cell_mass_kg", "node_mass_kg"))
    nodes = rows(reference / "shell-mass.csv.nodes.csv")
    donor = np.array([int(r["node_frame_owner_body"]) for r in nodes])
    group = old.group_for_parent
    assert np.array_equal(group[src], group[dst]) and len(typed) == 628
    assert np.all(w > 0.)
    mass_error = np.max(np.abs(np.bincount(src, weights=w, minlength=len(cells)) - old.source_kg))
    mapped = np.bincount(dst, weights=w, minlength=len(cells))
    assert mass_error < 1e-12 and np.array_equal(mapped, expected_cell)
    assert np.allclose(
        np.bincount(cells.ravel(), weights=np.repeat(mapped / 4., 4), minlength=len(nodes)),
        expected_node, rtol=1e-13, atol=0)
    per_node_old = np.bincount(cells.ravel(), weights=np.repeat(old.source_kg / 4., 4), minlength=len(nodes))
    zonal = np.bincount(donor, weights=expected_node - per_node_old)
    distance = np.linalg.norm(old.source_centroid[src] - positions[cells].mean(axis=1)[dst], axis=1)
    assert np.max(distance) <= .075
    volume = determinant(positions, cells) / 6.
    assert volume.min() > 0.
    lower, upper = old.density_limits.T
    density = mapped / volume
    assert np.all(density >= lower-1e-5) and np.all(density <= upper+1e-5)
    result = {"cells": len(cells), "nodes": len(nodes), "typed_pools": len(typed),
              "total_mass_kg": float(mapped.sum()), "maximum_source_mass_error_kg": float(mass_error),
              "maximum_skeletal_donor_debit_change_kg": float(np.abs(zonal).max()),
              "changed_source_nodes": int(np.any(positions != reference_nodes, axis=1).sum()),
              "maximum_route_m": float(distance.max()), "mass_sha256": digest(fitted / "balanced-mass.npz")}
    if native is not None:
        nn = rows(native / "shell-mass.csv.nodes.csv")
        cc = rows(native / "shell-mass.csv.tets.csv")
        generated = np.array([float(v["parent_mass_kg"]) for v in cc])
        cooks = np.array([float(r["compiled_lumped_mass_kg"]) for r in nn])
        emitted = np.array([[r[f"current_{a}_m"] for a in "xyz"] for r in nn], dtype=np.float32)
        expected_float_mass = expected_node.astype(np.float32)
        assert np.array_equal(np.array([int(r["node_frame_owner_body"]) for r in nn]), donor)
        assert np.array_equal(np.array([[int(v[f"parent_node{i}"]) for i in range(4)] for v in cc]), cells)
        assert np.array_equal(generated, expected_cell)
        nodal_relative_error = np.max(np.abs(cooks-expected_float_mass)/expected_float_mass)
        assert nodal_relative_error <= 2e-7
        error = np.max(np.linalg.norm(emitted.astype(float) - positions, axis=1))
        assert error <= 1e-7
        determinants = determinant(emitted.astype(float), cells)
        quals = quality(emitted.astype(float), cells, determinants)
        assert determinants.min() > 0. and quals.min() >= 1e-5
        result["native"] = {"emitted_reference_error_m": float(error),
                            "maximum_relative_node_mass_error": float(nodal_relative_error),
                            "minimum_signed_determinant_m3": float(determinants.min()),
                            "minimum_normalized_determinant": float(quals.min()),
                            "all_cell_masses_exact": True, "all_donor_frame_ids_match": True,
                            "launch_sha256": digest(native / "launch.json")}
    return result


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    for name in ("reference", "fitted", "shared", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("--native", type=Path)
    a = vars(p.parse_args())
    out = a.pop("out")
    data = verify(**a)
    with out.open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(data))
