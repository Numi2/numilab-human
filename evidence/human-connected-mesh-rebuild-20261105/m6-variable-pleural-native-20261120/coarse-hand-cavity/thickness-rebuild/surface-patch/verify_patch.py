"""Independently audit saved geometry, native source masses and ancestry."""
from argparse import ArgumentParser
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from rebuild import HERE, ROLE_COUNT, determinant, digest, quality, table
from regional_mass import RADIUS_M, DENSITY_KG_M3
from run_patch import locked_columns


def typed_surface(elements, material):
    incident = Counter()
    aggregate = Counter()
    for cell, typed in zip(elements.tolist(), material.tolist(), strict=True):
        for face in combinations(sorted(cell), 3):
            incident[face] += 1
            aggregate[(*face, typed)] += 1
    assert max(incident.values()) == 2
    return {tri[:3] for tri, count in aggregate.items() if count == 1}


def verify(path, baseline, original_rest, ledger):
    receipt = json.loads((path / "receipt.json").read_text())
    assert digest(path / "rebuild.npz") == receipt["candidate_sha256"]
    with np.load(path / "rebuild.npz", allow_pickle=False) as f:
        mesh = dict(f)
    rest, world = mesh["rest"], mesh["world"].astype(float)
    cells, owner = mesh["elements"], mesh["parent"]
    typed = mesh["material"]
    assert np.isfinite(rest).all() and np.isfinite(world).all()
    assert np.all((cells >= 0) & (cells < len(rest)))
    assert len(np.unique(np.sort(cells, axis=1), axis=0)) == len(cells)
    assert np.array_equal(typed, ledger["type"][owner])
    assert np.array_equal(mesh["donor"], ledger["donor"][owner])
    assert {tuple(sorted(face)) for face in mesh["surface"].tolist()} == typed_surface(cells, typed)
    columns = locked_columns(baseline)
    support = np.flatnonzero(baseline["world"][:ROLE_COUNT, 2] <= 1.e-5)
    np.testing.assert_array_equal(world[support], baseline["world"][support])
    rest_old = baseline["rest"]
    world_old = baseline["world"].astype(float)
    if len(rest) > len(rest_old):
        old_child, old_face = (mesh[key] for key in ("split_seed_cells", "split_seed_faces"))
        assert np.array_equal(owner, baseline["parent"][old_child])
        assert np.array_equal(mesh["surface_parent"], baseline["surface_parent"][old_face])
        seed = mesh["split_seed_nodes"]
        w = np.repeat(mesh["split_seed_barycentric"], 3, axis=0)
        assert np.all(w > 0.) and np.allclose(w.sum(axis=1), 1., rtol=0., atol=1.e-15)
        reference_add = (rest_old[seed] * w[:, :, None]).sum(axis=1)
        emitted_add = (world_old[seed] * w[:, :, None]).sum(axis=1).astype(np.float32)
        rest_old = np.vstack((rest_old, reference_add))
        world_old = np.vstack((world_old, emitted_add))
        columns = np.vstack((columns, np.arange(len(baseline["rest"]), len(rest)).reshape(-1, 3)))
    else:
        for key in ("elements", "parent", "surface", "surface_parent"):
            np.testing.assert_array_equal(mesh[key], baseline[key])
    original = baseline["elements"]
    det, emitted_det = determinant(rest, cells), determinant(world, cells)
    conditions = {}
    for name, before, after, current_det in (("rest", rest_old, rest, det),
                                            ("world", world_old, world, emitted_det)):
        source = before[:len(baseline["rest"])]
        assert current_det.min() >= determinant(source, original).min() > 0.
        q = quality(after, cells)
        assert q.min() >= quality(source, original).min() > 0.
        prev = np.linalg.norm(np.diff(before[columns], axis=1), axis=2)
        now = np.linalg.norm(np.diff(after[columns], axis=1), axis=2)
        assert np.all(now >= np.minimum(prev, .0005) - 2.e-7)
        assert np.all(now <= np.maximum(prev, (.0025, .010)) + 2.e-7)
        conditions[name] = {"min_det_m3": float(current_det.min()), "min_shape": float(q.min()),
                            "invalid_cells": 0, "width_failures": 0}
    mass, a, b, route = (mesh[name] for name in ("cell_mass_kg", "route_source",
                                                 "route_receiver", "route_mass_kg"))
    assert np.all(np.isfinite(mass) & (mass > 0)) and np.all(np.isfinite(route) & (route > 0))
    assert np.array_equal(ledger["type"][a], ledger["type"][b])
    assert np.array_equal(ledger["donor"][a], ledger["donor"][b])
    assert len(np.unique(np.column_stack((a, b)), axis=0)) == len(a)
    count = len(ledger["mass"])
    given = np.bincount(a, weights=route, minlength=count)
    received = np.bincount(b, weights=route, minlength=count)
    assigned = np.bincount(owner, weights=mass, minlength=count)
    np.testing.assert_allclose(given, ledger["mass"], rtol=0., atol=1.e-12)
    np.testing.assert_allclose(received, assigned, rtol=0., atol=1.e-12)
    child_centers = rest[cells].mean(axis=1)
    order = np.argsort(owner, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(owner, minlength=count))]
    source_original = original_rest[ledger["cells"]].mean(axis=1)
    longest = max(np.linalg.norm(child_centers[order[offsets[j]:offsets[j + 1]]] - source_original[i],
                                axis=1).max() for i, j in zip(a, b, strict=True))
    assert longest <= RADIUS_M
    density = 6. * mass / det
    density_by_tissue = {}
    for t, (low, high) in DENSITY_KG_M3.items():
        d = density[typed == t]
        assert d.min() >= low - 1.e-5 and d.max() <= high + 1.e-5
        density_by_tissue[t] = [float(d.min()), float(d.max())]
    print(json.dumps({"candidate": path.name, "sha256": receipt["candidate_sha256"],
                      "cells": len(cells), "support_nodes_unchanged": len(support),
                      "tissue_conditions": conditions, "mass_kg": float(mass.sum()),
                      "source_residual_kg": float(abs(given - ledger["mass"]).max()),
                      "receiver_residual_kg": float(abs(received - assigned).max()),
                      "maximum_actual_transport_m": float(longest),
                      "density_kg_m3": density_by_tissue}, indent=2, allow_nan=False))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("candidates", nargs="+", type=Path)
    args = parser.parse_args()
    source = HERE / "cavity-batch-037/local-deep-restoration-03"
    assert digest(source / "rebuild.npz") == json.loads((source / "receipt.json").read_text())["candidate_sha256"]
    with np.load(source / "rebuild.npz", allow_pickle=False) as f:
        baseline = dict(f)
    root = HERE / "compiled-reference"
    native = json.loads((root / "execution.json").read_text())
    sources = [root / f"shell-mass.csv{suffix}" for suffix in (".nodes.csv", ".tets.csv", "")]
    for path in sources:
        assert digest(path) == native[path.name]["sha256"]
    nodes, cells, masses = map(table, sources)
    points = np.array([[float(row[f"rest_{c}_m"]) for c in "xyz"] for row in nodes])
    ledger = {"type": np.array([int(row["material_index"]) for row in cells]),
              "donor": np.array([int(row["donor_body"]) for row in cells]),
              "cells": np.array([[int(row[f"parent_node{i}"]) for i in range(4)] for row in cells]),
              "mass": np.array([float(row["mass_kg"]) for row in masses])}
    assert np.array_equal([int(row["source_tet_row"]) for row in masses], np.arange(len(cells)))
    for path in args.candidates:
        verify(path, baseline, points, ledger)


if __name__ == "__main__":
    main()
