"""Independently verify saved source routes against every receiving child."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

import numpy as np

from rebuild import HERE, ROLE_COUNT, AUDIT, determinant, digest, table
from regional_mass import DENSITY_KG_M3, RADIUS_M, RegionalMass


def verify(candidate, base_points, world_points, base_cells, ledger, expected_masses):
    receipt = json.loads((candidate / "receipt.json").read_text())
    path = candidate / "rebuild.npz"
    assert digest(path) == receipt["candidate_sha256"]
    data = dict(np.load(path, allow_pickle=False))
    rest, world = data["rest"], data["world"].astype(np.float64)
    cells, parent = data["elements"], data["parent"]
    reference_mass, tissue, donor = ledger.source_kg, ledger.material, ledger.donor
    assert len(reference_mass) == len(base_cells) == len(expected_masses)
    assert np.array_equal(reference_mass, expected_masses)
    assert np.isfinite(rest).all() and np.isfinite(world).all()
    assert np.all((cells >= 0) & (cells < len(rest)))
    assert np.all((parent >= 0) & (parent < len(reference_mass)))
    assert np.array_equal(tissue[parent], data["material"])
    assert np.array_equal(donor[parent], data["donor"])
    assert len(np.unique(cells, axis=0)) == len(cells)
    det, world_det = determinant(rest, cells), determinant(world, cells)
    assert min(det.min(), world_det.min()) > 0.
    mass, source, destination, kilograms = (data[key] for key in
        ("cell_mass_kg", "route_source", "route_receiver", "route_mass_kg"))
    assert np.isfinite(mass).all() and np.all(mass > 0.)
    assert np.isfinite(kilograms).all() and np.all(kilograms > 0.)
    for indices in (source, destination):
        assert np.all((indices >= 0) & (indices < len(reference_mass)))
    assert len(np.unique(np.stack((source, destination), axis=1), axis=0)) == len(kilograms)
    assert np.array_equal(tissue[source], tissue[destination])
    assert np.array_equal(donor[source], donor[destination])
    given = np.bincount(source, weights=kilograms, minlength=len(reference_mass))
    received = np.bincount(destination, weights=kilograms, minlength=len(reference_mass))
    child_mass = np.bincount(parent, weights=mass, minlength=len(reference_mass))
    np.testing.assert_allclose(given, reference_mass, atol=1.e-12, rtol=0.)
    np.testing.assert_allclose(received, child_mass, atol=1.e-12, rtol=0.)

    centers = rest[cells].mean(axis=1)
    ordered = np.argsort(parent, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(parent, minlength=len(reference_mass)))]
    original = base_points[base_cells].mean(axis=1)
    route_lengths = np.empty(len(kilograms))
    for i, (a, b) in enumerate(zip(source, destination, strict=True)):
        children = ordered[offsets[b]:offsets[b + 1]]
        assert len(children) > 0
        route_lengths[i] = np.linalg.norm(centers[children] - original[a], axis=1).max()
    assert route_lengths.max() <= RADIUS_M
    density = 6. * mass / det
    assert np.all(np.isfinite(density))
    per_material = {}
    for t, (low, high) in DENSITY_KG_M3.items():
        rows = density[data["material"] == t]
        assert np.all((rows >= low - 1.e-5) & (rows <= high + 1.e-5))
        per_material[str(t)] = [float(rows.min()), float(rows.max())]

    changed = np.flatnonzero(np.any(rest[:len(base_points)] != base_points, axis=1))
    unchanged = np.setdiff1d(np.arange(len(base_points)), changed)
    support = np.flatnonzero(world_points[:ROLE_COUNT, 2] <= 1.e-5)
    assert np.array_equal(world[support], world_points[support])
    # Finite precision can round a rest move out of the emitted geometry.
    assert np.array_equal(world[unchanged], world_points[unchanged])

    added = len(rest) - len(base_points)
    assert added % 3 == 0
    cols = np.r_[np.arange(3 * ROLE_COUNT).reshape(3, ROLE_COUNT).T,
                 np.arange(len(base_points), len(rest)).reshape(-1, 3)]
    old_world = np.concatenate((world_points,
        world_points[data["edge_parent"]].mean(axis=1).astype(np.float32).astype(np.float64)))
    old_rest = np.concatenate((base_points, base_points[data["edge_parent"]].mean(axis=1)))
    thickness = {}
    for key, old, current in (("rest", old_rest, rest), ("world", old_world, world)):
        prev = np.linalg.norm(np.diff(old[cols], axis=1), axis=2)
        now = np.linalg.norm(np.diff(current[cols], axis=1), axis=2)
        rows = np.any(old[cols] != current[cols], axis=(1, 2))
        names = ("skin", "fat")
        thickness[key] = {}
        for layer, name in enumerate(names):
            width, before = now[rows, layer], prev[rows, layer]
            thickness[key][name] = {
                "min_mm": float(width.min() * 1000.),
                "max_mm": float(width.max() * 1000.),
                "below_0p5_mm_before": int(np.count_nonzero(before < .0005)),
                "below_0p5_mm_after": int(np.count_nonzero(width < .0005)),
                "newly_below_0p5_mm": int(np.count_nonzero((before >= .0005) & (width < .0005))),
            }
    return {
        "candidate": candidate.name, "sha256": receipt["candidate_sha256"],
        "source_rows": len(given), "total_mass_kg": float(mass.sum()),
        "typed_mass_balance": {str(t): float(mass[data["material"] == t].sum() - reference_mass[tissue == t].sum())
                               for t in DENSITY_KG_M3},
        "maximum_source_residual_kg": float(np.abs(given - reference_mass).max()),
        "maximum_receiving_residual_kg": float(np.abs(received - child_mass).max()),
        "independent_maximum_child_route_m": float(route_lengths.max()),
        "density_limits_observed_kg_m3": per_material,
        "invalid_rest_tets": int(np.count_nonzero(det <= 0.)),
        "invalid_float32_world_tets": int(np.count_nonzero(world_det <= 0.)),
        "support_vertices_unchanged": len(support),
        "affected_existing_vertices": len(changed),
        "affected_euclidean_thickness": thickness,
        "new_external_pairs": len(receipt["new_outside_pairs"]),
        "new_internal_pairs": len(receipt["new_self_pairs"]),
        "new_skull_pairs": len(receipt["new_skull_pairs"]),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("candidates", metavar="CANDIDATE", type=Path, nargs="+")
    args = parser.parse_args()
    reference = HERE / "compiled-reference"
    receipt = json.loads((reference / "execution.json").read_text())
    points = reference / "shell-mass.csv.nodes.csv"
    tets = reference / "shell-mass.csv.tets.csv"
    masses = reference / "shell-mass.csv"
    for path in (points, tets, masses):
        assert digest(path) == receipt[path.name]["sha256"]
    rows = table(tets)
    pts = table(points)
    base_cells = np.array([[int(row[f"parent_node{i}"]) for i in range(4)] for row in rows])
    rest_points, world_points = (np.array([[float(row[f"{frame}_{axis}_m"]) for axis in "xyz"] for row in pts])
                                for frame in ("rest", "current"))
    ledger = RegionalMass.from_original(rest_points, base_cells, rows, AUDIT / "build-commands.json")
    values = table(masses)
    assert np.array_equal([int(row["source_tet_row"]) for row in values], np.arange(len(rows)))
    assert np.array_equal([int(row["material_index"]) for row in values], ledger.material)
    expected = np.array([float(row["mass_kg"]) for row in values])
    labels = table(reference / "shell-mass.csv.labels.csv")
    assert len(labels) == len(rows)
    assert np.array_equal([int(row["geometric_donor_body"]) for row in labels], ledger.donor)
    source = table(reference / "shell-mass.csv.source.csv")
    nodal_mass = Counter()
    for row in source:
        nodal_mass[int(row["material_index"])] += float(row["mass_kg"])
    label_summary = {
        "source_nodal_rows": len(source), "native_ancestry_labels_sha256": digest(reference / "shell-mass.csv.labels.csv"),
        "nodal_source_mass_kg": dict(nodal_mass),
        "tet_source_mass_kg": {int(t): float(expected[ledger.material == t].sum()) for t in DENSITY_KG_M3},
    }
    print(json.dumps({"source": label_summary, "candidates": [
        verify(candidate, rest_points, world_points, base_cells, ledger, expected) for candidate in args.candidates
    ]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
