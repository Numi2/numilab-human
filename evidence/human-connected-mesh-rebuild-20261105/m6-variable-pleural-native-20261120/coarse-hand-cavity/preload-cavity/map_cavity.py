"""Transport every typed source contribution through a fixed-boundary tetrahedral remesh."""
from argparse import ArgumentParser
from hashlib import file_digest
from itertools import combinations
from pathlib import Path
import json

import numpy as np
from scipy.spatial import ConvexHull

from cavity import determinant, oriented_boundary


def digest(p):
    with p.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def barycentric_planes(vertices):
    transform = np.r_[vertices.T, np.ones((1, 4))]
    return np.linalg.inv(transform)


def overlap(a, b):
    planes = np.r_[barycentric_planes(a), barycentric_planes(b)]
    triples = planes[np.array(list(combinations(range(8), 3)))]
    active = np.abs(np.linalg.det(triples[:, :, :3])) > 1.e-11
    systems = triples[active]
    points = np.linalg.solve(systems[:, :, :3], -systems[:, :, 3, None])[:, :, 0]
    interior = np.all(points @ planes[:, :3].T + planes[:, 3] >= -2.e-10, axis=1)
    points = np.unique(np.round(points[interior], 13), axis=0)
    if len(points) < 4 or np.linalg.matrix_rank(points[1:] - points[0], tol=1.e-10) < 3:
        return 0.
    return ConvexHull(points).volume


def transfer(rest, previous, replacement):
    vertices = rest[np.unique(previous)]
    center = vertices.mean(axis=0)
    scale = float(np.ptp(vertices, axis=0).max())
    local = (rest - center) / scale
    remap = np.array([[overlap(local[old], local[new]) for new in replacement] for old in previous])
    before = determinant(local, previous) / 6.
    after = determinant(local, replacement) / 6.
    np.testing.assert_allclose(remap.sum(axis=1), before, rtol=2.e-8, atol=1.e-13)
    np.testing.assert_allclose(remap.sum(axis=0), after, rtol=2.e-8, atol=1.e-13)
    return remap / remap.sum(axis=1)[:, None]


def remap_anatomy(old, rest, previous, replacement, retired):
    transition = transfer(rest, previous, replacement)
    src, dest, mass = (old[k] for k in ("parent_cell", "target_cell", "mass_kg"))
    selected = np.isin(dest, retired)
    local = np.searchsorted(retired, dest[selected])
    split = mass[selected, None] * transition[local]
    nz, col = np.nonzero(split)
    combined = {
        "parent_cell": np.r_[src[~selected], src[selected][nz]],
        "target_cell": np.r_[dest[~selected], retired[col]],
        "mass_kg": np.r_[mass[~selected], split[nz, col]],
    }
    prior_source = np.bincount(src, weights=mass)
    actual_source = np.bincount(combined["parent_cell"], weights=combined["mass_kg"])
    np.testing.assert_allclose(prior_source, actual_source, rtol=1.e-12, atol=1.e-16)
    return combined, transition, float(np.abs(prior_source - actual_source).max())


def main():
    p = ArgumentParser(description=__doc__)
    for name in ("fit", "mass", "screen", "balance", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    screen = json.loads(a.screen.read_text())
    pick = min((r for r in screen if "replacement" in r), key=lambda r: (len(r["retired"]), r["retired"]))
    retired = np.array(pick["retired"])
    replacement = np.array(pick["replacement"])
    with np.load(a.fit) as fit, np.load(a.mass) as old:
        cells = fit["cells"].copy()
        previous = cells[retired]
        assert len(replacement) == len(retired)
        assert oriented_boundary(previous) == oriented_boundary(replacement)
        world, rest = fit["nodes_current"], fit["nodes_reference"]
        assert np.all(determinant(rest, replacement) > 0.)
        assert np.all(determinant(world.astype(float), replacement) > 0.)
        raw, transition, error = remap_anatomy(old, rest, previous, replacement, retired)
        cells[retired] = replacement
        cell_mass = np.bincount(raw["target_cell"], weights=raw["mass_kg"], minlength=len(cells))
        node_mass = np.bincount(cells.ravel(), weights=np.repeat(cell_mass/4., 4), minlength=len(world))
        old_mass = old["node_mass_kg"]
        np.testing.assert_allclose(node_mass.sum(), old_mass.sum(), rtol=0., atol=1.e-12)
        affected = np.unique(previous)
        unchanged = np.setdiff1d(np.arange(len(world)), affected)
        np.testing.assert_allclose(node_mass[unchanged], old_mass[unchanged], rtol=2.e-12, atol=1.e-15)
        assert np.all(node_mass > 0.) and np.all(cell_mass > 0.)
        raw["cell_mass_kg"], raw["node_mass_kg"] = cell_mass, node_mass
        old_density = old["cell_mass_kg"][retired] * 6. / determinant(rest, previous)
        new_density = cell_mass[retired] * 6. / determinant(rest, replacement)
        assert new_density.min() >= old_density.min() * (1. - 2.e-8)
        assert new_density.max() <= old_density.max() * (1. + 2.e-8)
        balance = json.loads(a.balance.read_text())
        center_delta = ((node_mass-old_mass) @ world.astype(float)) / balance["mass_kg"]
        balance["com_m"] = (np.array(balance["com_m"]) + center_delta).tolist()
        # First-touch geometric points are fixed, but their moment coefficients
        # must use the center of mass of the conservatively remapped stencil.
        support = balance["possible_normal_distribution_at_threshold"]
        indices = np.array([row["outer_shell_row"] for row in support])
        relative = world[indices, :2].astype(float) - np.array(balance["com_m"])[:2]
        loads = np.linalg.solve(np.c_[np.ones(3), relative[:, 1], -relative[:, 0]].T,
                                np.array([balance["mass_kg"] * 9.81, 0., 0.]))
        assert min(loads) > 0.
        for row, load in zip(support, loads, strict=True):
            row["normal_force_n"] = float(load)
        a.output.mkdir()
        np.savez_compressed(a.output / "posterior-interface-fit.npz", nodes_current=world, nodes_reference=rest,
                            cells=cells, original_cells=fit["cells"], changed_rows=retired)
        np.savez_compressed(a.output / "balanced-mass.npz", **raw)
        (a.output / "support-hull-01.json").write_text(json.dumps(balance, indent=2) + "\n")
        receipt = {"source_hashes": {f.name: digest(f) for f in (a.fit, a.mass, a.screen, a.balance)},
                   "retired": retired.tolist(), "replacement": replacement.tolist(),
                   "overlap_fraction": transition.tolist(), "source_mass_max_error_kg": error,
                   "mass_kg": float(node_mass.sum()), "maximum_nodal_mass_change_kg": float(abs(node_mass-old_mass).max()),
                   "com_delta_m": center_delta.tolist(), "prior_density_kg_per_m3": old_density.tolist(),
                   "new_density_kg_per_m3": new_density.tolist()}
        (a.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(receipt)


if __name__ == "__main__":
    main()
