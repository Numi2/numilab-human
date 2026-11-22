"""Conserve the measured material mass and its spatial moments across a witnessed local retetrahedralization."""

import json, itertools
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from fit_dorsal_material import inputs
from material_context import sha
from torso_cell_geometry import geom


def tet_faces(t):
    return [tuple(sorted(q)) for q in itertools.combinations(t, 3)]


def point_mass_features(cells, poses, nodal):
    if nodal is not None:
        return np.stack([(cells == n).sum(axis=1) for n in nodal]) * 0.25
    rows = [np.ones(len(cells))]
    for points in poses:
        center = points[cells].mean(axis=1)
        rows.extend(center.T)
    return np.asarray(rows)


def fit_mass(ref, poses, old, new, mass, bounds, all_nodes):
    # Work in units of kg per total cavity mass. Using normalized local
    # geometry avoids cancellation of small moments around a translated body.
    total = mass.sum()
    nodes = np.unique(old)
    centered = [
        (q - q[nodes].mean(axis=0)) / np.ptp(q[nodes], axis=0).max() for q in poses
    ]
    Ao = point_mass_features(old, centered, nodes if all_nodes else None)
    An = point_mass_features(new, centered, nodes if all_nodes else None)
    target = Ao @ (mass / total)
    vol = geom(ref, new)[0]
    lo = bounds[0] * vol / total
    hi = bounds[1] * vol / total
    if np.any(lo <= 0):
        raise ValueError("Invalid replacement geometry")
    ideal = vol / vol.sum()
    n = len(new)
    I = np.eye(n)
    # L1 departure from constant density bounds redistribution without
    # changing any material density limit or hard moment constraint.
    sol = linprog(
        np.r_[np.zeros(n), np.ones(n)],
        A_eq=np.c_[An, np.zeros_like(An)],
        b_eq=target,
        A_ub=np.block([[I, -I], [-I, -I]]),
        b_ub=np.r_[ideal, -ideal],
        bounds=[*zip(lo, hi), *[(0, None)] * n],
        method="highs",
        options={
            "primal_feasibility_tolerance": 1.0e-10,
            "dual_feasibility_tolerance": 1.0e-10,
        },
    )
    if not sol.success:
        return None
    m = sol.x[:n] * total
    if np.max(abs(An @ (m / total) - target)) > 1.1e-10 or not np.all(
        (m / vol >= bounds[0] - 1e-6) & (m / vol <= bounds[1] + 1e-6)
    ):
        raise ValueError("Uncertified constrained density route")
    return m


def search(
    native, first, ledger, inventory, out, pose_count, witness=1064386, kind="3-2"
):
    ref, y, t, tag, pins = inputs(native)
    ref = ref.astype(float)
    y = y.astype(float)
    if (
        sha(first / "physical-accepted.node")
        != json.loads((first / "physical-recovery.json").read_text())[
            "verified_member_sha256"
        ]["accepted.node"]
    ):
        raise ValueError("Stale accepted material")
    end = np.loadtxt(first / "physical-accepted.node", skiprows=1, usecols=(1, 2, 3))
    src = json.loads(inventory.read_text())
    entry = next(
        p for p in src["proposals"] if p["witness"] == witness and p["kind"] == kind
    )
    archive = ledger / "current-density-closed-flow.npz"
    if (
        sha(archive)
        != json.loads((ledger / "mass-transfer.json").read_text())["flow_sha256"]
    ):
        raise ValueError("Changed registered mass")
    mass = np.load(archive)["element_mass_kg"]
    core = np.array(entry["old"])
    new = np.array(entry["new"])
    if np.any(tag[core] != 16):
        raise ValueError("Incorrect tissue identity")
    vol = geom(ref, t[core])[0]
    density = mass[core] / vol
    out.mkdir(exist_ok=False)
    # Collect one-ring cells sharing any vertex of the changed bipyramid.
    # Connectivity growth always crosses a full face. The native owner will
    # independently validate exact oriented boundary, region and all ancestry.
    ids = np.flatnonzero((tag == 16) & np.isin(t, np.unique(t[core])).any(axis=1))
    faces = defaultdict(list)
    for i in ids:
        for f in tet_faces(t[i]):
            faces[f].append(i)
    seed = list(map(int, core))
    seen = set(seed)
    front = seed.copy()
    order = []
    while front and len(order) < 120:
        nxt = {
            j for i in front for f in tet_faces(t[i]) for j in faces[f] if j not in seen
        }
        front = sorted(
            nxt,
            key=lambda j: np.linalg.norm(ref[t[j]].mean(0) - ref[t[core[0]]].mean(0)),
        )
        order += front
        seen.update(front)
    supports = [0, 3, 6, 12, 20, 29]
    report = []
    solution = None
    for count in supports:
        extra = np.array(order[:count], dtype=int)
        oi = np.r_[core, extra]
        nt = np.vstack([new, t[extra]])
        for nodal in (True, False):
            m = fit_mass(
                ref,
                [ref, y, end][:pose_count],
                t[oi],
                nt,
                mass[oi],
                (850.0, 950.0),
                nodal,
            )
            report.append(
                dict(extra=len(extra), full_nodal_closure=nodal, feasible=m is not None)
            )
            if m is None:
                continue
            nodes = np.unique(t[oi])
            before = point_mass_features(t[oi], [], nodes) @ mass[oi]
            after = point_mass_features(nt, [], nodes) @ m
            inertia = []
            for coords in (ref, y, end):
                X = coords[nodes] - coords[nodes].mean(0)
                inertia.append(
                    float(
                        np.linalg.norm(np.einsum("i,ij,ik->jk", after - before, X, X))
                    )
                )
            solution = dict(
                retired_rows=oi.tolist(),
                replacement_tets=nt.tolist(),
                mass_kg=m.tolist(),
                density_kg_m3=(m / geom(ref, nt)[0]).tolist(),
                total_mass_kg=float(m.sum()),
                nodal_l1_change_kg=float(np.abs(after - before).sum()),
                first_moment_error_kg_m=[
                    np.max(abs((after - before) @ s[nodes])) for s in (ref, y, end)
                ],
                second_moment_tensor_change_kg_m2=inertia,
                full_nodal_closure=nodal,
                physical_nodes=nodes.tolist(),
                witness=entry["witness"],
            )
            if nodal:
                break
        if solution and solution["full_nodal_closure"]:
            break
        if solution and count >= 12:
            break
    result = dict(
        constrained_pose_count=pose_count,
        proposal=solution,
        attempts=report,
        old_density_kg_m3=density.tolist(),
        source=pins,
        ledger_sha256=sha(archive),
        inventory_sha256=sha(inventory),
        accepted_coordinates_sha256=sha(first / "physical-accepted.node"),
        scope="Feasibility under bounded P1 element densities. The count selects reference, then installed, then accepted coordinate sets for moment conservation. Source-carrier radius, native role labels, forces and dynamic acceptance remain separate gates.",
    )
    (out / "fit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    for arg in ("native", "first", "ledger", "inventory", "out"):
        parser.add_argument(arg, type=lambda v: Path(v).resolve())
    parser.add_argument("pose_count", type=int, choices=(1, 2, 3))
    parser.add_argument("--witness", type=int, default=1064386)
    parser.add_argument("--kind", choices=("3-2", "4-4"), default="3-2")
    search(**vars(parser.parse_args()))
