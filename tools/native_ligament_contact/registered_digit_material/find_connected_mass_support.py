"""Find face-connected conservative mass carriers for multiple observed material limits."""

import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from fit_cavity_mass import tet_faces, point_mass_features
from fit_dorsal_material import inputs
from material_context import sha
from torso_cell_geometry import geom


def route(x, y, t, old, new, mass):
    nodes = np.unique(t[old])
    total = mass[old].sum()
    coords = []
    for q in (x, y):
        loc = q[nodes]
        coords.append((loc - loc.mean(axis=0)) / np.ptp(loc, axis=0).max())
    a = point_mass_features(np.searchsorted(nodes, t[old]), coords, None)
    b = point_mass_features(np.searchsorted(nodes, new), coords, None)
    # Whiten the constraints on the *union* of old and new sample points.
    # Rigid components may cancel to roundoff between the two poses.
    u, s, _ = np.linalg.svd(np.c_[a, b], full_matrices=False)
    retain = s > s[0] * (64 * np.finfo(float).eps)
    pre = u[:, retain].T / s[retain, None]
    vol = geom(x, new)[0]
    lo = vol * 850 / total
    hi = vol * 950 / total
    if min(lo) <= 0:
        raise ValueError("Nonpositive candidate cell")
    orig = mass[old] / total
    rhs = a @ orig
    A = pre @ b
    # Most adipose cells are at the upper density bound. Solve for the
    # unused mass capacity so HiGHS does not subtract two nearly equal masses.
    sol = linprog(
        np.zeros(len(new)),
        A_eq=A,
        b_eq=pre @ (b @ hi - rhs),
        bounds=list(zip(np.zeros(len(new)), hi - lo)),
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-10,
            "dual_feasibility_tolerance": 1e-10,
        },
    )
    if not sol.success:
        return None
    v = (hi - sol.x) * total
    if (
        np.max(abs(b @ (v / total) - rhs)) > 1.0e-10
        or np.any(v / vol < 850 - 1e-6)
        or np.any(v / vol > 950 + 1e-6)
    ):
        return None
    return v


def search(x, y, t, tag, mass, core, new, ban, rng):
    mid = x[t[core]].mean(axis=(0, 1))
    r = np.linalg.norm(x[t].mean(axis=1) - mid, axis=1)
    allowed = tag == tag[core[0]]
    if len(ban):
        allowed &= ~np.isin(t, list(ban)).any(axis=1)
    if not allowed[core].all():
        return None, {"issue": "two proposals share physical vertices"}
    ids = np.flatnonzero(allowed)
    ids = ids[np.argpartition(r[ids], 2500)[:2500]]
    ids = np.unique(np.r_[ids, core])
    face = defaultdict(list)
    for i in ids:
        for f in tet_faces(t[i]):
            face[f].append(int(i))
    adj = {int(i): {j for f in tet_faces(t[i]) for j in face[f] if j != i} for i in ids}
    volumes = geom(x, t[ids])[0]
    merit = dict(zip(ids, np.log(np.maximum(volumes, 1e-20))))
    original = set(map(int, core))
    extra = []
    tests = 0

    def test(extra):
        nonlocal tests
        tests += 1
        oi = np.r_[core, extra].astype(int)
        nt = np.vstack([new, t[np.array(extra, dtype=int)]])
        return route(x, y, t, oi, nt, mass)

    def connected(extra):
        group = original | set(extra)
        seen = set(original)
        q = list(original)
        while q:
            v = q.pop()
            more = (adj[v] & group) - seen
            q.extend(more)
            seen.update(more)
        return len(seen) == len(group)

    largest = False
    sizes = []
    for trial in range(18):
        picked = set(original)
        front = set.union(*(adj[i] for i in original)) - picked
        extra = []
        ranking = {i: float(rng.gumbel()) for i in ids}
        while front and len(extra) < 185:
            if trial == 0:
                key = lambda i: -r[i]
            else:
                key = lambda i: (
                    merit[i]
                    - (1 + (trial % 5)) * r[i] / 0.006
                    + (0.25 + (trial % 4)) * ranking[i]
                )
            v = max(front, key=lambda i: (key(i), -i))
            front.remove(v)
            picked.add(v)
            extra.append(v)
            front.update(adj[v] - picked)
            if len(extra) not in {12, 20, 29, 52, 92, 155, 185}:
                continue
            solution = test(extra)
            if solution is None:
                continue
            largest = True
            sizes.append(len(extra) + 3)
            if len(extra) > 29:
                # Prune low-leverage carriers first. Removing an articulation
                # cell is allowed only if all remaining carriers stay connected.
                for n in sorted(extra, key=lambda i: merit[i]):
                    trial_extra = [k for k in extra if k != n]
                    if not connected(trial_extra):
                        continue
                    m = test(trial_extra)
                    if m is not None:
                        extra = trial_extra
                        solution = m
                    if len(extra) <= 29:
                        break
            if len(extra) <= 29:
                return (
                    np.r_[core, extra].astype(int),
                    np.vstack([new, t[np.array(extra, dtype=int)]]),
                    solution,
                ), dict(tests=tests, feasible_sizes=sizes)
            break
    return None, dict(tests=tests, larger_domain_feasible=largest, feasible_sizes=sizes)


def main(native, first, ledger, inventory, witnesses, prior, out):
    out.mkdir(exist_ok=False)
    x, y, t, tag, pins = inputs(native)
    x = x.astype(float)
    y = y.astype(float)
    flow = ledger / "current-density-closed-flow.npz"
    report = json.loads((ledger / "mass-transfer.json").read_text())
    if sha(flow) != report["flow_sha256"]:
        raise ValueError("Wrong source mass")
    mass = np.load(flow)["element_mass_kg"]
    inv = json.loads(inventory.read_text())
    start = json.loads(prior.read_text())
    if start["source"] != pins:
        raise ValueError("Wrong retained source")
    accepted = first / "physical-accepted.node"
    if sha(accepted) != start["accepted_coordinates_sha256"]:
        raise ValueError("Wrong matched pose")
    end = np.loadtxt(accepted, skiprows=1, usecols=(1, 2, 3))
    ban = set(start["proposal"]["physical_nodes"])
    rng = np.random.default_rng(51)
    for row in witnesses:
        rec = next(
            c for c in inv["proposals"] if c["witness"] == row and c["kind"] == "3-2"
        )
        found, diagnosis = search(
            x, y, t, tag, mass, np.array(rec["old"]), np.array(rec["new"]), ban, rng
        )
        prop = None
        if found:
            old, new, m = found
            nodes = np.unique(t[old])
            ban.update(nodes.tolist())
            delta = (
                point_mass_features(new, [], nodes) @ m
                - point_mass_features(t[old], [], nodes) @ mass[old]
            )
            prop = dict(
                retired_rows=old.tolist(),
                replacement_tets=new.tolist(),
                mass_kg=m.tolist(),
                total_mass_kg=float(m.sum()),
                nodal_l1_change_kg=float(np.sum(abs(delta))),
                first_moment_error_kg_m=[
                    float(max(abs(delta @ q[nodes]))) for q in (x, y, end)
                ],
                density_kg_m3=(m / geom(x, new)[0]).tolist(),
                physical_nodes=nodes.tolist(),
                witness=row,
            )
        result = dict(
            source=pins,
            proposal=prop,
            accepted_coordinates_sha256=sha(accepted),
            ledger_sha256=sha(flow),
            inventory_sha256=sha(inventory),
            constrained_pose_count=2,
            search=diagnosis,
        )
        (out / f"{row}.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main(
        *(Path(v).resolve() for v in sys.argv[1:5]),
        [int(i) for i in sys.argv[5].split(",")],
        *(Path(v).resolve() for v in sys.argv[6:]),
    )
