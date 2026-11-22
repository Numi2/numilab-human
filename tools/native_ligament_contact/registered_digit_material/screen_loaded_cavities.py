"""Screen the observed mechanical and mesh failures with the existing exact cavity owner."""

from itertools import combinations
from pathlib import Path
import json
import sys
import numpy as np
from scipy.optimize import linprog
from bind_posterior_equilibrium import coordinates
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha
from torso_cell_geometry import geom


def screen(run, parent, candidate, witness, lib, out):
    sys.path.insert(0, str(lib))
    sys.path.insert(0, str(lib / "bounded625-local-cavity-repair"))
    import fixed_boundary_volume_flip_198_root as owner
    import local_fixed_boundary_cavity_repair_625 as cavity

    r, initial, t, roles, identity = inputs(run)
    x = r.astype(float)
    y = initial.astype(float)
    z = coordinates(run / "physical-accepted.node")
    a = json.loads(witness.read_text())
    info = json.loads((parent / "loaded-cell-limits.json").read_text())
    if sha(run / "physical-accepted.node") != a["accepted"] or a["source"] != identity:
        raise ValueError("Witness source drift")
    rows = sorted(set(k["source_row"] for k in info["rows"] + a["rows"]))
    t = t.astype(np.int64)
    index = cavity.build_node_cell_index(t, len(x))
    mpath = parent / "fine-mass-routed/current-density-closed-flow.npz"
    mass = np.load(mpath)
    element = mass["element_mass_kg"]
    group = mass["group_for_cell"]
    props, pins = materials(candidate / "native-owner")
    labels = {}
    for row in rows:
        labels[row] = (
            "shape" if any(f["source_row"] == row for f in a["rows"]) else "determinant"
        )

    def metrics(cells, pos):
        v, q = geom(pos, cells)
        return v, q

    def strain(cells, cur, mu, K):
        X = x[cells]
        Y = cur[cells]
        T = np.linalg.solve(X[:, 1:] - X[:, :1], Y[:, 1:] - Y[:, :1])
        det = np.linalg.det(T)
        sv = np.linalg.svd(T, compute_uv=False)
        if min(det) <= 0:
            return det, sv, np.inf
        logj = np.log(det)
        V = geom(x, cells)[0]
        e = (
            0.5 * mu * (np.sum(T * T, axis=(1, 2)) - 3 - 2 * logj) + 0.5 * K * logj**2
        ) @ V
        return det, sv, float(e)

    proposals = []
    excluded = {}

    def reject(reason):
        excluded[reason] = excluded.get(reason, 0) + 1

    for row in rows:
        neighborhoods = []
        for corners, kinds in ((3, {2: "2-3"}), (2, {3: "3-2", 4: "4-4"})):
            for key in combinations(t[row], corners):
                old = cavity._incident(key, index, len(x), len(t))
                if len(old) in kinds:
                    neighborhoods.append((old, kinds[len(old)]))
        for old, kind in neighborhoods:
            if len(set(roles[old])) != 1:
                reject("material boundary")
                continue
            try:
                variants = cavity.propose(owner, r, t, old, kind, index)
            except ValueError as exc:
                reject(str(exc))
                continue
            for opt in variants:
                new = opt["new_tets"]
                before = t[old]
                check, reason = owner.certify(r, before, new)
                if not check:
                    reject("exact source " + reason)
                    continue
                if min(geom(z, new)[0]) <= 0:
                    reject("motion inversion")
                    continue
                m, q = zip(*(metrics(new, P) for P in (x, y, z)))
                if min(v.min() for v in m) <= 1e-18 or min(v.min() for v in q) < 1e-5:
                    reject("source, load or observed motion shape")
                    continue
                mu, K = props[roles[row] - 15]
                J, S, E0 = strain(new, y, mu, K)
                J1, _, E1 = strain(new, z, mu, K)
                if (
                    S.min() < 2 / 3
                    or S.max() > 1.5
                    or J.min() < 0.4
                    or J.max() > 2.5
                    or J1.min() <= 0.35
                    or J1.max() >= 2.5
                ):
                    reject("registered or observed strain bounds")
                    continue
                # Raw mesh can move inside an otherwise fixed closed cavity. Re-evaluate
                # material energy and the possible mass/first moment transfer before admission.
                total = float(element[old].sum())
                src_center = x[before].mean(axis=1)
                dst_center = x[new].mean(axis=1)
                if total <= 0:
                    reject("missing source mass")
                    continue
                origin = src_center[0]
                scale = max(np.linalg.norm(src_center - origin, axis=1).max(), 1e-9)
                A = np.vstack([np.ones(len(new)), ((dst_center - origin) / scale).T])
                b = np.r_[1.0, (element[old] / total) @ ((src_center - origin) / scale)]
                prob = linprog(
                    np.zeros(len(new)), A_eq=A, b_eq=b, bounds=(0, None), method="highs"
                )
                feasible = bool(prob.success and np.max(np.abs(A @ prob.x - b)) < 1e-7)
                previous = [strain(before, pos, mu, K)[2] for pos in (y, z)]
                error = [
                    abs(u - v) / max(1e-20, abs(u)) for u, v in zip(previous, (E0, E1))
                ]
                proposals.append(
                    dict(
                        witness=int(row),
                        type=labels[row],
                        kind=kind,
                        old=old.tolist(),
                        new=new.tolist(),
                        material=int(roles[row]),
                        groups=np.unique(group[old]).tolist(),
                        observed_min_J=float(J1.min()),
                        observed_max_J=float(J1.max()),
                        minimum_shape=float(min(v.min() for v in q)),
                        energy_relative_error=error,
                        energy_absolute_change_J=[
                            float(v - u) for u, v in zip(previous, (E0, E1))
                        ],
                        preserves_local_mass_and_first_moment=feasible,
                        whole_cavity_mass_kg=total,
                    )
                )
    result = dict(
        source=identity,
        accepted=a["accepted"],
        old_mass=sha(mpath),
        observed_rows=rows,
        eligible_geometry=len(proposals),
        excluded=excluded,
        proposals=proposals,
        interpretation="Fixed-boundary geometries evaluated at the observed native coordinates. Mass feasibility is an aggregate test. Source carrier assignment and recomputed native motion are required for promotion.",
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(len(proposals), excluded)
    for row in rows:
        found = [r for r in proposals if r["witness"] == row]
        trusted = [
            r
            for r in found
            if max(r["energy_relative_error"]) <= 0.05
            and r["preserves_local_mass_and_first_moment"]
        ]
        print(row, len(found), len(trusted))


if __name__ == "__main__":
    screen(*(Path(p).resolve() for p in sys.argv[1:]))
