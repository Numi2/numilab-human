"""Screen closed edge stars against the installed and accepted material deformation."""

from functools import lru_cache
from pathlib import Path
from itertools import combinations
import importlib.util, json, sys
import numpy as np
from audit_cavity_response import elastic
from fit_posterior_equilibrium import materials
from fit_dorsal_material import inputs
from material_context import sha


@lru_cache(None)
def triangulate(ring):
    if len(ring) < 3:
        return ((),)
    out = []
    for i in range(1, len(ring) - 1):
        for a in triangulate(ring[: i + 1]):
            for b in triangulate(ring[i:]):
                out.append(a + b + ((ring[0], ring[i], ring[-1]),))
    return tuple(out)


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def screen(lib, run, first, witness, owner, out):
    exact = load_module(lib / "fixed_boundary_volume_flip_198_root.py")
    cavity = load_module(lib / "local_fixed_boundary_cavity_repair_625.py")
    r, y, t, role, pins = inputs(run)
    x = r.astype(float)
    y = y.astype(float)
    p = first / "physical-accepted.node"
    if (
        sha(p)
        != json.loads((first / "physical-recovery.json").read_text())[
            "verified_member_sha256"
        ]["accepted.node"]
    ):
        raise ValueError("Changed matched pose")
    z = np.loadtxt(p, skiprows=1, usecols=(1, 2, 3))
    props, materials_pin = materials(owner)
    observed = json.loads(witness.read_text())
    ids = sorted(
        {
            v["parents"][0]
            for v in observed["material_witnesses"]
            if v["exactly_retained"]
        }
    )
    needed = set(t[ids].ravel())
    table = {}
    subset = np.flatnonzero(np.isin(t, list(needed)).any(axis=1))
    for row in subset:
        for v in t[row]:
            if v in needed:
                table.setdefault(int(v), set()).add(int(row))
    # Each new diagonal joins ring vertices already present in this one-ring.
    # Cache the global incident rows at those vertices once per invocation.
    ring_nodes = np.unique(t[subset])
    extra = np.flatnonzero(np.isin(t, ring_nodes).any(axis=1))
    incidence = {int(n): set() for n in ring_nodes}
    for row in extra:
        for v in t[row]:
            if v in incidence:
                incidence[v].add(int(row))
    filt = {}
    proposals = []
    attempts = []
    poly = {}
    for row in ids:
        for edge in combinations(sorted(t[row]), 2):
            oldr = np.array(sorted(table[edge[0]] & table[edge[1]]))
            old = t[oldr]
            n = len(oldr)
            if n < 3 or n > 8 or len(np.unique(role[oldr])) != 1:
                continue
            ring = cavity._ring(old, edge)
            if not ring:
                continue
            ring = tuple(ring)
            nodes = np.unique(old)
            minor = np.searchsorted(nodes, old)
            mu, K = props[int(role[row]) - 15]
            Es = [elastic(x, Y, old, nodes, mu, K)[0] for Y in (y, z)]
            key = (row, n)
            poly[key] = poly.get(key, 0) + 1
            for triangles in triangulate(ring):
                tet = exact.orient(
                    r, np.array([[*face, v] for face in triangles for v in edge])
                )
                small = np.searchsorted(nodes, tet)
                ok, why = exact.certify(r[nodes], minor, small)
                if not ok:
                    filt[why] = filt.get(why, 0) + 1
                    continue
                # Forbid any pre-existing separator or inner diagonal outside the complete old star.
                oldedges = {tuple(sorted(k)) for tt in old for k in combinations(tt, 2)}
                newedges = {tuple(sorted(k)) for tt in tet for k in combinations(tt, 2)}
                oldfaces = {tuple(sorted(k)) for tt in old for k in combinations(tt, 3)}
                newfaces = {tuple(sorted(k)) for tt in tet for k in combinations(tt, 3)}
                if any(
                    set.intersection(*(incidence[n] for n in item))
                    for item in (newedges - oldedges) | (newfaces - oldfaces)
                ):
                    filt["existing separator"] = filt.get("existing separator", 0) + 1
                    continue
                results = [exact.measurements(Q, tet) for Q in (r, y, z)]
                if any(not np.all(exact.passes(D, Q)) for D, Q in results):
                    continue
                before = float(exact.measurements(r, old)[1].min())
                after = float(results[0][1].min())
                if after < 1.02 * before:
                    continue
                E = []
                errors = []
                force = []
                minJ = 5.0
                maxJ = 0.0
                stretch = True
                for k, Y in enumerate((y, z)):
                    e, f, J = elastic(x, Y, tet, nodes, mu, K)
                    _, F, _ = elastic(x, Y, old, nodes, mu, K)
                    X = x[tet[:, 1:]] - x[tet[:, :1]]
                    U = Y[tet[:, 1:]] - Y[tet[:, :1]]
                    sv = np.linalg.svd(np.linalg.solve(X, U), compute_uv=False)
                    stretch &= sv.min() >= 2 / 3 and sv.max() <= 1.5
                    E.append(e)
                    force.append(float(np.linalg.norm(F - f)))
                    errors.append(abs(e - Es[k]) / Es[k])
                    minJ = min(minJ, float(J.min()))
                    maxJ = max(maxJ, float(J.max()))
                entry = dict(
                    witness=int(row),
                    kind=f"{len(old)}-{len(tet)}",
                    material=int(role[row]),
                    old=oldr.tolist(),
                    new=tet.tolist(),
                    reference_min_shape_before=before,
                    reference_min_shape_after=after,
                    min_J=minJ,
                    max_J=maxJ,
                    energy_relative_error=errors,
                    local_nodal_force_L2_difference_N=force,
                    stretch_gate=bool(stretch),
                )
                attempts.append(entry)
                if stretch and minJ >= 0.4 and maxJ <= 2.5 and max(errors) <= 0.05:
                    proposals.append(entry)
    out.write_text(
        json.dumps(
            dict(
                source=pins,
                accepted=sha(p),
                native_material=materials_pin,
                reference_modules={
                    n: sha(lib / n)
                    for n in (
                        "fixed_boundary_volume_flip_198_root.py",
                        "local_fixed_boundary_cavity_repair_625.py",
                    )
                },
                edge_stars=[
                    {"row": int(k[0]), "valence": k[1], "count": v}
                    for k, v in poly.items()
                ],
                rejection=filt,
                proposals=proposals,
                geometry_candidates=attempts,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    screen(*(Path(s).resolve() for s in sys.argv[1:]))
