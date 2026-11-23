"""Test a local internal-node tetrahedralization using the existing constitutive energy and condensed boundary forces."""

import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from audit_cavity_response import elastic
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha
from screen_layer_spanning_cavities import oriented_faces, quality
from screen_remaining_stars import load_module
from supported_material_admission import installed_topology


def interior_equilibrium(x, y, tet, mu, bulk):
    """Minimize the exact energy with respect to the single new interior point."""
    ref = x[tet[:, 1:]] - x[tet[:, :1]]
    inv = np.linalg.inv(ref)
    vol = np.linalg.det(ref) / 6
    if np.any(vol <= 0):
        raise ValueError("Nonpositive reference")
    h = -inv.sum(2)
    y = y.copy()
    for iteration in range(30):
        T = inv @ (y[tet[:, 1:]] - y[tet[:, :1]])
        J = np.linalg.det(T)
        if np.any(J <= 0):
            return None
        log = np.log(J)
        invT = np.linalg.inv(T)
        v = np.einsum("nij,nj->ni", invT, h)
        g = (
            vol[:, None]
            * (mu * np.einsum("nji,nj->ni", T, h) + (bulk * log - mu)[:, None] * v)
        ).sum(0)
        if np.linalg.norm(g) < 1e-9:
            return y, float(np.linalg.norm(g))
        H = mu * (vol @ np.sum(h * h, axis=1)) * np.eye(3) + np.einsum(
            "n,ni,nj->ij", vol * (bulk + mu - bulk * log), v, v
        )
        d = np.linalg.solve(H, g)

        # Monitor energy and inversion for every local trial, including all
        # tetrahedra of the candidate cavity.
        def energy(T, J):
            return vol @ (
                0.5 * mu * (np.sum(T * T, axis=(1, 2)) - 3 - 2 * np.log(J))
                + 0.5 * bulk * np.log(J) ** 2
            )

        old = energy(T, J)
        for k in range(20):
            step = 2.0**-k
            A = T - step * h[:, :, None] * d[None, None, :]
            j = np.linalg.det(A)
            if np.all(j > 0) and energy(A, j) <= old + max(1e-19, abs(old) * 1e-13):
                y[-1] -= step * d
                break
        else:
            return None
    return None


def candidates(x, old):
    faces = defaultdict(list)
    for t in old:
        for f in oriented_faces(t):
            faces[tuple(sorted(f))].append(f)
    if any(len(v) > 2 for v in faces.values()):
        return
    wall = np.asarray([v[0] for v in faces.values() if len(v) == 1])
    if set(wall.ravel()) != set(old.ravel()):
        return
    # Volume sign determines outward orientation of the existing boundary.
    probe = x[old[0]].mean(0)
    test = (
        next(i for i in range(len(wall)) if set(wall[i]).issubset(old[0]))
        if any(set(w).issubset(old[0]) for w in wall)
        else None
    )
    if test is not None and np.linalg.det(x[wall[test]] - probe) < 0:
        wall = wall[:, [0, 2, 1]]
    tri = x[wall]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1)[:, None]
    o = x.mean(0)
    bound = np.sum(n * (tri[:, 0] - o), axis=1)
    res = linprog(
        [0, 0, 0, -1],
        A_ub=np.c_[n, np.ones(len(n))],
        b_ub=bound,
        bounds=[(None, None)] * 3 + [(0, None)],
        method="highs",
    )
    if not res.success or res.x[3] < 1e-9:
        return
    # Near the Chebyshev center and toward the old mass centroid, consider
    # several legal material locations without moving a boundary vertex.
    leaf = x[old].mean(1)
    w = np.linalg.det(x[old[:, 1:]] - x[old[:, :1]])
    centroid = np.average(leaf, axis=0, weights=w)
    for alpha in (0.0, 0.5, 1.0):
        p = (1 - alpha) * (o + res.x[:3]) + alpha * centroid
        pos = np.vstack((x, p)).astype("f4").astype(float)
        new = np.c_[np.full(len(wall), len(x)), wall]
        volumes = np.linalg.det(pos[new[:, 1:]] - pos[new[:, :1]])
        if min(volumes) <= 0 or abs(volumes.sum() / w.sum() - 1) > 1e-9:
            continue
        uv = np.linalg.solve(
            (x[old[:, 1:]] - x[old[:, :1]]).transpose(0, 2, 1),
            (p - x[old[:, 0]])[:, :, None],
        )[:, :, 0]
        bary = np.c_[1 - uv.sum(1), uv]
        near = np.argmax(bary.min(1))
        if bary[near].min() < -1e-5:
            continue
        yield pos, new, old[near], bary[near]


def screen(frame, witness, former, exact):
    load = lambda p: json.loads(p.read_text())
    cfg = load(frame / "launch.json")
    e = cfg["environment"]
    trace = load(witness)
    prev = load(former)
    X, Y, src, roles, identity = inputs(frame)
    end = frame / "physical-accepted.node"
    if (
        sha(end) != prev["accepted"]
        or sha(end) != trace["accepted_geometry_sha256"]
        or identity != trace["geometry"]
        or identity != prev["source"]
    ):
        raise ValueError("Material state mismatch")
    t, role = installed_topology(src, roles, e)
    props, pins = materials(Path(cfg["argv"][0]).parent)
    if pins != prev["materials"]:
        raise ValueError("Different material law")
    X = X.astype(float)
    Y = Y.astype(float)
    Z = np.loadtxt(end, skiprows=1, usecols=(1, 2, 3))
    p = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    labels = Path(e[p + "CHILD_LABELS"])
    changed = Path(e["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ"])
    if (
        sha(labels) != e[p + "CHILD_LABELS_SHA256"]
        or sha(changed) != prev["previous_cavities"]
    ):
        raise ValueError("Lineage mismatch")
    tags = np.loadtxt(
        labels, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3), dtype="i4"
    )
    if not np.array_equal(tags[:, 0], np.arange(len(src))) or not np.array_equal(
        tags[:, 1], roles
    ):
        raise ValueError("Material tags mismatch")
    with np.load(changed) as data:
        kept, off, parents = [
            data[k]
            for k in (
                "retained_input_cell_rows",
                "replacement_parent_offsets",
                "replacement_parent_cell_rows",
            )
        ]

    def lineage(i):
        return (
            [kept[i]]
            if i < len(kept)
            else parents[off[i - len(kept)] : off[i - len(kept) + 1]]
        )

    exact = load_module(exact)
    out = []
    for site in trace["before_failure"]:
        k = site["native_cell"] - trace["global_element_offset"]
        corner = t[k]
        if list(corner) != site["nodes"]:
            raise ValueError("Different native tetrahedron")
        ids = np.unique(t[np.isin(t, corner).any(1)])
        rows = np.flatnonzero(np.isin(t, ids).any(1))
        faces = defaultdict(set)
        for i in rows:
            for f in oriented_faces(t[i]):
                faces[tuple(sorted(f))].add(int(i))
        tag = tags[lineage(k)[0], 1:]
        eligible = {int(i) for i in rows if np.all(tags[lineage(i), 1:] == tag)}

        def adj(patch):
            return {
                v
                for i in patch
                for f in oriented_faces(t[i])
                for v in faces[tuple(sorted(f))]
            } & eligible - set(patch)

        patches = [(k,)]
        stats = Counter()
        best = None
        for size in (2, 3, 4, 5):
            patches = sorted({tuple(sorted((*p, i))) for p in patches for i in adj(p)})
            for patch in patches:
                stats["patches"] += 1
                v, old = np.unique(t[list(patch)], return_inverse=True)
                old = old.reshape(-1, 4)
                x = X[v]
                y = Y[v]
                z = Z[v]
                mu, bulk = props[role[k] - 15]
                baselines = [
                    elastic(x, q, old, np.arange(len(x)), mu, bulk) for q in (y, z)
                ]
                for ref, new, donor, bar in candidates(x, old):
                    stats["positive_reference_fans"] += 1
                    if not exact.passes(*exact.measurements(ref, new)).all():
                        continue
                    result = []
                    qs = []
                    for state, (E, F, _) in zip((y, z), baselines):
                        init = np.vstack((state, bar @ state[donor]))
                        sol = interior_equilibrium(ref, init, new, mu, bulk)
                        if sol is None:
                            break
                        q, res = sol
                        e2, f2, _ = elastic(ref, q, new, np.arange(len(ref)), mu, bulk)
                        if np.linalg.norm(f2[-1]) > 1.1e-9:
                            raise ValueError("Unbalanced interior force")
                        qs.append(q)
                        result.append(
                            [
                                abs(e2 - E) / max(E, 1e-30),
                                np.linalg.norm(f2[:-1] - F)
                                / max(np.linalg.norm(F), 1e-30),
                            ]
                        )
                    if len(qs) != 2:
                        continue
                    stats["interior_equilibrated_both"] += 1
                    if not quality(ref, *qs, new, exact):
                        continue
                    stats["geometry_admissible"] += 1
                    score = max(map(max, result))
                    if best is None or score < best["worst"]:
                        best = dict(
                            worst=score,
                            errors=result,
                            patch=list(patch),
                            reference_point_m=ref[-1].tolist(),
                            cell=site["source_cell"],
                        )
                    if score <= 0.01:
                        stats["boundary_force_and_energy_pass"] += 1
        out.append(dict(cell=site["source_cell"], counts=stats, best=best))
    return dict(accepted=sha(end), source=identity, materials=pins, targets=out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("frame", "witness", "former", "exact"):
        p.add_argument(k, type=Path)
    print(json.dumps(screen(**vars(p.parse_args())), indent=2, allow_nan=False))
