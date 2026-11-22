"""Project the requested dorsal displacement onto the actual P1 mesh in both native coordinate states."""

from pathlib import Path
import json, sys
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack
from scipy.sparse.csgraph import connected_components
from scipy.optimize import linprog
from fit_dorsal_material import inputs, make_field, evaluate
from torso_cell_geometry import geom


def project(
    run,
    root,
    amplitude,
    contact_exe=None,
    mass_owner=None,
    local=False,
    targeted=False,
    dense=None,
):
    x, y, t, mat, pins = inputs(run)
    g = make_field(y.astype(float), json.loads((root / "geometry.json").read_text()))
    if local:
        from localized_mass_field import localize

        g = localize(root, y, g) / amplitude
    support = g != 0
    if targeted:
        from target_digit_interfaces import domain

        support = domain(root, y)
        g = np.zeros(len(y))
    ids = np.flatnonzero(np.any(support[t], axis=1))
    tt = t[ids]
    subset = np.unique(tt)
    loc = np.searchsorted(subset, tt)
    nv = len(subset)
    # Rigidly translate precision-limited edge stars in one coordinate quantum.
    # Their material reference and current-state volumes need a representable room for a split.
    v0, q0 = geom(x.astype(float), tt)
    v1, q1 = geom(y.astype(float), tt)
    quantum = max(
        abs(np.spacing(x[subset, 0])).max(), abs(np.spacing(y[subset, 0])).max()
    )
    state = []
    stiff = np.zeros(len(tt), bool)
    for z, v, q in ((x, v0, q0), (y, v1, q1)):
        pt = z[tt].astype(float)
        e = pt[:, 1:] - pt[:, :1]
        inv = np.linalg.inv(e)
        grad = np.concatenate((-inv.sum(2)[:, :, None], inv), axis=2)
        worst = np.max(np.sum(np.abs(grad), axis=2), axis=1) * 0.5 * quantum
        stiff |= (q < 1.5e-5) | (v < 1.5e-18) | (worst > 0.035)
        state.append((grad * 0.001, v, q))
    ij = loc[stiff]
    edges = coo_matrix(
        (np.ones(3 * len(ij), dtype="u1"), (np.repeat(ij[:, 0], 3), ij[:, 1:].ravel())),
        shape=(nv, nv),
    ).tocsr()
    ng, group = connected_components(edges, directed=False)
    fixed = np.zeros(ng, bool)
    fixed[group[~support[subset]]] = True
    pr = np.full(ng, -1, dtype="i4")
    pr[~fixed] = np.arange((~fixed).sum())
    N = (~fixed).sum()
    node_vars = pr[group]
    cell_vars = node_vars[loc]
    free = node_vars >= 0
    want = amplitude * g[subset]
    counts = np.bincount(node_vars[free], minlength=N)
    target = np.bincount(node_vars[free], weights=want[free], minlength=N) / counts

    def matvec(coeff, cell):
        vv = cell_vars[cell]
        i, j = np.nonzero(vv >= 0)
        return coo_matrix((coeff[i, j], (i, vv[i, j])), shape=(len(cell), N)).tocsr()

    # Positive contributions from the two authored configurations share one reference morph.
    # Keep an explicit shape safety factor in a cell-specific gradient bound.
    models = []
    for grad, vol, quality in state:
        # If |each gradient component|<=limit, every directional length increases
        # by at most 1+sqrt(3)*limit, and J>=1-limit. Bound both acceptance gates.
        a = np.zeros(len(tt))
        b = np.full(len(tt), 0.25)
        for _ in range(28):
            m = (a + b) * 0.5
            good = (vol * (1 - m) > 1.000001e-18) & (
                quality * (1 - m) / (1 + np.sqrt(3) * m) ** 3 > 1.000001e-5
            )
            a = np.where(good, m, a)
            b = np.where(good, b, m)
        round_error = 0.501 * quantum * 1000 * np.max(np.sum(abs(grad), 2), 1)
        limit = a - round_error
        if np.any((limit <= 0) & ~stiff):
            # Those cells are handled by an exact common shift, with zero rounded gradient.
            raise ValueError("Insufficient binary32 margin in a free cell")
        limit[stiff] = 0.25
        models.append((grad, limit))
    I = eye(N, format="csr")
    main = vstack((hstack([I, -I]), hstack([-I, -I])), format="csr")
    rhs = np.r_[target, -target]
    blocks = [main]
    rhsblocks = [rhs]
    accum = [set() for _ in models]
    vi = target.copy()
    history = []
    contacts = None
    if contact_exe is not None:
        from digital_contact_constraints import NoNewCrossings

        volume_vars = np.full(len(x), -1, dtype="i4")
        volume_vars[subset] = node_vars
        contacts = NoNewCrossings(
            root / "contact-domain.npz", contact_exe, volume_vars, N
        )
    mass = None
    frozen = set()
    if mass_owner:
        from digit_mass_constraints import FixedCarrierVolumes

        mass = FixedCarrierVolumes(
            mass_owner, pins, ids, mat, state[0][0][:, 0], v0, matvec, dense
        )
    if targeted:
        from target_digit_interfaces import targets

        A, b = targets(root, contacts, quantum)
        blocks.append(A)
        rhsblocks.append(b)
    for iteration in range(64):
        val = np.zeros(nv)
        val[free] = vi[node_vars[free]]
        site = val[loc]
        added = 0
        for i, (grad, limit) in enumerate(models):
            out = np.einsum("ijk,ik->ij", grad, site)
            bad = (abs(out) > limit[:, None] + 1e-7) & (~stiff[:, None])
            new = [
                (int(c), int(ax), 1 if out[c, ax] >= 0 else -1)
                for c, ax in zip(*np.nonzero(bad))
            ]
            new = [item for item in new if item not in accum[i]]
            if not new:
                continue
            accum[i].update(new)
            ci, ax, sg = map(np.array, zip(*new))
            M = matvec(grad[ci, ax] * sg[:, None], ci)
            scale = np.maximum(1.0, np.asarray(abs(M).max(1).toarray()).ravel())
            blocks.append(
                hstack(
                    [M.multiply((1 / scale)[:, None]), csr_matrix(M.shape)],
                    format="csr",
                )
            )
            rhsblocks.append(limit[ci] / scale)
            added += len(ci)
        hist = dict(iteration=iteration, new_constraints=added)
        history.append(hist)
        if targeted and iteration == 0:
            added += len(b)
        if not added and mass:
            matrix, bounds, preserve = mass.constraints(vi, quantum)
            if len(bounds):
                blocks.append(matrix)
                rhsblocks.append(bounds)
            frozen.update(preserve)
            added = len(bounds) + len(preserve)
            hist.update(
                density_constraints=len(bounds), pinned_density_variables=len(preserve)
            )
        if not added and contacts is not None:
            unacceptable, cleared = contacts.evaluate(vi, quantum)
            hist["new_contacts"] = len(unacceptable)
            hist["cleared"] = len(cleared)
            print("geometric certificate", len(unacceptable), len(cleared), flush=True)
            if unacceptable:
                matrix, bounds = contacts.constrain(unacceptable, quantum)
                if not len(bounds):
                    raise ValueError("New contact without movable variables")
                blocks.append(matrix)
                rhsblocks.append(bounds)
                added = len(bounds)
        if not added:
            break
        bounds = [
            (0.0, 0.0)
            if i in frozen
            else (-amplitude, amplitude)
            if targeted
            else (min(0.0, target[i]), max(0.0, target[i]))
            for i in range(N)
        ] + [(0, None)] * N
        result = linprog(
            np.r_[np.zeros(N), counts],
            A_ub=vstack(blocks, format="csr"),
            b_ub=np.concatenate(rhsblocks),
            bounds=bounds,
            method="highs",
            options={
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
            },
        )
        if not result.success:
            raise ValueError(result.message)
        vi = result.x[:N]
        hist["maximum_delta_mm"] = float(np.max(abs(vi)))
        print(
            iteration,
            added,
            round(float(np.average(abs(vi), weights=counts)), 4),
            flush=True,
        )
    else:
        raise ValueError("P1 projection did not certify")
    delta = np.zeros(len(x))
    delta[subset[free]] = np.rint(vi[node_vars[free]] * 0.001 / quantum) * quantum

    def pose(base):
        o = base.copy()
        o[:, 0] = (base[:, 0].astype(float) + delta).astype("f4")
        return o

    z = pose(x)
    w = pose(y)
    qual = {}
    svals = {}
    for coord, base, test in (("reference", x, z), ("loaded", y, w)):
        qual[coord] = {str(i): evaluate(test, tt[mat[ids] == i]) for i in (15, 16)}
        original = base[tt].astype(float)
        candidate = test[tt].astype(float)
        F = np.linalg.solve(
            original[:, 1:] - original[:, :1], candidate[:, 1:] - candidate[:, :1]
        )
        sv = np.linalg.svd(F, compute_uv=False)
        svals[coord] = [float(sv.min()), float(sv.max())]
    changed = np.flatnonzero(delta)
    case = root / (
        (
            "routed-target-"
            if dense
            else "target-mass-"
            if targeted
            else "local-mass-"
            if local
            else "mass-constrained-"
            if mass
            else "constrained-"
            if contacts
            else "projected-"
        )
        + f"{amplitude:g}mm"
    )
    case.mkdir(exist_ok=True)
    np.savez_compressed(
        case / "field.npz", nodes=changed, delta_x_m=delta[changed], cells=ids
    )
    report = dict(
        inputs=pins,
        requested_mm=amplitude,
        changed_nodes=len(changed),
        affected_cells=len(ids),
        limited_cells=int(stiff.sum()),
        free_parameters=int(N),
        minimum_and_maximum_stretch=svals,
        quality=qual,
        projection=history,
        contact_history=contacts.log if contacts else [],
        desired_mm=float(np.max(abs(want))),
        maximum_realized_mm=1000 * float(np.max(abs(delta))),
        mean_requested_displacement_retained=float(
            np.sum(abs(delta[subset]) * 1000) / np.sum(abs(want))
        )
        if np.any(want)
        else None,
    )
    (case / "quality.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(v["bad"] for stage in qual.values() for v in stage.values()):
        raise ValueError("Rounded geometry rejected")
    print(
        "valid",
        report["changed_nodes"],
        report["mean_requested_displacement_retained"],
        svals,
    )


if __name__ == "__main__":
    project(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        float(sys.argv[3]),
        Path(sys.argv[4]) if len(sys.argv) > 4 else None,
        Path(sys.argv[5]) if len(sys.argv) > 5 else None,
        sys.argv[6:] == ["--local"],
        sys.argv[6:7] == ["--target"],
        Path(sys.argv[7]) if len(sys.argv) > 7 else None,
    )
