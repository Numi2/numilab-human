"""Transfer posterior fine-mesh reactions into an admitted loaded pose of the existing native tissue."""

import argparse
import json
from pathlib import Path

import numpy as np
from conserved_cavity_rows import inventories, verify_native_mass
from material_bed_release import stationary_bed
from material_context import sha
from scipy.linalg import qr
from scipy.optimize import minimize
from support_patch import prepare
from supported_material_admission import certify, stress_limits
from torso_cell_geometry import geom

from compare_posterior_support_load import solve_load
from posterior_stretch import normal_stiffness, stretch_envelope
from refine_posterior_contact import mechanical, pullback


def fit(
    prep,
    variant,
    run,
    material,
    transfer,
    owner,
    bedfile,
    out,
    control_release=False,
    maximin=False,
    protect_stretch=False,
):
    out.mkdir(exist_ok=False)
    src = json.loads((prep / "source.json").read_text())
    bed = json.loads(bedfile.read_text())
    d = prepare(run, material, transfer, owner)
    if (
        src["registered"] != d["source"]
        or bed["receipt"] != src["matched_receipt"]
        or sha(prep / "patch.npz") != src["topology_patch_sha256"]
    ):
        raise ValueError("Different physical registration")
    with np.load(prep / "patch.npz") as a:
        g = dict(a)
    with np.load(variant) as a:
        ref = dict(a)
    if not np.array_equal(ref["local_source_nodes"], g["nodes"]):
        raise ValueError("Different posterior domain")
    x = g["x"]
    t = g["t"]
    y = g["y"]
    end = g["end"]
    nv = len(y)
    w = g["width"]
    dt = g["dt"]
    mass = g["mass"]
    pairs = ref["midpoint_parents"]
    nt = ref["tetrahedra"]
    use = (g["layer"] == 1) & (np.minimum(y[:, 2], end[:, 2]) < w)
    # Nodes of the repaired cavities are governed by their independent moment check.
    exc = np.unique(d["dm_tets"])
    free = np.flatnonzero(
        use & (y[:, 2] < w + 2e-7) & ~np.isin(g["nodes"], np.r_[exc, d["anchors"]])
    )
    rows = g["nodes"][free]
    measure = np.flatnonzero(use)
    idx = np.searchsorted(measure, free)
    if not np.array_equal(measure[idx], free):
        raise ValueError("Unknown bed rows")
    fine_mass = np.r_[mass, np.zeros(len(pairs))]
    for k in range(4):
        np.add.at(fine_mass, t[:, k], -g["cell_mass"] / 4)
        np.add.at(fine_mass, nt[:, k], ref["child_mass_kg"] / 4)
    fine_outer = np.r_[g["layer"] == 1, np.ones(len(pairs), bool)]
    outside = (d["layer"] == 1) & ~np.isin(np.arange(len(d["layer"])), g["nodes"])
    other = d["mass"][outside]
    xy = y[measure, :2]
    z = [y[measure, 2], end[measure, 2]]
    p0 = []
    goal = []
    el = []
    stiff = []
    for state, key in ((y, "current"), (end, "end")):
        local = np.empty((len(fine_mass), 3))
        local[:nv] = state
        for i, (a, b) in enumerate(pairs):
            local[nv + i] = ((local[a] + local[b]) * 0.5).astype("f4")
        pts = np.vstack((local[fine_outer], d[key][outside]))
        m = np.r_[fine_mass[fine_outer], other]
        _h, _, raw, _ = solve_load(pts, m, dt, w, bed["weight_N"])
        load = np.zeros(len(local))
        load[fine_outer] = raw[: fine_outer.sum()]
        proj = pullback(load, pairs)
        goal.append(proj[measure])
        bf = stationary_bed(state[measure, 2], mass[measure], dt, w)[1]
        p0.append(bf)
        f = mechanical((x, state), t, g["tags"], g["props"])[1]
        el.append(f[measure, 2] - mass[measure] * 9.81)
        stiff.append(
            normal_stiffness(x, state, g["tags"], t, g["props"], free, measure) * 1e-6
        )
    typed, groups = inventories(run, material)
    A = []
    labs = typed[g["cell_ids"], 1:]
    for group in np.unique(labs, axis=0):
        sel = (labs == group).all(1)
        nodal = np.bincount(
            t[sel].ravel(), weights=np.repeat(g["cell_mass"][sel] / 4, 4), minlength=nv
        )
        A.append(nodal[free])
    A = np.asarray(A)
    A = A[A.max(1) > 0]
    A /= np.linalg.norm(A, axis=1)[:, None]
    _, R, order = qr(A.T, mode="economic", pivoting=True)
    A = A[order[: np.linalg.matrix_rank(R)]]
    unit = np.eye(len(measure))[:, idx]
    mu = 1e-6
    cxy = np.mean(xy, axis=0)
    W = np.column_stack(
        (np.ones(len(xy)), (xy - cxy) / np.maximum(np.ptp(xy, axis=0), 1e-5))
    ).T
    target = W @ p0[0]

    def terms(v, phase):
        gaps = z[phase] + unit @ v * mu
        _, fb, curv = stationary_bed(gaps, mass[measure], dt, w)
        der = -curv[:, None] * unit * mu
        return fb, der, el[phase] + fb - stiff[phase] @ v, der - stiff[phase]

    def feature(v):
        rows = []
        jac = []
        for phase in (0, 1):
            fb, df, res, jr = terms(v, phase)
            for m, k, norm in (
                (fb - goal[phase], df, np.linalg.norm(p0[phase] - goal[phase])),
                (res, jr, np.linalg.norm(p0[phase] + el[phase])),
            ):
                rows.append(m / max(norm, 1e-6))
                jac.append(k / max(norm, 1e-6))
        return np.concatenate(rows), np.vstack(jac)

    def objective(v):
        r, j = feature(v)
        return 0.5 * (r @ r), r @ j

    def eq(v):
        fb, _, _, _ = terms(v, 0)
        return np.r_[(W @ fb - target) / bed["weight_N"], A @ v]

    def eq_jac(v):
        _, e, _, _ = terms(v, 0)
        return np.vstack((W @ e / bed["weight_N"], A))

    def release_limits(v):
        output = []
        gradient = []
        for i in (0, 1):
            fb, df, res, jr = terms(v, i)
            for current, J, base in [
                (res, jr, p0[i] + el[i]),
                (fb - goal[i], df, p0[i] - goal[i]),
            ]:
                sq = base @ base
                output.append(1 - current @ current / sq)
                gradient.append(-2 * current @ J / sq)
            if i == 1:
                scale = p0[i].sum() - bed["weight_N"]
                output.append((p0[i].sum() - fb.sum()) / scale)
                gradient.append(-df.sum(0) / scale)
        return np.array(output), np.array(gradient)

    constraints = [{"type": "eq", "fun": eq, "jac": eq_jac}]
    if control_release:
        # Retain at least a small positive physical margin after rounding to Float32.
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda v: release_limits(v)[0] - 2e-4,
                "jac": lambda v: release_limits(v)[1],
            }
        )
    keep, guarded = stretch_envelope(x, y, t, free, 2.001)
    if protect_stretch and len(guarded):
        constraints.append(
            {"type": "ineq", "fun": lambda v: keep(v)[0], "jac": lambda v: keep(v)[1]}
        )
    initial = np.zeros(len(free))
    results = []
    for limit in (0.25, 1.0, 2.0):
        opt = minimize(
            objective,
            initial,
            method="SLSQP",
            jac=True,
            constraints=constraints,
            bounds=[(-limit, limit)] * len(free),
            options={"maxiter": 140, "ftol": 2e-11},
        )
        if maximin:

            def score(v):
                value, jac = objective(v[:-1])
                return -v[-1] + 1e-3 * value, np.r_[1e-3 * jac, -1.0]

            opt = minimize(
                score,
                np.r_[opt.x, 0.0],
                method="SLSQP",
                jac=True,
                constraints=[
                    {
                        "type": "eq",
                        "fun": lambda v: eq(v[:-1]),
                        "jac": lambda v: np.column_stack(
                            (eq_jac(v[:-1]), np.zeros(len(eq(v[:-1]))))
                        ),
                    },
                    {
                        "type": "ineq",
                        "fun": lambda v: release_limits(v[:-1])[0] - v[-1],
                        "jac": lambda v: np.column_stack(
                            (release_limits(v[:-1])[1], -np.ones(5))
                        ),
                    },
                ]
                + (
                    [
                        {
                            "type": "ineq",
                            "fun": lambda v: keep(v[:-1])[0],
                            "jac": lambda v: np.column_stack(
                                (keep(v[:-1])[1], np.zeros(2 * len(guarded)))
                            ),
                        }
                    ]
                    if protect_stretch and len(guarded)
                    else []
                ),
                bounds=[(-limit, limit)] * len(free) + [(0.0, 1.0)],
                options={"maxiter": 240, "ftol": 2e-11},
            )
            opt.x = opt.x[:-1]
        delta = (y[free, 2] + mu * opt.x).astype("f4").astype(float) - y[free, 2]

        def exact(phase, shift):
            original = (y, end)[phase]
            new = original.copy()
            new[free, 2] += shift
            bf = stationary_bed(new[measure, 2], mass[measure], dt, w)[1]
            f = mechanical((x, new), t, g["tags"], g["props"])[1]
            local = f[measure, 2] + bf - mass[measure] * 9.81
            vol, quality = geom(new, t)
            return {
                "force_N": float(bf.sum()),
                "moment_Nm": (xy.T @ bf).tolist(),
                "local_residual_N": float(np.linalg.norm(local)),
                "projected_force_residual_N": float(np.linalg.norm(bf - goal[phase])),
                "min_volume_m3": float(vol.min()),
                "min_quality": float(quality.min()),
            }

        base = [exact(i, np.zeros(len(free))) for i in (0, 1)]
        after = [exact(i, delta) for i in (0, 1)]
        candidate = d["current"].copy()
        candidate[rows, 2] += delta
        if not np.array_equal(candidate, candidate.astype("f4")):
            raise ValueError("Non-emittable start")
        mrec = verify_native_mass(candidate, groups)
        valid = (
            opt.success
            and max(abs(eq(delta / mu))) < 1e-5
            and mrec["passed"]
            and abs(after[0]["force_N"] - base[0]["force_N"]) < 0.002
            and np.max(abs(np.array(after[0]["moment_Nm"]) - base[0]["moment_Nm"]))
            < 0.002
            and all(
                p["min_volume_m3"] > 1e-18 and p["min_quality"] >= 1e-5 for p in after
            )
            and after[1]["force_N"] < base[1]["force_N"]
            and all(
                after[i][k] < base[i][k]
                for i in (0, 1)
                for k in ("local_residual_N", "projected_force_residual_N")
            )
        )
        if protect_stretch:
            exact_start = certify(
                x,
                candidate[g["nodes"]],
                t,
                g["tags"],
                g["props"],
                stress_limits(d["materials"]),
            )
            valid = valid and exact_start["passed"]
        results.append(
            {
                "emitted_material": exact_start if protect_stretch else None,
                "step_limit_um": limit,
                "optimizer": str(opt.message),
                "iterations": int(opt.nit),
                "baseline": base,
                "after": after,
                "max_constraint_error": float(np.max(abs(eq(delta / mu)))),
                "cavity_moments": mrec,
                "gated": bool(valid),
            }
        )
        print(
            limit,
            opt.message,
            [(r["local_residual_N"], r["projected_force_residual_N"]) for r in after],
            valid,
            flush=True,
        )
        if valid:
            np.savez_compressed(
                out / f"pose-{limit:g}.npz",
                rows=rows,
                coordinates=candidate[rows].astype("f4"),
            )
    passed = [s for s in results if s["gated"]]
    best = (
        min(passed, key=lambda r: r["after"][1]["local_residual_N"]) if passed else None
    )
    report = {
        "registered": d["source"],
        "fine_mesh_sha256": sha(variant),
        "source_receipt": bed["receipt"],
        "trials": results,
        "chosen": best,
        "source_mass_unchanged": True,
        "stretch_guard_cells": len(guarded) if protect_stretch else 0,
        "active_coordinates": len(free),
    }
    if best:
        name = f"pose-{best['step_limit_um']:g}.npz"
        report.update(proposal=name, proposal_sha256=sha(out / name))
    (out / "qualification.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "prep",
        "variant",
        "run",
        "material",
        "transfer",
        "owner",
        "bedfile",
        "out",
    ):
        parser.add_argument(key, type=lambda p: Path(p).resolve())
    parser.add_argument("--control-release", action="store_true")
    parser.add_argument("--maximin", action="store_true")
    parser.add_argument("--protect-stretch", action="store_true")
    fit(**vars(parser.parse_args()))
