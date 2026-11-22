"""Initialize posterior support by matching the full external gravity wrench under the material constraints."""

import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from material_context import sha
from compare_bed_sampling import integrate
from posterior_stretch import stretch_envelope, normal_stiffness


def solve(data, observer, baseline, out, seed=None):
    inp = json.loads((data / "input.json").read_text())
    archive = data / "support.npz"
    if sha(archive) != inp["data"]:
        raise ValueError("Changed posterior material")
    with np.load(archive) as a:
        d = dict(a)
    lines = (observer / "stdout.log").read_text().splitlines()

    def records(prefix):
        return [
            json.loads(s.split("=", 1)[1]) for s in lines if s.startswith(prefix + "=")
        ]

    (grav,) = records("initial_gravity_total")
    rows = records("initial_gravity_component")
    weight = json.loads(baseline.read_text())["weight_N"]
    mass = sum(r["mass_kg"] for r in rows)
    center = np.array(grav["center_m"])
    moment = np.sum([r["first_moment_kg_m"] for r in rows], axis=0)
    if (
        [r["object"] for r in rows] != list(range(-1, 22))
        or abs(mass - grav["mass_kg"]) > 1e-8
        or np.linalg.norm(moment / mass - center) > 1e-12
        or abs(weight - 9.81 * mass) > 0.001
    ):
        raise ValueError("Incomplete gravitational load")
    x, y, f, t, A, free = [d[k] for k in ("x", "y", "f", "t", "area", "free")]
    props = d["props"]
    tags = d["tags"]
    W = d["W"]
    n = len(free)
    u = np.zeros(n)
    H = normal_stiffness(x, y, tags, t, props, free, free)
    H = (H + H.T) * 0.5
    eig, Q = np.linalg.eigh(H)
    # The material tangent defines the metric for the initial displacement.
    H = (Q * np.maximum(eig, 0)) @ Q.T
    H /= max(float(np.median(np.diag(H))), 1e-20)
    H += np.eye(n) * 0.02
    term, at_risk = stretch_envelope(x, y, t, free, 2.0)
    md = np.c_[
        np.ones(len(y)), (y[:, 0] - center[0]) / 0.15, (y[:, 1] - center[1]) / 0.5
    ]
    last = None

    def wrench(v):
        nonlocal last
        if last is None or not np.array_equal(last[0], v):
            cur = y.copy()
            cur[free, 2] += v * 1e-6
            args = (cur, f, A, d["density"], float(d["dt"]), float(d["width"]), 4)
            load, _, k0 = integrate(*args, normal_direction=md[:, 0])
            k1 = integrate(*args, normal_direction=md[:, 1])[2]
            k2 = integrate(*args, normal_direction=md[:, 2])[2]
            c = md.T @ load / weight - np.array([1.0, 0.0, 0.0])
            jac = -np.vstack([k0, k1, k2])[:, free] * 1e-6 / weight
            last = (v.copy(), c, jac, load)
        return last

    con = [
        dict(type="eq", fun=lambda v: W @ v, jac=lambda v: W),
        dict(type="eq", fun=lambda v: wrench(v)[1], jac=lambda v: wrench(v)[2]),
    ]
    if len(at_risk):
        con.append(
            dict(type="ineq", fun=lambda v: term(v)[0], jac=lambda v: term(v)[1])
        )
    # The closed material stars span every tetrahedron incident on a free row.
    # The independent physical comparison uses elastic + bed + gravity impulses.
    from fit_refined_preload import mechanical

    def actual(v, gradient):
        cur = y.copy()
        cur[free, 2] += v * 1e-6
        _, internal = mechanical((x, cur), t, tags, props)
        force = internal[free, 2] + wrench(v)[3][free] - d["mass"][free] * 9.81
        acc = force / d["mass"][free]
        obj = 0.5 * float(force @ acc)
        if not gradient:
            return dict(
                mass_weighted_acceleration_l2=float(np.sqrt(2 * obj)),
                free_normal_force_l2_N=float(np.linalg.norm(force)),
                max_nodal_normal_force_N=float(abs(force).max()),
                max_acceleration_m_s2=float(abs(acc).max()),
            )
        dir = np.zeros(len(y))
        dir[free] = acc
        rigidK = integrate(
            cur,
            f,
            A,
            d["density"],
            float(d["dt"]),
            float(d["width"]),
            4,
            normal_direction=dir,
        )[2][free]
        materialK = normal_stiffness(x, cur, tags, t, props, free, free)
        return obj, -1e-6 * (materialK @ acc + rigidK)

    start = u
    if seed:
        with np.load(seed / "preload.npz") as first:
            if not np.array_equal(first["rows"], d["nodes"][free]):
                raise ValueError("Different equilibrated support nodes")
            start = (first["coordinates"][:, 2].astype(float) - y[free, 2]) * 1e6
        norm = actual(start, True)[0]

        def objective(v):
            obj, jac = actual(v, True)
            return obj / norm, jac / norm
    else:

        def objective(v):
            return 0.5 * v @ H @ v, H @ v

    r = minimize(
        objective,
        start,
        jac=True,
        constraints=con,
        bounds=[(-2.0, 2.0)] * n,
        method="SLSQP",
        options={"maxiter": 120, "ftol": 2e-10},
    )

    def measure(v):
        _, c, j, load = wrench(v)
        return dict(
            normal_N=float(load.sum()),
            roll_pitch_residual_Nm=[
                float(c[2] * weight * 0.5),
                float(-c[1] * weight * 0.15),
            ],
            cop_xy_m=((y[:, :2].T @ load) / load.sum()).tolist(),
            weight_error_N=float(c[0] * weight),
            min_gap_m=float(
                np.min(
                    (y[:, 2] + np.bincount(free, weights=v * 1e-6, minlength=len(y)))[
                        d["density"] > 0
                    ]
                )
            ),
        )

    v = r.x
    changed = y[free].copy()
    changed[:, 2] = (changed[:, 2] + v * 1e-6).astype("f4")
    emitted = (changed[:, 2] - y[free, 2]) * 1e6
    result = measure(emitted)
    residual = float(np.max(abs(d["raw"] @ emitted)) * 1e-6)
    limit = float(np.min(term(emitted)[0])) if len(at_risk) else 1.0
    passed = bool(
        r.success
        and abs(result["weight_error_N"]) < 0.02
        and np.linalg.norm(result["roll_pitch_residual_Nm"]) < 0.01
        and limit >= 0
        and residual < 8.0e-11
    )
    local = {
        k: actual(vv, False)
        for k, vv in [("original", u), ("seed", start), ("emitted", emitted)]
    }
    out.mkdir(exist_ok=True)
    np.savez_compressed(
        out / "preload.npz", rows=d["nodes"][free], coordinates=changed.astype("f4")
    )
    summary = dict(
        source=inp,
        observer_launch=sha(observer / "launch.json"),
        gravity=grav,
        components=rows,
        weight_N=weight,
        initial=measure(u),
        emitted=result,
        local_balance=local,
        reference_seed=(sha(seed / "preload.npz") if seed else None),
        converged=r.success,
        message=str(r.message),
        iterations=r.nit,
        admitted=passed,
        mass_group_drift_kg_m=residual,
        minimum_stretch_inequality=limit,
        maximum_displacement_um=float(abs(emitted).max()),
        source_files={"proposal": sha(out / "preload.npz")},
    )
    (out / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        {
            k: summary[k]
            for k in [
                "initial",
                "emitted",
                "message",
                "admitted",
                "mass_group_drift_kg_m",
                "minimum_stretch_inequality",
            ]
        },
        flush=True,
    )


if __name__ == "__main__":
    solve(*(Path(s).resolve() for s in sys.argv[1:]))
