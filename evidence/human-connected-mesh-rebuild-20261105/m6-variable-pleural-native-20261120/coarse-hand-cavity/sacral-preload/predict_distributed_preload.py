"""Solve a bounded offline preload on the exact current native shell."""
from argparse import ArgumentParser
import json
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.optimize import brentq

from audit import digest, rows
from lateral import LateralSeedEnergy
from predict import TISSUE_MATERIAL, SeedEnergy, ip_contact, match_ipc_seed, material_parameters, solve

FLOOR_LOADS = (.01, .025, .05, .075, .10, .20, .35, .60, 1.)


def main():
    p = ArgumentParser(description=__doc__)
    for name in ("run", "fit", "balance", "build", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--full-lateral", action="store_true")
    p.add_argument("--descent-fallback", action="store_true")
    p.add_argument("--deep-tangential", action="store_true")
    p.add_argument("--warm", type=float, help="temporary offline contact penalty stiffness in N/m")
    p.add_argument("--lateral-strip-m", type=float, help="three-dimensional motion in the bed-facing strip")
    a = p.parse_args()
    start = monotonic()
    support = json.loads(a.balance.read_text())
    nodes_path = a.run / "shell-mass.csv.nodes.csv"
    cells_path = a.run / "shell-mass.csv.tets.csv"
    assert digest(nodes_path) == support["sha256"][nodes_path.name]
    cell_rows = rows(cells_path)
    ids = np.array([int(r["material_index"]) for r in cell_rows])
    cells = np.array([[int(r[f"parent_node{i}"]) for i in range(4)] for r in cell_rows])
    materials = {}
    commands = a.build / "build-commands.json"
    compile_recipe = json.loads(commands.read_text())
    prms = {i: material_parameters(materials, compile_recipe, i)
            for i in TISSUE_MATERIAL}
    parameters = np.array([prms[i] for i in ids])
    fit_path = a.fit / "posterior-interface-fit.npz"
    with np.load(fit_path) as fit:
        assert np.array_equal(cells, fit["cells"])
        world, rest = (fit[key].astype(float) for key in ("nodes_current", "nodes_reference"))
    mass_path = a.fit / "balanced-mass.npz"
    with np.load(mass_path) as m:
        mass = m["node_mass_kg"].copy()
    assert np.isclose(mass.sum(), 13.939679838742224, rtol=0., atol=1.e-11)
    outer = len(world) // 3
    assert len(world) == 3 * outer
    width = support["contact_width_m"]
    shift = np.array([0., 0., support["bed_offset_m"]])
    world, rest = world - shift, rest - shift
    launch = json.loads((a.run / "launch.json").read_text())
    argv = launch["argv"]
    dt = float(argv[argv.index("--muscle-step-seconds") + 1])
    args = (world, rest, cells, parameters[:, 0], parameters[:, 1], mass, outer,
            support["mass_kg"], np.array(support["com_m"]) - shift, width, dt)
    lateral = a.full_lateral or a.lateral_strip_m is not None
    if lateral:
        width_m = float(world[:, 2].max()) if a.full_lateral else a.lateral_strip_m
        assert width_m > 0.
        extra = []
        if a.deep_tangential:
            # Keep physically represented elastic retaining sites fixed in
            # this diagnostic. All other selected deep nodes retain their
            # normal constraint and are permitted to slide laterally.
            with (a.run / "stdout.log").open() as stream:
                anchors = {2 * outer + int(line.partition(" local_vertex=")[2].split()[0])
                           for line in stream if line.startswith("whole_body_skin_attachment ")}
            assert len(anchors) == 14
            extra = [i for i in range(2 * outer, len(world))
                     if world[i, 2] <= width_m and i not in anchors]
            assert extra
        extra_args = {"tangential_deep": extra} if a.deep_tangential else {}
        model = LateralSeedEnergy(*args, width_m=width_m, **extra_args)
        if a.full_lateral and not a.deep_tangential:
            assert len(model.lateral) == 2 * outer
    else:
        assert not a.deep_tangential
        model = SeedEnergy(*args)
    support_rows = support["possible_normal_distribution_at_threshold"]
    probes = np.array([r["outer_shell_row"] for r in support_rows])
    force = np.array([r["normal_force_n"] * FLOOR_LOADS[0] for r in support_rows])
    assert len(force) == 3
    gap = np.array([brentq(lambda g: ip_contact(np.array([g]), width, np.array([node_mass]), dt)[1][0] - f,
                            1.e-9, width, xtol=1.e-15)
                    for f, node_mass in zip(force, mass[probes], strict=True)])
    v = np.zeros(model.n)
    v[model.q] = np.linalg.solve(model.a[probes], gap - model.base_gap[probes])
    assert model.state(v, FLOOR_LOADS[0]) is not None
    a.output.mkdir()
    loads = FLOOR_LOADS
    extra = {"regularize": True} if a.descent_fallback else {}
    if a.warm is not None:
        assert a.warm > 0.
        v[model.q[0]] -= width * 2.
        v, warm = solve(model, v, 65, weights=loads, penalty=a.warm, guard=True, **extra)
        (a.output / "warm-trace.json").write_text(json.dumps(warm, indent=2, allow_nan=False) + "\n")
        reached = max((r["load_fraction"] for r in warm if r.get("outcome") == "converged"), default=0.)
        v = match_ipc_seed(model, v, a.warm)
        if model.state(v, reached) is None:
            raise ValueError("temporary warm solution fails the exact IPC and volume admission")
        loads = tuple(fraction for fraction in loads if fraction >= reached)
    accepted = {}
    if a.descent_fallback:
        extra["converged"] = accepted.__setitem__
    v, trace = solve(model, v, 65, weights=loads, guard=True, **extra)
    if accepted:
        weight = max(accepted)
        k = accepted[weight]
        _, residual, force, jac = model.state(k, weight)
        assert np.abs(residual).max() < 1.e-4
        np.savez_compressed(a.output / "last-equilibrated.npz",
                            coordinates=k, contact_force_n=force, determinant=jac,
                            load_fraction=weight, residual=residual)
    rate = trace[-1]["load_fraction"]
    _, residual, normal, jac = model.state(v, rate)
    successful = [r["load_fraction"] for r in trace if r.get("outcome") == "converged"]
    shape = model.full_positions(v)[0] if lateral else np.c_[np.zeros((len(world), 2)), model.positions(v)[0]]
    np.savez_compressed(a.output / "seed.npz", deformation_m=shape, rigid_seed=v[model.q],
                        contact_force_n=normal, determinant=jac, load_fraction=rate)
    report = {
        "scope": "Offline three-layer elastic preload under one fixed flat bed. The deep boundary is prescribed relative to the three floating coordinates. No state is installed. Lateral and normal modes use the same constitutive and bed contact energies.",
        "hashes": {f.name: digest(f) for f in (nodes_path, cells_path, fit_path, mass_path, a.balance, commands)},
        "constitutive_hashes": materials, "dt_s": dt, "width_m": width, "initializer_penalty_n_m": a.warm, "descent_fallback": a.descent_fallback, "deep_tangential": a.deep_tangential,
        "lo_j": model.min_j, "hi_j": model.max_j, "lateral_coordinates": model.n - 3 - 2 * outer,
        "last_converged_fraction": max(successful, default=0.), "attempted_fraction": rate,
        "physical_weight_n": model.body_weight, "last_normal_force_n": float(normal.sum()),
        "last_max_residual": float(np.abs(residual).max()), "last_active_sites": int((normal > 0.).sum()),
        "maximum_displacement_m": float(np.linalg.norm(shape, axis=1).max()),
        "last_smallest_J": float(jac.min()), "last_largest_J": float(jac.max()),
        "last_tet": int(np.argmin(np.minimum(jac-model.min_j, model.max_j-jac))),
        "outcome": trace[-1].get("outcome", "continued"),
        "elapsed_s": monotonic() - start,
    }
    (a.output / "trace.json").write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n")
    (a.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
