"""Balance released dermis/fat elastic volume modes with fixed boundary, mass and witness volumes."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.linalg import qr

from admissible import emit_conservative, material_direction
from cavity_mass import cavity_delta
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha
from refine_posterior_contact import mechanical
from supported_material_admission import certify, installed_topology, stress_limits


def read(p):
    return json.loads(p.read_text())


def fit(run, ledger, witness, out, translate=False):
    cfg = read(run / "launch.json")
    env = cfg["environment"]
    owner = Path(cfg["argv"][0]).parent
    X, Y, base, source_tags, pins = inputs(run)
    t, tags = installed_topology(base, source_tags, env)
    w = read(witness)
    if pins != w["geometry"]:
        raise ValueError("Different installed mesh")
    rows = [v["native_cell"] - w["global_element_offset"] for v in w["before_failure"]]
    if not np.array_equal(t[rows], [v["nodes"] for v in w["before_failure"]]):
        raise ValueError("Mismatched native cell witnesses")
    protected = np.unique(t[rows])
    flow = ledger / "fine-mass-routed/current-density-closed-flow.npz"
    labels = ledger / "mesh/current-child.labels.csv"
    if (
        sha(flow) != read(flow.parent / "mass-transfer.json")["flow_sha256"]
        or sha(labels) != read(labels.with_suffix(".json"))["labels_sha256"]
    ):
        raise ValueError("Different material ownership")
    donors = np.loadtxt(labels, delimiter=",", usecols=2, skiprows=1, dtype=int)
    cavity = Path(env["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ"])
    if sha(cavity) != env["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256"]:
        raise ValueError("Installed cavity topology changed")
    with np.load(flow) as mass, np.load(cavity) as stars:
        old = mass["element_mass_kg"]
        ret, nt, mnew = (
            stars[k]
            for k in (
                "retained_input_cell_rows",
                "replacement_tetrahedra",
                "replacement_mass_kg",
            )
        )
        off, par = (
            stars["replacement_parent_offsets"],
            stars["replacement_parent_cell_rows"],
        )
        dst = donors[par[off[:-1]]]
        if not np.array_equal(np.repeat(dst, np.diff(off)), donors[par]):
            raise ValueError("Ambiguous anatomical donor")
        cm = np.r_[old[ret], mnew]
        donor = np.r_[donors[ret], dst]
        dm, remesh, remass = cavity_delta(base, old, ret, nt, mnew, len(Y))
    node_map = Path(
        env[
            "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_CSV"
        ]
    )
    if (
        sha(node_map)
        != env[
            "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_SHA256"
        ]
    ):
        raise ValueError("Different layer map")
    layer = np.full(len(Y), -1, dtype=int)
    with node_map.open() as f:
        for row in csv.DictReader(f):
            n = int(row["joint_node_id"]) - 1
            kind = {"": 0, "O": 1, "I": 2, "D": 3}[row["registered_surface_layer"]]
            if layer[n] not in (-1, kind):
                raise ValueError("Conflicting node labels")
            layer[n] = kind
    if (layer < 0).any():
        raise ValueError("Unregistered node")
    header = (owner / "NumiHumanLoadedSkinPoseData.inc").read_text()
    anchor = re.search(r"constexpr Anchor anchors\[\]=\{(.*?)\};", header, re.S)
    anchors = np.array(re.findall(r"\{\s*\d+u?\s*,\s*(\d+)u?\s*,", anchor[1]), int)
    if len(anchors) != 14:
        raise ValueError("Anchor count changed")
    first = np.flatnonzero(np.isin(t, protected).any(axis=1))
    if translate:
        # Enlarge the neighborhood for first-moment compensation. All four
        # corners of each target may translate together; its shape is fixed.
        first = np.flatnonzero(np.isin(t, np.unique(t[first])).any(axis=1))
    free = np.setdiff1d(
        np.unique(t[first]), anchors if translate else np.r_[protected, anchors]
    )
    free = free[np.isin(layer[free], [0, 2])]
    active = np.flatnonzero(np.isin(t, np.r_[free, protected]).any(axis=1))
    nodes = np.unique(t[active])
    tet, corners, f = (np.searchsorted(nodes, v) for v in (t[active], t[rows], free))
    x, y = X[nodes].astype(float), Y[nodes].astype(float)
    gids, group = np.unique(
        np.c_[tags[active], donor[active]], axis=0, return_inverse=True
    )
    weights = np.zeros((len(gids), len(nodes)))
    for col in tet.T:
        np.add.at(weights, (group, col), cm[active] / 4)
    m = weights.sum(axis=0)
    W = weights[:, f]
    rel = Y[remesh].astype(float) - Y[remesh[0]].astype(float)
    allowance = 1e-10 * remass * abs(rel).max()
    reserve = 0.5 * (allowance - abs(dm[remesh] @ rel).max())
    if reserve <= 0:
        raise ValueError("Cavity moment admission failed on input")
    W = np.vstack((W, dm[free] * (8e-11 / reserve)))
    Q, R, _ = qr(
        (W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-99)).T,
        mode="economic",
        pivoting=True,
    )
    diag = abs(np.diag(R))
    C = np.kron(Q[:, diag > diag.max() * 1e-11].T, np.eye(3))
    if translate:
        rigid_rows = []
        for cell in t[rows]:
            loc = np.searchsorted(free, cell)
            if not np.array_equal(free[loc], cell):
                raise ValueError("Witness shares a locked boundary")
            for vertex in loc[1:]:
                row = np.zeros(len(free))
                row[[loc[0], vertex]] = [1.0, -1.0]
                rigid_rows.append(row)
        C = np.vstack((C, np.kron(rigid_rows, np.eye(3))))
    props, identities = materials(owner)
    limits = stress_limits(identities)

    def evaluate(p):
        en, force = mechanical((x, p), tet, tags[active], props)
        force[:, 2] -= 9.81 * m
        return en + 9.81 * (m @ (p[:, 2] - y[:, 2])), force

    direction = []
    for cell in corners:
        H = np.linalg.inv(y[cell[1:]] - y[cell[0]]).T
        direction.append(np.vstack((-H.sum(axis=0), H)) / m[cell, None])

    def modes(force):
        return np.array(
            [np.sum(d * force[cell]) for d, cell in zip(direction, corners)]
        )

    def derivative(p):
        # Symmetry of the constitutive Hessian gives all free gradients in two
        # force evaluations per mode. Fixed corners keep dlogJ and mass constant.
        jac = []
        for d, cell in zip(direction, corners):
            scale = abs(d).max()
            step = np.zeros_like(p)
            step[cell] = d * (1.0e-9 / scale)
            jac.append(
                (evaluate(p + step)[1][f] - evaluate(p - step)[1][f]) * (scale / 2.0e-9)
            )
        return np.asarray(jac)

    start, F = evaluate(y)
    initial = modes(F)
    J = derivative(y)
    probe = np.zeros_like(y)
    probe[f] = np.random.default_rng(221).normal(size=(len(f), 3))
    check = (
        modes(evaluate(y + 1.0e-9 * probe)[1]) - modes(evaluate(y - 1.0e-9 * probe)[1])
    ) / 2.0e-9
    tangent_error = np.linalg.norm(
        np.einsum("ijk,jk->i", J, probe[f]) - check
    ) / np.linalg.norm(check)
    if tangent_error > 1.0e-4:
        raise ValueError(("Constitutive derivative mismatch", tangent_error))
    pos = y.copy()
    history = []
    cert0 = certify(x, y, tet, tags[active], props, limits)
    if not cert0["passed"]:
        raise ValueError(cert0)
    for iteration in range(48 if translate else 10):
        en, F = evaluate(pos)
        ratio = modes(F) / abs(initial)
        jac = derivative(pos) / abs(initial)[:, None, None]
        target = -np.einsum("i,ijk->jk", ratio, jac)
        # Both mode error and real material potential must decrease.
        fn = np.linalg.norm(F[f])
        tn = np.linalg.norm(target)
        best = None
        for en_weight in (0.0, 0.12, 0.5, 1.5):
            search = target / tn + en_weight * F[f] / max(fn, 1.0e-30)
            delta = material_direction(x, pos, tet, f, search, C, trust_um=8.0)
            for half in range(16):
                test = pos.copy()
                test[f] += delta * 2.0**-half
                test = emit_conservative(test, y, f, W)
                if np.max(abs(W @ (test[f] - y[f]))) > 8e-11:
                    continue
                if translate:
                    original = y[corners[:, 1:]] - y[corners[:, :1]]
                    fitted = test[corners[:, 1:]] - test[corners[:, :1]]
                    # Quantization can change a component by one ULP. Admit only
                    # translations that leave all six relative values identical.
                    if not np.array_equal(original, fitted):
                        continue
                Et, Ft = evaluate(test)
                rt = modes(Ft) / abs(initial)
                if (
                    Et > en
                    or np.any(abs(rt) >= abs(ratio) - 1e-8)
                    or not certify(x, test, tet, tags[active], props, limits)["passed"]
                ):
                    continue
                score = float(np.sum(rt * rt))
                if best is None or score < best[0]:
                    best = score, test, rt, Et
                break
        if best is None:
            break
        _, pos, rt, Et = best
        history.append(
            dict(iteration=iteration, relative_modes=rt.tolist(), potential_J=Et)
        )
        if np.max(abs(rt)) < 0.02:
            break
    final, F = evaluate(pos)
    moved = np.any(pos[f] != y[f], axis=1)
    out.mkdir(exist_ok=True)
    if moved.any():
        np.savez_compressed(
            out / "preload.npz",
            rows=free[moved],
            coordinates=pos[f[moved]].astype("f4"),
        )
    if not np.array_equal(
        y[corners[:, 1:]] - y[corners[:, :1]], pos[corners[:, 1:]] - pos[corners[:, :1]]
    ):
        raise ValueError("Witness geometry moved")
    result = dict(
        source=pins,
        registered_map=sha(node_map),
        mass_ledger=sha(flow),
        material_identities=identities,
        cells=[v["source_cell"] for v in w["before_failure"]],
        held_nodes=protected.tolist(),
        eligible_bonded=int(sum(layer[free] == 2)),
        eligible_interior=int(sum(layer[free] == 0)),
        material_cells=len(tet),
        constrained_rows=len(C),
        owner_anchor_count=len(anchors),
        force_gradient_directional_error=tangent_error,
        initial_modes_s2=initial.tolist(),
        final_modes_s2=modes(F).tolist(),
        potential_before_J=start,
        potential_after_J=final,
        initial_admission=cert0,
        final_admission=certify(x, pos, tet, tags[active], props, limits),
        mass_moment_drift_kg_m=float(abs(weights[:, f] @ (pos[f] - y[f])).max()),
        maximum_displacement_m=float(np.linalg.norm(pos - y, axis=1).max()),
        moved=int(moved.sum()),
        iterations=history,
        exterior_deep_and_anchor_coordinates_fixed=True,
        witness_edge_matrices_fixed=True,
        rigid_witness_translation=translate,
    )
    (out / "fit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("run", "ledger", "witness", "out"):
        p.add_argument(k, type=Path)
    p.add_argument("--translate", action="store_true")
    fit(**vars(p.parse_args()))
