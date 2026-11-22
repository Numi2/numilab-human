"""Jointly enforce registered carrier capacity, cell density and spatial moments."""

from pathlib import Path
import json, sys
import numpy as np
from scipy.optimize import linprog
from fit_dorsal_material import inputs
from material_context import sha
from fit_cavity_mass import point_mass_features
from torso_cell_geometry import geom
from registered_density import density_bounds


def solve(x, y, old, new, prior, supply, bounds=(850.0, 950.0)):
    n = len(new)
    nodes = np.unique(old)
    oi = np.searchsorted(nodes, old)
    ni = np.searchsorted(nodes, new)
    xyz = [
        (r[nodes] - r[nodes].mean(0)) / np.ptp(r[nodes], axis=0).max() for r in (x, y)
    ]
    b = point_mass_features(ni, xyz, None)
    a = point_mass_features(oi, xyz, None)
    u, s, _ = np.linalg.svd(np.c_[a, b], full_matrices=False)
    keep = s > s[0] * 64 * np.finfo(float).eps
    pre = u[:, keep].T / s[keep, None]
    total = prior.sum()
    vol = geom(x, new)[0]
    hi = bounds[1] * vol / total
    lo = bounds[0] * vol / total
    groups = {}
    for row in supply:
        mask = tuple(row["allowed"])
        if not mask or len(set(mask)) != len(mask) or min(mask) < 0 or max(mask) >= n:
            raise ValueError("Missing or invalid native carrier edges")
        if len(mask) < n:
            groups[mask] = groups.get(mask, 0.0) + row["mass_kg"] / total
    if abs(sum(r["mass_kg"] for r in supply) - total) > 1e-10 * total:
        raise ValueError("Unmatched native carrier mass")
    slots = sum(len(mask) for mask in groups)
    eq = np.zeros((len(pre) + len(groups), n + slots))
    eq[: len(pre), :n] = pre @ b
    upper = np.zeros((n, n + slots))
    upper[:, :n] = np.eye(n)
    slot = 0
    for g, (mask, amount) in enumerate(groups.items()):
        for sink in mask:
            eq[len(pre) + g, n + slot] = 1.0
            upper[sink, n + slot] = amount
            slot += 1
    # Full-support sources may serve any remaining sink capacity.
    goal = np.r_[pre @ (b @ hi - a @ (prior / total)), np.ones(len(groups))]
    result = linprog(
        np.zeros(n + slots),
        A_eq=eq,
        b_eq=goal,
        A_ub=upper,
        b_ub=hi,
        bounds=list(zip(np.zeros(n), hi - lo)) + [(0.0, 1.0)] * slots,
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-10,
            "dual_feasibility_tolerance": 1e-10,
        },
    )
    if not result.success:
        return None
    m = (hi - result.x[:n]) * total
    if np.max(abs(b @ (m / total) - a @ (prior / total))) > 1.0e-10:
        raise ValueError("Mass moment mismatch")
    return m


def main(run, trial, proposal, fit, ledger, first, out):
    x, y, t, tag, pins = inputs(run)
    x = x.astype(float)
    y = y.astype(float)
    d = json.loads(fit.read_text())
    launch = json.loads((trial / "launch.json").read_text())
    if (
        sha(proposal)
        != launch["environment"]["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256"]
        or d["source"] != pins
        or sha(ledger) != d["ledger_sha256"]
    ):
        raise ValueError("Source identity changed")
    r = d["proposal"]
    old = np.array(r["retired_rows"])
    new = np.array(r["replacement_tets"])
    with np.load(proposal) as p:
        offsets = p["replacement_tet_offsets"]
        hits = [
            i
            for i, j in zip(offsets[:-1], offsets[1:])
            if np.array_equal(p["replacement_tetrahedra"][i:j], new)
        ]
        if len(hits) != 1:
            raise ValueError("Native cavity order does not match this fit")
    lines = [
        json.loads(l.partition("=")[2])["supply"]
        for l in (trial / "stderr.log").read_text().splitlines()
        if l.startswith("cavity_carrier_route=")
    ]
    if len(lines) != 1:
        raise ValueError("Expected one explicit carrier witness")
    with np.load(ledger) as f:
        prior = f["element_mass_kg"][old]
    bounds = density_bounds(
        Path(json.loads((run / "launch.json").read_text())["argv"][0]).parent,
        tag[old[0]],
    )
    m = solve(x, y, t[old], new, prior, lines[0], bounds)
    if m is None:
        raise ValueError(
            "No feasible joint route at these coordinates and density bounds"
        )
    nodes = np.unique(new)
    delta = (
        point_mass_features(new, [], nodes) @ m
        - point_mass_features(t[old], [], nodes) @ prior
    )
    endpoint = first / "physical-accepted.node"
    if sha(endpoint) != d["accepted_coordinates_sha256"]:
        raise ValueError("Accepted positions changed")
    end = np.loadtxt(endpoint, skiprows=1, usecols=(1, 2, 3))
    r.update(
        mass_kg=m.tolist(),
        total_mass_kg=float(m.sum()),
        density_kg_m3=(m / geom(x, new)[0]).tolist(),
        nodal_l1_change_kg=float(sum(abs(delta))),
        first_moment_error_kg_m=[
            float(max(abs(delta @ q[nodes]))) for q in (x, y, end)
        ],
    )
    d["previous_proposal_sha256"] = sha(proposal)
    d["native_route_graph"] = lines[0]
    out.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main(*(Path(v).resolve() for v in sys.argv[1:]))
