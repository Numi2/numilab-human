"""Measure receiver capacity using the retained typed carrier graph and actual emitted volumes."""

import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
from transfer_connected_volume import geometry


def run(data, nodes, owner, executable, out, all_edges):
    source = owner / "refit-density-closed-flow.npz"
    ref = json.loads((owner / "refit-density-closure.json").read_text())
    with source.open("rb") as stream:
        pin = hashlib.file_digest(stream, "sha256").hexdigest()
    if ref["candidate_sha256"] != pin:
        raise ValueError("Carrier allocation changed")
    z = np.load(data)
    allocation = np.load(source)
    volume, q = geometry(np.load(nodes), z["tets"])
    if min(volume) <= 1e-18 or min(q) <= 1e-5:
        raise ValueError("Invalid emitted volume")
    c = allocation["carrier_row"]
    g = allocation["group"]
    cg = allocation["group_for_cell"]
    f = allocation["mass_flow_kg"]
    compensation = allocation["archived_compensation"]
    count = len(allocation["source_parent"])
    lo = np.where(z["tags"] == 15, 900.0, 850.0)
    hi = np.where(z["tags"] == 15, 1100.0, 950.0)
    reserve = ref["reserve_kg_m3"]
    low = np.bincount(cg, weights=volume * (lo + reserve))
    up = np.bincount(cg, weights=volume * (hi - reserve))
    quota = np.bincount(c, weights=f, minlength=count)
    before = np.bincount(g, weights=f, minlength=len(low))
    take = np.flatnonzero(
        ((~compensation.astype(bool)) if all_edges else np.zeros(len(c), bool))
        | (f > 0)
    )
    problem = out.with_suffix(".bin")
    solution = out.with_suffix(".flow")
    cut = out.with_suffix(".cut")
    with problem.open("wb") as stream:
        for array, dt in (
            ([count, len(low), len(take)], "u4"),
            (c[take], "u4"),
            (g[take], "u4"),
            (f[take], "f8"),
            (quota, "f8"),
            (low, "f8"),
            (up, "f8"),
            (compensation[take], "u1"),
        ):
            np.asarray(array, dtype=dt).tofile(stream)
    result = subprocess.run(
        [executable, problem, solution, cut], text=True, capture_output=True, check=True
    )
    flow = f.copy()
    flow[take] = np.fromfile(solution, dtype="f8")
    assigned = np.bincount(g, weights=flow, minlength=len(low))
    q1 = np.bincount(c, weights=flow, minlength=count)
    # After a saturated upper-capacity cut, continue the opposite flow
    # direction from its conserved carrier state. A below-minimum receiver
    # can only grow to its original lower bound. Strict physical bounds
    # are checked afresh for both sides at the end.
    interim = np.bincount(g, weights=flow, minlength=len(low))
    interim_upper = np.maximum(up, interim + np.maximum(2e-16, interim * 1e-12))
    with problem.open("wb") as stream:
        for array, dt in (
            ([count, len(low), len(take)], "u4"),
            (c[take], "u4"),
            (g[take], "u4"),
            (flow[take], "f8"),
            (quota, "f8"),
            (low, "f8"),
            (interim_upper, "f8"),
            (compensation[take], "u1"),
        ):
            np.asarray(array, dtype=dt).tofile(stream)
    second = subprocess.run(
        [executable, problem, solution, cut], text=True, capture_output=True, check=True
    )
    flow[take] = np.fromfile(solution, dtype="f8")
    q1 = np.bincount(c, weights=flow, minlength=count)
    assigned = np.bincount(g, weights=flow, minlength=len(low))
    if (
        not np.isfinite(flow).all()
        or np.any(flow < 0)
        or np.any(flow[compensation.astype(bool)] > f[compensation.astype(bool)])
        or max(abs(q1 - quota)) > 1e-14
    ):
        raise ValueError("Carrier mass changed")

    def measure(y):
        err = np.maximum(low - y, y - up)
        k = np.flatnonzero(err > 1e-15)
        return dict(
            groups=len(k),
            sum_violation_kg=err[k].sum(),
            maximum_group_violation_kg=err.max(),
            largest_witness=int(err.argmax()),
        )

    stat = dict(
        input=pin,
        selected_edges=len(take),
        before=measure(before),
        after=measure(assigned),
        carrier_error_kg=np.max(abs(q1 - quota)),
        carrier_count=count,
        carrier_mass_kg=quota.sum(),
        solver=(result.stdout + second.stdout).strip().splitlines(),
        geometry_sha256=hashlib.sha256(np.load(nodes).tobytes()).hexdigest(),
    )
    np.savez_compressed(
        out.with_suffix(".npz"), group_mass=assigned, lower=low, upper=up
    )
    out.write_text(json.dumps(stat, indent=2) + "\n")
    print(stat, flush=True)


if __name__ == "__main__":
    run(*map(Path, sys.argv[1:6]), "--all" in sys.argv[6:])
