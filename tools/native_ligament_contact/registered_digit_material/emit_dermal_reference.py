"""Emit source/loaded reference-star coordinates and the conservative per-cell mass candidate."""

import json
import sys
from pathlib import Path

import numpy as np
from emit_constrained_material import write_coordinates
from fit_dorsal_material import inputs
from material_context import sha


def emit(native, material, axis_fit, joint_fit, out):
    x, y, t, tag, pins = inputs(native)
    x = x.astype(float)
    y = y.astype(float)
    A, B = [json.loads(p.read_text()) for p in (axis_fit, joint_fit)]
    if A["source"] != pins or B["source"] != pins or B["control"] != sha(axis_fit):
        raise ValueError("Unmatched matched-pose fits")
    flow = material / "fine-mass-routed/current-density-closed-flow.npz"
    if sha(flow) != A["mass"]:
        raise ValueError("Changed source masses")
    with np.load(flow) as f:
        oldmass = f["element_mass_kg"]
        groups = f["group_for_cell"]
    m = oldmass.copy()
    x0 = x.copy()
    y0 = y.copy()
    prior = np.load(material / "proposal.npz")
    edits = {}
    retired = set()
    cases = []
    joint = {r["row"]: r for r in B["results"]}
    for a in A["results"]:
        row = a["source_row"]
        free = np.array(a["free"])
        candidates = []
        for attempt in a["attempts"]:
            for v in attempt["trials"]:
                if v["status"] == "passed":
                    moves = np.zeros((len(free), 3))
                    moves[:, v["axis"]] = v["shift_m"]
                    candidates.append(
                        (
                            moves,
                            np.array(attempt["cells"]),
                            v["new_mass"],
                            v["mechanical_checks"],
                        )
                    )
        if not candidates:
            for v in joint[row]["attempts"]:
                if v["status"] == "passed":
                    candidates.append(
                        (
                            np.array(v["moves_m"]),
                            np.array(v["cells"]),
                            v["mass_kg"],
                            v["region_checks"],
                        )
                    )
        moves, cells, masses, mech = min(
            candidates, key=lambda c: np.linalg.norm(c[0], axis=1).max()
        )
        if retired.intersection(cells.tolist()) or not np.all(tag[cells] == 15):
            raise ValueError("Overlapping or foreign material")
        retired.update(cells.tolist())
        m[cells] = masses
        for n, move in zip(free, moves):
            if np.any(move):
                if n in edits:
                    raise ValueError("Duplicate point")
                edits[int(n)] = move
                for s in (x, y):
                    s[n] = (s[n] + move).astype("f4").astype(float)
        cases.append(
            {
                "source_cell": row,
                "force_and_energy": mech,
                "changed_nodes": int(np.count_nonzero(np.linalg.norm(moves, axis=1))),
            }
        )
    ids = np.array(sorted(edits))
    cells = np.array(sorted(retired))
    if not np.array_equal(x - x0, y - y0) or not np.array_equal(
        ids, np.flatnonzero(np.any(x != x0, 1))
    ):
        raise ValueError("Material and loaded offsets differ")
    group_error = float(
        np.max(abs(np.bincount(groups[cells], weights=(m - oldmass)[cells])))
    )
    moments = []
    for before, after in [(x0, x), (y0, y)]:
        delta = (m[cells] - oldmass[cells]) @ before[t[cells]].mean(1) + m[cells] @ (
            after[t[cells]] - before[t[cells]]
        ).mean(1)
        moments.append(delta.tolist())
    if group_error > 2e-12 or np.max(abs(np.array(moments))) > 1e-10:
        raise ValueError("Source group or mechanical first moment changed")
    out.mkdir(exist_ok=False)
    env = json.loads((native / "launch.json").read_text())["environment"]
    P = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    for s, key, name in [
        (x, P + "TETGEN_NODE", "reference.node"),
        (y, "NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE", "current.node"),
    ]:
        write_coordinates(Path(env[key]), out / name, ids, s[ids])
    affected = np.unique(np.r_[prior["cell"], np.flatnonzero(np.isin(t, ids).any(1))])
    nodes = np.union1d(prior["nodes"], np.unique(t[affected]))
    np.savez_compressed(
        out / "proposal.npz",
        nodes=nodes,
        cell=affected,
        x=x[nodes].astype("f4"),
        t=np.searchsorted(nodes, t[affected]).astype("i4"),
        role=tag[affected],
    )
    np.savez_compressed(
        out / "mass-patch.npz",
        cells=cells,
        element_mass_kg=m[cells],
        old_mass_kg=oldmass[cells],
    )
    report = {
        "source": pins,
        "reference": sha(out / "reference.node"),
        "current": sha(out / "current.node"),
        "proposal": sha(out / "proposal.npz"),
        "mass_patch": sha(out / "mass-patch.npz"),
        "prior_flow": sha(flow),
        "rows": ids.tolist(),
        "moves_m": [edits[i].tolist() for i in ids],
        "max_source_group_mass_change_kg": group_error,
        "first_moment_changes_kg_m": moments,
        "cases": cases,
    }
    report["native_source"] = pins
    report["regions"] = [
        {"axis": j, "rows": [int(i)], "movement_m": [float(edits[i][j])]}
        for i in ids
        for j in range(3)
        if edits[i][j] != 0
    ]
    (out / "material-identity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "material coordinates",
        len(ids),
        "source group mass error",
        group_error,
        "moments",
        moments,
        flush=True,
    )


if __name__ == "__main__":
    emit(*(Path(v).resolve() for v in sys.argv[1:]))
