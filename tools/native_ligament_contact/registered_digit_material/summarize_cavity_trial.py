"""Trace a composed mesh through the native element ordering and fresh certificate."""

from pathlib import Path
import json, sys
import numpy as np
from fit_dorsal_material import inputs
from material_context import sha
from torso_cell_geometry import geom


def extract(path, name):
    prefix = name + "="
    return [json.loads(l[len(prefix) :]) for l in path.open() if l.startswith(prefix)]


def summarize(run, control, original_map, proposal, output):
    log = run / "stderr.log"
    j = json.loads((run / "launch.json").read_text())
    if sha(proposal) != j["environment"]["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256"]:
        raise ValueError("Wrong composed topology")
    fail = json.loads((run / "failure.json").read_text())
    prior = json.loads((control / "failure.json").read_text())
    for key in (
        "matter_code",
        "matter_diagnostics",
        "segment_start",
        "accepted_common_coordinates",
    ):
        if fail[key] != prior[key]:
            raise ValueError("Different replay outcome " + key)
    old = json.loads(original_map.read_text())
    x, y, t, role, pins = inputs(run)
    if pins != old["source"]:
        raise ValueError("Material input changed")
    with np.load(proposal) as a:
        kept = a["retained_input_cell_rows"]
        new = a["replacement_tetrahedra"]
        parent = a["replacement_parent_cell_rows"]
        offset = a["replacement_parent_offsets"]
    bt = extract(log, "fem_backtracking_trials")
    if len(bt) != 1 or bt[0]["object"] != 15:
        raise ValueError("Expected the material consumer")
    steps = extract(log, "fem_newton_iterate")
    stages = [
        dict(
            iteration=d["iteration"],
            stage=d["stage"],
            global_alpha=d["alpha"][0],
            material_line_search=d["object_line_search"][15],
            solver=d["fgmres"][0],
        )
        for d in steps
    ]
    case = bt[0]
    w = []
    start = old["source_element_offset"]
    x = x.astype(float)
    y = y.astype(float)
    for s in case["trials"]:
        if not s["tile_witnesses_complete"]:
            raise ValueError("Incomplete native witnesses")
        for g in s["tile_witness_tets"]:
            i = g - start
            if i < 0 or i >= len(kept) + len(new):
                raise ValueError("Missing material row")
            if i < len(kept):
                orig = [int(kept[i])]
                nodes = t[orig[0]]
            else:
                local = i - len(kept)
                nodes = new[local]
                orig = parent[offset[local] : offset[local + 1]].tolist()
            if any(q["native_global_element"] == g for q in w):
                continue
            V, q = geom(x, nodes[None])
            U, _ = geom(y, nodes[None])
            w.append(
                dict(
                    native_global_element=g,
                    composed_material_row=i,
                    parents=orig,
                    exactly_retained=i < len(kept),
                    material=int(role[orig[0]]),
                    nodes=nodes.tolist(),
                    reference_shape=float(q[0]),
                    installed_volume_ratio=float(U[0] / V[0]),
                )
            )
    tcert = extract(log, "fem_newton_terminal_certificate")
    output.write_text(
        json.dumps(
            dict(
                composition_sha256=sha(proposal),
                material_sources=pins,
                native_global_element_offset=start,
                trials=case,
                material_witnesses=w,
                stages=stages,
                terminal=[
                    dict(
                        root=c["root"],
                        material_volume_ratio_min=c["objects"][15]["validity"][0],
                        fresh_residual=c["objects"][15]["nonlinear"][0],
                    )
                    for c in tcert
                ],
                deformable_limits=extract(log, "fem_deformable_limit_minimum"),
                replay_failure=fail["matter_diagnostics"],
                accepted_steps=fail["segment_start"],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    summarize(*(Path(s).resolve() for s in sys.argv[1:]))
