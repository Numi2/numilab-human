"""Measure the differential response of the same accepted body to a registered bed displacement."""

import json
import sys
from pathlib import Path

import numpy as np
from compile_connected_joint import digest

from measure_applied_route import get, rows


def compare(control, trial, normal, work, out):
    test = get(trial / "launch.json")
    if test["bed_pair"]["control"] != digest(control / "launch.json"):
        raise ValueError("Different bed experiment")
    a = get(work / "matched.json")
    b = get(work / "paired-forces.json")
    if [c["launch"] for c in (a, b)] != [
        digest(x / "launch.json") for x in (control, trial)
    ]:
        raise ValueError("Unmatched measurements")
    n = np.array(get(normal)["bed_normal"])
    per = []
    for x, y in zip(a["source_steps"], b["source_steps"], strict=True):
        k = x["source_step"]
        ids, r = rows(control, k)
        other, s = rows(trial, k)
        ia, ib = (
            np.union1d(ids[:, 0], other[:, 0]),
            np.intersect1d(ids[:, 0], other[:, 0]),
        )
        # Physical cached forces are already zero for absent unilateral rows.
        q = np.zeros((2, len(ia), 3))
        for p, (i, z) in enumerate(((ids, r), (other, s))):
            q[p, np.searchsorted(ia, i[:, 0])] = z[:, 3, :3]
        dq = q[1] - q[0]
        rigid = []
        for body in sorted(set(x["by_body"]) | set(y["by_body"]), key=int):
            f = [
                np.asarray(
                    v["by_body"].get(body, {}).get("resultant_on_bone_N", [0, 0, 0])
                )
                for v in (x, y)
            ]
            rigid.append((body, (f[1] - f[0]).tolist()))
        per.append(
            dict(
                step=k,
                matching_rows=len(ib),
                union_rows=len(ia),
                same_ids=bool(np.array_equal(ids, other)),
                same_row_values=bool(np.array_equal(r, s)),
                max_row_force_delta_N=float(np.linalg.norm(dq, axis=1).max()),
                skin_fat_all_delta_normal_N=y["skin_fat_all"]["bone_bed_normal_N"]
                - x["skin_fat_all"]["bone_bed_normal_N"],
                skin_fat_to_scapula_delta_normal_N=y["skin_fat_to_scapula"][
                    "bone_bed_normal_N"
                ]
                - x["skin_fat_to_scapula"]["bone_bed_normal_N"],
                bone_vector_deltas_N={i: f for i, f in rigid if any(f)},
            )
        )
    x, y = [
        np.loadtxt(p / "physical-accepted.node", skiprows=1, usecols=(1, 2, 3))
        for p in (control, trial)
    ]
    delta = y - x
    dist = np.linalg.norm(delta, axis=1)
    peak = int(dist.argmax())
    node = a["source_bed_node"]
    result = dict(
        control=a["launch"],
        trial=b["launch"],
        bed_normal=n.tolist(),
        bed_load_N=[a["bed_N"], b["bed_N"]],
        bed_load_delta_N=b["bed_N"] - a["bed_N"],
        terminal_step_skin_nodes=[a["bed_active_nodes"], b["bed_active_nodes"]],
        max_skin_displacement_difference_m=float(dist[peak]),
        max_displacement_node=peak,
        dominant_bed_node_delta_m=delta[node].tolist(),
        scapular_sample_delta_m=delta[a["scapula_closest_sample"]].tolist(),
        sources=per,
        accepted_material=[a["emitted_end"], b["emitted_end"]],
        scope="Force source i is assembled from the beginning of accepted step i+1. Bed load is the terminal impulse divided by the physical root duration. This short perturbation resolves an initial response only.",
    )
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    compare(*(Path(v).resolve() for v in sys.argv[1:]))
