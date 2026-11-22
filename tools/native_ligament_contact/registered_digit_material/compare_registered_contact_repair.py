"""Verify the measured vertex/face clearance and coupled response at matching accepted endpoints."""

import json, sys
from pathlib import Path
import numpy as np
from material_context import sha


def compare(initial, modified, fit, source_trace, new_trace):
    patch = json.loads(fit.read_text())
    ids = patch["physical_rows"]
    results = []
    for run in (initial, modified):
        r = json.loads((run / "native-outcome.json").read_text())
        m = json.loads((run / "matched-digital-interfaces.json").read_text())[
            "repaired"
        ]
        node = run / "physical-accepted.node"
        if (
            r["exit_code"]
            or r["accepted_steps"] != m["step"]
            or sha(node) != m["accepted_material"]
        ):
            raise ValueError("Unbound material endpoint")
        v = np.loadtxt(node, skiprows=1, usecols=(1, 2, 3))[ids]
        a, b = v[2] - v[1], v[3] - v[1]
        n = np.cross(a, b)
        n /= np.linalg.norm(n)
        coords = np.linalg.lstsq(np.stack((a, b), axis=1), v[0] - v[1], rcond=None)[0]
        bary = np.array([1 - sum(coords), *coords])
        if bary.min() < 0:
            raise ValueError("Closest point moved outside the physical triangle")
        results.append(
            dict(
                pack=m["pack"],
                step=m["step"],
                time_s=m["time_s"],
                distance_m=float(abs((v[0] - v[1]) @ n)),
                barycentric=bary.tolist(),
                bed_reaction_N=float(r["bed_reaction"][0]["mean_normal_force_N"]),
            )
        )
    if (
        results[0]["step"] != results[1]["step"]
        or results[0]["time_s"] != results[1]["time_s"]
    ):
        raise ValueError("Different physical horizons")

    def minima(run):
        out = []
        for line in (run / "stderr.log").open():
            if line.startswith("fem_deformable_limit_minimum="):
                out.append(json.loads(line.partition("=")[2]))
        if [r["iteration"] for r in out] != list(range(16)) or any(
            r["root"] != 1 or r["microtick"] for r in out
        ):
            raise ValueError("Incomplete matching nonlinear solves")
        return out

    old, new = minima(source_trace), minima(new_trace)
    return dict(
        before=results[0],
        after=results[1],
        physical_rows=ids,
        original_limiting_contact_iterations=sum(
            r["contact_slot"] == old[0]["contact_slot"] for r in old
        ),
        same_slot_limiting_after_repair=sum(
            r["contact_slot"] == old[0]["contact_slot"] for r in new
        ),
        same_slot_scope="Both runs have the same indexed material topology. The repaired gap is also verified directly with the four physical nodes.",
        original_minima=old,
        repaired_minima=new,
    )


if __name__ == "__main__":
    a, b, c, d, e, f = map(Path, sys.argv[1:])
    f.write_text(json.dumps(compare(a, b, c, d, e), indent=2) + "\n")
