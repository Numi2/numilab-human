"""Compare per-forearm pressure resultants at the same accepted substeps, in metres and newtons."""

import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from registered_forearms import read, sha
from compare_supported_forearm_runs import table


def centers(case):
    w = table(case / "resting-support-selected-skin-witness.csv")
    j = table(case / "resting-com-support-impulses.csv")
    t = read(case / "terminal.receipt.json")
    args = read(case / "launch.json")["argv"]
    dt = float(args[args.index("--muscle-step-seconds") + 1])
    if dt <= 0 or t["accepted_step"] != 8000:
        raise ValueError("Different trajectory")
    group = defaultdict(list)
    for x, y in zip(w, j, strict=True):
        step = int(y["accepted_step"])
        sid = int(y["source_geometry_index"])
        if (
            step != int(x["accepted_step"])
            or step != int(x["pre_dynamics_step_index"]) + 1
            or x["region"] != y["contact_index"]
            or sid != int(x["source_geometry_index"])
        ):
            raise ValueError("Unmatched impulse point")
        if sid not in (6890, 47596):
            continue
        p = np.array([float(x["selected_point_world_" + k + "_m"]) for k in "xyz"])
        F = float(y["normal_impulse_ns"]) / dt
        if F < 0 or not np.all(np.isfinite(np.r_[p, F])):
            raise ValueError("Invalid physical sample")
        group[sid, step].append((F, p))
    result = {}
    for sid in (6890, 47596):
        samples = []
        prev = None
        jumps = []
        for step in range(32, t["accepted_step"] + 1, 32):
            part = group[sid, step]
            normal = sum(F for F, p in part)
            if not len(part):
                raise ValueError("Missing source region")
            if normal <= 0:
                prev = None
                continue
            center = sum(F * p for F, p in part) / normal
            if prev is not None:
                jumps.append(
                    (float(np.linalg.norm(center[:2] - prev[:2]) * 1000), step)
                )
            prev = center
            samples.append(
                dict(
                    step=step,
                    normal_N=normal,
                    center_m=center.tolist(),
                    normal_joint_rows=sum(F > 0 for F, _ in part),
                )
            )
        final = [s for s in samples if s["step"] * dt > 15]
        result[sid] = dict(
            samples_with_force=len(samples),
            max_simultaneously_loaded=max(
                (q["normal_joint_rows"] for q in samples), default=0
            ),
            mean_last_second_N=sum(s["normal_N"] for s in final)
            / sum(k * dt > 15 for k in range(32, t["accepted_step"] + 1, 32)),
            consecutive_loaded_CoP_jumps=len(jumps),
            median_CoP_change_mm=float(np.median([v for v, _ in jumps]))
            if jumps
            else None,
            max_CoP_change_mm=max(jumps, default=(None, None))[0],
            max_change_step=max(jumps, default=(None, None))[1],
        )
    return dict(
        pack=t["pack_file_sha256"],
        witness_sha256=sha(case / "resting-support-selected-skin-witness.csv"),
        impulse_sha256=sha(case / "resting-com-support-impulses.csv"),
        sample_interval_s=32 * dt,
        sources=result,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("control", "candidate"):
        p.add_argument(k, type=Path)
    args = p.parse_args()
    print(
        json.dumps(
            dict(control=centers(args.control), candidate=centers(args.candidate)),
            indent=2,
        )
    )
