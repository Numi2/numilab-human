"""Locate the first recorded stage with a shortened correction at a rejected physical root."""

import json
import sys
from pathlib import Path

from material_context import sha


def read(p):
    return json.loads(p.read_text())


def summarize(trial):
    failure = read(trial / "failure.json")
    root = failure["segment_start"]
    stages = [
        "before_contact_limits",
        "after_deformable_limit",
        "after_rigid_limit",
        "after_support_limit",
        "after_vascular_limit",
        "final_line_search",
    ]
    events = {}
    for line in (trial / "stderr.log").read_text().splitlines():
        k, _, value = line.partition("=")
        if k in (
            "fem_newton_iterate",
            "matter_fgmres_microtick_work",
            "matter_solver_domain_norms",
            "fem_newton_terminal_certificate",
        ):
            record = json.loads(value)
            if record["root"] == root:
                events.setdefault(k, []).append(record)
    if (
        len(events["matter_fgmres_microtick_work"]) != 1
        or len(events["matter_solver_domain_norms"]) != 1
    ):
        raise ValueError("Ambiguous physical root")
    (work,) = events["matter_fgmres_microtick_work"]
    rows = events["fem_newton_iterate"]
    result = []
    for i in range(work["active_newton_begins"]):
        snapshots = [r for r in rows if r["iteration"] == i]
        if [r["stage"] for r in snapshots] != stages or any(
            r["environment_count"] != 1 or not r["object_line_search"]
            for r in snapshots
        ):
            raise ValueError("Incomplete Newton snapshots")
        s = snapshots[0]
        fractions = [
            min(r["alpha"][0], min(v[0] for v in r["object_line_search"]))
            for r in snapshots
        ]
        result.append(
            {
                "iteration": i,
                "physical_residual": s["fgmres"][0]["nonlinear"][1],
                "minimum_fraction_per_stage": dict(zip(stages, fractions, strict=True)),
            }
        )
    return {
        "failure": sha(trial / "failure.json"),
        "log": sha(trial / "stderr.log"),
        "root": root,
        "requested_steps": int(
            read(trial / "launch.json")["argv"][
                read(trial / "launch.json")["argv"].index("--muscle-step-count") + 1
            ]
        ),
        "rollback_common_coordinates_equal": failure["candidate_common_coordinates"]
        == failure["accepted_common_coordinates"],
        "rollback_respiratory_status_equal": failure["respiratory_accepted_status"]
        == failure["respiratory_candidate_status"],
        "domain_norms": events["matter_solver_domain_norms"][0],
        "work": work,
        "iterations": result,
        "already_limited_before_contact": all(
            min(r["minimum_fraction_per_stage"].values())
            == r["minimum_fraction_per_stage"][stages[0]]
            < 1
            for r in result
        ),
    }


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1]).resolve()), indent=2, allow_nan=False))
