"""Compare the native prefix with its matched loaded-volume control and check rollback."""

import csv
import json
import sys
from pathlib import Path
import numpy as np
from material_context import sha


def read(p):
    return json.loads(p.read_text())


def measure(run, prior, fit, material_control):
    cfg = read(run / "launch.json")
    control = read(prior / "launch.json")
    exe = read(run / "execution.json")
    step = lambda conf, key: conf["argv"][conf["argv"].index(key) + 1]
    if step(cfg, "--muscle-step-seconds") != step(control, "--muscle-step-seconds"):
        raise ValueError("Changed load timestep")
    if sha(Path(cfg["argv"][0])) != read(fit / "installation.json")["binary"]:
        raise ValueError("Changed native owner")
    witness = read(material_control / "launch.json")
    if cfg["parent_launch"] != witness["parent_launch"]:
        raise ValueError("Anatomical registration differs")
    allowed = {
        "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT",
        "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS",
        "NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE",
        "NUMI_HUMAN_SKIN_REFERENCE_DUMP",
        "NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP",
        "NUMI_HUMAN_COUPLED_TRACE_CSV",
        "NUMI_HUMAN_REGISTERED_CELL_NODE_PREFIX",
        "NM_FEM_NEWTON_TRACE_ROOT",
        "NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT",
        "NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT",
        "NM_FEM_BACKTRACKING_TRACE_ROOT",
    }
    changed = {
        k
        for k in cfg["environment"].keys() | witness["environment"].keys()
        if cfg["environment"].get(k) != witness["environment"].get(k)
    }
    if (
        changed - allowed
        or len(cfg["argv"]) != len(witness["argv"])
        or any(
            x != y and i not in (0, 4, cfg["argv"].index("--muscle-step-count") + 1)
            for i, (x, y) in enumerate(zip(cfg["argv"], witness["argv"]))
        )
    ):
        raise ValueError("Changed trial physics")
    observed = sum(
        line.startswith("accepted_fem_body_load_capture=accepted ")
        for line in (run / "stdout.log").read_text().splitlines()
    )
    failure = run / "failure.json"
    if failure.exists():
        f = read(failure)
        n = observed
        reject = dict(
            code=f["matter_code"],
            object=f["matter_object_index"],
            index=f["matter_failing_index"],
            values=f["matter_diagnostics"],
        )
        if (
            n != f["segment_start"]
            or exe["exit"] == 0
            or f["candidate_common_coordinates"] != f["accepted_common_coordinates"]
            or f["respiratory_candidate_status"] != f["respiratory_accepted_status"]
        ):
            raise ValueError("Invalid rejected transaction")
    else:
        r = read(run / "terminal.receipt.json")
        n = r["accepted_step"]
        reject = None
        if (
            n != observed
            or n != int(step(cfg, "--muscle-step-count"))
            or exe["exit"]
            or r["physical_endpoint"] != "accepted"
        ):
            raise ValueError("Uncertified state")
    clock = list(csv.DictReader((run / "native-traces/resting-coupled.csv").open()))
    secs = float(clock[-1]["time_s"]) if clock else 0.0
    if not np.isclose(
        secs, n * float(step(cfg, "--muscle-step-seconds")), rtol=2e-7, atol=1e-14
    ):
        raise ValueError("Controller clock disagrees with accepted mechanics")
    before = read(prior.parent / "six-summary.json")
    return dict(
        matched_control_launch=sha(material_control / "launch.json"),
        parent_launch=sha(prior / "launch.json"),
        trial_launch=sha(run / "launch.json"),
        fitted_pose=sha(fit / "current.node"),
        matched_step_s=float(step(cfg, "--muscle-step-seconds")),
        baseline_steps=before["accepted"],
        trial_steps=n,
        trial_seconds=secs,
        requested_steps=int(step(cfg, "--muscle-step-count")),
        rejection=reject,
        trial_wall_s=exe["wall_s"],
        trial_cold_factor=secs / exe["wall_s"],
        accepted_clock_samples=len(clock),
    )


if __name__ == "__main__":
    print(json.dumps(measure(*(Path(s).resolve() for s in sys.argv[1:])), indent=2))
