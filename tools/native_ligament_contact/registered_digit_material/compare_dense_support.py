"""Compare the physical skin support trial with its matched native control and observer."""

import json
import re
import sys
from pathlib import Path

import numpy as np
from material_context import sha


def read(p):
    return json.loads(p.read_text())


def observed(path):
    lines = (path / "stdout.log").read_text().splitlines()
    state = read(path / "terminal.receipt.json")
    base = next(
        l.split("=", 1)[1]
        for l in lines
        if l.startswith("persistent_initial_force_reference=")
    )
    gravity = next(
        r["gravity_target_n"] for r in json.loads(base)["rows"] if r["dof"] == 2
    )
    out = {}
    for line in lines:
        if line.startswith("accepted_fem_body_load_capture=accepted "):
            step = int(re.search("source_step=(\\d+)", line)[1])
            root = json.loads(
                re.search(r"root_generalized_force_xyz_n=(\[[^\]]+\])", line)[1]
            )
            if step in out or len(root) != 3 or not np.isfinite(root).all():
                raise ValueError("Invalid physical load sample")
            out[step] = root
    if state["physical_endpoint"] != "accepted" or sorted(out) != list(
        range(state["accepted_step"])
    ):
        raise ValueError("Incomplete accepted skeleton loads")
    return gravity, out


def compare(control, observer, candidate, base_bed, new_bed):
    A, B = [read(p / "launch.json") for p in (control, candidate)]
    a, b = [read(p / "terminal.receipt.json") for p in (control, candidate)]
    o = read(observer / "terminal.receipt.json")
    key = "NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"
    flag = "NUMI_HUMAN_ACCEPTED_FEM_BODY_LOAD_CAPTURE"
    # Sample recording must not alter the reference dynamics.
    if not (
        a["accepted_body_poses"] == o["accepted_body_poses"]
        and sha(control / "physical-accepted.node")
        == sha(observer / "physical-accepted.node")
        and sha(control / "native-traces/resting-coupled.csv")
        == sha(observer / "native-traces/resting-coupled.csv")
    ):
        raise ValueError("Unmatched load observer")
    controls = []
    for job in (A, B):
        e = job["environment"].copy()
        pose = e.pop(key)
        e.pop(flag, None)
        tmp = str(Path(job["argv"][4]).parent)
        pins = job["pins"].copy()
        for p in (pose, job["argv"][0]):
            if sha(Path(p)) != pins.pop(p):
                raise ValueError("Native or coordinate digest mismatch")
        controls.append(
            (
                job["cwd"],
                [v.replace(tmp, "<out>") for v in job["argv"][1:]],
                {k: v.replace(tmp, "<out>") for k, v in e.items()},
                pins,
            )
        )
    if (
        controls[0] != controls[1]
        or a["accepted_step"] != b["accepted_step"]
        or a["accepted_time_s"] != b["accepted_time_s"]
        or B["environment"].get(flag) != "1"
    ):
        raise ValueError("Run configuration changed beyond the registered experiment")
    target, root0 = observed(observer)
    g, root1 = observed(candidate)
    if g != target:
        raise ValueError("Reference gravity changed")
    u, v = [read(p) for p in (base_bed, new_bed)]
    for d, path in [(u, control), (v, candidate)]:
        if d["receipt"] != sha(path / "terminal.receipt.json") or d[
            "source_launch"
        ] != sha(path / "launch.json"):
            raise ValueError("Different pressure or force report")
    if (
        u["weight_N"] != v["weight_N"]
        or u["initial_rows"] != v["initial_rows"]
        or abs(u["initial_barrier_N"] - v["initial_barrier_N"]) > 1e-9
    ):
        raise ValueError("Initial physical contact changed")
    w = v["weight_N"]
    rows = []
    den = b["accepted_step"] - 1
    for path, bed, root in [(control, u, root0), (candidate, v, root1)]:
        result = read(path / "verified-endpoint.json")
        trace = (path / "stdout.log").read_text()
        times = [
            float(v)
            for v in re.findall(
                r"^human_execution_stage=native_horizon_(?:begin|end) wall_elapsed_ms=([\d.eE+-]+)",
                trace,
                re.MULTILINE,
            )
        ]
        if len(times) != 2 or times[1] <= times[0]:
            raise ValueError("Incomplete native horizon measurement")
        rows.append(
            {
                "launch": sha(path / "launch.json"),
                "bed_N": bed["native"]["mean_normal_force_N"],
                "bed_excess_N": bed["native"]["mean_normal_force_N"] - w,
                "root_FEM_z_N": root[den][2],
                "root_load_by_source_step": root,
                "top_four_fraction": bed["top_four_endpoint_load_fraction"],
                "checked_geometry": result["endpoint_anatomy"],
                "accepted_time_s": result["simulated_s"],
                "cold_s": result["complete_wall_s"],
                "cold_factor": result["end_to_end_factor"],
                "native_horizon_s": (times[1] - times[0]) / 1000,
            }
        )
    old, new = rows
    reduction = 1 - abs(new["bed_excess_N"]) / abs(old["bed_excess_N"])
    root_improves = abs(new["root_FEM_z_N"] - g) < abs(old["root_FEM_z_N"] - g)
    return {
        "control": old,
        "candidate": new,
        "weight_N": w,
        "skeletal_gravity_target_N": g,
        "starting_barrier_N": v["initial_barrier_N"],
        "starting_bed_rows": v["initial_rows"],
        "excess_fraction_removed": reduction,
        "predicted_20_percent": reduction >= 0.2,
        "skeletal_support_error_improved": root_improves,
        "physiology_trace_identical": sha(control / "native-traces/resting-coupled.csv")
        == sha(candidate / "native-traces/resting-coupled.csv"),
    }


if __name__ == "__main__":
    print(
        json.dumps(
            compare(*(Path(v).resolve() for v in sys.argv[1:])),
            indent=2,
            allow_nan=False,
        )
    )
