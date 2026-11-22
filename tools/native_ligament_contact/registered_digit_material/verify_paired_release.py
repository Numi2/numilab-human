"""Compare a two-step release after updating the material reference and carrier mapping."""

import json
import re
import sys
from pathlib import Path

import numpy as np
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha
from supported_material_admission import (
    installed_topology,
    material_state,
    stress_limits,
)

from compare_dense_support import observed, read


def verify(control, trial, model, bed, out):
    control_job = read(control / "launch.json")
    job = read(trial / "launch.json")
    parent = read(model / "qualified-parent/launch.json")
    registration = read(model / "material-identity.json")
    role = read(model / "material/native-role-verification.json")
    if (
        not role["passed"]
        or sha(model / "material/native-role-verification.json")
        != parent["connected_material_volume"]["role_reader_sha256"]
    ):
        raise ValueError("Role writer/reader attribution failed")
    source = read(control / "terminal.receipt.json")
    rec = read(trial / "terminal.receipt.json")
    for j in (job, control_job):
        if any(sha(Path(p)) != h for p, h in j["pins"].items()):
            raise ValueError("Different pinned physical assets")

    def argv(j):
        return [v for i, v in enumerate(j["argv"]) if i not in (0, 4)]

    if (
        argv(job) != argv(control_job)
        or rec["accepted_step"] != source["accepted_step"]
        or rec["accepted_time_s"] != source["accepted_time_s"]
    ):
        raise ValueError("Different physical clocks, muscles or anatomy")

    def cfg(j):
        allowed = set(parent["connected_material_volume"]["changed_keys"])
        for k in allowed:
            if job["environment"][k] != parent["environment"][k]:
                raise ValueError(f"Unknown material operator {k}")
        tmp = str(Path(j["argv"][4]).parent)
        return {
            k: v.replace(tmp, "<capture>")
            for k, v in j["environment"].items()
            if k not in allowed and k != "NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"
        }

    if cfg(job) != cfg(control_job):
        raise ValueError("Other physical operators changed")
    ep = read(trial / "verified-endpoint.json")
    p = read(bed)
    if p["receipt"] != sha(trial / "terminal.receipt.json") or ep[
        "physical_nodes_sha256"
    ] != sha(trial / "physical-accepted.node"):
        raise ValueError("Stale native state")
    x, _, t, r, pins = inputs(trial)
    if (
        pins["reference_sha256"] != registration["reference"]
        or pins["installed_pose_sha256"] != registration["current"]
    ):
        raise ValueError("Wrong native physical pose")
    t, r = installed_topology(t, r, job["environment"])
    props, mat = materials(Path(job["argv"][0]).parent)
    caps = stress_limits(mat)
    coords = np.loadtxt(trial / "physical-accepted.node", skiprows=1, usecols=(1, 2, 3))
    stats = {
        k: {"outside_registration": 0, "outside_stress_cap": 0, "severe": 0}
        for k in (15, 16)
    }
    for n in range(0, len(t), 32768):
        c = t[n : n + 32768]
        roles = r[n : n + len(c)]
        _, s, stress = material_state(x[c].astype(float), coords[c], props[roles - 15])
        for k in (15, 16):
            m = roles == k
            for name, v in [
                ("outside_registration", (s[:, 0] > 1.5) | (s[:, -1] < 2 / 3)),
                ("severe", (s[:, 0] > 3.0) | (s[:, -1] < 1 / 3)),
                ("outside_stress_cap", stress > caps[k - 15]),
            ]:
                stats[k][name] += int(np.sum(m & v))
    m = read(model / "fine-mass-routed/mass-transfer.json")
    raw = (trial / "stdout.log").read_text()
    horizon = [
        float(v)
        for v in re.findall(
            r"^human_execution_stage=native_horizon_(?:begin|end) wall_elapsed_ms=([\d.eE+-]+)",
            raw,
            re.MULTILINE,
        )
    ]
    if len(horizon) != 2 or horizon[1] <= horizon[0]:
        raise ValueError("Missing physical timing")
    g, root = observed(trial)
    trace = trial / "native-traces/resting-coupled.csv"
    result = {
        "material": registration,
        "role_reader": sha(model / "material/native-role-verification.json"),
        "launch": sha(trial / "launch.json"),
        "terminal": ep,
        "unchanged_physical_arguments": True,
        "carrier_assignments_unchanged": m["changed_edges"] == 0,
        "native_density_outliers": m["density_outliers"],
        "poses": str(pins),
        "stress_and_strain": stats,
        "gravity_N": g,
        "root_by_step": root,
        "weight_N": p["weight_N"],
        "initial_bed_N": p["initial_barrier_N"],
        "terminal_bed_N": p["native"]["mean_normal_force_N"],
        "horizon_s": (horizon[1] - horizon[0]) / 1000,
        "body_factor": rec["accepted_time_s"] / ((horizon[1] - horizon[0]) / 1000),
        "coupled_trace_sha256": sha(trace),
        "accepted_variable_pleural_owner": "resting_vascular_external_pressure=accepted_absolute_pleural_gauge"
        in raw,
    }
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    verify(*(Path(v).resolve() for v in sys.argv[1:]))
