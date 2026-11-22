"""Compare matching native accepted tissue states, contact and the six prior limiting elements."""

from pathlib import Path
import json
import re
import sys
import numpy as np
from bind_posterior_equilibrium import coordinates
from fit_dorsal_material import inputs
from fit_native_bed_barrier_load import force
from material_context import sha
from torso_cell_geometry import geom


def compare(run, case, control, base, split, bed, out):
    native = json.loads((run / "native-outcome.json").read_text())
    layer = json.loads((run / "accepted-layers.json").read_text())
    reg = json.loads((case / "supported-registration.json").read_text())
    launch = json.loads((run / "launch.json").read_text())
    step = native["accepted_steps"]
    receipt = json.loads(
        (run / f"native/accepted-geometry/step-{step}.receipt.json").read_text()
    )
    finish = run / "physical-accepted.node"
    other = json.loads((control / "native-outcome.json").read_text())
    parent = json.loads((base / "loaded-cell-limits.json").read_text())
    support = json.loads((bed / "load.json").read_text())
    massfile = case / "fine-mass-routed/current-density-closed-flow.npz"
    if (
        native["exit_code"]
        or int(step) != receipt["accepted_step"]
        or receipt["physical_endpoint"] != "accepted"
        or layer["inputs"]["NATIVE_ACCEPTED_NODE_SHA256"] != sha(finish)
        or layer["inputs"]["NATIVE_LOADED_START_NODE_SHA256"] != reg["current"]
        or launch["pins"].get(str(case / "native-owner/numi-human-native"))
        != reg["native_binary"]
        or not native["variable_pleural_configured"]
        or native["dt_s"] != other["dt_s"]
        or sha(massfile) != reg["source_mass"]
    ):
        raise ValueError("Native state, geometry, mass or operating point mismatch")
    ref, current, tets, mat, ids = inputs(run)
    if (
        ids["reference_sha256"] != parent["source"]["reference_sha256"]
        or ids["elements_sha256"] != parent["source"]["elements_sha256"]
    ):
        raise ValueError("Prior witness indexing differs")
    end = coordinates(finish)
    target = np.array([row["source_row"] for row in parent["rows"]])
    region = tets[target]
    source_volume = geom(ref.astype(float), region)[0]
    startJ = geom(current.astype(float), region)[0] / source_volume
    endJ = geom(end, region)[0] / source_volume
    rows = [
        dict(
            source_row=int(r), role=int(mat[r]), initial_J=float(s), accepted_J=float(e)
        )
        for r, s, e in zip(target, startJ, endJ)
    ]
    outer = np.load(split / "physical.npz")["Orows"]
    mass = np.load(massfile)["nodal_mass_kg"]
    state = {}
    for name, xyz in [("initial", current.astype(float)), ("accepted", end)]:
        active = outer[xyz[outer, 2] < support["band_m"]]
        if xyz[outer, 2].min() <= 0 or (
            name == "initial" and set(active) != set(support["nodes"])
        ):
            raise ValueError("Invalid bed footprint")
        forces = force(xyz[active, 2], mass[active], support["dt_s"], support["band_m"])
        state[name] = dict(
            rows=len(active),
            stationary_bed_N=float(forces.sum()),
            top_ten_fraction=float(np.sort(forces)[-10:].sum() / forces.sum()),
        )
    measured = float(native["bed_reaction"][-1]["mean_normal_force_N"])
    result = dict(
        step=step,
        step_hash=receipt["accepted_root_fingerprint_hex"],
        seconds=native["accepted_seconds"],
        end_to_end_factor=native["end_to_end_factor"],
        physical_cells=native["material_endpoints"],
        layer_pairs=layer["pair_counts"],
        poses=receipt["accepted_body_pose_count"],
        pose=ids["installed_pose_sha256"],
        terminal_native_node_sha256=sha(finish),
        mass=sha(massfile),
        bed=state,
        reported_dynamic_bed_N=measured,
        weight_N=support["weight_N"],
        prior_limiting_cells=rows,
        pleural_samples_pa=native["pleural_samples_pa"],
        variable_pleural_configured=native["variable_pleural_configured"],
    )
    if step == other["accepted_steps"] and other["exit_code"] == 0:
        before = float(other["bed_reaction"][-1]["mean_normal_force_N"])
        result["matched_contact"] = dict(
            control_N=before,
            candidate_N=measured,
            fractional_reduction_of_excess=(before - measured)
            / (before - support["weight_N"]),
        )
    log = (run / "stdout.log").read_text()
    spans = re.findall(
        r"human_execution_stage=native_horizon_end wall_elapsed_ms=([0-9.e+-]+)", log
    )
    if spans:
        result["native_horizon_wall_s"] = float(spans[-1]) / 1000
        result["native_horizon_factor"] = (
            native["accepted_seconds"] / result["native_horizon_wall_s"]
        )
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        step, measured, result.get("matched_contact"), [r["accepted_J"] for r in rows]
    )


if __name__ == "__main__":
    compare(*(Path(p).resolve() for p in sys.argv[1:]))
