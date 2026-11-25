"""Compare the constitutive response of two accepted native skin/fat volumes."""

import argparse
import json
from pathlib import Path

import numpy as np

from registered_cavity_materials import apply_registered_cavities
from normal_gravity_step import constitutive_gradient
from solve_connected_normal_transfer import digest, material_parameters


def load_endpoint(run, cortex_report):
    manifest = run / "launch.json"
    cfg = json.loads(manifest.read_bytes())
    env = cfg["environment"]
    observation = json.loads(cortex_report.read_bytes())["accepted_observation"]
    state = run / "physical-accepted.node"
    receipt = run / "native-traces/accepted-geometry" / f"step-{observation['step']}.receipt.json"
    if (observation["launch_sha256"] != digest(manifest)
            or observation["accepted_node_sha256"] != digest(state)
            or observation["receipt_sha256"] != digest(receipt)
            or observation["time_s"] != json.loads(receipt.read_bytes())["accepted_time_s"]):
        raise ValueError("Endpoint differs from the audited native material capture")

    def verify(path, pinned=True):
        path = Path(path)
        if ((pinned or str(path) in cfg["pins"])
                and cfg["pins"].get(str(path)) != digest(path)):
            raise ValueError(f"Changed native input {path}")
        return path

    prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_"
    reference = verify(env[prefix + "NODE"])
    raw = np.loadtxt(verify(env[prefix + "ELEMENTS"]), skiprows=1,
                     usecols=(1, 2, 3, 4, 5), dtype=np.int64)
    cells, labels = raw[:, :4] - 1, raw[:, 4]
    if "NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ" in env:
        with np.load(verify(env["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ"])) as cavity:
            cells, labels = apply_registered_cavities(cells, labels, cavity)
    if set(np.unique(labels)) != {15, 16}:
        raise ValueError("Unrecognized tissue material set")
    positions = np.loadtxt(state, skiprows=1, usecols=(1, 2, 3))
    return (observation, reference, positions, cells, labels, material_parameters(cfg, verify))


def regional_response(reference, positions, cells, labels, params):
    result = {}
    for label in (15, 16):
        selected = labels == label
        tets = cells[selected]
        materials = labels[selected]
        columns = []
        for axis in np.eye(3):
            column, energy = constitutive_gradient(reference, positions, tets, axis, materials, params)
            columns.append(column)
        result[label] = np.column_stack(columns), energy
    return result


def differences(base, trial):
    results = {}
    for label in (15, 16):
        force, energy = base[label]
        after, final_energy = trial[label]
        delta = np.linalg.norm(after - force, axis=1)
        peak = int(np.argmax(delta))
        denominator = np.linalg.norm(force)
        results[label] = dict(control_J=energy, candidate_J=final_energy,
                              relative_energy=abs(final_energy-energy)/abs(energy) if energy else None,
                              relative_force_L2=float(np.linalg.norm(delta)/denominator) if denominator else None,
                              force_delta_L2_N=float(np.linalg.norm(delta)),
                              peak_nodal_force_delta_N=float(delta[peak]), peak_row=peak)
    return results


def compare(control, trial, control_report, trial_report):
    a, b = load_endpoint(control, control_report), load_endpoint(trial, trial_report)
    if ((a[0]["step"], a[0]["time_s"]) != (b[0]["step"], b[0]["time_s"])
            or digest(a[1]) != digest(b[1]) or a[-1] != b[-1]):
        raise ValueError("Different physical clocks, reference coordinates or constitutive laws")
    reference = np.loadtxt(a[1], skiprows=1, usecols=(1, 2, 3)).astype("f4").astype(float)
    before, after = [regional_response(reference, *entry[2:]) for entry in (a, b)]
    return dict(control_launch=a[0]["launch_sha256"], trial_launch=b[0]["launch_sha256"],
                reference=digest(a[1]), matched_time_s=a[0]["time_s"], regions=differences(before, after))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "trial", "control_report", "trial_report", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    report = compare(args.control, args.trial, args.control_report, args.trial_report)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
