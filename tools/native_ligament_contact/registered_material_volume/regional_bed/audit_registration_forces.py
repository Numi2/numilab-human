"""Compare native skin/fat stored energy and nodal forces for a sparse registration."""

import argparse
import json
from pathlib import Path

import numpy as np

from normal_gravity_step import constitutive_gradient
from solve_connected_normal_transfer import digest, material_parameters


def compare(reference, current, cells, labels, params, rows, delta):
    rows = np.asarray(rows)
    delta = np.asarray(delta)
    if (
        rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer)
        or len(np.unique(rows)) != len(rows)
        or np.any((rows < 0) | (rows >= len(current)))
        or delta.shape != (len(rows), 3) or not np.isfinite(delta).all()
    ):
        raise ValueError("Invalid or duplicate material displacement rows")
    which = np.isin(cells, rows).any(axis=1)
    patch = cells[which]
    nodes, local = np.unique(patch, return_inverse=True)
    local = local.reshape(-1, 4)
    if len(np.setdiff1d(rows, nodes)):
        raise ValueError("Moved node has no incident material")
    start = current[nodes]
    end = start.copy()
    end[np.searchsorted(nodes, rows)] += delta
    forces = np.zeros_like(start)
    energy = {}
    for tag in (15, 16):
        selected = labels[which] == tag
        t = local[selected]
        material = np.full(len(t), tag)
        if not len(t):
            energy[str(tag)] = [0., 0.]
            continue
        for i, axis in enumerate(np.eye(3)):
            before, initial_energy = constitutive_gradient(
                reference[nodes], start, t, axis, material, params
            )
            after, final_energy = constitutive_gradient(
                reference[nodes], end, t, axis, material, params
            )
            forces[:, i] += after - before
        energy[str(tag)] = [initial_energy, final_energy]
    change = np.linalg.norm(forces, axis=1)
    worst = np.argsort(-change, kind="stable")[:10]
    return {
        "changed_nodes": len(rows),
        "incident_cells": int(which.sum()),
        "incident_energy_J": energy,
        "nodal_force_change_l2_N": float(np.linalg.norm(change)),
        "peak_force_change_N": float(change.max(initial=0)),
        "net_force_change_N": forces.sum(axis=0).tolist(),
        "largest_node_changes": [
            {"node_row": int(nodes[i]), "force_delta_N": forces[i].tolist()} for i in worst
        ],
    }


def audit(run, proposal):
    cfg = json.loads((run / "launch.json").read_bytes())
    env = cfg["environment"]
    hashes = {}
    def verify(path, pinned=True):
        p = Path(path)
        h = digest(p)
        if (pinned or str(p) in cfg["pins"]) and cfg["pins"].get(str(p)) != h:
            raise ValueError(f"Changed native input {p}")
        hashes[str(p)] = h
        return p
    key = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_"
    raw = np.loadtxt(verify(env[key + "ELEMENTS"]), skiprows=1,
                     usecols=(1, 2, 3, 4, 5), dtype=np.int64)
    cells, labels = raw[:, :4] - 1, raw[:, 4]
    if "NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ" in env:
        with np.load(verify(env["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ"])) as z:
            keep = z["retained_input_cell_rows"]
            parents = z["replacement_parent_cell_rows"]
            offsets = z["replacement_parent_offsets"]
            if np.any(np.diff(offsets) <= 0):
                raise ValueError("Replacement cell has no material ancestor")
            owners = labels[parents]
            first = owners[offsets[:-1]]
            if not np.array_equal(owners, np.repeat(first, np.diff(offsets))):
                raise ValueError("A cavity spans different tissue materials")
            cells = np.vstack((cells[keep], z["replacement_tetrahedra"]))
            labels = np.r_[labels[keep], first]
    if set(np.unique(labels)) != {15, 16}:
        raise ValueError("Different material identities")
    def load_xyz(p):
        return np.loadtxt(verify(p), skiprows=1, usecols=(1, 2, 3))
    ref = load_xyz(env[key + "NODE"]).astype(np.float32).astype(float)
    current = load_xyz(env["NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"])
    params = material_parameters(cfg, verify)
    with np.load(proposal) as z:
        result = compare(ref, current, cells, labels, params, z["node_rows"], z["delta_m"])
    result.update(launch_sha256=digest(run / "launch.json"), proposal_sha256=digest(proposal),
                  inputs=hashes)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("proposal", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    result = audit(args.run, args.proposal)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Peak nodal force change {result['peak_force_change_N']:.9g} N")
