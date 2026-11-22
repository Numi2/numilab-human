"""Recover the actual connected material neighborhoods carrying flat-bed load."""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from audit_cavity_response import elastic
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_bed_release import stationary_bed
from material_context import sha


def prepare(run, material, transfer, owner):
    launch = json.loads((run / "launch.json").read_text())
    env = launch["environment"]
    audit = json.loads((run / "accepted-layers.json").read_text())
    model = json.loads(transfer.read_text())
    ref, current, source, role, pins = inputs(run)
    if pins["installed_pose_sha256"] != model["current"]:
        raise ValueError("Loaded material mismatch")
    prop_path = Path(env["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ"])
    flow = material / "fine-mass-routed/current-density-closed-flow.npz"
    if sha(prop_path) != model["topology"] or sha(flow) != model["mass"]:
        raise ValueError("Connectivity or mass differs")
    with np.load(flow) as f:
        mass = f["nodal_mass_kg"].copy()
        elem_mass = f["element_mass_kg"]
    with np.load(prop_path) as p:
        keep = p["retained_input_cell_rows"]
        new = p["replacement_tetrahedra"]
        lost = np.setdiff1d(np.arange(len(source)), keep)
        dtets = np.vstack((source[lost], new))
        dweights = np.r_[-elem_mass[lost], p["replacement_mass_kg"]]
        for j in range(4):
            np.add.at(mass, dtets[:, j], dweights / 4)
        tags = np.r_[
            role[keep],
            role[
                p["replacement_parent_cell_rows"][p["replacement_parent_offsets"][:-1]]
            ],
        ]
        mesh = np.vstack([source[keep], new])
        per_cell = np.r_[elem_mass[keep], p["replacement_mass_kg"]]
    table = Path(
        env[
            "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_CSV"
        ]
    )
    if sha(table) != audit["inputs"]["COMBINED_NODE_MAP_CSV"]:
        raise ValueError("Source node mapping differs")
    donors = np.full(len(current), -1, int)
    layer = np.zeros(len(current), dtype="u1")
    with table.open() as stream:
        for row in csv.DictReader(stream):
            n = int(row["joint_node_id"]) - 1
            value = int(row["donor_body"])
            tag = row["registered_surface_layer"]
            if donors[n] != -1 and value != donors[n]:
                raise ValueError("Inconsistent joined node owner")
            donors[n] = value
            layer[n] = {"": 0, "O": 1, "I": 2, "D": 3}[tag]
    nodes = run / "physical-accepted.node"
    if sha(nodes) != audit["inputs"]["NATIVE_ACCEPTED_NODE_SHA256"]:
        raise ValueError("Stale endpoint")
    end = np.loadtxt(nodes, skiprows=1, usecols=(1, 2, 3))
    props, mat_sha = materials(owner)
    argv = launch["argv"]
    step = float(argv[argv.index("--muscle-step-seconds") + 1])
    width = float.fromhex("0x1.4f8b580000000p-17")
    anchor_text = (owner / "NumiHumanLoadedSkinPoseData.inc").read_text()
    text = re.search(
        r"constexpr Anchor anchors\[\]=\{(.*?)\};", anchor_text, re.DOTALL
    )[1]
    anchors = np.array(
        [int(n) for n in re.findall(r"\{\s*\d+u?\s*,\s*(\d+)u?\s*,", text)]
    )
    if len(anchors) != 14 or np.any(donors < 0):
        raise ValueError("Invalid anchor or material roster")
    return {
        "ref": ref.astype(float),
        "current": current.astype(float),
        "end": end,
        "t": mesh,
        "tags": tags,
        "cell_mass": per_cell,
        "dm_tets": dtets,
        "dm_weights": dweights,
        "mass": mass,
        "donor": donors,
        "layer": layer,
        "dt": step,
        "width": width,
        "props": props,
        "materials": mat_sha,
        "anchors": anchors,
        "source": pins,
    }


def motion(run, rows):
    rec = json.loads((run / "terminal.receipt.json").read_text())
    launch = json.loads((run / "launch.json").read_text())
    receipt = json.loads((run / "admission.json").read_text())
    if (
        rec["physical_endpoint"] != "accepted"
        or receipt["terminal_pack_sha256"] != rec["pack_file_sha256"]
    ):
        raise ValueError("Unmatched release capture")
    for name in ["reference", "accepted"]:
        path = run / f"physical-{name}.node"
        if (run / "accepted-layers.json").exists():
            pin = json.loads((run / "accepted-layers.json").read_text())["inputs"][
                f"NATIVE_{name.upper()}_NODE_SHA256"
            ]
            if pin != sha(path):
                raise ValueError("Unmatched physical coordinates")
    start, end = [
        np.loadtxt(run / f"physical-{k}.node", skiprows=1, usecols=(1, 2, 3))[rows]
        for k in ("reference", "accepted")
    ]
    return end - start, {
        "initial": sha(run / "physical-reference.node"),
        "accepted": sha(run / "physical-accepted.node"),
        "topology": launch["environment"]["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256"],
        "receipt": sha(run / "terminal.receipt.json"),
    }


def probes(d):
    outer = d["layer"] == 1
    start = d["current"]
    end = d["end"]
    t = d["t"]
    threshold = d["width"]
    subjects = np.flatnonzero(
        outer & ((start[:, 2] < threshold) | (end[:, 2] < threshold))
    )
    ids = np.flatnonzero(np.isin(t, subjects).any(1))
    nodes = np.unique(t[ids])

    def examine(y):
        cols = [np.zeros((len(nodes), 3)) for _ in (15, 16)]
        for m in (15, 16):
            local = ids[d["tags"][ids] == m]
            if len(local):
                cols[m - 15] = elastic(
                    d["ref"], y, t[local], nodes, *d["props"][m - 15]
                )[1]
        p = stationary_bed(y[subjects, 2], d["mass"][subjects], d["dt"], threshold)[1]
        idx = np.searchsorted(nodes, subjects)
        g = d["mass"][subjects] * 9.81
        return np.column_stack(
            [
                p,
                cols[0][idx, 2],
                cols[1][idx, 2],
                cols[0][idx, 2] + cols[1][idx, 2] + p - g,
            ]
        )

    A = examine(start)
    B = examine(end)
    data = []
    for row, n in enumerate(subjects):
        data.append(
            {
                "node": int(n),
                "donor": int(d["donor"][n]),
                "layer": int(d["layer"][n]),
                "initial": A[row].tolist(),
                "endpoint": B[row].tolist(),
                "delta_z_um": float((end[n, 2] - start[n, 2]) * 1e6),
            }
        )
    return {
        "columns": ["bed_N", "dermis_N", "fat_N", "forces_plus_gravity_N"],
        "rows": data,
        "material_sha256": d["materials"],
        "geometry": d["source"],
    }


if __name__ == "__main__":
    data = prepare(*(Path(s).resolve() for s in sys.argv[1:5]))
    result = probes(data)
    Path(sys.argv[5]).write_text(json.dumps(result, indent=2) + "\n")
    for row in sorted(result["rows"], key=lambda r: -r["endpoint"][0])[:6]:
        print(row)
