"""Emit a qualified posterior starting pose through the existing native tissue loader."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from conserved_cavity_rows import inventories, verify_native_mass
from emit_constrained_material import write_coordinates
from material_bed_release import bind_contact_footprint, stationary_bed
from material_context import sha
from support_patch import prepare
from supported_material_admission import certify, stress_limits


def bind(run, material, transfer, owner, parent, prepared, fit, fine, out):
    d = prepare(run, material, transfer, owner)
    src = json.loads((prepared / "source.json").read_text())
    q = json.loads((fit / "qualification.json").read_text())
    a = json.loads((owner / "identities.json").read_text())
    launch = json.loads((run / "launch.json").read_text())
    prev = json.loads((parent / "launch.json").read_text())
    key = "NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"
    old = Path(launch["environment"][key])
    pin = a["loaded_pose_override"]
    header = "NumiHumanLoadedSkinPoseData.inc"
    checker = "NumiHumanLoadedSkinPose.hpp"
    proposal = fit / q["proposal"]
    if not (
        q["chosen"]["gated"]
        and q["registered"] == src["registered"] == d["source"]
        and sha(fine) == q["fine_mesh_sha256"]
        and src["matched_receipt"]
        == q["source_receipt"]
        == sha(run / "terminal.receipt.json")
        and sha(prepared / "patch.npz") == src["topology_patch_sha256"]
        and sha(proposal) == q["proposal_sha256"]
        and sha(old) == pin["sha256"] == d["source"]["installed_pose_sha256"]
        and sha(owner / header) == pin["header_sha256"]
        and sha(owner / checker) == a["registered_bed_footprint"]["loader_sha256"]
        and sha(owner / "numi-human-native")
        == a["binary_sha256"]
        == launch["pins"][launch["argv"][0]]
        and prev["pins"][prev["environment"][key]] == pin["parent_sha256"]
    ):
        raise ValueError("Changed material, physical owner, or refinement evidence")
    with np.load(proposal) as p:
        rows = p["rows"]
        xyz = p["coordinates"]
    y = d["current"].copy()
    if not (
        len(rows)
        and np.array_equal(np.unique(rows), rows)
        and np.all((rows >= 0) & (rows < len(y)))
        and np.all(d["layer"][rows] == 1)
        and not np.isin(rows, np.r_[d["dm_tets"].ravel(), d["anchors"]]).any()
        and xyz.shape == (len(rows), 3)
        and xyz.dtype == np.float32
        and np.isfinite(xyz).all()
        and np.array_equal(xyz[:, :2], y[rows, :2])
        and np.max(np.abs(xyz[:, 2] - y[rows, 2])) <= 2.001e-6
    ):
        raise ValueError("Changed unqualified coordinates")
    y[rows] = xyz
    labels, groups = inventories(run, material)
    cm = verify_native_mass(y, groups)
    if not cm["passed"]:
        raise ValueError(cm)
    delta = y[:, 2] - d["current"][:, 2]
    t = d["t"]
    involved = np.flatnonzero(np.isin(t, rows).any(1))
    moments = []
    for g in np.unique(labels[involved, 1:], axis=0):
        idx = involved[np.all(labels[involved, 1:] == g, axis=1)]
        moments.append(
            {
                "material": int(g[0]),
                "donor": int(g[1]),
                "delta_kg_m": float(d["cell_mass"][idx] @ delta[t[idx]].mean(1)),
            }
        )
    if max(abs(r["delta_kg_m"]) for r in moments) > 8e-11:
        raise ValueError("Source/donor first moment drift")
    admission = certify(
        d["ref"], y, t, d["tags"], d["props"], stress_limits(d["materials"])
    )
    if not admission["passed"]:
        raise ValueError(admission)
    outer = np.flatnonzero(d["layer"] == 1)
    minimum = float(y[outer, 2].min())
    bed = stationary_bed(y[outer, 2], d["mass"][outer], d["dt"], d["width"])[1]
    if abs(bed.sum() - q["chosen"]["after"][0]["force_N"]) > 0.002:
        raise ValueError("Unsupported loaded force")
    out.mkdir(exist_ok=False)
    new = out / "current.node"
    dest = out / "native-owner"
    write_coordinates(old, new, rows, xyz.astype(float))
    if not np.array_equal(y, np.loadtxt(new, skiprows=1, usecols=(1, 2, 3))):
        raise ValueError("Changed emission")
    subprocess.run(["cp", "-cR", str(owner), str(dest)], check=True)
    s = (dest / header).read_text()
    c = (dest / checker).read_text()
    old_min = a["registered_bed_footprint"]["minimum_gap_m"].hex()
    if s.count(pin["sha256"]) != 1 or c.count(old_min) != 2:
        raise ValueError("Unknown native pose or support pin")
    new_hash = sha(new)
    (dest / header).write_text(s.replace(pin["sha256"], new_hash))
    (dest / checker).write_text(c.replace(old_min, minimum.hex()))
    contacts = bind_contact_footprint(dest, outer, y, minimum, d["width"])
    commands = [
        [s.replace(str(owner) + "/", str(dest) + "/") for s in cmd]
        for cmd in a["commands"]
    ]
    for cmd in commands:
        subprocess.run(cmd, cwd=launch["cwd"], check=True)
    exe = dest / "numi-human-native"
    binary = sha(exe)
    fit_hash = sha(fit / "qualification.json")

    def write(path, data):
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

    a.update(
        commands=commands,
        binary=binary,
        binary_sha256=binary,
        binary_path=str(exe),
        registered_bed_footprint={
            "outer_nodes": len(outer),
            "contact_nodes": len(contacts),
            "minimum_gap_m": minimum,
            "loader_sha256": sha(dest / checker),
        },
        parent_pose_owner_sha256=sha(owner / "numi-human-native"),
        installed_tissue_admission=admission,
        loaded_pose_override={
            "key": key,
            "parent_sha256": pin["parent_sha256"],
            "immediate_parent_sha256": pin["sha256"],
            "path": str(new),
            "sha256": new_hash,
            "header_sha256": sha(dest / header),
            "fit_sha256": fit_hash,
            "nodes": rows.tolist(),
        },
    )
    write(dest / "identities.json", a)
    report = {
        "current": new_hash,
        "parent": d["source"],
        "material_admission": admission,
        "cavity_moments": cm,
        "source_donor_moments": moments,
        "bed_normal_N": float(bed.sum()),
        "support_rows": len(contacts),
        "registered_outer_nodes": len(outer),
        "nodes_changed": int(np.count_nonzero(delta)),
        "max_delta_m": float(abs(delta).max()),
        "source_material_mass_unchanged": True,
        "native_binary_sha256": binary,
    }
    write(out / "qualification.json", report)
    tr = json.loads(transfer.read_text())
    before = q["chosen"]["baseline"][0]
    after = q["chosen"]["after"][0]
    tr.update(
        parent=d["source"],
        parent_transfer_sha256=sha(transfer),
        current=new_hash,
        changed_nodes=report["nodes_changed"],
        changed_reference=[],
        before_N=before["force_N"],
        after_N=report["bed_normal_N"],
        support_rows=len(contacts),
        minimum_gap_m=minimum,
        bed_moment_change_Nm=(
            np.array(after["moment_Nm"]) - before["moment_Nm"]
        ).tolist(),
        mass_moments=cm,
        source_donor_moments=moments,
        admission=admission,
        qualification_sha256=sha(out / "qualification.json"),
        matched_bed_rows_and_masses=bool(
            np.array_equal(contacts, outer[d["current"][outer, 2] < d["width"]])
        ),
    )
    write(out / "transfer.json", tr)
    print(
        binary,
        report["nodes_changed"],
        report["max_delta_m"],
        len(contacts),
        flush=True,
    )


if __name__ == "__main__":
    bind(*(Path(s).resolve() for s in sys.argv[1:]))
