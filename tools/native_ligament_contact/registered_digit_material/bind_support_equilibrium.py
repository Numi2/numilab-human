"""Qualify and pin the physical tissue initialization used by the native owner."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from emit_constrained_material import write_coordinates
from material_context import sha
from supported_material_admission import certify, stress_limits

from support_patch import prepare


def bind(run, material, transfer, owner, parent, prepared, fit, out):
    d = prepare(run, material, transfer, owner)
    source = json.loads((prepared / "source.json").read_text())
    f = json.loads((fit / "fit.json").read_text())
    manifest = json.loads((owner / "identities.json").read_text())
    control = json.loads((run / "launch.json").read_text())
    before = json.loads((parent / "launch.json").read_text())
    key = "NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"
    old = Path(control["environment"][key])
    orig = manifest["loaded_pose_override"]
    header = "NumiHumanLoadedSkinPoseData.inc"
    if not (
        sha(old) == d["source"]["installed_pose_sha256"] == orig["sha256"]
        and sha(owner / header) == orig["header_sha256"]
        and sha(owner / "numi-human-native")
        == manifest["binary_sha256"]
        == control["pins"][control["argv"][0]]
        and before["pins"][before["environment"][key]] == orig["parent_sha256"]
        and source["geometry"] == d["source"]
        and f["source"] == source
        and sha(prepared / "material.npz") == source["prepared_sha256"]
        and sha(fit / "fit.npz") == f["fit_sha256"]
        and source["source_launch_sha256"] == sha(run / "launch.json")
    ):
        raise ValueError("Inputs differ from qualified source")
    y = d["current"].copy()
    with np.load(fit / "fit.npz") as patch:
        rows = patch["rows"]
        coords = patch["coords"]
    if not (
        len(rows) > 0
        and np.array_equal(np.unique(rows), rows)
        and np.all(np.isfinite(coords))
        and np.all(np.isin(d["layer"][rows], [0, 2]))
        and not np.isin(rows, d["anchors"]).any()
        and not np.isin(rows, d["dm_tets"]).any()
    ):
        raise ValueError("Unowned nodes or changed attachment/cavity")
    y[rows] = coords
    with np.load(prepared / "material.npz") as p:
        free = p["full_rows"][p["free"]]
        if not np.isin(rows, free).all():
            raise ValueError("Point outside qualified patch")
        mass_error = float(np.max(np.abs(p["W"] @ (y[free] - d["current"][free]))))
        if mass_error > 8e-11:
            raise ValueError("Regional mass moment changed")
    admission = certify(
        d["ref"], y, d["t"], d["tags"], d["props"], stress_limits(d["materials"])
    )
    if not admission["passed"]:
        raise ValueError(admission)
    out.mkdir(exist_ok=False)
    new = out / "current.node"
    dest = out / "native-owner"
    write_coordinates(old, new, rows, coords)
    if not np.array_equal(y, np.loadtxt(new, skiprows=1, usecols=(1, 2, 3))):
        raise ValueError("Emitted pose differs")
    src = (owner / header).read_text()
    if src.count(orig["sha256"]) != 1:
        raise ValueError("Unknown material pin")
    pin = sha(new)
    subprocess.run(["cp", "-cR", str(owner), str(dest)], check=True)
    (dest / header).write_text(src.replace(orig["sha256"], pin))
    commands = [
        [s.replace(str(owner) + "/", str(dest) + "/") for s in c]
        for c in manifest["commands"]
    ]
    for c in commands:
        subprocess.run(c, check=True, cwd=control["cwd"])
    exe = dest / "numi-human-native"
    manifest.update(
        commands=commands,
        binary=sha(exe),
        binary_sha256=sha(exe),
        binary_path=str(exe),
        parent_pose_owner_sha256=sha(owner / "numi-human-native"),
        loaded_pose_override={
            "key": key,
            "parent_sha256": orig["parent_sha256"],
            "immediate_parent_sha256": orig["sha256"],
            "path": str(new),
            "sha256": pin,
            "header_sha256": sha(dest / header),
            "fit_sha256": sha(fit / "fit.json"),
            "nodes": rows.tolist(),
        },
        installed_tissue_admission=admission,
    )
    (dest / "identities.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = {
        "loaded_pose_sha256": pin,
        "reference": d["source"],
        "initial_material": admission,
        "mass_group_max_abs_kg_m": mass_error,
        "mass_source_unchanged": True,
        "changed_nodes": len(rows),
        "protected_surfaces_identical": True,
        "anchors_identical": len(d["anchors"]),
        "native_binary_sha256": sha(exe),
    }
    (out / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    bind(*(Path(v).resolve() for v in sys.argv[1:]))
