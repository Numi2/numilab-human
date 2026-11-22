"""Publish a fitted initial displacement to the existing full Human FEM and validate its loaded mesh."""

import json, sys, subprocess
from pathlib import Path
import numpy as np
from support_patch import prepare
from material_context import sha
from conserved_cavity_rows import verify_native_mass, inventories
from material_bed_release import bind_contact_footprint
from emit_constrained_material import write_coordinates
from supported_material_admission import certify, stress_limits


def read(p):
    return json.loads(p.read_text())


def write(p, v):
    p.write_text(json.dumps(v, indent=2, allow_nan=False) + "\n")


def bind(root, base, out):
    fit = root / "wrench-fit"
    proposal = fit / "preload.npz"
    report = read(fit / "result.json")
    source = base / "owner"
    run = base / "native-prefix"
    mat = base.parent / "paired-reference"
    observer = root / "gravity"
    old = observer / "owner"
    ids = read(old / "identities.json")
    if (
        not report["admitted"]
        or sha(proposal) != report["source_files"]["proposal"]
        or report["observer_launch"] != sha(observer / "one/launch.json")
    ):
        raise ValueError("Unqualified wrench fit")
    d = prepare(run, mat, source / "transfer.json", source / "native-owner")
    if d["source"] != report["source"]["source"] or ids["binary_sha256"] != sha(
        old / "numi-human-native"
    ):
        raise ValueError("Changed parent mechanics")
    sourcepose = Path(ids["loaded_pose_override"]["path"])
    pin = ids["loaded_pose_override"]
    if (
        sha(sourcepose) != pin["sha256"]
        or pin["sha256"] != d["source"]["installed_pose_sha256"]
    ):
        raise ValueError("Different source pose")
    with np.load(proposal) as p:
        rows = p["rows"]
        xy = p["coordinates"]
    y = d["current"].copy()
    if (
        not np.array_equal(np.unique(rows), rows)
        or np.isin(rows, np.r_[d["anchors"], d["dm_tets"].ravel()]).any()
        or not np.array_equal(y[rows, :2], xy[:, :2])
    ):
        raise ValueError("Moved a protected structure")
    y[rows] = xy
    labels, groups = inventories(run, mat)
    cm = verify_native_mass(y, groups)
    if not cm["passed"]:
        raise ValueError(cm)
    cell = np.flatnonzero(np.isin(d["t"], rows).any(1))
    delta = y[:, 2] - d["current"][:, 2]
    moments = []
    for group in np.unique(labels[cell, 1:], axis=0):
        selected = cell[(labels[cell, 1:] == group).all(1)]
        moments.append(
            float(d["cell_mass"][selected] @ delta[d["t"][selected]].mean(1))
        )
    if max(abs(v) for v in moments) > 8e-11:
        raise ValueError("Donor first moment")
    admission = certify(
        d["ref"], y, d["t"], d["tags"], d["props"], stress_limits(d["materials"])
    )
    if not admission["passed"]:
        raise ValueError(admission)
    out.mkdir(exist_ok=True)
    dest = out / "owner"
    pose = out / "current.node"
    write_coordinates(sourcepose, pose, rows, xy.astype(float))
    if not np.array_equal(y, np.loadtxt(pose, skiprows=1, usecols=(1, 2, 3))):
        raise ValueError("Different emitted coordinates")
    subprocess.run(["cp", "-cR", str(old), str(dest)], check=True)
    header = dest / "NumiHumanLoadedSkinPoseData.inc"
    checker = dest / "NumiHumanLoadedSkinPose.hpp"
    main = dest / "numilab_human_myosim_visual_probe.mm"
    if (
        sha(header) != pin["header_sha256"]
        or sha(checker) != ids["registered_bed_footprint"]["loader_sha256"]
        or sha(main) != ids["source"]
    ):
        raise ValueError("Unrecognized registered loader")
    outer = np.flatnonzero(d["layer"] == 1)
    min_gap = float(y[outer, 2].min())
    old_gap = float(ids["registered_bed_footprint"]["minimum_gap_m"]).hex()

    def replace(path, before, after, expected):
        s = path.read_text()
        if s.count(before) != expected:
            raise ValueError("Unrecognized source in " + path.name + " for " + before)
        path.write_text(s.replace(before, after))

    replace(header, pin["sha256"], sha(pose), 1)
    replace(checker, old_gap, min_gap.hex(), 2)
    replace(
        main,
        "targetGap=loadedBed ? " + old_gap,
        "targetGap=loadedBed ? " + min_gap.hex(),
        1,
    )
    feet = bind_contact_footprint(dest, outer, y, min_gap, d["width"])
    commands = [
        [s.replace(str(old) + "/", str(dest) + "/") for s in cmd]
        for cmd in ids["commands"]
    ]
    for cmd in commands:
        subprocess.run(cmd, check=True, cwd=read(run / "launch.json")["cwd"])
    exe = dest / "numi-human-native"
    ids.update(
        binary=sha(exe),
        binary_sha256=sha(exe),
        binary_path=str(exe),
        commands=commands,
        source=sha(main),
        installed_tissue_admission=admission,
    )
    ids["registered_bed_footprint"] = dict(
        outer_nodes=len(outer),
        contact_nodes=len(feet),
        minimum_gap_m=min_gap,
        loader_sha256=sha(checker),
    )
    ids["loaded_pose_override"] = dict(
        pin,
        path=str(pose),
        sha256=sha(pose),
        header_sha256=sha(header),
        immediate_parent_sha256=pin["sha256"],
        fit_sha256=sha(fit / "result.json"),
        nodes=rows.tolist(),
    )
    write(dest / "identities.json", ids)
    write(
        out / "qualification.json",
        dict(
            pose=sha(pose),
            prior_pose=pin["sha256"],
            delta_z_um=float(np.max(abs(delta)) * 1e6),
            mass_cavities=cm,
            maximum_donor_first_moment_change_kg_m=max(abs(v) for v in moments),
            whole_material=admission,
            gravity=report["gravity"],
            stationary_bed=report["emitted"],
            fitted_rows=len(rows),
            new_contact_rows=len(feet),
            outer_nodes=len(outer),
            binary=sha(exe),
            scene_pose_sha=sha(main),
        ),
    )
    parent = out / "parent"
    parent.mkdir()
    j = read(observer / "parent/launch.json")
    write(parent / "launch.json", j)
    print(admission, flush=True)


if __name__ == "__main__":
    bind(*(Path(s).resolve() for s in sys.argv[1:]))
