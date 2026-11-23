"""Install a source-pinned equilibrium update in the same full-volume native owner."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from cavity_mass import cavity_delta
from emit_constrained_material import write_coordinates
from fit_dorsal_material import inputs
from material_context import sha
from supported_material_admission import check_installed, installed_topology


def read(p):
    return json.loads(p.read_text())


def save(p, data):
    p.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def install(run, fitdir, ledger, audit, predicate):
    cfg = read(run / "launch.json")
    old = Path(cfg["argv"][0]).parent
    e = cfg["environment"]
    prior = read(old / "identities.json")
    fit = read(fitdir / "fit.json")
    x, y, base, roles, pins = inputs(run)
    if (
        pins != fit["source"]
        or not fit["final_admission"]["passed"]
        or any(
            abs(b) >= abs(a)
            for a, b in zip(fit["initial_modes_s2"], fit["final_modes_s2"])
        )
    ):
        raise ValueError("Unmatched fit or no force relief")
    with np.load(fitdir / "preload.npz") as z:
        rows, values = z["rows"], z["coordinates"]
    if (
        values.dtype != np.float32
        or not np.array_equal(rows, np.unique(rows))
        or len(rows) != fit["moved"]
    ):
        raise ValueError("Incomplete Float32 update")
    original = Path(e["NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"])
    target = fitdir / "current.node"
    write_coordinates(original, target, rows, values)
    cur = np.loadtxt(target, skiprows=1, usecols=(1, 2, 3))
    if not np.array_equal(cur[rows], values) or not np.array_equal(
        np.flatnonzero(np.any(cur != y, axis=1)), rows
    ):
        raise ValueError("Updated unrelated coordinates")
    flow = ledger / "fine-mass-routed/current-density-closed-flow.npz"
    if sha(flow) != fit["mass_ledger"]:
        raise ValueError("Mass map differs")
    tet, tags = installed_topology(base, roles, e)
    labels = ledger / "mesh/current-child.labels.csv"
    if sha(labels) != read(labels.with_suffix(".json"))["labels_sha256"]:
        raise ValueError("Different anatomical donors")
    donor = np.loadtxt(labels, delimiter=",", skiprows=1, usecols=2, dtype=int)
    with np.load(flow) as f, np.load(e["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ"]) as z:
        retained = z["retained_input_cell_rows"]
        oldm = f["element_mass_kg"]
        mass = np.r_[oldm[retained], z["replacement_mass_kg"]]
        parents = z["replacement_parent_cell_rows"][
            z["replacement_parent_offsets"][:-1]
        ]
        offsets = z["replacement_parent_offsets"]
        all_parents = z["replacement_parent_cell_rows"]
        if not np.array_equal(
            np.repeat(donor[parents], np.diff(offsets)), donor[all_parents]
        ):
            raise ValueError("Replacement spans anatomical owners")
        groups = np.r_[donor[retained], donor[parents]]
        original_groups = f["group_for_cell"]
        errors = []
        for i, j in zip(
            z["replacement_tet_offsets"][:-1], z["replacement_tet_offsets"][1:]
        ):
            replaced = np.unique(all_parents[offsets[i] : offsets[j]])
            errors.append(
                abs(oldm[replaced].sum() - z["replacement_mass_kg"][i:j].sum())
            )
        if max(errors) > 1e-14:
            raise ValueError("Cavity mass differs")
        dm, r, remass = cavity_delta(
            base,
            oldm,
            retained,
            z["replacement_tetrahedra"],
            z["replacement_mass_kg"],
            len(y),
        )
        nodal = f["nodal_mass_kg"] + dm
    changed = np.flatnonzero(np.isin(tet, rows).any(axis=1))
    disp = cur - y.astype(float)
    g, ix = np.unique(
        np.c_[groups[changed], tags[changed]], axis=0, return_inverse=True
    )
    moments = np.zeros((len(g), 3))
    np.add.at(moments, ix, mass[changed, None] * disp[tet[changed]].mean(axis=1))
    # Many source groups contribute to each replacement. Advect their original
    # mass ledger and add the verified cavity transfer exactly once.
    affected = np.flatnonzero(np.isin(base, rows).any(axis=1))
    original_ids, idx = np.unique(original_groups[affected], return_inverse=True)
    advected = np.zeros((len(original_ids), 3))
    np.add.at(advected, idx, oldm[affected, None] * disp[base[affected]].mean(axis=1))
    mapped = nodal[rows] @ disp[rows]
    remap_error = float(
        abs(advected.sum(axis=0) + dm[rows] @ disp[rows] - mapped).max()
    )
    if remap_error > 1e-14:
        raise ValueError("Source moments disagree with installed nodal transport")
    rel = cur[r] - cur[r[0]]
    residual = abs(dm[r] @ rel).max()
    tol = 1e-10 * remass * abs(rel).max()
    if (
        residual > tol
        or abs(moments).max() > 8e-11
        or abs(moments.sum(0) - mapped).max() > 1e-14
        or abs(mass.sum() - oldm.sum()) > 1e-12
        or mass.min() <= 0
    ):
        raise ValueError(
            dict(
                cavity=residual,
                limit=tol,
                group_moment=abs(moments).max(),
                transport=abs(moments.sum(0) - mapped).max(),
                total_mass_delta=mass.sum() - oldm.sum(),
                min_mass=mass.min(),
            )
        )
    admission = check_installed(x, cur, base, roles, e, old)
    if not admission["passed"]:
        raise ValueError(admission)
    expected = read(run / "accepted-layers.json")["library_sha256"]
    if sha(predicate) != expected:
        raise ValueError("Changed intersection predicate")
    subprocess.run(
        [
            sys.executable,
            str(audit),
            str(run / "launch.json"),
            str(predicate),
            str(fitdir / "layers.json"),
            "--accepted-nodes",
            str(run / "physical-reference.node"),
            str(target),
        ],
        check=True,
        env=dict(
            os.environ,
            PYTHONPATH=str(predicate.parent)
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        ),
    )
    if np.any(read(fitdir / "layers.json")["pair_counts"]):
        raise ValueError("New loaded tissue boundary intersection")
    header = "NumiHumanLoadedSkinPoseData.inc"
    loader = prior["loaded_pose_override"]
    body = (old / header).read_text()
    if (
        sha(original) != loader["sha256"]
        or sha(old / header) != loader["header_sha256"]
        or body.count(loader["sha256"]) != 1
    ):
        raise ValueError("Unmatched pose loader")
    anchor = re.search(r"constexpr Anchor anchors\[\]=\{(.*?)\};", body, re.S)
    fixed = np.asarray(re.findall(r"\{\s*\d+u?\s*,\s*(\d+)u?\s*,", anchor[1]), int)
    if np.intersect1d(fixed, rows).size or len(fixed) != 14:
        raise ValueError("Attachment drift")
    dest = fitdir / "owner"
    if dest.exists():
        raise FileExistsError(dest)
    subprocess.run(["cp", "-cR", old, dest], check=True)
    (dest / header).write_text(body.replace(loader["sha256"], sha(target)))
    cmds = [
        [v.replace(str(old) + "/", str(dest) + "/") for v in c]
        for c in prior["commands"]
    ]
    if (
        str(dest / "numilab_human_myosim_visual_probe.mm") not in cmds[0]
        or str(dest / "numi-human-native") not in cmds[1]
    ):
        raise ValueError("Stale compile path")
    for cmd in cmds:
        subprocess.run(cmd, cwd=cfg["cwd"], check=True)
    exe = dest / "numi-human-native"
    prior.update(
        binary=sha(exe),
        binary_sha256=sha(exe),
        binary_path=str(exe),
        commands=cmds,
        generated_header=sha(dest / header),
        installed_tissue_admission=admission,
        parent_pose_owner_sha256=sha(old / "numi-human-native"),
    )
    prior["loaded_pose_override"] = dict(
        loader,
        path=str(target),
        sha256=sha(target),
        header_sha256=sha(dest / header),
        immediate_parent_sha256=sha(original),
        fit_sha256=sha(fitdir / "fit.json"),
        nodes=rows.tolist(),
    )
    save(dest / "identities.json", prior)
    save(
        fitdir / "installation.json",
        dict(
            loaded_pose=sha(target),
            parent=sha(original),
            binary=sha(exe),
            material=admission,
            layer_pairs=read(fitdir / "layers.json")["pair_counts"],
            installed_cells=len(tet),
            installed_mass_kg=float(mass.sum()),
            changed_material_stars=len(changed),
            source_mass_groups=len(original_ids),
            anatomical_donor_material_groups=len(g),
            worst_cavity_mass_difference_kg=float(max(errors)),
            remap_moment_error_kg_m=remap_error,
            first_moment_max_kg_m=float(abs(moments).max()),
            cavity_moment_kg_m=float(residual),
            cavity_moment_limit=float(tol),
            reference=pins["reference_sha256"],
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("run", "fitdir", "ledger", "audit", "predicate"):
        p.add_argument(k, type=lambda s: Path(s).resolve())
    install(**vars(p.parse_args()))
