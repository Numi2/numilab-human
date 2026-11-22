"""Recover exact local force/mass ownership from both sides of the six registered fillings."""

import itertools
import json
from pathlib import Path

import numpy as np
from fit_dorsal_material import inputs
from material_context import sha
from relax_bonded_interface import loaded_mass_moments


def inventories(trial, material):
    _, _, src, source_roles, _ = inputs(trial)
    env = json.loads((trial / "launch.json").read_text())["environment"]
    prop = Path(env["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ"])
    flow = material / "fine-mass-routed/current-density-closed-flow.npz"
    if (
        sha(prop) != env["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256"]
        or sha(flow)
        != json.loads((flow.parent / "mass-transfer.json").read_text())["flow_sha256"]
    ):
        raise ValueError("Unsigned input")
    with np.load(flow) as f:
        mass = f["element_mass_kg"]
    with np.load(prop) as f:
        keep = f["retained_input_cell_rows"]
        new = f["replacement_tetrahedra"]
        gm = f["replacement_mass_kg"]
        p = f["replacement_parent_cell_rows"]
        po = f["replacement_parent_offsets"]
        co = f["replacement_tet_offsets"]
    lab = material / "mesh/current-child.labels.csv"
    if sha(lab) != json.loads(lab.with_suffix(".json").read_text())["labels_sha256"]:
        raise ValueError("Changed source label")
    label = np.loadtxt(lab, delimiter=",", skiprows=1, usecols=(0, 1, 2), dtype="i4")
    if not np.array_equal(label[:, 0], np.arange(len(src))) or not np.array_equal(
        label[:, 1], source_roles
    ):
        raise ValueError("Unmatched source rows")
    typed = []
    groups = []
    retired = []
    for a, b in itertools.pairwise(co):
        rows = np.unique(p[po[a] : po[b]])
        if len(np.unique(label[rows, 1:], axis=0)) != 1:
            raise ValueError("Mixed ancestry in replacement group")
        typed.extend([label[rows[0]].copy() for _ in range(b - a)])
        groups.append((src[rows], mass[rows], new[a:b], gm[a:b]))
        retired.extend(rows.tolist())
    if not np.array_equal(np.sort(retired), np.setdiff1d(np.arange(len(src)), keep)):
        raise ValueError("Cavities overlap or omit source volume")
    return np.vstack([label[keep], typed]), groups


def moment_rows(free, groups, count):
    out = []
    for old, m0, new, m1 in groups:
        row = np.zeros(count)
        for i in range(4):
            np.add.at(row, old[:, i], -m0 / 4)
            np.add.at(row, new[:, i], m1 / 4)
        out.append(row[free])
    return np.array(out)


def verify_native_mass(y, groups):
    fields = list(zip(*groups, strict=True))
    return loaded_mass_moments(y, *(np.concatenate(col) for col in fields))
