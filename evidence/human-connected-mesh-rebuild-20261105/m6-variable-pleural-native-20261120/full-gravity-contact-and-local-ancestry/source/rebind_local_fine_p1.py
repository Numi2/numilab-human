"""Rebind exact fractional ancestry on repaired fine cells in the current paired frame."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

root = Path.cwd()
prev = root.parent / "resting-clearance-and-atrium-011"
sys.path.insert(0, str(prev))
from candidate import COMPILER, load
from paired_reference_carriers import paired_reference_carriers

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

pins = json.loads((COMPILER / "mass-admission/mass-receipt.json").read_text())["input_sha256"]
sources = {}
for key in ("source330_nodes", "source330_tets", "source330_mass", "source330_labels"):
    entry = pins[key]
    sources[key] = Path(entry["path"])
    assert sha(sources[key]) == entry["sha256"], key

script = Path(json.loads((COMPILER / "p1-Skin.command.json").read_text())["command"][1])
spec = importlib.util.spec_from_file_location("pinned_role_p1", script)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert mod.PARENT == sources["source330_mass"]
owner = mod._load_230(mod.OWNER230)
parent = owner.native.read_parent(mod.PARENT)
paired_rest, paired_tets, _, frame = paired_reference_carriers(
    COMPILER, sources["source330_nodes"], sources["source330_tets"])
assert np.array_equal(parent["tets"], paired_tets)
old_nodes = np.loadtxt(sources["source330_nodes"], delimiter=",", skiprows=1)
current = old_nodes[:, 2:5] + (paired_rest - old_nodes[:, 5:8])
parent["points"] = current
edges = current[paired_tets[:, 1:]] - current[paired_tets[:, :1]]
volume6 = np.einsum("ni,ni->n", edges[:, 0], np.cross(edges[:, 1], edges[:, 2]))
assert np.all(np.isfinite(volume6)) and np.all(volume6 > 0.)
index = owner.native.build_index(parent)
material_index = owner.build_material_indices(parent)

_, xyz, cells, mat, origin, _, _, _, _ = load()
with np.load(prev / "fine-mass-local-position-delta.npz", allow_pickle=False) as patch:
    modified = patch["changed_rows"]
    xyz[modified] = patch["positions_f32"]
edited = (origin < 0) | np.isin(cells, modified).any(axis=1)
screen = json.loads((root / "fine-repaired-semantic-domain.json").read_text())
assert hashlib.sha256(xyz.astype("<f4", copy=False).tobytes()).hexdigest() == screen["emitted_float32_sha256"]
assert hashlib.sha256(cells.astype("<u4", copy=False).tobytes()).hexdigest() == screen["emitted_index_sha256"]
results = {}
for role, material in mod.ROLE_MATERIAL.items():
    rows = np.flatnonzero(edited & (mat == material))
    nodes, local = np.unique(cells[rows], return_inverse=True)
    groups, lookup = mod._source_groups(parent, role)
    support = mod.build_node_support(xyz[nodes], role, parent, index, material_index, groups, lookup, owner)
    data = mod.integrate_p1_cells(local.reshape(-1, 4), support["maps"], groups, parent, role)
    for j in range(len(rows)):
        start, end = data["cell_group_offsets"][j:j+2]
        denominators = data["cell_group_p1_score_denominators"][start:end]
        numerators = data["cell_group_p1_score_numerators"][start:end]
        assert np.all(denominators == denominators[0])
        assert numerators.sum() == denominators[0]
    output = root / f"fine-local-current-p1-{role.lower()}.npz"
    np.savez_compressed(output, child_rows=rows, child_node_rows=nodes,
        child_nodes_f32=xyz[nodes], local_tetrahedra=local.reshape(-1, 4),
        queried_node_distance_m=support["distance"], source_groups=groups, **data)
    results[role] = {
        "cells": len(rows),
        "nodes": len(nodes),
        "unique_P1_winner_cells": len(data["resolved_child_tet_rows"]),
        "tied_P1_winner_cells": len(data["unresolved_child_tet_rows"]),
        "maximum_local_parent_distance_m": support["query_distance_max_m"],
        "evidence": output.name,
        "sha256": sha(output),
    }
report = {
    "tested_scope": "retriangulated cells and cells incident to the final nodal mass correction",
    "method": "exact same-material current-frame query and integer P1 ancestry weights",
    "parent_reference_rebase": frame,
    "minimum_rebased_parent_tet_volume_m3": float(volume6.min()) / 6.,
    "child_geometry": {key: screen[key] for key in ("emitted_float32_sha256", "emitted_index_sha256")},
    "parent_inputs": {key: pins[key] for key in sources},
    "owner_sha256": sha(script),
    "outputs": results,
    "admission": "source-only local probe; full paired current-domain ancestry and compiler flow receipt remain required",
}
(root / "fine-local-current-p1.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(results, indent=2))
