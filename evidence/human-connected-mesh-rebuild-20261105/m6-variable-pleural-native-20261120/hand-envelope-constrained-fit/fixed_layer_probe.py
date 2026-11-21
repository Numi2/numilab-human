"""Test a skin-only envelope fit with the bonded inner and deep layers fixed."""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

from envelope_grouped import solve
from envelope_lp import prepare, selected_sites, digest, exact_summary
from screen import banks


launch, proposal, bed, faces = map(Path, sys.argv[2:6])
points, tets, role, locked = prepare(launch, proposal, bed)
source_locked = locked.copy()
tree = cKDTree(points.astype(float))
for key in ((51007, 4294966545), (51007, 4294966546)):
    surface, _ = banks[key]
    distance, idx = tree.query(surface)
    assert (distance == 0).all() and len(np.unique(idx)) == len(idx)
    locked[idx] = True
locked[np.unique(tets[role == 16])] = True
targets = selected_sites(faces)
moved, groups = solve(points, tets, role, locked, targets, 8.5, preserve_material_volume=True)
changed = np.flatnonzero(np.any(points != moved, axis=1))
assert np.array_equal(points[locked], moved[locked]) and np.array_equal(points[:, 1:], moved[:, 1:])
quality_before = exact_summary(points, tets, role)
quality_after = exact_summary(moved, tets, role)
sites = []
for site in targets:
    axial = moved[site["nodal_rows"], 0].astype(float) - points[site["nodal_rows"], 0]
    gap_mm = 1000 * (site["side"] * (axial @ site["bary"]) - site["gap_m"])
    sites.append(dict(route=site["route"], site=site["site"], intended_slack_mm=gap_mm))
output = {"locked_total": int(locked.sum()), "locked_before": int(source_locked.sum()),
          "groups": groups, "sites": sites, "changed_nodes": len(changed),
          "whole_volume_before": quality_before, "whole_volume": quality_after}
assert quality_after["volume_floor_failures"] == quality_after["shape_floor_failures"] == 0
if len(changed):
    asset = Path("fixed-layer-envelope.npz")
    np.savez_compressed(asset, node_rows=changed, position_f32=moved[changed])
    output["proposal_sha256"] = digest(asset)
print(json.dumps(output, indent=2))
