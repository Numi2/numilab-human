"""Determine whether the original, authenticated skin binding crosses the mass-capacity cut."""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

here = Path(__file__).resolve().parent
prior = here.parent / "resting-clearance-and-atrium-016"
sys.path.insert(0, str(prior))
from paired_route_projection import PairedRouteProjection
from screen_paired_mass_routes import COMPILER, sha

audit = json.loads((prior / "full-paired-route-audit.json").read_text())
cut = json.loads((prior / "paired-region-density-capacity-bound.json").read_text())["Skin"]["analytic_deficit_cut"]
flow_file = here.parent / "resting-clearance-and-atrium-014/fine-current-paired-semantic-cycle-mass.npz"
assert sha(flow_file) == audit["source_mass_sha256"]
with np.load(flow_file) as source:
    flow = {name: source[name] for name in ("source_parent", "group_for_cell")}
p = PairedRouteProjection(flow, audit, flow["group_for_cell"])

region_receipt = audit["role_reports"]["Skin"]
bounds = prior / region_receipt["region_bounds_file"]
assert sha(bounds) == region_receipt["region_bounds_sha256"]
with np.load(bounds) as s:
    masks = s["group_and_regions"][:, 1:].copy()
    for region in cut["region_ids"]:
        bit = int(np.searchsorted(s["region_ids"], region))
        assert s["region_ids"][bit] == region
        masks[:, bit // 64] &= ~(np.uint64(1) << np.uint64(bit % 64))
    child_rows = s["child_rows"][~np.any(masks[s["child_mask_index"]], axis=1)]
assert len(child_rows) == cut["constrained_cells"]
nodes, local_tets = np.unique(p.cells[child_rows], return_inverse=True)
local_tets = local_tets.reshape(-1, 4)
four = p.points[nodes[local_tets]].astype(np.float64)
lower = np.linalg.det(four[:, 1:] - four[:, :1]) * 150.
assert lower.min() > 0 and abs(lower.sum() - cut["minimum_required_kg"]) < 1e-12

parent_pins = json.loads((here.parent / "resting-clearance-and-atrium-013/fine-local-current-p1.json").read_text())
old_path = Path(parent_pins["parent_inputs"]["source330_nodes"]["path"])
assert sha(old_path) == parent_pins["parent_inputs"]["source330_nodes"]["sha256"]
recorded = np.loadtxt(old_path, delimiter=",", skiprows=1)[:, 2:5]
binding = COMPILER / "bindings/combined-node-map.csv.zst"
digest = hashlib.sha256()
matches = []
index = {int(i) + 1: k for k, i in enumerate(nodes)}
origin = here.parent / "resting-clearance-and-atrium-010/fine-transfer-screen-map.npz"
with np.load(origin) as orig:
    original_xyz = orig["old_world_m"][nodes]
with subprocess.Popen(["/opt/homebrew/bin/zstd", "-qcd", str(binding)], stdout=subprocess.PIPE) as proc:
    def decoded():
        for line in proc.stdout:
            digest.update(line)
            yield line.decode()
    for row in csv.DictReader(decoded()):
        j = index.get(int(row["joint_node_id"]))
        if j is not None and row["source_partition"] == "skin":
            matches.append((j, row))
    assert proc.wait() == 0
recovery = json.loads(binding.with_name(binding.name + ".recovery.json").read_text())
assert digest.hexdigest() == recovery["sha256"]
assert len({j for j, _ in matches}) == len(matches)

methods = {}
for key, bary_prefix, residual_prefix in (
    ("parent_tet_row", "parent_bary", "parent_current_residual_"),
    ("operational_reference_tet_row", "operational_reference_bary", "operational_reference_residual_"),
):
    outside = np.zeros(len(nodes), dtype=bool)
    error = []
    changed_projection = []
    wrong_material = 0
    other_row_domain = 0
    typed_region_mismatches = 0
    new_regions = set()
    for j, row in matches:
        triangle = int(row[key])
        if triangle < 0:
            continue
        bary = np.array([float(row[bary_prefix + str(i)]) for i in range(4)])
        residual = np.array([float(row[residual_prefix + xyz + "_m"]) for xyz in "xyz"])
        assert np.isfinite(bary).all() and np.min(bary) >= -1e-12 and abs(bary.sum() - 1) < 1e-10
        if triangle >= len(p.tets):
            other_row_domain += 1
            continue
        tet = p.tets[triangle]
        source_frame = p.rest if key.startswith("operational") else recorded
        expected = np.array([float(row["current_" + xyz + "_m"]) for xyz in "xyz"])
        assert np.linalg.norm(expected - original_xyz[j]) < 3e-6
        e = float(np.linalg.norm(bary @ source_frame[tet] + residual - expected))
        error.append(e)
        if e >= 3.e-6:
            continue
        same_material = p.parent["materials"][triangle] == 15
        if not same_material:
            wrong_material += 1
            continue
        region = int(p.parent["regions"][triangle])
        declared = int(row["source_parent_anatomical_region_id"]) if key.startswith("operational") else region
        if region != declared:
            typed_region_mismatches += 1
            continue
        d = float(np.linalg.norm(p.points[nodes[j]] - bary @ p.parent["points"][tet]))
        changed_projection.append(d)
        if d <= p.projection_limit and region not in cut["region_ids"]:
            outside[j] = True
            new_regions.add(region)
    possible = np.any(outside[local_tets], axis=1)
    moved = float(lower[possible].sum())
    methods[key] = {
        "binding_records": len(error),
        "original_frame_replay_passes": sum(e < 3e-6 for e in error),
        "maximum_original_frame_replay_error_m": max(error, default=0.),
        "wrong_material": wrong_material,
        "outside_source330_row_domain": other_row_domain,
        "typed_region_mismatches": typed_region_mismatches,
        "maximum_repaired_projection_m": max(changed_projection, default=0.),
        "additional_typed_regions": sorted(new_regions),
        "nodes_with_outside_binding": int(outside.sum()),
        "cells_with_outside_binding": int(possible.sum()),
        "maximum_cut_relief_kg": moved,
        "remaining_cut_shortage_lower_bound_kg": max(0., cut["shortage_kg"] - moved),
    }

result = {
    "source_binding_decoded_sha256": digest.hexdigest(),
    "position_array_sha256": audit["geometry_sha256"],
    "topology_array_sha256": audit["tetrahedron_sha256"],
    "cut_child_nodes": len(nodes),
    "cut_child_nodes_with_retained_skin_partition_rows": len(matches),
    "unsupported_cut_kg": cut["shortage_kg"],
    "projection_limit_m": p.projection_limit,
    "methods": methods,
    "note": "For each method relief credits whole cells at minimum density for one validated node to upper-bound possible relief. No flow or native admission.",
}
(here / "retained-skin-binding-audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
