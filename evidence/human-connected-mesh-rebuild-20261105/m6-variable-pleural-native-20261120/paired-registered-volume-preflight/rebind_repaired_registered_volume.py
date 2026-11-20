"""Rebuild the existing registered volume sidecars from the repaired native geometry."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np

root = Path(__file__).resolve().parent
original = root.parent / "captured-fat-sliver-495/proposal/compiler"
baseline = root.parent / "resting-clearance-and-atrium-016/repaired-tetgen"
floor_fix = os.environ.get("NUMI_HUMAN_ABSOLUTE_VOLUME_PREP") == "1"
repair = root / "absolute-volume-repaired-tetgen" if floor_fix else baseline
out = root / ("paired-repaired-registration-floor" if floor_fix else "paired-repaired-registration")
out.mkdir(exist_ok=False)
nodes_path = repair / "mesh.node"
tets_path = repair / "mesh.ele"
label_path = root / "paired-current-child.labels.csv"
donors_path = original / "volume-export/registered/donor-frames.csv"
map_path = original / "bindings/combined-node-map.csv.zst"
binding_receipt = original / "bindings/receipt.json"
parent = root / "paired-native-parent-ledger/compiled-parent-cell-mass.csv"
zstd = "/opt/homebrew/bin/zstd"

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def fill(row, names, values):
    for name, val in zip(names, values):
        row[name] = format(float(val), ".17g")

def xyz(stem, suffix=""):
    return tuple(stem + axis + suffix for axis in "xyz")

points = np.loadtxt(nodes_path, skiprows=1, usecols=(1, 2, 3))
tets = np.loadtxt(tets_path, skiprows=1, usecols=(1, 2, 3, 4), dtype=np.int32) - 1
labels = np.loadtxt(label_path, delimiter=",", skiprows=1, usecols=(1, 2, 3), dtype=np.int32)
parent_nodes = np.loadtxt(str(parent)+".nodes.csv", delimiter=",", skiprows=1, usecols=(2, 3, 4))
parent_tets = np.loadtxt(str(parent)+".tets.csv", delimiter=",", skiprows=1,
                        usecols=(3, 4, 5, 6), dtype=np.int32)
geometry_receipt = json.loads((baseline / "geometry-receipt.json").read_text())
source_receipt = json.loads((repair / ("receipt.json" if floor_fix else "geometry-receipt.json")).read_text())
assert len(points) == geometry_receipt["node_count"]
assert len(labels) == len(tets) == geometry_receipt["cell_count"]
assert sha(nodes_path) == source_receipt["mesh_node_sha256"]
assert sha(tets_path) == geometry_receipt["mesh_ele_sha256"]
assert sha(label_path) == json.loads((root/"paired-current-child-labels-receipt.json").read_text())["sha256"]
receipt = json.loads(binding_receipt.read_text())
assert not receipt["source_ids_changed"] and not receipt["full86_weights_changed"]

with donors_path.open() as f:
    frames = {int(row["donor"]): (
        np.array([float(row["current_"+a]) for a in "xyz"]),
        np.array([[float(row[f"current_R{i}{j}"]) for j in range(3)] for i in range(3)]),
        int(row["binding"])) for row in csv.DictReader(f)}
inverses = {body: np.linalg.inv(matrix) for body, (_, matrix, _) in frames.items()}
role_paths = {role: original/f"bindings/registered-{role}.csv" for role in "OID"}
for path in role_paths.values():
    assert sha(path) == receipt["outputs"][str(path)]["sha256"]

with subprocess.Popen([zstd, "-dc", map_path], stdout=subprocess.PIPE, text=True) as proc:
    reader = csv.DictReader(proc.stdout)
    rows = list(reader)
    assert proc.wait() == 0
assert len(rows) >= len(points)
old_points = np.full_like(points, np.nan)
old_donor = np.full(len(points), -1, dtype=np.int32)
old_region = old_donor.copy()
role_nodes = {role: {} for role in "OID"}
for row in rows:
    n = int(row["joint_node_id"]) - 1
    p = [float(row[key]) for key in xyz("current_", "_m")]
    donor, region = (int(row[key]) for key in ("geometric_donor_body", "anatomical_region_id"))
    if old_donor[n] >= 0:
        assert donor == old_donor[n] and region == old_region[n] and np.array_equal(old_points[n], p)
    old_points[n], old_donor[n], old_region[n] = p, donor, region
    role = row["registered_surface_layer"]
    if role in role_nodes:
        compact = int(row["registered_surface_vertex_row"])
        assert role_nodes[role].setdefault(compact, n) == n
assert np.all(np.isfinite(old_points))
moved = np.any(points != old_points, axis=1)
if floor_fix:
    previous_points = np.loadtxt(baseline/"mesh.node", skiprows=1, usecols=(1, 2, 3))
    updated_nodes = np.flatnonzero(np.any(previous_points!=points,axis=1))
    declared = {node for region in source_receipt["patches"] for node in region["moved_node_rows"]}
    assert set(updated_nodes) <= declared
else:
    assert moved.sum() == geometry_receipt["nodes_moved"]

# Keep the existing regional labels where they appear on incident current
# cells. Else use an actual incident cell, preferring the same geometric body.
fallback = np.full(len(points), len(tets), dtype=np.int32)
same_body = fallback.copy()
same_region = fallback.copy()
mask = np.zeros(len(points), dtype=np.uint32)
cell_rows = np.arange(len(tets), dtype=np.int32)
for corner in range(4):
    n = tets[:, corner]
    body_matches = labels[:, 1] == old_donor[n]
    regional_matches = body_matches & (labels[:, 2] == old_region[n])
    np.minimum.at(fallback, n, cell_rows)
    np.minimum.at(same_body, n[body_matches], cell_rows[body_matches])
    np.minimum.at(same_region, n[regional_matches], cell_rows[regional_matches])
    np.bitwise_or.at(mask, n, np.left_shift(np.uint32(1), labels[:, 0].astype(np.uint32)))
selected = np.minimum(np.where(same_region < len(tets), same_region, len(tets)),
                      np.where((same_region == len(tets)) & (same_body < len(tets)), same_body, len(tets)))
selected = np.where(selected == len(tets), fallback, selected)
assert np.all(selected < len(tets))
weights = tets[selected] == np.arange(len(points))[:, None]
assert np.all(weights.sum(axis=1) == 1)
operations = labels[selected, 1:]
registration = {role: {} for role in "OID"}
parent_error = 0.
parent_distance = 0.
axis = "xyz"
with (out / "combined-node-map.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        n = int(row["joint_node_id"]) - 1
        body = int(row["donor_body"])
        position, basis, binding = frames[body]
        assert binding == int(row["donor_binding"])
        p = points[n]
        local = inverses[body] @ (p - position)
        assert np.max(np.abs(position + basis @ local - p)) < 1.e-12
        pt = int(row["parent_tet_row"])
        betas = [float(row[f"parent_bary{i}"]) for i in range(4)]
        anchor = betas @ parent_nodes[parent_tets[pt]]
        residual = p - anchor
        old_residual = np.array([float(row[key]) for key in xyz("parent_current_residual_", "_m")])
        parent_error = max(parent_error, np.linalg.norm(anchor + old_residual - old_points[n]))
        parent_distance = max(parent_distance, np.linalg.norm(residual))
        fill(row, xyz("current_", "_m"), p)
        fill(row, xyz("donor_local_", "_m"), local)
        fill(row, xyz("parent_current_residual_", "_m"), residual)
        row["parent_registration_distance_m"] = format(np.linalg.norm(residual), ".17g")
        row["incident_material_mask"] = str(mask[n])
        row["geometric_donor_body"], row["anatomical_region_id"] = map(str, operations[n])
        row["operational_reference_tet_row"] = str(selected[n])
        for i in range(4):
            row[f"operational_reference_bary{i}"] = "1" if weights[n, i] else "0"
        fill(row, xyz("operational_reference_residual_", "_m"), [0., 0., 0.])
        row["operational_reference_distance_m"] = "0"
        role = row["registered_surface_layer"]
        if role in registration:
            compact = int(row["registered_surface_vertex_row"])
            side = dict(row, layer=role, surface_vertex_row=str(compact))
            for a in axis:
                side[f"parent_residual_{a}_m"] = row[f"parent_current_residual_{a}_m"]
            prior = registration[role].setdefault(compact, side)
            assert all(prior[key] == side[key] for key in (
                "donor_body", "donor_local_x_m", "donor_local_y_m", "donor_local_z_m",
                "parent_tet_row", "parent_registration_distance_m", "source_domain"))
        writer.writerow(row)
del rows

# Write the existing per-role CSVs. Update normals only in the one-ring
# affected by repaired boundary vertices and retain source identities.
with (out / "node-parent-registration.csv").open("w", newline="") as f:
    headings = ("layer,surface_vertex_row,parent_registration_kind,parent_tet_row,"
                "parent_bary0,parent_bary1,parent_bary2,parent_bary3,parent_residual_x_m,"
                "parent_residual_y_m,parent_residual_z_m,parent_registration_distance_m,"
                "parent_registration_tie_count,frame_owner_method,source_domain,source_vertex_row,"
                "source_face_row,source_bary0,source_bary1,source_bary2,source_residual_x_m,"
                "source_residual_y_m,source_residual_z_m").split(",")
    writer = csv.DictWriter(f, fieldnames=headings, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    moved_by_role = {}
    for role, path in role_paths.items():
        n = np.array([role_nodes[role][r] for r in range(len(role_nodes[role]))])
        moved_by_role[role] = int(moved[n].sum())
        faces_path = original/f"volume-export/registered/registered-{role}-faces.csv"
        faces = np.loadtxt(faces_path, delimiter=",", skiprows=1, usecols=(1, 2, 3), dtype=np.int32)
        p = points[n]
        normal = np.zeros_like(p)
        q = p[faces]
        cross = np.cross(q[:, 1]-q[:, 0], q[:, 2]-q[:, 0])
        assert np.all(np.linalg.norm(cross, axis=1) > 0.)
        for corner in range(3):
            np.add.at(normal, faces[:, corner], cross)
        change = np.unique(faces[np.any(moved[n[faces]], axis=1)])
        length = np.linalg.norm(normal[change], axis=1)
        assert np.all(length > 0.)
        normal[change] /= length[:, None]
        touched = set(change)
        with path.open() as source, (out/f"registered-{role}.csv").open("w", newline="") as dest:
            reader = csv.DictReader(source)
            surface = csv.DictWriter(dest, fieldnames=reader.fieldnames, lineterminator="\n")
            surface.writeheader()
            for r, row in enumerate(reader):
                assert int(row["vertex"]) == r
                side = registration[role][r]
                assert row["donor"] == side["donor_body"] and row["source_domain"] == side["source_domain"]
                fill(row, xyz("w"), p[r])
                for a in axis:
                    row[f"l{a}"] = side[f"donor_local_{a}_m"]
                if r in touched:
                    fill(row, xyz("n"), normal[r])
                surface.writerow(row)
                writer.writerow(side)
    if not floor_fix:
        assert moved_by_role == {r: geometry_receipt["nodes_moved_by_role"][r] for r in "OID"}

summary = {
    "status": "PASS_REPAIRED_REFERENCE_SIDECARS", "native_steps": 0, "physical_admission": False,
    "source_world_replay_max_m": float(parent_error), "historical_parent_distance_max_m": float(parent_distance),
    "nodes": len(points), "cells": len(tets), "moved_nodes_by_role": moved_by_role,
    "retained_regional_labels": int(np.count_nonzero(same_region < len(tets))),
    "rebound_same_donor": int(np.count_nonzero((same_region == len(tets)) & (same_body < len(tets)))),
    "rebound_other_incident_donor": int(np.count_nonzero(same_body == len(tets))),
    "inputs": {str(p): sha(p) for p in (nodes_path, tets_path, label_path, binding_receipt, donors_path, map_path)},
    "outputs": {p.name: sha(p) for p in sorted(out.iterdir())},
}
(out/"receipt.json").write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps(summary, indent=2))
