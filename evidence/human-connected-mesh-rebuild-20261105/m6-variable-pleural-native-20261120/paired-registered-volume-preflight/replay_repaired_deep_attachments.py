"""Replay the existing stable deep attachment gates on the repaired volume."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
compiler = root.parent / "captured-fat-sliver-495/proposal/compiler"
mesh = root.parent / "resting-clearance-and-atrium-016/repaired-tetgen"
receipt_path = compiler / "bindings/remap-receipt.json"
request_path = compiler / "bindings/deep-attachment-requests.csv"
role_path = compiler / "bindings/D-source.json"
registration_path = root / "paired-repaired-registration/registered-D.csv"
new_receipt = json.loads((registration_path.parent / "receipt.json").read_text())

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

incumbent = json.loads(receipt_path.read_text())
assert sha(request_path) == incumbent["output"]["sha256"]
bindings_path = compiler / "bindings/receipt.json"
bindings = json.loads(bindings_path.read_text())
assert not bindings["source_ids_changed"] and not bindings["full86_weights_changed"]
assert sha(role_path) == bindings["outputs"][str(role_path)]["sha256"]
assert sha(registration_path) == new_receipt["outputs"][registration_path.name]
with request_path.open() as f:
    requests = {int(row["stable_attachment_id"]): row for row in csv.DictReader(f)}
assert len(requests) == len(incumbent["rows"]) == 14
rows = [row["new_compact_surface_row"] for row in incumbent["rows"]]
want = set(rows)
with registration_path.open() as f:
    register = {i: r for i, r in enumerate(csv.DictReader(f)) if i in want}
roles = json.loads(role_path.read_text())["vertices"]
points = np.loadtxt(mesh / "mesh.node", skiprows=1, usecols=(1, 2, 3))
elements = np.loadtxt(mesh / "mesh.ele", skiprows=1, usecols=(1, 2, 3, 4, 5), dtype=np.int32)
tets = elements[:, :4] - 1
sites = np.array([row["volume_node_row"] for row in incumbent["rows"]], dtype=np.int32)
affected = np.flatnonzero(np.any(np.isin(tets, sites), axis=1))
tt = tets[affected]
q = points[tt]
e = q[:, 1:] - q[:, :1]
volume6 = np.einsum("ij,ij->i", e[:, 0], np.cross(e[:, 1], e[:, 2]))
pairs = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]
areas2 = np.stack([np.linalg.norm(np.cross(q[:, b] - q[:, a], q[:, c] - q[:, a]), axis=1)
                   for a, b, c in pairs], axis=1)
assert np.all(volume6 > 0.) and np.all(areas2 > 0.)
altitude = volume6[:, None] / areas2
limits = incumbent["limits"]
result = []
for previous, site, compact in zip(incumbent["rows"], sites, rows):
    identity = previous["stable_attachment_id"]
    request = requests[identity]
    registration = register[compact]
    role = roles[compact]
    assert int(request["donor_body"]) == int(registration["donor"]) == previous["donor_body"]
    assert int(request["stable_source_vertex"]) == int(registration["source_vertex_row"]) == role["source_vertex_row"]
    assert request["source_domain"] == registration["source_domain"]
    assert role["child_volume_node_row"] == site
    old_point = np.array(previous["current_world_m"])
    moved = float(np.linalg.norm(points[site] - old_point))
    i, j = np.nonzero(tt == site)
    minimum = float(altitude[i, j].min())
    registered = np.array([float(registration[a]) for a in ("wx", "wy", "wz")])
    assert np.array_equal(registered, points[site])
    original_point = np.array(previous["old_world_m"])
    displacement = float(np.linalg.norm(registered - original_point))
    other_sites = [s for p, s in zip(incumbent["rows"], sites)
                   if p["donor_body"] == previous["donor_body"] and s != site]
    spacing = min((np.linalg.norm(registered-points[s]) for s in other_sites), default=np.inf)
    passes = (minimum >= limits["minimum_incident_altitude_m"] and
              displacement <= limits["maximum_displacement_m"] and
              previous["donor_weight"] >= limits["minimum_donor_weight"] and
              spacing >= limits["minimum_spacing_m"])
    result.append({
        "stable_attachment_id": identity,
        "donor_body": previous["donor_body"],
        "volume_node_row": int(site),
        "stable_source_vertex": previous["new_source_vertex"],
        "position_change_m": moved, "displacement_from_original_m": displacement,
        "minimum_same_donor_spacing_m": float(spacing) if np.isfinite(spacing) else None,
        "incident_cells": len(i),
        "minimum_incident_altitude_m": minimum,
        "minimum_incident_altitude_cell": int(affected[i[altitude[i, j].argmin()]]),
        "old_minimum_incident_altitude_m": previous["minimum_incident_altitude_m"],
        "meets_retained_reference_gate": bool(passes),
    })
report = {
    "status": "PASS" if all(row["meets_retained_reference_gate"] for row in result) else "REJECT",
    "mesh_node_sha256": sha(mesh / "mesh.node"), "mesh_ele_sha256": sha(mesh / "mesh.ele"),
    "incumbent_receipt_sha256": sha(receipt_path), "request_sha256": sha(request_path),
    "dominant_binding_receipt_sha256": sha(bindings_path),
    "repaired_registration_sha256": sha(registration_path),
    "limits": limits, "rows": result, "physical_admission": False, "native_steps": 0,
}
(root / "paired-fine-attachment-replay.json").write_text(json.dumps(report, indent=2) + "\n")
print(report["status"], "attachments", len(result), "test cells", len(affected),
      "min altitude", min(row["minimum_incident_altitude_m"] for row in result),
      "max motion", max(row["position_change_m"] for row in result))
raise SystemExit(report["status"] != "PASS")
