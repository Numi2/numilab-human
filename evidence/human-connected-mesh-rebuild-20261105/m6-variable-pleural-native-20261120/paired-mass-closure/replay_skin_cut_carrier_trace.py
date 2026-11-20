"""Recover the previously certified source flow for byte-identical skin cells."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

root = Path(__file__).resolve().parent
prior = root.parent / "resting-clearance-and-atrium-016"
sys.path.insert(0, str(prior))
from screen_paired_mass_routes import COMPILER, PREVIOUS, load, sha
from export_repaired_tetgen import array_hash

audit = json.loads((prior / "full-paired-route-audit.json").read_text())
cut = json.loads((prior / "paired-region-density-capacity-bound.json").read_text())["Skin"]["analytic_deficit_cut"]
bounds = prior / audit["role_reports"]["Skin"]["region_bounds_file"]
assert sha(bounds) == audit["role_reports"]["Skin"]["region_bounds_sha256"]
with np.load(bounds) as data:
    bits = data["group_and_regions"][:, 1:].copy()
    for region in cut["region_ids"]:
        k = int(np.searchsorted(data["region_ids"], region))
        assert data["region_ids"][k] == region
        bits[:, k // 64] &= ~(np.uint64(1) << np.uint64(k % 64))
    rows = data["child_rows"][~np.any(bits[data["child_mask_index"]], axis=1)]
assert len(rows) == cut["constrained_cells"]

old, xyz, tets, material, origins, _, _, old_tets, old_materials = load()
with np.load(PREVIOUS / "fine-mass-local-position-delta.npz") as patch:
    xyz[patch["changed_rows"]] = patch["positions_f32"]
assert array_hash(xyz, "<f4") == audit["geometry_sha256"]
assert array_hash(tets, "<i4") == audit["tetrahedron_sha256"]
old_rows = origins[rows]
assert (old_rows >= 0).all()
assert np.array_equal(tets[rows], old_tets[old_rows])
assert np.array_equal(material[rows], old_materials[old_rows])
assert np.array_equal(xyz[tets[rows]], old[tets[rows]])

receipt = json.loads((COMPILER / "mass-admission/mass-receipt.json").read_text())
label_pin = receipt["source330"]["labels"]
label_path = Path(label_pin["path"])
assert sha(label_path) == label_pin["sha256"]
labels = np.loadtxt(label_path, delimiter=",", skiprows=1, dtype=np.int32)
assert np.array_equal(labels[:, 0], np.arange(len(labels)))
trace_pin = receipt["output_sha256"]["carrier_trace"]
trace = Path(trace_pin["path"] + ".zst")
digest = hashlib.sha256()
lookup = {int(i): j for j, i in enumerate(old_rows)}
received = np.zeros(len(rows))
outside_received = np.zeros(len(rows))
outside_receipts = []
source_summary = {}
with subprocess.Popen(["/opt/homebrew/bin/zstd", "-qcd", str(trace)], stdout=subprocess.PIPE) as proc:
    header = proc.stdout.readline()
    digest.update(header)
    assert header.startswith(b"flow_edge_row,child_row,expanded_parent_row,source_parent_row,carrier_row,")
    for line in proc.stdout:
        digest.update(line)
        fields = line.split(b",")
        slot = lookup.get(int(fields[1]))
        if slot is None:
            continue
        source = int(fields[3])
        mass = float(fields[5])
        distance = float(fields[7])
        assert labels[source, 1] == 15
        received[slot] += mass
        region = int(labels[source, 3])
        if region not in cut["region_ids"] and mass > 0.:
            outside_received[slot] += mass
            source_summary[region] = source_summary.get(region, 0.) + mass
            outside_receipts.append((int(rows[slot]), int(old_rows[slot]), source,
                                     int(fields[4]), mass, distance))
    assert proc.wait() == 0
assert digest.hexdigest() == trace_pin["sha256"]
p = xyz[tets[rows]].astype(np.float64)
volume = np.linalg.det(p[:, 1:] - p[:, :1]) / 6.
assert np.all(volume > 0.)
density = received / volume
result = {
    "exact_geometry_rows": len(rows),
    "cut_regions": cut["region_ids"],
    "archived_parent_label_sha256": label_pin["sha256"],
    "archived_carrier_trace_decoded_sha256": digest.hexdigest(),
    "received_mass_kg": float(received.sum()),
    "minimum_density_kg_per_m3": float(density.min()),
    "maximum_density_kg_per_m3": float(density.max()),
    "outside_mass_kg": float(outside_received.sum()),
    "outside_flow_rows": len(outside_receipts),
    "outside_recipients": int(np.count_nonzero(outside_received)),
    "outside_regions": source_summary,
    "outside_carrier_distance_max_m": max((row[-1] for row in outside_receipts), default=0.),
    "original_transport_distance_bound_m": receipt["transport_distance_limit_m"],
    "scope": "Previously certified identical child geometry and parent label lanes. No current-frame rebinding or full mass allocation claimed.",
}
assert len(outside_receipts) or result["outside_mass_kg"] == 0.
if outside_receipts:
    with (root / "skin-cut-outside-carriers.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("new_child_row", "archived_child_row", "parent_row", "carrier_row",
                         "mass_kg", "archived_distance_m"))
        writer.writerows(outside_receipts)
    result["outside_witness_sha256"] = sha(root / "skin-cut-outside-carriers.csv")
(root / "skin-cut-carrier-trace.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
