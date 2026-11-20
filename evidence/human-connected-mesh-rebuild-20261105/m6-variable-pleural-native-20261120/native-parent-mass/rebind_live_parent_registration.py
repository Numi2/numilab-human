"""Reconstruct the physical volume's retained barycentric bindings on the live parent."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

root = Path(__file__).resolve().parent
prior = root / "paired-repaired-registration-floor"
out = root / "paired-repaired-registration-live-parent"
out.mkdir()
state_path = root / "complete-fine-p1-live-parent/current-paired-volume-state.json"
state = json.loads(state_path.read_text())

def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def load_source(key, columns, dtype=float):
    pin = state["source330_identity"][key]
    assert digest(pin["path"]) == pin["sha256"]
    return np.loadtxt(pin["path"], usecols=columns, dtype=dtype, delimiter=",", skiprows=1)

tets = load_source("source330_tets", (3, 4, 5, 6), np.int32)
corners = load_source("source330_nodes", (2, 3, 4))[tets]
points = np.loadtxt(root / "absolute-volume-repaired-tetgen/mesh.node",
                    skiprows=1, usecols=(1, 2, 3))
live_rows = {role: {} for role in "OID"}
counts = dict(cells=0, role_rows=0)
maximum = 0.0
with subprocess.Popen(["/opt/homebrew/bin/zstd", "-dc", prior / "combined-node-map.csv.zst"],
                      stdout=subprocess.PIPE) as restored:
    import io
    source = io.TextIOWrapper(restored.stdout, newline="")
    with (out / "combined-node-map.csv").open("w", newline="") as result:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(result, reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        keys = [f"parent_current_residual_{axis}_m" for axis in "xyz"]
        distance_key = "parent_registration_distance_m"
        for row in reader:
            node = int(row["joint_node_id"]) - 1
            current = points[node]
            tet = int(row["parent_tet_row"])
            bary = np.asarray([float(row[f"parent_bary{i}"]) for i in range(4)])
            residual = current - bary @ corners[tet]
            distance = float(np.linalg.norm(residual))
            assert np.isfinite(distance) and distance <= .02
            maximum = max(maximum, distance)
            for key, val in zip(keys, residual):
                row[key] = format(val, ".17g")
            row[distance_key] = format(distance, ".17g")
            role = row["registered_surface_layer"]
            if role in live_rows:
                vertex = int(row["registered_surface_vertex_row"])
                values = tuple(row[k] for k in keys + [distance_key])
                assert values == live_rows[role].setdefault(vertex, values)
            writer.writerow(row)
            counts["cells"] += 1
    assert restored.wait() == 0

with (prior / "node-parent-registration.csv").open(newline="") as source, \
     (out / "node-parent-registration.csv").open("w", newline="") as result:
    reader = csv.DictReader(source)
    writer = csv.DictWriter(result, reader.fieldnames, lineterminator="\n")
    writer.writeheader()
    keys = [f"parent_residual_{axis}_m" for axis in "xyz"] + [distance_key]
    for row in reader:
        values = live_rows[row["layer"]][int(row["surface_vertex_row"])]
        for key, value in zip(keys, values):
            row[key] = value
        writer.writerow(row)
        counts["role_rows"] += 1
assert counts["role_rows"] == sum(map(len, live_rows.values()))
for name in ("all-body-current-rest.csv", "source-domain-pins.tsv", "facet-ancestry-current.csv",
             "registered-O.csv", "registered-I.csv", "registered-D.csv"):
    (out / name).hardlink_to(prior / name)
report = {"current_parent_state_sha256": digest(state_path), "node_map_rows": counts["cells"],
          "role_rows": counts["role_rows"], "maximum_parent_residual_m": maximum,
          "outputs": {p.name: digest(p) for p in out.iterdir()}}
(out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
