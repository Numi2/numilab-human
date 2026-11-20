"""Retain exact boundary-face lineage inside the selected P1 volumetric group."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
registration = root / "paired-repaired-registration-floor"
ancestor = root / "paired-repaired-registration/facet-ancestry.csv"
labels_path = root / "paired-current-child.labels.csv"
parent_path = root / "paired-native-parent-ledger/compiled-parent-cell-mass.csv.labels.csv"
def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

assert digest(ancestor) == "714aa22acc00784bf3f8ce0d77e8550b7a3bf0319f037264220f7e86be3999de"
assert digest(labels_path) == "3d543fa86a3b7319b047ca8dbb11d6691c3f81b3101f82d02049ea9baef2a634"
with ancestor.open() as stream:
    walls = [row for row in csv.DictReader(stream) if "headwall" in row["face_source_role"]]
assert len(walls) == 180
labels = np.loadtxt(labels_path, delimiter=",", skiprows=1, dtype=np.int64)
parent = np.loadtxt(parent_path, delimiter=",", skiprows=1, dtype=np.int64)
overrides = {}
for name, material in (("skin", 15), ("fat", 16)):
    receipt = json.loads((root / f"complete-fine-p1-current-parent/{name}-p1-receipt.json").read_text())
    p1 = Path(receipt["p1_one_hop_path"])
    assert digest(p1) == receipt["p1_one_hop_sha256"]
    with np.load(p1) as source:
        starts = source["cell_group_offsets"]
        offsets = source["cell_group_stable_face_offsets"]
        faces = source["cell_group_stable_face_rows"]
        scores = source["cell_group_p1_score_numerators"]
        group_ids = source["cell_group_ids"]
        groups = source["group_tuples"]
    role_rows = np.flatnonzero(labels[:, 1] == material)
    for row in walls:
        if row["source_partition"] != name:
            continue
        child = int(row["child_tet_row"])
        donor = int(row["face_parent_tet_row"])
        source_id, = json.loads(row["face_ancestry_source_rows"])
        assert row["face_ancestry_source_original_ids"] == f"[{source_id}]"
        assert row["parent_source_node_rows"] == "[-1,-1,-1]"
        assert parent[donor, 0] == donor and parent[donor, 1] == material and parent[donor, 4] == source_id
        local = np.searchsorted(role_rows, child)
        assert role_rows[local] == child
        lo, hi = starts[local:local+2]
        slots = np.arange(lo, hi)
        winning = slots[(scores[lo:hi] == scores[lo:hi].max()) &
                        np.all(groups[group_ids[lo:hi]] == labels[child, 1:4], axis=1)]
        slot, = winning
        assert source_id in faces[offsets[slot]:offsets[slot+1]]
        assert labels[child, 4] == 2**31 + int(group_ids[slot])
        assert overrides.setdefault(child, source_id) == source_id
final_facets = registration / "facet-ancestry-current.csv"
with ancestor.open(newline="") as stream, final_facets.open("w", newline="") as output:
    source = csv.DictReader(stream)
    writer = csv.DictWriter(output, source.fieldnames, lineterminator="\n")
    writer.writeheader()
    rows = 0
    for row in source:
        if "headwall" in row["face_source_role"]:
            row["face_ancestry_parent_face_rows"] = f"[{row['child_tet_row']}]"
        writer.writerow(row)
        rows += 1
assert rows == 833108
final_labels = root / "paired-floor-child.labels.csv"
with labels_path.open() as original, final_labels.open("w") as output:
    output.write(original.readline())
    for i, line in enumerate(original):
        if i in overrides:
            prefix, _ = line.rsplit(",", 1)
            line = f"{prefix},{overrides[i]}\n"
        output.write(line)
receipt = {
    "facets": rows, "head_wall_facets_rebound": len(walls),
    "child_lineage_ids_resolved_within_unchanged_p1_tuple": len(overrides),
    "head_wall_centroids_or_nodes_changed": 0,
    "original_labels_sha256": digest(labels_path),
    "labels_sha256": digest(final_labels),
    "facets_sha256": digest(final_facets),
    "exact_face_assignments": sorted(overrides.items()),
}
(root / "paired-floor-wall-lineage.json").write_text(json.dumps(receipt,indent=2)+"\n")
print({k:v for k,v in receipt.items() if k != "exact_face_assignments"})
