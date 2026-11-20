"""Identify which retained geometry transfer created the capacity mismatch."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

here = Path(__file__).resolve().parent
prior = here.parent / "resting-clearance-and-atrium-016"
sys.path.insert(0, str(prior))
from screen_paired_mass_routes import PREVIOUS, load, sha
from export_repaired_tetgen import array_hash

proof = json.loads((prior / "full-paired-route-audit.json").read_text())
cut = json.loads((prior / "paired-region-density-capacity-bound.json").read_text())["Skin"]["analytic_deficit_cut"]
mask = prior / proof["role_reports"]["Skin"]["region_bounds_file"]
assert sha(mask) == proof["role_reports"]["Skin"]["region_bounds_sha256"]
with np.load(mask) as archive:
    words = archive["group_and_regions"][:, 1:].copy()
    for region in cut["region_ids"]:
        k = int(np.searchsorted(archive["region_ids"], region))
        assert archive["region_ids"][k] == region
        words[:, k // 64] &= ~(np.uint64(1) << np.uint64(k % 64))
    cells = archive["child_rows"][~np.any(words[archive["child_mask_index"]], axis=1)]
assert len(cells) == cut["constrained_cells"]
old, new, tets, material, origins, ancestry, roles, source_tets, _ = load()
assert array_hash(tets, "<i4") == proof["tetrahedron_sha256"]
with np.load(PREVIOUS / "fine-mass-local-position-delta.npz", allow_pickle=False) as patch:
    final = new.copy()
    final[patch["changed_rows"]] = patch["positions_f32"]
assert array_hash(final, "<f4") == proof["geometry_sha256"]
assert np.all(material[cells] == 15)

def report(xyz, connectivity):
    points = xyz[connectivity].astype(np.float64)
    volume = np.linalg.det(points[:, 1:] - points[:, :1]) / 6.
    return {
        "signed_volume_m3": float(volume.sum()),
        "nonpositive_count": int(np.count_nonzero(volume <= 0.)),
        "positive_volume_m3": float(volume[volume > 0.].sum()),
        "mass_at_900_kg_per_m3": float(900*volume.sum()),
    }

owner_pins = json.loads((here.parent / "resting-clearance-and-atrium-013/fine-local-current-p1.json").read_text())["parent_inputs"]
paths = {name: Path(data["path"]) for name, data in owner_pins.items()}
assert all(sha(path) == owner_pins[name]["sha256"] for name, path in paths.items())
parent_source_mass = np.loadtxt(paths["source330_mass"], skiprows=1, delimiter=",")
parent_label = np.loadtxt(paths["source330_labels"], skiprows=1, delimiter=",", dtype=int)
assert np.array_equal(parent_source_mass[:, :2], parent_label[:, :2])
m = (parent_label[:,1] == 15) & np.isin(parent_label[:,3], cut["region_ids"])
old_cells = origins[cells]
unchanged = old_cells >= 0
result = {
    "original_native_source_mass_kg": float(parent_source_mass[m,2].sum()),
    "paired_source_mass_kg": cut["maximum_available_kg"],
    "retopologized_cut_count": int((~unchanged).sum()),
    "cut_support": cut["region_ids"],
    "frames": {
        "original_native": report(old, tets[cells]),
        "paired_transfer": report(new, tets[cells]),
        "geometry_repair": report(final, tets[cells]),
        "original_native_same_retained_topology": report(old, source_tets[old_cells[unchanged]]),
        "geometry_repair_same_retained_topology": report(final, tets[cells[unchanged]]),
    },
    "scope": "Identical current cut rows. Historical signed measurements are diagnostic and do not attest their old registration.",
}
assert abs(result["frames"]["geometry_repair"]["mass_at_900_kg_per_m3"] - cut["minimum_required_kg"]) < 1e-12
(here / "skin-cut-frame-comparison.json").write_text(json.dumps(result,indent=2) + "\n")
print(json.dumps(result,indent=2))
