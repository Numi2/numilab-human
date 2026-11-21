"""Classify exact crossing witnesses using the native ocular presentation receipt."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

crossing_path, ocular_path, output = map(Path,sys.argv[1:])
crossing_bytes, ocular_bytes = crossing_path.read_bytes(), ocular_path.read_bytes()
crossing, ocular = json.loads(crossing_bytes), json.loads(ocular_bytes)
assert ocular["presentation_only"] is True and ocular["physical_state_created"] is False
by_primitive = {row["primitive"]: row for row in crossing["surfaces"]}
assert len(by_primitive) == len(crossing["surfaces"])
for item in ocular["windows"] + ocular["posterior_lens_backings"]:
    derived = item["derived_skin_identity"]
    row = by_primitive[item["derived_primitive_index"]]
    assert (row["semantic"],row["stable_id"]) == (derived["semantic"],derived["stable_id"])
    assert row["count"] == len(row["triangle_pairs"])
    row["ocular_presentation_only"] = True
updated = Counter()
for row in crossing["surfaces"]:
    assert row["count"] == len(row["triangle_pairs"])
    category = ("ocular_presentation_only" if row.get("ocular_presentation_only")
                else "ocular_source" if row["ocular_source_id"] else "other_tissue")
    updated[category] += row["count"]
crossing["typed_scope"] = {
    "input_geometry_audit_sha256":hashlib.sha256(crossing_bytes).hexdigest(),
    "native_ocular_presentation_receipt_sha256":hashlib.sha256(ocular_bytes).hexdigest(),
    "counts":dict(updated),
    "geometry_modified":False,
    "tissue_contact_qualification":False,
    "ocular_physical_clearance_qualification":False,
}
with output.open("x") as f:
    json.dump(crossing,f,indent=2)
    f.write("\n")
print(json.dumps(crossing["typed_scope"],indent=2))
