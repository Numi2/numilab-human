"""Retain the pre-contact admissibility witness and its exact compiled fascia owner."""
import json
from pathlib import Path
import re

from compare_contact_witness import extract

run = Path("motion-backtracking")
status = json.loads((run / "comparison.json").read_text())
stderr = (run / "stderr.log").read_text()
stdout = (run / "stdout.log").read_text()
reports = extract(stderr, "fem_backtracking_trials=")
assert reports and all(row["root"] == 21 and row["microtick"] == 191 for row in reports)
compiled = [
    {"source_object": int(object_index), "element_offset": int(offset),
     "element_count": int(count), "source_region_range": [int(first), int(end)]}
    for object_index, offset, count, first, end in re.findall(
        r"resting_fascia_object=compiled_reference_mode object=(\d+) .*?"
        r"element_offset=(\d+) element_count=(\d+) .*?"
        r"region_begin=(\d+) region_end=(\d+)", stdout)
]
limited = []
for row in reports:
    if not row["trials"] or row["selected_alpha"] >= 1.0:
        continue
    invalid = []
    for trial in row["trials"]:
        if trial["reason"] == 0:
            continue
        tet = trial["first_bad_tet"]
        owners = [p for p in compiled if p["element_offset"] <= tet < p["element_offset"] + p["element_count"]]
        assert len(owners) <= 1
        invalid.append({**trial, "fascia_source_owner": owners[0] if owners else None})
    limited.append({"iteration": row["iteration"], "runtime_object": row["object"],
                    "selected_alpha": row["selected_alpha"], "invalid_trials": invalid})
result = {
    "root": 21,
    "microtick": 191,
    "observed_rows": len(reports),
    "row_failure_legend": {
        "1": "nonfinite determinant",
        "2": "below material or runtime determinant floor",
        "4": "above material determinant ceiling",
        "8": "constitutive validity failure",
    },
    "geometry_or_constitutive_limits": limited,
    "upstream_evidence": [
        {"iteration": minimum["iteration"], "incoming_contact_alpha": minimum["incoming_alpha"],
         "observations_with_same_upstream_alpha": [
             row for row in limited
             if row["iteration"] == minimum["iteration"] and row["selected_alpha"] == minimum["incoming_alpha"]],
         "nominal_contact_bound": minimum["bound"]}
        for minimum in status.get("reported_contact_bounds", [])
    ],
}
(run / "backtracking-witness.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({
    "observed_rows": len(reports),
    "constrained_object_rows": len(limited),
    "incoming_bounds_explained": sum(bool(row["observations_with_same_upstream_alpha"])
                                    for row in result["upstream_evidence"]),
    "minimum": min(limited, key=lambda row: row["selected_alpha"]) if limited else None,
}, indent=2))
