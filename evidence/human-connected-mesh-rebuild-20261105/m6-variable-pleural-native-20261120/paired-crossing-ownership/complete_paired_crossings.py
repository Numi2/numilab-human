"""Complete the zero-to-accepted pair audit using a matched native time-zero receipt."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from load_complete_reference import banks, clearance, exact, run, skin, skin_faces

initial_report_file, typed_report_file, output = map(Path,sys.argv[2:5])
seed = json.loads(initial_report_file.read_text())
typed = json.loads(typed_report_file.read_text())
raw = run/"exact-skin-surface.json"
assert hashlib.sha256(raw.read_bytes()).hexdigest() == seed["terminal_report_sha256"]
assert typed["typed_scope"]["input_geometry_audit_sha256"] == seed["terminal_report_sha256"]
previous = {tuple(row["key"]): row for row in seed["banks"]}
records = clearance._exact_surface_records(skin, skin_faces)
index = exact._prepare_surface_aabb(records)
broad = clearance._float32_skin_aabb_index(records)
totals = Counter()
counts = {}
results = []
for row in typed["surfaces"]:
    key = row["semantic"], row["stable_id"]
    if row["ocular_source_id"] or row.get("ocular_presentation_only"):
        continue
    prior = previous[key]
    if not prior.get("unavailable_initial_registration"):
        result = prior
    else:
        xyz, faces = banks[key]
        assert len(faces) == row["target_faces"]
        chosen, count = clearance._candidate_target_rows_for_skin_aabb(faces, xyz, broad)
        ids = np.unique(faces[chosen])
        recs = clearance._exact_surface_records(xyz[ids], np.searchsorted(ids,faces[chosen]))
        recs = [(a,b,c,int(chosen[d]),e) for a,b,c,d,e in recs]
        verified = exact._audit_pair_prepared_first(index, recs)
        assert verified["aabb_candidate_pairs"] == count
        initial = set(map(tuple,verified["triangle_pairs"]))
        terminal = set(map(tuple,row["triangle_pairs"]))
        result = dict(key=key,initial=len(initial),terminal=len(terminal),
                      persistent=len(initial&terminal),appeared=len(terminal-initial),
                      cleared=len(initial-terminal),initial_pose_recovered=True,
                      initial_only_pairs=sorted(initial-terminal),
                      terminal_only_pairs=sorted(terminal-initial))
    results.append(result)
    summary = counts.setdefault(str(key[0]),Counter())
    for column in ("initial","terminal","persistent","appeared","cleared"):
        summary[column] += result[column]
        totals[column] += result[column]
assert totals["terminal"] == typed["typed_scope"]["counts"]["other_tissue"]
assert totals["persistent"]+totals["appeared"] == totals["terminal"]
assert totals["persistent"]+totals["cleared"] == totals["initial"]
skin_self = exact._audit_pair_prepared_first(index,records,same_surface=True)
receipt = dict(scope="All nonocular common topology banks at initialized source and captured step 16",
               seed_audit_sha256=hashlib.sha256(initial_report_file.read_bytes()).hexdigest(),
               typed_audit_sha256=hashlib.sha256(typed_report_file.read_bytes()).hexdigest(),
               totals=dict(totals), per_semantic=counts, banks=results,
               initial_outer_self_pairs=skin_self,
               terminal_outer_self_pairs=typed["self_intersections"])
with output.open("x") as f:
    json.dump(receipt,f,indent=2)
    f.write("\n")
print(json.dumps(dict(totals=totals, recovered=[r for r in results
    if r.get("initial_pose_recovered") and r["terminal"]]),indent=2))
