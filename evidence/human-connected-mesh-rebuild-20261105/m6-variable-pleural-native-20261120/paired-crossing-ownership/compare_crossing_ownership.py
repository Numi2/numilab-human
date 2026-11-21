"""Attribute frozen endpoint crossing pairs to initial registration or motion."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[2])
from native_source import banks, clearance, exact, run, skin, skin_faces

start = time.monotonic()
observed_path = run / "exact-skin-surface.json"
observed_bytes = observed_path.read_bytes()
observed = json.loads(observed_bytes)
records = clearance._exact_surface_records(skin, skin_faces)
index = exact._prepare_surface_aabb(records)
broad_index = clearance._float32_skin_aabb_index(records)
results = []
for row in observed["surfaces"]:
    key = row["semantic"], row["stable_id"]
    if row["ocular_source_id"]:
        continue
    terminal = set(map(tuple, row["triangle_pairs"]))
    geometry = banks.get(key)
    if geometry is None:
        results.append(dict(key=key, unavailable_initial_registration=True, terminal=len(terminal)))
        continue
    positions, faces = geometry
    assert len(faces) == row["target_faces"]
    chosen, n = clearance._candidate_target_rows_for_skin_aabb(faces, positions, broad_index)
    ids = np.unique(faces[chosen])
    probes = clearance._exact_surface_records(positions[ids], np.searchsorted(ids,faces[chosen]))
    probes = [(a,b,c,int(chosen[d]),e) for a,b,c,d,e in probes]
    reference = exact._audit_pair_prepared_first(index, probes)
    assert reference["aabb_candidate_pairs"] == n
    initial = set(map(tuple, reference["triangle_pairs"]))
    shared = initial & terminal
    appeared = terminal - initial
    cleared = initial - terminal
    results.append(dict(key=key, initial=len(initial), terminal=len(terminal), persistent=len(shared),
                        appeared=len(appeared), cleared=len(cleared),
                        initial_only_pairs=sorted(cleared), terminal_only_pairs=sorted(appeared)))
    if len(results) % 50 == 0:
        print(f"compared {len(results)} banks",file=sys.stderr,flush=True)

totals = Counter()
per_semantic = {}
for result in results:
    summary = per_semantic.setdefault(str(result["key"][0]), Counter())
    if "initial" not in result:
        summary["unpaired_banks"] += 1
        summary["unpaired_terminal"] += result["terminal"]
        totals["unpaired_banks"] += 1
        totals["unpaired_terminal"] += result["terminal"]
        continue
    for label in ("initial", "terminal", "persistent", "appeared", "cleared"):
        value = result[label]
        summary[label] += value
        totals[label] += value

out = dict(scope="Exact triangle-pair comparison of pinned initial and one accepted endpoint",
    terminal_report_sha256=hashlib.sha256(observed_bytes).hexdigest(), terminal_root=observed["accepted_step"],
    terminal_time_s=observed["accepted_time_s"], paired_reference_sha256=json.loads(
        (run/"native/accepted-geometry/step-16.receipt.json").read_text())["base_pack_file_sha256"],
    per_semantic={k:dict(v) for k,v in sorted(per_semantic.items())}, totals=dict(totals),
    banks=results)
path = Path(sys.argv[3])
with path.open("x") as f:
    json.dump(out, f, indent=2)
    f.write("\n")
print(json.dumps(dict(wall_s=time.monotonic()-start, **out["totals"])))
