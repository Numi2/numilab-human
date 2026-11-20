"""Audit edited-cell mass routes against the same-material one-hop proof."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

here = Path.cwd()
previous = here.parent/"resting-clearance-and-atrium-013"
screening = json.loads((previous/"fine-local-current-p1.json").read_text())
compiler = here.parent/"captured-fat-sliver-495/proposal/compiler"
owner_path = Path(json.loads((compiler/"p1-Skin.command.json").read_text())["command"][1])

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

assert digest(owner_path) == screening["owner_sha256"]
spec = importlib.util.spec_from_file_location("pinned_role_p1", owner_path)
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)
native = owner._load_230(owner.OWNER230).native
for name,source in screening["parent_inputs"].items():
    assert digest(Path(source["path"])) == source["sha256"], name
parent = native.read_parent(owner.PARENT)
flow_file = here/"fine-current-paired-semantic-cycle-mass.npz"
with np.load(flow_file,allow_pickle=False) as data:
    flow = data["mass_flow_kg"]
    carrier = data["carrier_row"]
    parent_rows = data["source_parent"]
    receiver = data["group"]
    child_group = data["group_for_cell"]
    child_mass = data["element_mass_kg"]
assert np.all(flow>=0) and np.all(child_mass>0)
group_mass = np.bincount(child_group,weights=child_mass)
expected = np.bincount(receiver,weights=flow,minlength=len(group_mass))
assert np.allclose(expected, group_mass,rtol=1e-12,atol=1.e-15)
order = np.flatnonzero(flow>0)
order = order[np.argsort(receiver[order],kind="stable")]
off = np.r_[0,np.cumsum(np.bincount(receiver[order],minlength=len(group_mass)))]
report = {}
for role, result in screening["outputs"].items():
    proof = previous/result["evidence"]
    assert digest(proof)==result["sha256"], role
    with np.load(proof,allow_pickle=False) as packet:
        cells = packet["child_rows"]
        adjacency,_ = owner.build_region_adjacency(parent,role)
        evidence = owner.expand_onehop(packet,packet["source_groups"],adjacency,role,len(cells))
    ao = evidence["cell_admissible_region_offsets"]
    ar = evidence["cell_admissible_region_ids"]
    misses = []
    route_count = 0
    for local,child in enumerate(cells):
        g = child_group[child]
        edges = order[off[g]:off[g+1]]
        source = parent_rows[carrier[edges]]
        assert np.all(parent["materials"][source] == owner.ROLE_MATERIAL[role])
        region = parent["regions"][source]
        allowed = np.isin(region,ar[ao[local]:ao[local+1]])
        fraction = child_mass[child]/group_mass[g]
        route_count += len(edges)
        if not np.all(allowed):
            misses.append({
                "child":int(child),
                "route_count":int(np.count_nonzero(~allowed)),
                "mass_kg":float((flow[edges[~allowed]]*fraction).sum()),
                "source_region_ids":np.unique(region[~allowed]).tolist(),
                "source_parent_rows":np.unique(source[~allowed]).tolist(),
                "one_hop_region_ids":ar[ao[local]:ao[local+1]].tolist(),
            })
    report[role] = {
        "cells_checked": len(cells),
        "positive_expanded_routes_checked":route_count,
        "cells_with_beyond_one_hop_routes":len(misses),
        "positive_routes_needing_more_evidence":sum(m["route_count"] for m in misses),
        "mass_needing_more_evidence_kg":sum(m["mass_kg"] for m in misses),
        "witnesses":misses,
    }
output = {
    "scope": "353 corrected or incident child cells in the current exact Float32 frame",
    "method": "positive flow expanded by each child's conserved group mass fraction; source typed region intersected with frozen source330 one-hop adjacency",
    "interpretation": "A route outside the authenticated one-hop domain cannot be admitted by supplemental exact projection support alone. Passing this region screen still requires exact typed compiler admission.",
    "flow_sha256":digest(flow_file),
    "role_reports":report,
}
(here/"repaired-edited-cell-flow-evidence.json").write_text(json.dumps(output,indent=2)+"\n")
print(json.dumps({role:{k:v for k,v in row.items() if k!="witnesses"}
                  for role,row in report.items()},indent=2))
