import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

here = Path(__file__).resolve().parent
prior = here.parent / "resting-clearance-and-atrium-016"
sys.path.insert(0,str(prior))
from paired_route_projection import PairedRouteProjection
from reflow_paired_mass import sha

audit = json.loads((prior/"full-paired-route-audit.json").read_text())
cut = json.loads((prior/"paired-region-density-capacity-bound.json").read_text())["Skin"]["analytic_deficit_cut"]
path = here.parent/"resting-clearance-and-atrium-014/fine-current-paired-semantic-cycle-mass.npz"
assert sha(path) == audit["source_mass_sha256"]
with np.load(path) as pack:
    flow = {k:pack[k] for k in pack.files}
projection = PairedRouteProjection(flow,audit,np.zeros(len(flow["group_for_cell"]),np.int32))
parent = projection.parent
points = parent["points"]
rows = np.flatnonzero(parent["materials"] == 15)
tet = parent["tets"][rows]
region = parent["regions"][rows]
donor = parent["donors"][rows]
cut_mask = np.isin(region,cut["region_ids"])
roles = [["in",cut_mask],["out",~cut_mask]]
aabb_low = points[tet[cut_mask]].min(axis=(0,1))
aabb_hi = points[tet[cut_mask]].max(axis=(0,1))

faces = np.sort(tet[:,((1,2,3),(0,3,2),(0,1,3),(0,2,1))],axis=2).reshape(-1,3)
unique, inverse, counts = np.unique(faces,axis=0,return_inverse=True,return_counts=True)
inverse = inverse.ravel()
is_boundary = counts[inverse] == 1
owners = np.repeat(np.arange(len(tet)),4)
inside = cut_mask[owners]
exact_shared = np.unique(inverse[inside & ~is_boundary])
faces_across = np.flatnonzero(~inside & np.isin(inverse,exact_shared))
cut_boundary = faces[is_boundary & inside]
other_boundary = faces[is_boundary & ~inside]
other_owners = owners[is_boundary & ~inside]

# Check unordered coordinate equality across duplicated identifiers, including
# exact emission precision, before considering a geometric seam a neighbor.
def lex_coords(x):
    order = np.lexsort((x[:,:,2],x[:,:,1],x[:,:,0]),axis=1)
    return np.take_along_axis(x,order[:,:,None],axis=1)

x = lex_coords(points[cut_boundary])
y = lex_coords(points[other_boundary])
tree = cKDTree(y.reshape(-1,9))
distance, match = tree.query(x.reshape(-1,9))
closest = np.argsort(distance)[:16]
nearby = [{
    "gap_m":float(distance[i]),
    "inside_nodes":cut_boundary[i].tolist(),
    "outside_nodes":other_boundary[match[i]].tolist(),
    "outside_region":int(region[other_owners[match[i]]]),
    "outside_donor":int(donor[other_owners[match[i]]]),
} for i in closest]
interesting = np.unique(np.r_[rows[cut_mask],rows[owners[faces_across]]])
manifest = Path("/Users/n/numi-human-separation-20261018/C/publication-source-001/docs/media/numi-human-passive-fit-v4/source-manifest.json")
body_names = json.loads(manifest.read_text())["core_tree"]["body_order"]
result={
    "cut_regions":cut["region_ids"],
    "source_cut_cell_count":int(cut_mask.sum()),
    "source_cut_boundary_face_count":len(x),
    "shared_topology_neighbors":np.unique(region[owners[faces_across]]).tolist(),
    "shared_face_crossing_count":len(faces_across),
    "seam_match_f64_count":int(np.count_nonzero(distance == 0.)),
    "seam_match_1um_count":int(np.count_nonzero(distance <= 1e-6)),
    "nearest_other_boundary_faces":nearby,
    "donor_ids_and_candidate_names":[[int(d),body_names[d]] for d in np.unique(donor[cut_mask]) if d<len(body_names)],
    "source_cut_aabb":[aabb_low.tolist(),aabb_hi.tolist()],
    "child_cut_aabb":[cut["min_world_position_m"],cut["max_world_position_m"]],
    "source_manifest_sha256":sha(manifest),
}
(here/"parent-region-cut-topology.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!="nearest_other_boundary_faces"},indent=2))
