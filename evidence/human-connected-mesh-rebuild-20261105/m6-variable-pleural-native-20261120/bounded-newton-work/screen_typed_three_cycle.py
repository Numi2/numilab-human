"""Find a three-receiver mass exchange with exact typed support and unchanged quotas."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

root = Path.cwd()
prev = root.parent / "resting-clearance-and-atrium-011"
sys.path.insert(0, str(prev))
from candidate import COMPILER, load
from paired_reference_carriers import paired_reference_carriers

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

pins = json.loads((COMPILER / "mass-admission/mass-receipt.json").read_text())["input_sha256"]
sources = {}
for key in ("source330_nodes", "source330_tets", "source330_mass", "source330_labels"):
    entry = pins[key]
    sources[key] = Path(entry["path"])
    assert sha(sources[key]) == entry["sha256"], key

script = Path(json.loads((COMPILER / "p1-Skin.command.json").read_text())["command"][1])
spec = importlib.util.spec_from_file_location("pinned_role_p1", script)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert mod.PARENT == sources["source330_mass"]
owner = mod._load_230(mod.OWNER230)
parent = owner.native.read_parent(mod.PARENT)
paired_rest, paired_tets, _, frame = paired_reference_carriers(
    COMPILER, sources["source330_nodes"], sources["source330_tets"])
assert np.array_equal(parent["tets"], paired_tets)
old_nodes = np.loadtxt(sources["source330_nodes"], delimiter=",", skiprows=1)
current = old_nodes[:, 2:5] + (paired_rest - old_nodes[:, 5:8])
parent["points"] = current
edges = current[paired_tets[:, 1:]] - current[paired_tets[:, :1]]
volume6 = np.einsum("ni,ni->n", edges[:, 0], np.cross(edges[:, 1], edges[:, 2]))
assert np.all(np.isfinite(volume6)) and np.all(volume6 > 0.)
index = owner.native.build_index(parent)
material_index = owner.build_material_indices(parent)


# Use exact P1 group membership for any new route; retain all mass moments.
_, xyz, tets, mat, _, _, _, _, _ = load()
with np.load(prev/"fine-mass-local-position-delta.npz", allow_pickle=False) as patch:
    xyz[patch["changed_rows"]] = patch["positions_f32"]
flow_file = root.parent/"resting-clearance-and-atrium-012/fine-current-paired-relative-carrier-mass.npz"
with np.load(flow_file,allow_pickle=False) as pack:
    carrier = pack["carrier_row"]
    receiver = pack["group"]
    cell_group = pack["group_for_cell"]
    source_parent = pack["source_parent"]
    flow = pack["mass_flow_kg"]
    cell_mass = pack["element_mass_kg"]
groups, lookup = mod._source_groups(parent,"Skin")
donor_tuple = np.stack((parent["materials"], parent["donors"], parent["regions"]),axis=1)
target = 1448620
target_group = int(cell_group[target])
permit_group=lambda p: tuple(int(x) for x in donor_tuple[int(p)])
typed = [permit_group(p) for p in source_parent]
p1packet = root.parent/"resting-clearance-and-atrium-013/fine-local-current-p1-skin.npz"
with np.load(p1packet,allow_pickle=False) as packet:
    row = int(np.flatnonzero(packet["child_rows"]==target)[0])
    begin,end = packet["cell_group_offsets"][row:row+2]
    target_ids = {tuple(x.tolist()) for x in packet["source_groups"][packet["cell_group_ids"][begin:end]]}
assert target_ids == {(15,75,69),(15,76,69)}
bad = np.flatnonzero((receiver==target_group)&(flow>0))
assert len(bad)==1 and typed[carrier[bad[0]]][2]==57
bad = int(bad[0])
original = int(carrier[bad])
mass = float(cell_mass[target])
assert np.isclose(mass,4.332529375330995e-08,rtol=1.e-13)

# Geometry and quadrature positions are the exact authenticated paired
# reference frames. Source projection for P1 uses current-world positions.
tet_vertices = paired_rest[paired_tets]
carrier_pos, carrier_ids = [], []
basin_pins = json.loads((COMPILER/"mass-admission/mass-receipt.json").read_text())["input_sha256"]
for role in ("skin","fat"):
    packet = basin_pins[f"carrier_{role}_packet"]
    file = Path(packet["path"])
    assert sha(file)==packet["sha256"]
    with np.load(file,allow_pickle=False) as data:
        rows = np.repeat(data["global_source330_tet_rows"],np.diff(data["parent_offsets"]).astype(int))
        carrier_ids.extend(rows)
        carrier_pos.extend(np.einsum("na,naj->nj",data["carrier_barycentric"],tet_vertices[rows]))
centroid_only=np.setdiff1d(np.arange(len(paired_tets)),carrier_ids)
carrier_pos = np.vstack((carrier_pos,tet_vertices[centroid_only].mean(axis=1)))
assert np.array_equal(np.r_[carrier_ids,centroid_only],source_parent)

distance_limit = float(json.loads((COMPILER/"mass-admission/mass-receipt.json").read_text())["transport_distance_limit_m"])
target_centroid = xyz[tets[target]].mean(axis=0,dtype=np.float64)
allowed_parent = np.asarray([typed[c] in target_ids for c in range(len(typed))])
distance_to_target = np.linalg.norm(carrier_pos-target_centroid,axis=1)
possible = np.flatnonzero((flow>=mass)&allowed_parent[carrier]&(receiver!=target_group)&
                        (distance_to_target[carrier]<=distance_limit))
row_sort=np.argsort(cell_group,kind="stable")
offset=np.r_[0,np.cumsum(np.bincount(cell_group))]
group_rows = {g:row_sort[offset[g]:offset[g+1]] for g in np.unique(receiver[possible])}

# Lookup exact P1 support once per tested cell, in batches.
def p1_group_ids(rows):
    nodes, local = np.unique(tets[rows], return_inverse=True)
    support = mod.build_node_support(xyz[nodes], "Skin", parent, index, material_index,
                                     groups, lookup, owner)
    data = mod.integrate_p1_cells(local.reshape(-1, 4), support["maps"], groups, parent, "Skin")
    offsets = data["cell_group_offsets"]
    return [set(int(x) for x in data["cell_group_ids"][lo:hi])
            for lo, hi in zip(offsets[:-1], offsets[1:])]

group_sizes = np.diff(offset)
all_candidate_a = possible[group_sizes[receiver[possible]] == 1]
rows_a = row_sort[offset[receiver[all_candidate_a]]]
groups_a = p1_group_ids(rows_a)
center_a = xyz[tets[rows_a]].mean(axis=1, dtype=np.float64)
group_eligible = {}
for i, support in enumerate(groups_a):
    for group_id in support:
        group_eligible.setdefault(group_id, []).append(i)
carrier_group = np.array([lookup.get(t, -1) for t in typed], dtype=np.int32)
ids_for_b = np.array(sorted(group_eligible))
outgoing_b = np.flatnonzero((flow >= mass) & (receiver != target_group) &
                           (group_sizes[receiver] == 1) &
                           np.isin(carrier_group[carrier], ids_for_b) &
                           (carrier != original))
orig_group_id = lookup[typed[original]]
rows_b = row_sort[offset[receiver[outgoing_b]]]
center_b = xyz[tets[rows_b]].mean(axis=1, dtype=np.float64)
origin_distance = np.linalg.norm(center_b-carrier_pos[original], axis=1)
# Use shorter local exchanges before more distant feasible routes.
indices_b = np.argsort(origin_distance, kind="stable")
indices_b = indices_b[origin_distance[indices_b] <= distance_limit]
pairs_by_b = {}
for i in indices_b:
    e_b = outgoing_b[i]
    a_candidates = np.asarray(group_eligible[carrier_group[carrier[e_b]]], dtype=int)
    a_candidates = a_candidates[receiver[all_candidate_a[a_candidates]] != receiver[e_b]]
    a_distances = np.linalg.norm(center_a[a_candidates]-carrier_pos[carrier[e_b]], axis=1)
    admitted = a_candidates[a_distances <= distance_limit]
    if len(admitted):
        pairs_by_b[int(i)] = int(admitted[np.argmin(a_distances[a_distances <= distance_limit])])

result = {
    "target_cell":target, "bad_route_index":bad,
    "mass_kg":mass, "maximum_route_distance_m":distance_limit,
    "base_flow_sha256":sha(flow_file), "target_donor_edges_tested":len(all_candidate_a),
    "geometrically_feasible_intermediate_edges":len(pairs_by_b),
    "intermediate_exact_support_tested":0,
}
b_options = np.array(list(pairs_by_b), dtype=int)
for start in range(0, len(b_options), 768):
    candidate = b_options[start:start+768]
    b_support = p1_group_ids(rows_b[candidate])
    result["intermediate_exact_support_tested"] += len(candidate)
    match = [int(i) for i,support in zip(candidate,b_support) if orig_group_id in support]
    if not match:
        continue
    j = match[0]
    i = pairs_by_b[j]
    edge_a, edge_b = int(all_candidate_a[i]), int(outgoing_b[j])
    c_a, c_b = int(carrier[edge_a]), int(carrier[edge_b])
    assert len({original,c_a,c_b}) == 3
    row_a,row_b = int(rows_a[i]),int(rows_b[j])
    result["exchange"] = [
        {"carrier": original, "parent": int(source_parent[original]),
         "source_tuple": list(typed[original]), "from_cell":target, "to_cell":row_b,
         "distance_m":float(origin_distance[j])},
        {"carrier":c_b,"parent": int(source_parent[c_b]),
         "source_tuple":list(typed[c_b]),"from_cell":row_b,"to_cell":row_a,
         "distance_m":float(np.linalg.norm(center_a[i]-carrier_pos[c_b]))},
        {"carrier":c_a,"parent": int(source_parent[c_a]),
         "source_tuple":list(typed[c_a]),"from_cell":row_a,"to_cell":target,
         "distance_m":float(distance_to_target[c_a])},
    ]
    # Remaining routes retain their existing balance. Split only the edited
    # receiver row so the other two cells in the original group keep their flow.
    donors = np.r_[carrier, [original,c_b,c_a]]
    new_group = len(offset)-1
    recipients = np.r_[receiver, [receiver[edge_b],receiver[edge_a],new_group]]
    new_flow = np.r_[flow,np.full(3,mass)]
    new_flow[[bad,edge_b,edge_a]] -= mass
    new_cell_groups = cell_group.copy()
    new_cell_groups[target] = new_group
    expected_carrier = np.bincount(carrier,weights=flow)
    actual_carrier = np.bincount(donors,weights=new_flow)
    expected_group = np.bincount(new_cell_groups,weights=cell_mass)
    actual_group = np.bincount(recipients,weights=new_flow,minlength=len(expected_group))
    assert np.all(new_flow >= 0)
    assert np.allclose(expected_carrier,actual_carrier,rtol=1e-13,atol=0)
    assert np.allclose(expected_group,actual_group,rtol=1e-12,atol=0)
    result["carrier_closure_max_difference_kg"] = float(np.abs(expected_carrier-actual_carrier).max())
    result["group_closure_max_difference_kg"] = float(np.abs(expected_group-actual_group).max())
    result["flow_delta"] = {
        "removed_edge_mass":[[bad,mass],[edge_b,mass],[edge_a,mass]],
        "appended_routes":[[original,int(receiver[edge_b]),mass],[c_b,int(receiver[edge_a]),mass],[c_a,new_group,mass]],
        "changed_cell_group":[target,new_group],
    }
    break
(root/"skin-typed-three-cycle-search.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
