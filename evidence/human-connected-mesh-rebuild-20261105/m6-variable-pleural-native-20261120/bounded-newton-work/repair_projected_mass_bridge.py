"""Find a three-receiver exchange through independently qualified exact projections."""
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


physical_root = script.parents[1]
projection_path = physical_root/"current-native-1024-root/current-217-1036/register_current_combined_volume_nodes_217_1036.py"
spec = importlib.util.spec_from_file_location("existing_tuple_projection_owner", projection_path)
projection = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = projection
spec.loader.exec_module(projection)
projection_cap = 0.019353  # existing exact tuple registration owner limit
exact_lookup = projection.make_lookup(parent,owner.native)
adjacency,_ = mod.build_region_adjacency(parent,"Skin")
target_regions = set(json.loads((root/"edited-cell-flow-evidence.json").read_text())["role_reports"]["Skin"]["witnesses"][0]["one_hop_region_ids"])

def exact_projection(point, sid):
    reg = projection.nearest_for_tuple(point,sid,exact_lookup,owner.native,projection_cap,1e-8)
    pr = reg["tet"]
    bary = reg["beta"]
    rest_point = bary@paired_rest[paired_tets[pr]]
    rest_projection_distance = np.linalg.norm(point-rest_point)
    if reg["distance"]>projection_cap or rest_projection_distance>projection_cap:
        return None
    assert np.isfinite(bary).all() and np.min(bary)>=0 and abs(bary.sum()-1)<2.e-12
    return dict(source_parent=pr,source_tuple=list(sid),barycentric=bary.tolist(),
                current_world_m=reg["registered"].tolist(),rest_world_m=rest_point.tolist(),
                current_distance_m=float(reg["distance"]),rest_distance_m=float(rest_projection_distance))

extra_support={}
allowed_groups=set(target_ids)
for sid in sorted(lookup):
    if sid[2] not in target_regions or sid in target_ids:
        continue
    evidence=exact_projection(target_centroid,sid)
    if evidence:
        extra_support[sid]=evidence
        allowed_groups.add(sid)

candidate_edges=np.flatnonzero((flow>=mass)&(receiver!=target_group)&
    (distance_to_target[carrier]<=distance_limit)&
    np.asarray([typed[c] in allowed_groups for c in carrier]))
group_sizes=np.diff(offset)
candidate_edges=candidate_edges[group_sizes[receiver[candidate_edges]]==1]
candidate_rows=row_sort[offset[receiver[candidate_edges]]]
centers=xyz[tets[candidate_rows]].mean(axis=1,dtype=np.float64)
distance_orig=np.linalg.norm(centers-carrier_pos[original],axis=1)
competing=np.flatnonzero(distance_orig<=distance_limit)
competing=competing[np.argsort(distance_orig[competing],kind="stable")]

orig_tuple = typed[original]
positive_skin = np.flatnonzero((flow>=mass)&(receiver!=target_group)&
                              (group_sizes[receiver]==1)&(carrier!=original)&
                              (mat[row_sort[offset[receiver]]]==15))
cells_b = row_sort[offset[receiver[positive_skin]]]
centers_b = xyz[tets[cells_b]].mean(axis=1,dtype=np.float64)
source_mesh = exact_lookup[orig_tuple][0]
source_pts = source_mesh["points"][np.unique(source_mesh["tets"])]
lo = source_pts.min(axis=0)-projection_cap
hi = source_pts.max(axis=0)+projection_cap
# Supplementary projection is impossible outside the expanded source AABB.
within_projection = np.all((centers_b>=lo)&(centers_b<=hi),axis=1)
from_origin_distance = np.linalg.norm(centers_b-carrier_pos[original],axis=1)
b_options = np.flatnonzero(within_projection & (from_origin_distance<=distance_limit))
b_options = b_options[np.argsort(from_origin_distance[b_options],kind="stable")]

def allows(rows):
    for ids in p1_group_ids(rows):
        exact = {tuple(int(x) for x in groups[j]) for j in ids}
        regions = {int(g[2]) for g in exact}
        yield exact, regions.union(*(set(adjacency[r]) for r in regions))

base_a = list(allows(candidate_rows[competing]))
report = {
    "target_cell":target,"mass_kg":mass,
    "base_flow_sha256":sha(flow_file),
    "projection_owner_sha256":sha(projection_path),
    "source_projection_limit_m":projection_cap,
    "distance_limit_m":distance_limit,
    "candidate_first_receivers":len(b_options),
    "candidate_second_receivers":len(competing),
    "first_receivers_admissible":0,
    "second_receiver_projected_slots_checked":0,
}
qual = {}
for start in range(0,len(b_options),768):
    tests = b_options[start:start+768]
    for i,(direct,hops) in zip(tests, allows(cells_b[tests])):
        # No one-hop exception, even for an independently projected tuple.
        if orig_tuple[2] not in hops:
            continue
        incoming = None if orig_tuple in direct else exact_projection(centers_b[i],orig_tuple)
        if orig_tuple not in direct and incoming is None:
            continue
        report["first_receivers_admissible"]+=1
        edge_b = int(positive_skin[i])
        carrier_b = int(carrier[edge_b])
        source_b = typed[carrier_b]
        dist_a = np.linalg.norm(centers[competing]-carrier_pos[carrier_b],axis=1)
        candidate_links = np.flatnonzero(dist_a<=distance_limit)
        order = candidate_links[np.argsort(dist_a[candidate_links],kind="stable")]
        for j in order:
            k = competing[j]
            edge_a = int(candidate_edges[k])
            carrier_a = int(carrier[edge_a])
            if carrier_a in (original,carrier_b) or receiver[edge_a]==receiver[edge_b]:
                continue
            directly_a, hops_a = base_a[j]
            if source_b[2] not in hops_a:
                continue
            if source_b in directly_a:
                middle = None
            else:
                key=(int(candidate_rows[k]),source_b)
                if key not in qual:
                    qual[key]=exact_projection(centers[k],source_b)
                    report["second_receiver_projected_slots_checked"]+=1
                middle=qual[key]
                if middle is None:
                    continue
            new_group=len(offset)-1
            receivers=np.r_[receiver,[receiver[edge_b],receiver[edge_a],new_group]]
            donors=np.r_[carrier,[original,carrier_b,carrier_a]]
            new_flow=np.r_[flow,np.full(3,mass)]
            new_flow[[bad,edge_a,edge_b]]-=mass
            new_cell_groups=cell_group.copy()
            new_cell_groups[target]=new_group
            before=np.bincount(carrier,weights=flow)
            after=np.bincount(donors,weights=new_flow)
            expected=np.bincount(new_cell_groups,weights=cell_mass)
            actual=np.bincount(receivers,weights=new_flow,minlength=len(expected))
            assert np.all(new_flow>=0) and np.allclose(before,after,rtol=1.e-13,atol=0)
            assert np.allclose(expected,actual,rtol=1.e-12,atol=0)
            directions=(
                (original,int(cells_b[i]),from_origin_distance[i],incoming,hops),
                (carrier_b,int(candidate_rows[k]),dist_a[j],middle,hops_a),
                (carrier_a,target,distance_to_target[carrier_a],extra_support.get(typed[carrier_a]),target_regions),
            )
            report.update({
                "repair_found":True,
                "admitted_routes":[dict(carrier=c, source_tuple=list(typed[c]),receiver=r,
                    distance_m=float(d),projection=w,authenticated_one_hop_regions=sorted(regions))
                    for c,r,d,w,regions in directions],
                "source_mass_change_max_kg":float(np.abs(before-after).max()),
                "cell_group_closure_max_kg":float(np.abs(expected-actual).max()),
                "flow_delta":{
                    "removed_edge_mass":[[e,mass] for e in [bad,edge_b,edge_a]],
                    "appended_routes":[[original,int(receiver[edge_b]),mass],
                                       [carrier_b,int(receiver[edge_a]),mass],[carrier_a,new_group,mass]],
                    "changed_cell_group":[target,new_group],
                },
            })
            break
        if report.get("repair_found"):
            break
    if report.get("repair_found"):
        break
(root/"skin-typed-projected-bridge.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
