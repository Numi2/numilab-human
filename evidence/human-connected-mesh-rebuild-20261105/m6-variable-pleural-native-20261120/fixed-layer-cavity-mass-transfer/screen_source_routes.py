"""Measure local topology/mass proposals against the unchanged source-route policy."""
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

launch_path,proposal_path,transfer_root,owner_path = map(Path,sys.argv[1:])
env = json.loads(launch_path.read_text())["environment"]
result = json.loads((transfer_root/"density-transfer.json").read_text())
proposal = json.loads(proposal_path.read_text())
def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

assert digest(proposal_path) == result["proposal_sha256"]
shares_path = transfer_root/"cavity-mass-shares.csv"
assert digest(shares_path) == result["shares_sha256"]
ops = {i:proposal["replacements"][i] for i in result["eligible_operation_ids"]}
mapping = defaultdict(list)
op_for_old = {}
for i,e in ops.items():
    for old in e["retired"]:
        assert old not in op_for_old
        op_for_old[old] = i
for i,old,new,share in np.loadtxt(shares_path,delimiter=",",skiprows=1):
    i,old,new = map(int,(i,old,new))
    assert i == op_for_old[old] and share > 0
    mapping[old].append((new,float(share)))

prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_"
node,cell = (Path(env[prefix+k]) for k in ("NODE","ELEMENTS"))
for k,p in zip(("NODE","ELEMENTS"),(node,cell)):
    assert digest(p) == env[prefix+k+"_SHA256"]
points = np.loadtxt(node,skiprows=1,usecols=(1,2,3))
table = np.loadtxt(cell,skiprows=1,dtype=np.int32)
nodes,roles = table[:,1:5]-1,table[:,5]
del table
def pinned(key,hash_key=None):
    path = Path(env[key])
    assert digest(path) == env[hash_key or key+"_SHA256"],path
    return path

keys = "NUMI_H706_FRACTIONAL_"
parent_nodes = np.loadtxt(pinned(keys+"SOURCE330_NODES"),delimiter=",",skiprows=1,usecols=(5,6,7))
parent_cells = np.loadtxt(pinned(keys+"SOURCE330_TETS"),delimiter=",",skiprows=1,
                          usecols=(1,2,3,4,5,6),dtype=np.int32)
records = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_RECORDED_ALLOCATION_"
parent_faces = np.empty(len(parent_cells),np.uint32)
parent_regions = np.full(len(parent_cells),-1,np.int32)
parent_local = {}
for name,material in (("SKIN",15),("FAT",16)):
    side = np.loadtxt(pinned(records+name+"_SOURCE_PARENT_CSV"),delimiter=",",skiprows=1,
                      usecols=(0,1,2,3,4,5),dtype=np.int32)
    order = side[:,1]
    assert np.array_equal(side[:,0],np.arange(len(side)))
    assert np.all(side[:,2] == material)
    assert np.array_equal(parent_cells[order,:2],side[:,2:4])
    assert np.all(parent_regions[order] == -1)
    parent_faces[order],parent_regions[order] = side[:,5],side[:,4]
    parent_local[name] = order
assert np.all(parent_regions >= 0)
parent = dict(tets=parent_cells[:,2:],materials=parent_cells[:,0],faces=parent_faces,regions=parent_regions)
spec = importlib.util.spec_from_file_location("eligible_native_p1",owner_path)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
limit_key = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MAXIMUM_MASS_TRANSPORT_DISTANCE_M"
limit = float(env[limit_key])
reasons = defaultdict(set)
stats = {}
for name,material in (("SKIN",15),("FAT",16)):
    role = name.title()
    role_ids = [i for i,o in ops.items() if o["material_id"] == material]
    old_role_rows = np.flatnonzero(roles == material)
    old_role_nodes = np.unique(nodes[old_role_rows])
    p1 = pinned(keys+"P1_"+name+"_NPZ",keys+"P1_"+name+"_SHA256")
    with np.load(p1) as data:
        info = dict(data)
    assert np.array_equal(info["points_world_m_f32"],points[old_role_nodes])
    tets = np.concatenate([ops[i]["new"] for i in role_ids]).astype(np.int32)
    offsets = np.r_[0,np.cumsum([len(ops[i]["new"]) for i in role_ids])]
    corners,local = np.unique(tets,return_inverse=True)
    local = local.reshape(-1,4)
    local_nodes = np.searchsorted(old_role_nodes,corners)
    assert np.array_equal(old_role_nodes[local_nodes],corners)
    node_group = info["node_group_ids"]
    node_offsets = info["node_group_offsets"]
    parent_offsets = info["node_group_parent_tet_offsets"]
    parent_rows = info["node_group_parent_tet_rows"]
    corner_map = [
        {int(node_group[s]):set(parent_rows[parent_offsets[s]:parent_offsets[s+1]])
         for s in range(node_offsets[node],node_offsets[node+1])}
        for node in local_nodes
    ]
    groups = info["group_tuples"]
    p1_cells = api.integrate_p1_cells(local,corner_map,groups,parent,role)
    adjacency,_ = api.build_region_adjacency(parent,role)
    p1_regions = api.expand_onehop(p1_cells,groups,adjacency,role,len(local))
    starts,region_ids = p1_regions["cell_admissible_region_offsets"],p1_regions["cell_admissible_region_ids"]
    allowed = [set(region_ids[a:b]) for a,b in zip(starts[:-1],starts[1:])]
    centers = (points[tets]*.25).sum(axis=1)
    child_start = {i:int(o) for i,o in zip(role_ids,offsets)}
    carriers = pinned(keys+"CARRIERS_"+name+"_NPZ",keys+"CARRIERS_"+name+"_SHA256")
    with np.load(carriers) as data:
        carrier = dict(data)
    parent_for_carrier = np.repeat(carrier["global_source330_tet_rows"],np.diff(carrier["parent_offsets"]))
    carrier_regions = parent_regions[parent_for_carrier]
    carrier_position = np.einsum("ni,nij->nj",carrier["carrier_barycentric"],
                                 parent_nodes[parent["tets"][parent_for_carrier]])
    compensation_rows = set(carrier.get("regional_compensation_child_rows",()))
    for old_local in compensation_rows:
        old = old_role_rows[old_local]
        if old in op_for_old:
            reasons[op_for_old[old]].add("regional_compensation_receiver")
    primary_target = carrier["carrier_target_child_rows"]
    targeted_primaries = 0
    for c, old_local in enumerate(primary_target):
        old = old_role_rows[old_local]
        if old not in op_for_old:
            continue
        targeted_primaries += 1
        op = op_for_old[old]
        dest = np.arange(child_start[op],child_start[op]+len(ops[op]["new"]))
        if not any(carrier_regions[c] in allowed[j] and
                   np.linalg.norm(carrier_position[c]-centers[j]) <= limit for j in dest):
            reasons[op].add("no_primary_route_in_cavity")

    flow_path = pinned(records+name+"_SOURCE_CARRIER_FLOW_CSV")
    affected = supported = mass_eligible = 0
    transported = reroute_needed_mass = 0.
    maximum_distance = 0.
    with flow_path.open() as stream:
        reader = csv.reader(stream)
        assert next(reader) == ["child_row","parent_row","expanded_parent_row","carrier_row","mass_kg","source_measure_m3","distance_m"]
        for row in reader:
            old = old_role_rows[int(row[0])]
            if old not in mapping:
                continue
            affected += 1
            c = int(row[3])
            assert parent_for_carrier[c] == parent_local[name][int(row[1])]
            redirect_mass = float(row[4])
            transported += redirect_mass
            op = op_for_old[old]
            first = child_start[op]
            can_transport = []
            for local_dest in range(len(ops[op]["new"])):
                j = first + local_dest
                if carrier_regions[c] not in allowed[j]:
                    continue
                distance = float(np.linalg.norm(carrier_position[c]-centers[j]))
                if distance <= limit:
                    can_transport.append(local_dest)
                    maximum_distance = max(maximum_distance,distance)
            if not can_transport:
                reasons[op].add("no_positive_route_in_cavity")
                continue
            supported += 1
            if any(j not in can_transport for j,_ in mapping[old]):
                reasons[op].add("geometric_split_needs_constrained_transport")
                reroute_needed_mass += redirect_mass
            else:
                mass_eligible += 1
    stats[name] = dict(checked_operations=len(role_ids),
                       affected_primary_receivers=targeted_primaries,
                       affected_carrier_flows=affected,
                       flows_with_at_least_one_allowed_neighbor=supported,
                       flows_with_all_overlap_targets_allowed=mass_eligible,
                       cavity_input_mass_kg=transported,
                       source_mass_on_flows_needing_rerouting_kg=reroute_needed_mass,
                       maximum_eligible_transport_distance_m=maximum_distance,
                       p1_cells_resolved=int(np.count_nonzero(p1_cells["cell_semantic_status"]==1)),
                       p1_cells_unresolved=int(np.count_nonzero(p1_cells["cell_semantic_status"]!=1)))
eligible = [i for i in ops if not reasons.get(i)]
retry = [i for i,r in reasons.items() if r == {"geometric_split_needs_constrained_transport"}]
print(json.dumps(dict(
    scope="Exact node-derived P1 region and centroid distance screen; not a native allocation.",
    proposal_sha256=result["proposal_sha256"],
    overlap_remap_sha256=result["shares_sha256"],
    p1_owner_sha256=digest(owner_path),
    transport_distance_limit_m=limit,
    direct_overlap_eligible_operations=len(eligible),
    direct_overlap_eligible_ids=eligible,
    constrained_transport_operations=len(retry),
    constrained_transport_ids=sorted(retry),
    rejected_ops_by_reason=dict(Counter(k for values in reasons.values() for k in values)),
    roles=stats),indent=2))
