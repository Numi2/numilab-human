"""Screen multi-cell recipient groups for an exact individual cell exchange."""
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

from collections import deque

# The serving group's existing quota is valid on its other cells. Isolate
# one deficient receiver and leave their densities and ancestry untouched.
assert np.isclose(flow[bad], cell_mass[cell_group == target_group].sum(), rtol=1e-12)
assert np.count_nonzero(cell_group == target_group) > 1
isolated = int(cell_group.max()) + 1
cell_group[target] = isolated
center = xyz[tets].mean(axis=1, dtype=np.float64)
frozen_flow = flow.copy()
flow = np.r_[flow, np.zeros(np.count_nonzero(
    np.asarray([sid in allowed_groups for sid in typed]) &
    (distance_to_target <= distance_limit)))]
new_carriers = np.flatnonzero(
    np.asarray([sid in allowed_groups for sid in typed]) &
    (distance_to_target <= distance_limit))
before_carrier = carrier
before_receiver = receiver
carrier = np.r_[carrier, new_carriers]
receiver = np.r_[receiver, np.full(len(new_carriers), isolated)]
flow[bad] -= mass
ngroups = isolated + 1
nc = len(source_parent)

def csr_rows(rows, count):
    order = np.argsort(rows, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(rows, minlength=count))]
    return order, offsets

donor_order, donor_offsets = csr_rows(carrier, nc)
receiver_order, receiver_offsets = csr_rows(receiver, ngroups)
cell_order, cell_offsets = csr_rows(cell_group, ngroups)
points_cache = {}
support_cache = {}
approved = {}
excluded = {}

def cells_of(g):
    lo, hi = cell_offsets[g:g+2]
    return cell_order[lo:hi]

def p1_support(g):
    if g not in support_cache:
        support_cache[g] = [
            {tuple(int(x) for x in groups[i]) for i in group_ids}
            for group_ids in p1_group_ids(cells_of(g))
        ]
    return support_cache[g]

def verify_edge(e):
    c, g = int(carrier[e]), int(receiver[e])
    if g == isolated:
        return {"target_projection": extra_support.get(typed[c])}
    if frozen_flow[e] > 0:
        return {"retained_positive_exact_transport_edge": int(e)}
    rows = cells_of(g)
    sid = typed[c]
    d = np.linalg.norm(center[rows] - carrier_pos[c], axis=1)
    if d.max() > distance_limit:
        return None
    if sid not in points_cache:
        mesh = exact_lookup[sid][0]
        points = mesh["points"][np.unique(mesh["tets"])]
        points_cache[sid] = (points.min(axis=0)-projection_cap,
                             points.max(axis=0)+projection_cap)
    # Exact P1 support can be valid away from the cell centroid. Only
    # apply the bounding box test when looking for supplemental support.
    lo, hi = points_cache[sid]
    evidence = []
    for row, direct in zip(rows, p1_support(g)):
        if sid in direct:
            continue
        regions = {s[2] for s in direct}
        if not regions or sid[2] not in regions.union(*(set(adjacency[r]) for r in regions)):
            return None
        point = center[row]
        if np.any(point < lo) or np.any(point > hi):
            return None
        cert = exact_projection(point, sid)
        if cert is None:
            return None
        evidence.append({"child": int(row), **cert})
    return {
        "group":g, "carrier":c, "tuple":list(sid),
        "cells":len(rows), "maximum_distance_m":float(d.max()),
        "supplemental_projections":evidence,
    }

def augmenting_path():
    # Single residual search spans all existing carrier and group capacities.
    # A group can only accept an increase after an equal removal.
    queue = deque([original])
    via_carrier = {original: None}
    via_group = {}
    while queue and isolated not in via_group:
        c = queue.popleft()
        lo, hi = donor_offsets[c:c+2]
        for e in donor_order[lo:hi]:
            g = int(receiver[e])
            if g in via_group or e in excluded:
                continue
            via_group[g] = int(e)
            if g == isolated:
                break
            lo_r, hi_r = receiver_offsets[g:g+2]
            for back in receiver_order[lo_r:hi_r]:
                if flow[back] <= 1e-18:
                    continue
                prior = int(carrier[back])
                if prior not in via_carrier:
                    via_carrier[prior] = int(back)
                    queue.append(prior)
    if isolated not in via_group:
        return None, None, len(via_carrier), len(via_group)
    forward, reverse = [], []
    g = isolated
    while True:
        edge = via_group[g]
        forward.append(edge)
        back = via_carrier[int(carrier[edge])]
        if back is None:
            return forward[::-1], reverse[::-1], len(via_carrier), len(via_group)
        reverse.append(back)
        g = int(receiver[back])

unplaced = float(mass)
moves = []
paths = 0
while unplaced > 1e-17:
    forward, reverse, visited_c, visited_g = augmenting_path()
    if forward is None:
        break
    paths += 1
    # Validate newest nonzero arcs lazily. Recompute paths when a specific
    # route fails instead of rebuilding the large source projection.
    valid = True
    for e in forward[::-1]:
        if e not in approved:
            evidence = verify_edge(e)
            if evidence is None:
                excluded[e] = {
                    "carrier": int(carrier[e]), "group": int(receiver[e]),
                    "tuple": list(typed[carrier[e]]),
                }
                valid = False
                break
            approved[e] = evidence
    if not valid:
        continue
    moved = min(unplaced, *(flow[e] for e in reverse)) if reverse else unplaced
    flow[forward] += moved
    flow[reverse] -= moved
    unplaced -= float(moved)
    moves.append({"forward_arcs": forward, "reverse_arcs":reverse, "mass_kg":float(moved)})
    print(f"routed {moved:.12g} kg with {len(forward)} forward and {len(reverse)} reverse arcs", flush=True)

repair_found = unplaced <= 1e-17
report = {
    "target_cell":target, "target_kg":mass, "target_carrier":original,
    "unplaced_kg":unplaced, "repair_found":repair_found,
    "searches":paths, "last_search_carriers":visited_c, "last_search_groups":visited_g,
    "base_flow_sha256":sha(flow_file), "projection_owner_sha256":sha(projection_path),
    "distance_limit_m":distance_limit, "projection_limit_m":projection_cap,
    "approved_routes":approved, "excluded_routes":excluded, "moves":moves,
}
if repair_found:
    old_quota = np.bincount(before_carrier, weights=frozen_flow, minlength=nc)
    new_quota = np.bincount(carrier, weights=flow, minlength=nc)
    old_mass = np.bincount(cell_group, weights=cell_mass, minlength=ngroups)
    new_mass = np.bincount(receiver, weights=flow, minlength=ngroups)
    assert np.all(flow >= 0.)
    assert np.allclose(old_quota, new_quota, rtol=1e-10, atol=0)
    assert np.allclose(old_mass, new_mass, rtol=1e-10, atol=0)
    assert not np.any((receiver==isolated)&(carrier==original)&(flow>0.))
    with np.load(flow_file, allow_pickle=False) as base:
        output = dict(base)
    output["carrier_row"] = carrier
    output["group"] = receiver
    output["mass_flow_kg"] = flow
    output["group_for_cell"] = cell_group
    out_path = root/"fine-current-paired-semantic-cycle-mass.npz"
    np.savez_compressed(out_path, **output)
    report["candidate_mass_sha256"] = sha(out_path)
    report["maximum_source_quota_delta_kg"] = float(np.max(abs(new_quota - old_quota)))
    report["maximum_receiver_mass_delta_kg"] = float(np.max(abs(new_mass - old_mass)))
(root/"skin-typed-capacity-graph.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k not in {"approved_routes","excluded_routes"}},indent=2))
