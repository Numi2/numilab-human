"""Repair a typed route with conservative density slack in two skin cells."""
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


def geometry(rows):
    vertices=xyz[tets[rows]].astype(np.float64)
    edges=vertices[:,1:]-vertices[:,:1]
    volumes=np.einsum("ij,ij->i",edges[:,0],np.cross(edges[:,1],edges[:,2]))/6.
    assert np.all(volumes>0)
    return vertices.mean(axis=1),volumes


group_volume={}
for g in np.unique(receiver[possible]):
    rows=row_sort[offset[g]:offset[g+1]]
    group_volume[int(g)] = float(geometry(rows)[1].sum())
group_mass=np.bincount(cell_group,weights=cell_mass)
report=[]
for g,v in group_volume.items():
    e=possible[receiver[possible]==g]
    slack=group_mass[g]-900*v
    report.append([int(g), len(e), float(slack), float(group_mass[g]/v)])
print(json.dumps(dict(target_center=target_centroid.tolist(),
    original_source=carrier_pos[original].tolist(),distance=float(distance_to_target[original]),
    qualified_supply_groups=report),indent=2))
