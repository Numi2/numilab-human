"""Check retained receipt routes against the exact producer parent from the scene."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
live=root/"floor-live-parent-capture-002/compiled-parent-cell-mass.csv"
mesh=root/"absolute-volume-repaired-tetgen"
points=np.loadtxt(mesh/"mesh.node",skiprows=1,usecols=(1,2,3))
tets=np.loadtxt(mesh/"mesh.ele",skiprows=1,usecols=(1,2,3,4),dtype=np.int32)-1
centers=sum(points[tets[:,i]]*0.25 for i in range(4))
del points,tets
parent_nodes=np.loadtxt(str(live)+".nodes.csv",delimiter=",",skiprows=1,usecols=(5,6,7))
parent_tets=np.loadtxt(str(live)+".tets.csv",delimiter=",",skiprows=1,usecols=(3,4,5,6),dtype=int)
expected_mass=np.loadtxt(live,delimiter=",",skiprows=1,usecols=2)
flow_source=root/"paired-floor-density-closed-flow.npz"
with np.load(flow_source) as packet:
    flow={name:packet[name] for name in packet.files}
reference=json.loads((root/"paired-floor-flow-witness/receipt.json").read_text())
source_parents,source_centroids,source_weights=[],[],[]
for role in reference["roles"]:
    with np.load(role["carrier_path"]) as packet:
        rows=np.repeat(packet["global_source330_tet_rows"],np.diff(packet["parent_offsets"]))
        source_parents.append(rows)
        source_centroids.append(np.einsum("ni,nid->nd",packet["carrier_barycentric"],parent_nodes[parent_tets[rows]]))
        source_weights.append(packet["carrier_weights"])
parents=np.concatenate(source_parents)
assert np.array_equal(parents,flow["source_parent"])
source=np.concatenate(source_centroids)
quota=np.concatenate(source_weights)*expected_mass[parents]
recipient=flow["group_for_cell"]
order=np.argsort(recipient,kind="stable")
rows=np.bincount(recipient)
offsets=np.r_[0,rows.cumsum()]
counts=rows[flow["group"]]
edges=np.repeat(np.arange(len(counts)),counts)
start=np.repeat(np.r_[0,counts.cumsum()[:-1]],counts)
children=order[offsets[flow["group"][edges]]+np.arange(len(edges))-start]
live_distance=np.linalg.norm(centers[children]-source[flow["carrier_row"][edges]],axis=1)
limit=json.loads((root/"archived-compensation-independent-check.json").read_text())["transport_distance_limit_m"]
blocked=np.unique(edges[live_distance>limit])
carriers=flow["carrier_row"]
original=flow["mass_flow_kg"]
old_quota=np.bincount(carriers,weights=original,minlength=len(parents))
old_allocation=np.bincount(parents,weights=old_quota)
assert np.max(np.abs(old_allocation-np.loadtxt(root/"paired-native-parent-ledger/compiled-parent-cell-mass.csv",
                                               delimiter=",",skiprows=1,usecols=2))) < 1.e-12
assert np.all(old_quota>0.0)
report=dict(parent_mass=float(expected_mass.sum()),old_total=float(original.sum()),
            maximum_mass_quota_delta_kg=float(abs(quota-old_quota).max()),
            blocked_positive_route_count=len(blocked),
            blocked_positive_mass_kg=float(original[blocked].sum()),
            maximum_new_distance_m=float(live_distance.max()),distance_limit_m=limit)
print(json.dumps(report),flush=True)
(root/"live-parent-route-supply-audit.json").write_text(json.dumps(report,indent=2)+"\n")
if len(blocked):
    raise RuntimeError("retained positive routes exceed the live separation limit")
new_flow=original*np.divide(quota,old_quota)[carriers]
flow["mass_flow_kg"]=new_flow
group_mass=np.bincount(flow["group"],weights=new_flow,minlength=len(rows))
reference_volume=np.fromfile(mesh/"native-reference-volumes.bin",dtype=np.float64)
group_volume=np.bincount(recipient,weights=reference_volume,minlength=len(rows))
flow["element_mass_kg"]=group_mass[recipient]*reference_volume/group_volume[recipient]
np.savez_compressed(root/"live-parent-warm-quota-flow.npz",**flow)
np.savez(root/"live-parent-route-geometry.npz",source=source,centers=centers,limit=limit)
