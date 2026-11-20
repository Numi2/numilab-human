"""Expand the feasible graph using already admitted donor/material/region tuples."""
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree

def add_live_routes(root, output, lower, upper, neighbors):
    with np.load(root/"live-parent-route-geometry.npz") as z:
        origin, centers, limit = z["source"], z["centers"], float(z["limit"])
    receiver=output["group_for_cell"]
    order=np.argsort(receiver,kind="stable")
    counts=np.bincount(receiver)
    offsets=np.r_[0,counts.cumsum()]
    center=np.add.reduceat(centers[order],offsets[:-1],axis=0)/counts[:,None]
    masses=output["mass_flow_kg"]
    dest=output["group"]
    source=output["carrier_row"]
    initial=np.bincount(dest,weights=masses,minlength=len(center))
    shortage=initial<lower
    excess=initial>upper
    supplied_by_excess=np.unique(source[excess[dest]])
    labels=np.loadtxt(root/"floor-live-parent-capture-002/compiled-parent-cell-mass.csv.labels.csv",
                      delimiter=",",skiprows=1,dtype=np.int64)[:,1:4]
    _,typed=np.unique(labels,axis=0,return_inverse=True)
    kind=typed.ravel()[output["source_parent"]]
    plan=defaultdict(set)
    ordinary=(masses>0)&(~output["archived_compensation"])
    for k,g in zip(kind[source[ordinary]],dest[ordinary]):
        plan[k].add(int(g))
    pairs=set()
    def admit(carrier, group):
        route=(int(carrier),int(group))
        if route not in pairs:
            children=order[offsets[group]:offsets[group+1]]
            if np.max(np.linalg.norm(centers[children]-origin[carrier],axis=1))<=limit:
                pairs.add(route)
    for k,groups in plan.items():
        groups=np.asarray(sorted(groups),dtype=np.int32)
        carriers=np.flatnonzero(kind==k)
        hungry=groups[shortage[groups]]
        if len(hungry):
            _,near=cKDTree(origin[carriers]).query(center[hungry],k=np.arange(1,min(neighbors,len(carriers))+1))
            for g,neighbors_of_group in zip(hungry,carriers[near]):
                for c in neighbors_of_group:
                    admit(c,g)
        needing=supplied_by_excess[kind[supplied_by_excess]==k]
        target=groups[~excess[groups]]
        if len(needing) and len(target):
            _,near=cKDTree(center[target]).query(origin[needing],k=np.arange(1,min(neighbors,len(target))+1))
            for c,neighbors_of_source in zip(needing,target[near]):
                for g in neighbors_of_source:
                    admit(c,g)
    pairs.difference_update(zip(source,dest))
    add=np.asarray(sorted(pairs),dtype=np.int32).reshape(-1,2)
    print("live-proven tuple routes outside prior active flow",len(add),flush=True)
    n=len(add)
    for key, values in (
        ("carrier_row",add[:,0]),("group",add[:,1]),("mass_flow_kg",np.zeros(n)),
        ("prior_edge_row",np.full(n,-1,dtype=np.int32)),("archived_compensation",np.zeros(n,dtype=bool)),
    ):
        output[key]=np.concatenate((output[key],values))
