"""Bounded annulus bridge-and-ear fallback using the existing exact predicates.

The two oppositely oriented boundary cycles are cut along one duplicated
directed bridge. Ears may use vertices from only one original boundary, unlike
a zipper. The bridge is only a search construction; output has the original
two cycles and no duplicated vertices or edges. No coordinates move.
Reference: Eberly, Triangulation by Ear Clipping, section 3:
https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf
"""
from __future__ import annotations
from collections import Counter, defaultdict
import math
import time
import numpy as np

def validate_annulus(tris, loops, external_edges):
    if len(loops)!=2 or any(len(x)<3 or len(x)!=len(set(x)) for x in loops):
        raise ValueError("annulus requires two simple boundary cycles")
    if set(loops[0]) & set(loops[1]):
        raise ValueError("annulus boundary cycles share vertices")
    tris=[tuple(map(int,t)) for t in tris]
    allowed=set(loops[0])|set(loops[1])
    if not tris or any(len(set(t))!=3 or not set(t)<=allowed for t in tris):
        raise ValueError("invalid annulus triangle")
    if len({tuple(sorted(t)) for t in tris})!=len(tris):
        raise ValueError("duplicate annulus triangle")
    expected=Counter((int(u),int(v)) for loop in loops for u,v in zip(loop,loop[1:]+loop[:1]))
    edges=defaultdict(list);dirs=Counter()
    for fi,t in enumerate(tris):
        for u,v in zip(t,t[1:]+t[:1]):
            dirs[(u,v)]+=1;edges[tuple(sorted((u,v)))].append(fi)
    boundary=Counter();adj=[set() for _ in tris]
    for (u,v),owners in edges.items():
        f,r=dirs[(u,v)],dirs[(v,u)]
        if len(owners)==1 and f+r==1:
            boundary[(u,v) if f else (v,u)]=1
        elif len(owners)==2 and f==r==1:
            if (u,v) in external_edges:
                raise ValueError("annulus interior edge is already owned by retained geometry")
            a,b=owners;adj[a].add(b);adj[b].add(a)
        else:
            raise ValueError("annulus edge incidence/orientation invalid")
    if boundary!=expected:
        raise ValueError("annulus changed its original directed boundary")
    reached={0};todo=[0]
    while todo:
        for j in adj[todo.pop()]:
            if j not in reached:reached.add(j);todo.append(j)
    if len(reached)!=len(tris):
        raise ValueError("annulus is disconnected")
    used=set(v for t in tris for v in t)
    if used!=allowed or len(used)-len(edges)+len(tris)!=0:
        raise ValueError("annulus Euler characteristic is not zero")
    return {"vertices":len(used),"edges":len(edges),"faces":len(tris),
            "euler_characteristic":0,"boundary_components":2,
            "same_boundary_only_triangles":sum(len(set(t)&set(loops[0])) in (0,3) for t in tris)}

def collision_aware_bridged_annulus(
        loops,P,P32,external_edges,base_index,pred,lattice,previous_records,
        first_candidate_id,*,ear_clip,max_nodes=20000,max_seconds=4.0,max_bridges=16):
    loops=[list(map(int,x)) for x in loops]
    if len(loops)!=2 or any(len(x)<3 or len(set(x))!=len(x) for x in loops) or set(loops[0])&set(loops[1]):
        raise ValueError("invalid annulus boundary cycles")
    a,b=loops;start=time.monotonic();trials=[]
    pairs=sorted((float(np.sum((P[u]-P[v])**2)),i,j)
                 for i,u in enumerate(a) for j,v in enumerate(b)
                 if tuple(sorted((u,v))) not in external_edges)
    # Different bridges get disjoint exact-predicate identity ranges. The
    # existing helper caches one record per unordered triangle of real IDs.
    stride=math.comb(len(a)+len(b),3)+1
    for bi,(_,ai,bj) in enumerate(pairs[:max_bridges]):
        remaining=max_seconds-(time.monotonic()-start)
        if remaining<=0:break
        aa=a[ai:]+a[:ai];bb=b[bj:]+b[:bj]
        cut=aa+[aa[0]]+bb+[bb[0]]
        try:
            tris,recs,details=ear_clip(
                cut,P,P32,external_edges,base_index,pred,lattice,
                previous_records,int(first_candidate_id)+bi*stride,
                max_nodes=max_nodes,max_seconds=min(0.75,remaining))
            fresh=recs[len(previous_records):]
            if len(fresh)!=len(tris):raise ValueError("ear helper record-prefix mismatch")
            topology=validate_annulus(tris,loops,external_edges)
            return tris,fresh,{"method":"exact_collision_aware_annulus_bridge_and_ear_search",
                "coordinate_changes":0,"bridge_vertices":[aa[0],bb[0]],"bridge_trial":bi,
                "prior_failed_trials":trials,"topology":topology,"ear_search":details,
                "elapsed_s":time.monotonic()-start}
        except (ValueError,TimeoutError) as exc:
            trials.append({"bridge_vertices":[aa[0],bb[0]],"error":str(exc)[:1000]})
    raise ValueError("no bounded exact-clean bridged annulus: "+repr(trials))
