"""Collision-aware genus-zero patch construction with several fixed boundaries.

Bridges form a cut tree. Virtual copies describe its disk only; exact predicates
use original physical vertex identities, so no seam or contact exemptions arise.
The output retains every original directed boundary and has Euler 2-boundaries.
Reference: Eberly, Triangulation by Ear Clipping, sections3/5.
"""
from collections import Counter,defaultdict
import itertools,time,numpy as np

def validate_surface(tris,loops,external_edges):
    allowed=set(v for loop in loops for v in loop)
    if any(len(x)<3 or len(set(x))!=len(x)for x in loops)or sum(map(len,loops))!=len(allowed):
        raise ValueError("boundary cycles must be simple and disjoint")
    tris=[tuple(map(int,t))for t in tris]
    if not tris or any(len(set(t))!=3 or not set(t)<=allowed for t in tris):raise ValueError("invalid patch face")
    if len({tuple(sorted(t))for t in tris})!=len(tris):raise ValueError("duplicate physical face")
    expected=Counter((u,v)for loop in loops for u,v in zip(loop,loop[1:]+loop[:1]))
    dirs=Counter((u,v)for t in tris for u,v in zip(t,t[1:]+t[:1]))
    owners=defaultdict(list)
    for i,t in enumerate(tris):
        for u,v in zip(t,t[1:]+t[:1]):owners[tuple(sorted((u,v)))].append(i)
    boundary=Counter();adj=[set()for _ in tris]
    for (u,v),rows in owners.items():
        f,r=dirs[(u,v)],dirs[(v,u)]
        if len(rows)==1 and f+r==1:boundary[(u,v)if f else(v,u)]=1
        elif len(rows)==2 and f==r==1:
            if (u,v)in external_edges:raise ValueError("retained diagonal reused")
            a,b=rows;adj[a].add(b);adj[b].add(a)
        else:raise ValueError("physical edge incidence/orientation failed")
    if boundary!=expected:raise ValueError("physical directed boundaries changed")
    reached={0};todo=[0]
    while todo:
        for j in adj[todo.pop()]:
            if j not in reached:reached.add(j);todo.append(j)
    used=set(v for t in tris for v in t);chi=len(used)-len(owners)+len(tris)
    if used!=allowed or len(reached)!=len(tris)or chi!=2-len(loops):
        raise ValueError("physical connectivity/Euler failed")
    return {"vertices":len(used),"edges":len(owners),"faces":len(tris),"euler_characteristic":chi,"boundary_components":len(loops),"genus":0}

class PhysicalIdentityPredicate:
    def __init__(self,owner,physical):
        self.owner=owner;self.physical=list(map(int,physical));self.added_topological_conflicts=0
    def __getattr__(self,key):return getattr(self.owner,key)
    def _records(self,lattice,faces):
        recs=self.owner._records(lattice,faces)
        return [(r[0],r[1],r[2],r[3],tuple(self.physical[v]for v in r[4]))for r in recs]
    def _audit_record_pairs(self,pairs,**kw):
        pairs=list(pairs);res=self.owner._audit_record_pairs(pairs,**kw)
        present={tuple(sorted(map(int,p)))for p in res["triangle_pairs"]}
        extra=set()
        for a,b in pairs:
            va=tuple(a[4]);vb=tuple(b[4])
            da={(u,v)for u,v in zip(va,va[1:]+va[:1])}
            db={(u,v)for u,v in zip(vb,vb[1:]+vb[:1])}
            if set(va)==set(vb)or da&db:
                pair=tuple(sorted((int(a[3]),int(b[3]))))
                if pair not in present:extra.add(pair)
        if extra:
            res=dict(res);res["count"]=int(res["count"])+len(extra);res["triangle_pairs"]=sorted(present|extra)
            self.added_topological_conflicts+=len(extra)
        return res

def collision_aware_multi_boundary(loops,P,P32,external_edges,base_index,pred,
        first_candidate_id,*,dp_disk,max_seconds=24.,max_cuts=8,max_branches=1024,chord_filter=None):
    loops=[list(map(int,l))for l in loops]
    if len(loops)<2:raise ValueError("at least two physical cycles required")
    if sum(map(len,loops))!=len(set(v for l in loops for v in l)):raise ValueError("boundary cycles intersect")
    start=time.monotonic();xyz=np.asarray(P32,np.float32).astype(float)
    first=max(range(len(loops)),key=lambda i:(len(loops[i]),-i))
    states=[(0.,tuple(loops[first]),tuple(i for i in range(len(loops))if i!=first))]
    for depth in range(len(loops)-1):
        nexts={}
        for score,cut,left in states:
            for hole in left:
                h=loops[hole];pairs=sorted((float(np.sum((xyz[u]-xyz[v])**2)),i,j)
                    for i,u in enumerate(cut)for j,v in enumerate(h)if tuple(sorted((u,v)))not in external_edges)
                for cost,i,j in pairs[:4]:
                    hh=h[j:]+h[:j]
                    c=tuple(list(cut[:i+1])+hh+[hh[0],cut[i]]+list(cut[i+1:]))
                    rem=tuple(k for k in left if k!=hole);key=(c,rem)
                    value=score+cost
                    if key not in nexts or value<nexts[key]:nexts[key]=value
        states=sorted((value,c,rem)for(c,rem),value in nexts.items())[:max_cuts]
        if not states:raise ValueError("no unoccupied boundary bridge tree")
    trials=[]
    original_boundary={tuple(sorted((u,v)))for l in loops for u,v in zip(l,l[1:]+l[:1])}
    for ci,(_,cut,_)in enumerate(states):
        remaining=max_seconds-(time.monotonic()-start)
        if remaining<=0:break
        physical=list(cut);vp=np.asarray(P32,np.float32)[physical]
        proxy=PhysicalIdentityPredicate(pred,physical)
        virtual_external={(i,j)for i in range(len(cut))for j in range(i+1,len(cut))
          if tuple(sorted((cut[i],cut[j])))in external_edges-original_boundary}
        keys=tuple(pred.float32_point_lattice_key(v)for v in vp)
        try:
            chord_info=None
            if chord_filter is not None:
                blocked,chord_info=chord_filter(list(range(len(cut))),keys,base_index,proxy)
                if chord_info["blocked_boundary_edges"]:
                    raise ValueError("cut-tree boundary/bridge has an exact retained-surface contact: "+repr(chord_info["witnesses"][:2]))
                virtual_external=virtual_external|blocked
            remaining=max_seconds-(time.monotonic()-start)
            if remaining<=0:raise TimeoutError("multi-boundary chord budget exhausted")
            tris,recs,info=dp_disk(list(range(len(cut))),vp.astype(float),vp,virtual_external,base_index,proxy,keys,[],
                int(first_candidate_id)+ci*1000000,max_seconds=min(remaining,max_seconds/max(1,min(len(states),4))),max_branches=max_branches)
            actual=[tuple(physical[v]for v in t)for t in tris]
            topology=validate_surface(actual,loops,external_edges)
            exact=pred._audit_record_pairs(itertools.combinations(recs,2),same_surface=True)
            if exact["count"]:raise ValueError("physical output has exact geometric conflict")
            return actual,recs,{"method":"virtual_cut_tree_exact_constraint_DP","coordinate_changes":0,"topology":topology,
                "cut_index":ci,"virtual_slot_count":len(cut),"physical_vertex_count":sum(map(len,loops)),
                "prior_failed_cuts":trials,"added_topological_conflicts":proxy.added_topological_conflicts,
                "chord_pruning":chord_info,"dp":info,"elapsed_s":time.monotonic()-start}
        except (ValueError,TimeoutError)as exc:trials.append({"cut_index":ci,"error":str(exc)[:2000]})
    raise ValueError("no bounded exact-clean multi-boundary patch: "+repr(trials))
