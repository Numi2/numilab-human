"""Bounded disk triangulation using polygon DP and exact geometric conflicts.

Offline asset construction only. All triangle eligibility and pair conflicts use
the retained exact-predicate owner. The DP enumerates combinatorially valid disk
triangulations; branch constraints exclude one endpoint of each measured 3-D
intersection. Acceptance still requires whole-candidate anatomical/volume gates.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import heapq,itertools,time
import numpy as np

def validate_disk(triangles,loop,external_edges):
    loop=list(map(int,loop));allowed=set(loop)
    if len(loop)<3 or len(allowed)!=len(loop):
        raise ValueError("disk boundary must be one simple identity cycle")
    tris=[tuple(map(int,t)) for t in triangles]
    if len(tris)!=len(loop)-2 or any(len(set(t))!=3 or not set(t)<=allowed for t in tris):
        raise ValueError("invalid boundary-only disk triangle count or vertex set")
    if len({tuple(sorted(t))for t in tris})!=len(tris):
        raise ValueError("duplicate disk triangle")
    dirs=Counter((u,v)for t in tris for u,v in zip(t,t[1:]+t[:1]))
    owners=defaultdict(list)
    for i,t in enumerate(tris):
        for u,v in zip(t,t[1:]+t[:1]):owners[tuple(sorted((u,v)))].append(i)
    expected=Counter((u,v)for u,v in zip(loop,loop[1:]+loop[:1]))
    boundary=Counter();adj=[set()for _ in tris]
    for (u,v),rows in owners.items():
        f,r=dirs[(u,v)],dirs[(v,u)]
        if len(rows)==1 and f+r==1:boundary[(u,v)if f else(v,u)]=1
        elif len(rows)==2 and f==r==1:
            if (u,v)in external_edges:raise ValueError("interior edge already retained")
            a,b=rows;adj[a].add(b);adj[b].add(a)
        else:raise ValueError("disk edge multiplicity or orientation invalid")
    if boundary!=expected:raise ValueError("directed disk boundary changed")
    reached={0};todo=[0]
    while todo:
        for j in adj[todo.pop()]:
            if j not in reached:reached.add(j);todo.append(j)
    used=set(v for t in tris for v in t)
    if len(reached)!=len(tris)or used!=allowed or len(used)-len(owners)+len(tris)!=1:
        raise ValueError("disk connectivity/Euler gate failed")
    return {"vertices":len(used),"edges":len(owners),"faces":len(tris),"euler_characteristic":1}

def collision_aware_dp_disk(loop,P,P32,external_edges,base_index,pred,lattice,
        previous_records,first_candidate_id,*,max_seconds=12.,max_branches=512,chunk_size=64,minimum_altitude_m=0.,minimum_angle_degrees=0.):
    started=time.monotonic();loop=list(map(int,loop));n=len(loop)
    if n<3 or len(set(loop))!=n:raise ValueError("invalid simple disk cycle")
    xyz=np.asarray(P32,np.float32).astype(np.float64)
    if not np.all(np.isfinite(xyz[loop])):raise ValueError("nonfinite disk boundary")
    if not np.isfinite(minimum_altitude_m) or minimum_altitude_m < 0. or not np.isfinite(minimum_angle_degrees) or not 0. <= minimum_angle_degrees <= 60.:
        raise ValueError("invalid optional triangle-quality eligibility thresholds")
    quality_rejected = 0
    edges={tuple(sorted((loop[i],loop[(i+1)%n])))for i in range(n)}
    forbidden={tuple(sorted(map(int,e)))for e in external_edges}-edges
    records=[];triples=[];local=[];costs=[];by_id={}
    scale=max(float(np.ptp(xyz[loop],axis=0).max()),1e-30)
    for i,k,j in itertools.combinations(range(n),3):
        tri=(loop[i],loop[k],loop[j])
        if any(tuple(sorted(e))in forbidden for e in [(tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])]):continue
        v=xyz[list(tri)];a=v[1]-v[0];b=v[2]-v[0];cross=np.cross(a,b)
        area2=float(np.linalg.norm(cross))
        if not np.isfinite(area2)or area2==0:continue
        if minimum_altitude_m > 0. or minimum_angle_degrees > 0.:
            aa=float(np.dot(a,a));bb=float(np.dot(b,b));ab=float(np.dot(a,b))
            longest=float(max(np.linalg.norm(a),np.linalg.norm(b),np.linalg.norm(a-b)))
            altitude=area2/longest
            smallest_angle=float(np.min(np.arctan2(area2,np.asarray([ab,aa-ab,bb-ab]))))*180./np.pi
            if altitude < minimum_altitude_m or smallest_angle < minimum_angle_degrees:
                quality_rejected += 1
                continue
        rec=pred._records(lattice,[tri])[0];tid=int(first_candidate_id)+len(records)
        rec=(rec[0],rec[1],rec[2],tid,rec[4]);by_id[tid]=len(records)
        records.append(rec);triples.append(tri);local.append((i,k,j))
        lens=float(np.dot(a,a)+np.dot(b,b)+np.dot(a-b,a-b))
        # Area and bounded shape terms only choose among physically admissible faces.
        costs.append(area2/(scale*scale)+1e-6*min(lens/(area2+1e-300),1e6))
    rejected=set();external_pairs=0;query_candidates=0
    for offset in range(0,len(records),chunk_size):
        if time.monotonic()-started>max_seconds:raise TimeoutError("disk triangle eligibility budget exhausted")
        chunk=records[offset:offset+chunk_size]
        audit=pred._audit_pair_prepared_first(base_index,chunk,same_surface=True)
        external_pairs+=int(audit["count"]);query_candidates+=int(audit.get("aabb_candidate_pairs",0))
        queryids={r[3]for r in chunk}
        for a,b in audit["triangle_pairs"]:
            for key in (a,b):
                if key in queryids:rejected.add(by_id[key])
        if previous_records:
            prioraudit=pred._audit_record_pairs(((a,b)for a in chunk for b in previous_records),same_surface=True)
            external_pairs+=int(prioraudit["count"])
            for a,b in prioraudit["triangle_pairs"]:
                for key in (a,b):
                    if key in queryids:rejected.add(by_id[key])
    candidates=defaultdict(list)
    for tid,(i,k,j)in enumerate(local):
        if tid not in rejected:candidates[(i,j)].append((k,tid,costs[tid]))
    stats={"rejected_quality_triangles":quality_rejected,
           "minimum_altitude_m":float(minimum_altitude_m),"minimum_angle_degrees":float(minimum_angle_degrees),
           "triangle_candidates":len(records),"eligible_triangles":len(records)-len(rejected),
           "rejected_external_triangles":len(rejected),"external_exact_pairs":external_pairs,
           "external_aabb_candidates":query_candidates,"dp_solves":0,"branches_popped":0,"self_pair_conflicts":0}
    def solve(blocked):
        stats["dp_solves"]+=1
        d=[[float("inf")]*n for _ in range(n)];split={}
        for i in range(n-1):d[i][i+1]=0.
        for length in range(2,n):
            for i in range(n-length):
                j=i+length;best=(float("inf"),n,-1)
                for k,tid,cost in candidates.get((i,j),()):
                    if tid in blocked:continue
                    value=cost+d[i][k]+d[k][j]
                    key=(value,k,tid)
                    if key<best:best=key
                if best[2]>=0 and np.isfinite(best[0]):d[i][j]=best[0];split[(i,j)]=(best[1],best[2])
        if not np.isfinite(d[0][n-1]):return None
        ids=[];todo=[(0,n-1)]
        while todo:
            i,j=todo.pop()
            if j-i<2:continue
            k,tid=split[(i,j)];ids.append(tid);todo.extend([(i,k),(k,j)])
        return d[0][n-1],tuple(ids)
    first=solve(frozenset())
    if first is None:raise ValueError("no disk triangulation passes individual exact constraints: "+str(stats))
    queue=[(first[0],0,frozenset(),first[1])];seen={frozenset()};serial=0
    conflicts=[]
    while queue and stats["branches_popped"]<max_branches:
        if time.monotonic()-started>max_seconds:raise TimeoutError("disk conflict branch budget exhausted: "+str(stats))
        score,_,blocked,ids=heapq.heappop(queue);stats["branches_popped"]+=1
        chosen=[records[i]for i in ids]
        audit=pred._audit_record_pairs(itertools.combinations(chosen,2),same_surface=True)
        if not audit["count"]:
            tris=[triples[i]for i in ids];topology=validate_disk(tris,loop,external_edges)
            return tris,chosen,{"method":"exact_constraint_polygon_dp_with_3D_conflict_branching",
                "coordinate_changes":0,"topology":topology,"statistics":stats,"selection_cost":score,
                "conflict_samples":conflicts[:16],"elapsed_s":time.monotonic()-started}
        stats["self_pair_conflicts"]+=int(audit["count"])
        a,b=min(tuple(map(int,p))for p in audit["triangle_pairs"])
        ia,ib=by_id[a],by_id[b]
        if len(conflicts)<16:conflicts.append([list(triples[ia]),list(triples[ib])])
        for bad in (ia,ib):
            child=blocked|{bad}
            if child in seen:continue
            seen.add(child);sol=solve(child)
            if sol:
                serial+=1;heapq.heappush(queue,(sol[0],serial,child,sol[1]))
    raise ValueError("no bounded exact-clean disk triangulation: "+str(stats))
