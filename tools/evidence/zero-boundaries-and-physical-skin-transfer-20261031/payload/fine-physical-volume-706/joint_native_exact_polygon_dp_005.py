"""Interval-DP enumeration for exact-clean fixed-coordinate triangulations of a disk loop.
Candidate faces pass the unchanged joint score/altitude gate, external exact checks,
and forbidden-edge checks. Ranked full plans are then checked pairwise with the
existing exact owner before acceptance. The DP is geometric construction only.
"""
from functools import lru_cache
import itertools, json, math, time
import numpy as np

def triangulate(loop, points32, external_edges, base_index, pred, filtered,
                lattice, previous_records, first_candidate_id, disk_mod,
                score_faces, *, max_seconds=30.0, max_plans=100000,
                minimum_altitude_m=20e-9, quality_floor=1e-5):
    started=time.monotonic(); loop=tuple(map(int,loop)); n=len(loop)
    if n<3 or len(set(loop))!=n:
        raise ValueError("invalid simple disk cycle")
    xyz=np.asarray(points32,np.float32).astype(np.float64)
    boundary={tuple(sorted((loop[i],loop[(i+1)%n]))) for i in range(n)}
    forbidden={tuple(sorted(map(int,e))) for e in external_edges}-boundary
    stats={"boundary_vertices":n,"all_triples":0,"forbidden_edge_rejects":0,
           "degenerate_or_altitude_rejects":0,"quality_floor_rejects":0,
           "geometric_candidates":0,"external_exact_calls":0,"external_exact_pairs":0,
           "interval_states":0,"complete_plans":0,"plan_cap_hit":False,
           "self_exact_calls":0,"self_exact_pairs":0,"self_pair_cache_hits":0,
           "previous_exact_calls":0,"previous_exact_pairs":0,
           "accepted_plans":0,"elapsed_s":0.0}
    triangles=[]; positional={};
    for i in range(n-2):
        for k in range(i+1,n-1):
            for j in range(k+1,n):
                stats["all_triples"]+=1
                tri=(loop[i],loop[k],loop[j])
                edges=(tuple(sorted((tri[0],tri[1]))),tuple(sorted((tri[1],tri[2]))),tuple(sorted((tri[2],tri[0]))))
                if any(e in forbidden for e in edges):
                    stats["forbidden_edge_rejects"]+=1;continue
                v=xyz[np.asarray(tri,np.int64)];a=v[1]-v[0];b=v[2]-v[0];c=v[2]-v[1]
                area=float(np.linalg.norm(np.cross(a,b)))
                longest=max(float(np.linalg.norm(a)),float(np.linalg.norm(b)),float(np.linalg.norm(c)))
                if not math.isfinite(area) or area<=0 or not math.isfinite(longest) or longest<=0 or area/longest<minimum_altitude_m:
                    stats["degenerate_or_altitude_rejects"]+=1;continue
                triangles.append((i,k,j,tri))
    if triangles:
        scored=np.asarray(score_faces(np.asarray([x[3] for x in triangles],np.int64)),np.float64)
        assert scored.shape==(len(triangles),) and np.all(np.isfinite(scored))
    else:scored=np.empty(0,np.float64)
    records={};qvalue={};ext_cache={};next_id=int(first_candidate_id)
    for item,q in zip(triangles,scored):
        i,k,j,tri=item
        if q<quality_floor:
            stats["quality_floor_rejects"]+=1;continue
        raw=pred._records(lattice,[tri])[0]
        rec=(raw[0],raw[1],raw[2],next_id,raw[4]);next_id+=1
        stats["external_exact_calls"]+=1
        gate=filtered._audit_pair_prepared_first(base_index,(rec,),same_surface=True)
        stats["external_exact_pairs"]+=int(gate.get("count",0))
        if int(gate.get("count",0)):
            ext_cache[(i,k,j)]=False;continue
        ext_cache[(i,k,j)]=True;records[(i,k,j)]=rec;qvalue[(i,k,j)]=float(q)
    stats["geometric_candidates"]=len(records)
    if not records:
        stats["elapsed_s"]=time.monotonic()-started
        raise ValueError("no external-clean triangle passes fixed gates: "+json.dumps(stats,separators=(",",":")))
    cap=[False]
    @lru_cache(None)
    def subplans(i,j):
        stats["interval_states"]+=1
        if j<=i+1:return ((),)
        out=[]
        for k in range(i+1,j):
            key=(i,k,j)
            if key not in records:continue
            left=subplans(i,k);right=subplans(k,j)
            if not left or not right:continue
            for lp in left:
                for rp in right:
                    if len(out)>=max_plans:
                        cap[0]=True;break
                    out.append(lp+rp+(key,))
                if cap[0]:break
            if cap[0]:break
        return tuple(out)
    plans=subplans(0,n-1);stats["plan_cap_hit"]=cap[0];stats["complete_plans"]=len(plans)
    if not plans:
        stats["elapsed_s"]=time.monotonic()-started
        raise ValueError("no interval-DP triangulation passes per-face gates: "+json.dumps(stats,separators=(",",":")))
    ranked=sorted(plans,key=lambda p:(-min(qvalue[x] for x in p),-sum(qvalue[x] for x in p),p))
    pair_cache={}
    def pair_clean(k1,k2):
        key=tuple(sorted((k1,k2)))
        if key in pair_cache:
            stats["self_pair_cache_hits"]+=1;return pair_cache[key]
        stats["self_exact_calls"]+=1
        gate=pred._audit_record_pairs(((records[k1],records[k2]),),same_surface=True)
        stats["self_exact_pairs"]+=int(gate.get("count",0));good=int(gate.get("count",0))==0
        pair_cache[key]=good;return good
    previous_keys=[]
    for rec in previous_records:
        previous_keys.append((rec,tuple(sorted(map(int,rec[4])))))
    for plan in ranked:
        if time.monotonic()-started>max_seconds:break
        clean=True
        for key in plan:
            rec=records[key]
            for prev,prevtri in previous_keys:
                stats["previous_exact_calls"]+=1
                gate=pred._audit_record_pairs(((rec,prev),),same_surface=True)
                stats["previous_exact_pairs"]+=int(gate.get("count",0))
                if int(gate.get("count",0)):clean=False;break
            if not clean:break
        if not clean:continue
        for ii in range(len(plan)):
            for jj in range(ii+1,len(plan)):
                if not pair_clean(plan[ii],plan[jj]):clean=False;break
            if not clean:break
        if not clean:continue
        newfaces=np.asarray([(loop[x[0]],loop[x[1]],loop[x[2]]) for x in plan],np.int64)
        topology=disk_mod.validate_disk(newfaces,loop,external_edges)
        tri_xyz=xyz[newfaces];u=tri_xyz[:,1]-tri_xyz[:,0];v=tri_xyz[:,2]-tri_xyz[:,0];w=tri_xyz[:,2]-tri_xyz[:,1]
        area2=np.linalg.norm(np.cross(u,v),axis=1);longest=np.maximum.reduce((np.linalg.norm(u,axis=1),np.linalg.norm(v,axis=1),np.linalg.norm(w,axis=1)))
        altitude=area2/longest
        stats["accepted_plans"]+=1
        stats["elapsed_s"]=time.monotonic()-started
        detail={"method":"ranked_interval_dp_all_admissible_cyclic_triangles_exact_context_and_pair_replay",
                "minimum_new_joint_qbound":float(min(qvalue[x] for x in plan)),
                "minimum_new_altitude_m":float(np.min(altitude)),
                "topology":topology,"statistics":dict(stats),"elapsed_s":stats["elapsed_s"]}
        return newfaces,[records[x] for x in plan],detail
    stats["elapsed_s"]=time.monotonic()-started
    kind="bounded exact-pair replay exhausted" if stats["elapsed_s"]>max_seconds else "no exact-clean interval-DP plan"
    raise ValueError(kind+": "+json.dumps(stats,separators=(",",":")))
