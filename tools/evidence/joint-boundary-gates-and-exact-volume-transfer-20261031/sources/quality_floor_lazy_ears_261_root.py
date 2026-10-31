"""Existing 024 bounded exact ear search; construction quality floor is now explicit. Native thresholds are untouched."""
import math,time
import numpy as np
SOURCE_PATH='/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/fixed-boundary-quality-topology-196/run_fixed_boundary_earclip_current_024.py'
SOURCE_SHA256='90d346febf6495d12bf7650984bafbab229bd96f3eb8af9aa792ad1c2731ba2f'
def collision_aware_lazy_ears(loop,P32,external_edges,base_index,pred,filtered,lattice,previous_records,first_candidate_id,disk_mod,*,max_nodes=30000,max_seconds=8.0,minimum_altitude_m=20e-9,max_solutions=32,quality_floor=1e-5):
    """Bounded lazy ear clipping; retain the best qbound among exact-clean plans."""
    started=time.monotonic(); loop=list(map(int,loop)); n=len(loop)
    if n<3 or len(set(loop))!=n: raise ValueError('invalid simple disk cycle')
    xyz=np.asarray(P32,np.float32).astype(np.float64)
    uv=np.stack((np.cos(2*np.pi*np.arange(n)/n),np.sin(2*np.pi*np.arange(n)/n)),axis=1)
    eps=np.finfo(np.float64).eps*128
    boundary={tuple(sorted((loop[i],loop[(i+1)%n]))) for i in range(n)}
    forbidden={tuple(sorted(map(int,e))) for e in external_edges}-boundary
    stats={'search_nodes':0,'candidate_ears':0,'quality_rejects':0,'external_exact_calls':0,'external_exact_pairs':0,'internal_exact_calls':0,'internal_exact_pairs':0,'external_cache_hits':0,'internal_cache_hits':0,'backtracks':0,'max_depth':0,'complete_exact_clean_solutions':0,'best_bad_faces':None,'budget_truncated':False}
    nextid=[int(first_candidate_id)]; rec_cache={}; ext_cache={}; pair_cache={}; best=[None]; stop=[False]
    def tri_key(tri): return tuple(sorted(map(int,tri)))
    def rec_for(tri):
        key=tri_key(tri)
        if key not in rec_cache:
            raw=pred._records(lattice,[tuple(map(int,tri))])[0]
            rec_cache[key]=(raw[0],raw[1],raw[2],nextid[0],raw[4]); nextid[0]+=1
        return rec_cache[key]
    def quality(tri):
        v=xyz[np.asarray(tri,np.int64)]; a=v[1]-v[0]; b=v[2]-v[0]
        area2=float(np.linalg.norm(np.cross(a,b)))
        longest=max(float(np.linalg.norm(a)),float(np.linalg.norm(b)),float(np.linalg.norm(v[2]-v[1])))
        if not math.isfinite(area2) or area2<=0 or not math.isfinite(longest) or longest<=0: return None
        altitude=area2/longest; q=area2/(longest*longest)
        if altitude<minimum_altitude_m: stats['quality_rejects']+=1; return None
        return q,altitude
    def cross2(x,y,z): return (y[0]-x[0])*(z[1]-x[1])-(y[1]-x[1])*(z[0]-x[0])
    def inside(k,a,b,c):
        p=uv[k]; va,vb,vc=uv[a],uv[b],uv[c]
        return cross2(va,vb,p)>=-eps and cross2(vb,vc,p)>=-eps and cross2(vc,va,p)>=-eps
    def external_clean(rec,tri):
        key=tri_key(tri)
        if key in ext_cache: stats['external_cache_hits']+=1; return ext_cache[key]
        stats['external_exact_calls']+=1
        gate=filtered._audit_pair_prepared_first(base_index,(rec,),same_surface=True)
        stats['external_exact_pairs']+=int(gate.get('count',0))
        good=int(gate.get('count',0))==0; ext_cache[key]=good; return good
    def internal_clean(rec,tri,chosen):
        pairs=[(other,otri) for other,otri in chosen]+[(other,tuple(map(int,other[4]))) for other in previous_records]
        for other,otri in pairs:
            key=tuple(sorted((tri_key(tri),tri_key(otri))))
            if key in pair_cache: stats['internal_cache_hits']+=1; good=pair_cache[key]
            else:
                stats['internal_exact_calls']+=1
                gate=pred._audit_record_pairs(((rec,other),),same_surface=True)
                stats['internal_exact_pairs']+=int(gate.get('count',0)); good=int(gate.get('count',0))==0; pair_cache[key]=good
            if not good: return False
        return True
    def remember(faces,records,qs):
        stats['complete_exact_clean_solutions']+=1
        bad=sum(q<quality_floor for q in qs)
        obj=(bad,-min(qs),-sum(qs))
        if best[0] is None or obj<best[0][0]: best[0]=(obj,faces,records,qs)
        stats['best_bad_faces']=int(best[0][0][0])
        if bad==0 or stats['complete_exact_clean_solutions']>=max_solutions: stop[0]=True
    def search(poly,chosen_faces,chosen_records,chosen_q,depth):
        if stop[0]: return
        stats['search_nodes']+=1; stats['max_depth']=max(stats['max_depth'],depth)
        if stats['search_nodes']>max_nodes or time.monotonic()-started>max_seconds:
            if best[0] is None: raise TimeoutError('lazy ear search budget reached without exact-clean plan: '+json.dumps(stats,separators=(',',':')))
            stats['budget_truncated']=True; stop[0]=True; return
        if len(poly)==3:
            a,b,c=poly; tri=(loop[a],loop[b],loop[c]); q=quality(tri)
            if q is None: return
            rec=rec_for(tri); stats['candidate_ears']+=1
            if external_clean(rec,tri) and internal_clean(rec,tri,chosen_records):
                remember(chosen_faces+[tri],chosen_records+[(rec,tri)],chosen_q+[q[0]])
            return
        ears=[]; m=len(poly)
        for j,cur in enumerate(poly):
            prev=poly[(j-1)%m]; nxt=poly[(j+1)%m]
            tri=(loop[prev],loop[cur],loop[nxt])
            edges=[tuple(sorted((tri[0],tri[1]))),tuple(sorted((tri[1],tri[2]))),tuple(sorted((tri[2],tri[0])))]
            if any(e in forbidden for e in edges): continue
            if cross2(uv[prev],uv[cur],uv[nxt])<=eps: continue
            if any(k not in (prev,cur,nxt) and inside(k,prev,cur,nxt) for k in poly): continue
            q=quality(tri)
            if q is not None: ears.append((q[0],int(loop[cur]),j,prev,cur,nxt,tri))
        for q,_,j,prev,cur,nxt,tri in sorted(ears,key=lambda x:(-x[0],x[1])):
            if stop[0]: break
            rec=rec_for(tri); stats['candidate_ears']+=1
            if not external_clean(rec,tri) or not internal_clean(rec,tri,chosen_records): continue
            search(poly[:j]+poly[j+1:],chosen_faces+[tri],chosen_records+[(rec,tri)],chosen_q+[q],depth+1)
            stats['backtracks']+=1
    search(tuple(range(n)),[],[],[],0)
    if best[0] is None: raise ValueError('no exact-clean lazy ear triangulation: '+json.dumps(stats,separators=(',',':')))
    obj,newfaces,recs,qs=best[0]
    topo=disk_mod.validate_disk(newfaces,loop,external_edges)
    return newfaces,[r for r,t in recs],{'method':'lazy_exact_checked_oriented_ear_clipping_bounded_best_qbound','boundary_vertex_count':n,'minimum_new_qbound_ratio':float(min(qs)),'new_bad_face_count':int(obj[0]),'topology':topo,'statistics':stats,'elapsed_s':time.monotonic()-started}
