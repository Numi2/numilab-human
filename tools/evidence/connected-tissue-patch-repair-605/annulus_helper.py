import collections, json, math, time
import numpy as np

def collision_aware_annulus(loops,P,P32,external_edges,base_index,pred,lattice,previous_pairs,first_candidate_id,max_nodes=25000,max_seconds=2.0,max_offsets=3):
    A=list(map(int,loops[0])); Bdirect=list(map(int,loops[1]))
    m,n=len(A),len(Bdirect)
    if m<3 or n<3 or len(set(A)&set(Bdirect)):
        raise ValueError("annulus boundary cycles are not simple vertex-disjoint loops")
    Brev=[Bdirect[0]]+list(reversed(Bdirect[1:]))
    offsets=sorted(range(n),key=lambda k:float(np.sum((P[A[0]]-P[Brev[k]])**2)))[:max_offsets]
    boundary={tuple(sorted((int(loop[i]),int(loop[(i+1)%len(loop)])))) for loop in (A,Bdirect) for i in range(len(loop))}
    forbidden={tuple(sorted(map(int,e))) for e in external_edges}-boundary
    total_stats={"offsets_tried":[],"nodes":0,"exact_external_pairs":0,"exact_internal_pairs":0,"backtracks":0}
    start=time.monotonic()
    nextid=[int(first_candidate_id)]
    for offset in offsets:
        B=[Brev[(offset+j)%n] for j in range(n)]
        stats={"nodes":0,"triangle_candidates":0,"exact_external_calls":0,"exact_external_pairs":0,"exact_internal_calls":0,"exact_internal_pairs":0,"backtracks":0,"max_depth":0}
        rec_cache={}; ext_cache={}; pair_cache={}
        def tri_key(tri):return tuple(sorted(map(int,tri)))
        def make_rec(tri):
            key=tri_key(tri)
            if key not in rec_cache:
                raw=pred._records(lattice,[tuple(map(int,tri))])[0]
                rec_cache[key]=(raw[0],raw[1],raw[2],nextid[0],raw[4])
                nextid[0]+=1
            return rec_cache[key]
        def valid(tri):
            if len(set(tri))!=3:return None
            if any(tuple(sorted((int(u),int(v)))) in forbidden for u,v in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0]))):return None
            xyz=P32[np.asarray(tri,np.int64)].astype(np.float64)
            cross=np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0])
            area=float(np.linalg.norm(cross))
            if not np.all(np.isfinite(cross)) or area==0:return None
            e2=sum(float(np.sum((xyz[(k+1)%3]-xyz[k])**2)) for k in range(3))
            return float(2.0*math.sqrt(3.0)*area/max(e2,1e-300))
        def clean_external(rec,tri):
            key=tri_key(tri)
            if key in ext_cache:return ext_cache[key]
            stats["exact_external_calls"]+=1
            result=pred._audit_pair_prepared_first(base_index,(rec,),same_surface=True)
            stats["exact_external_pairs"]+=int(result["count"])
            ext_cache[key]=int(result["count"])==0
            return ext_cache[key]
        def clean_chosen(rec,tri,chosen):
            for other,othertri in chosen:
                key=tuple(sorted((tri_key(tri),tri_key(othertri))))
                if key not in pair_cache:
                    stats["exact_internal_calls"]+=1
                    result=pred._audit_record_pairs(((rec,other),),same_surface=True)
                    pair_cache[key]=int(result["count"])==0
                    stats["exact_internal_pairs"]+=int(result["count"])
                if not pair_cache[key]:return False
            return True
        def search(i,j,chosen,triangles,depth):
            stats["nodes"]+=1
            stats["max_depth"]=max(stats["max_depth"],depth)
            if stats["nodes"]>max_nodes or time.monotonic()-start>max_seconds*max(1,len(total_stats["offsets_tried"])+1):
                raise TimeoutError("bounded annulus triangulation search limit reached")
            if i==m and j==n:return triangles,chosen
            opts=[]
            if i<m:
                tri=(A[i%m],A[(i+1)%m],B[j%n])
                q=valid(tri)
                if q is not None:opts.append((q,"A",tri,i+1,j))
            if j<n:
                tri=(A[i%m],B[(j+1)%n],B[j%n])
                q=valid(tri)
                if q is not None:opts.append((q,"B",tri,i,j+1))
            for q,kind,tri,ni,nj in sorted(opts,key=lambda x:(-x[0],x[1])):
                stats["triangle_candidates"]+=1
                rec=make_rec(tri)
                if not clean_external(rec,tri) or not clean_chosen(rec,tri,chosen):continue
                result=search(ni,nj,chosen+[(rec,tri)],triangles+[tri],depth+1)
                if result is not None:return result
                stats["backtracks"]+=1
            return None
        try:
            result=search(0,0,list(previous_pairs),[],0)
        except TimeoutError as exc:
            result=None
            stats["error"]=str(exc)
        total_stats["offsets_tried"].append({"offset":int(offset),"stats":stats})
        total_stats["nodes"]+=stats["nodes"]
        total_stats["exact_external_pairs"]+=stats["exact_external_pairs"]
        total_stats["exact_internal_pairs"]+=stats["exact_internal_pairs"]
        total_stats["backtracks"]+=stats["backtracks"]
        if result is None:continue
        tris,chosen=result
        if len(tris)!=m+n:raise ValueError("annulus triangulation does not have m+n faces")
        edge_dirs=collections.defaultdict(list)
        for tri in tris:
            a,b,c=map(int,tri)
            for u,v in ((a,b),(b,c),(c,a)):edge_dirs[tuple(sorted((u,v)))].append((u,v))
        actual=collections.Counter()
        for edge,dirs in edge_dirs.items():
            if len(dirs)==1:actual[dirs[0]]+=1
            elif len(dirs)==2:
                if dirs[0]!=(dirs[1][1],dirs[1][0]):raise ValueError("annulus internal edge winding mismatch")
            else:raise ValueError("annulus has nonmanifold internal edge")
        expected=collections.Counter()
        for loop in (A,Bdirect):
            for k in range(len(loop)):
                expected[(int(loop[k]),int(loop[(k+1)%len(loop)]))]+=1
        if actual!=expected:raise ValueError("annulus directed boundary does not match both source cycles")
        fresh=[rec for rec,tri in chosen[-len(tris):]]
        quality=[valid(tri) for tri in tris]
        details={"method":"exact collision-aware annulus zipper","boundary_loop_sizes":[m,n],"offset":int(offset),"min_face_quality":float(min(quality)),"exact_search":stats,"elapsed_s":time.monotonic()-start}
        return tris,fresh,details
    raise ValueError("no exact-clean annulus zipper found: "+json.dumps(total_stats,separators=(",",":")))

