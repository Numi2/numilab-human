"""Exact necessary edge constraints for the existing offline patch triangulators.

A triangle containing a segment that crosses an unshared retained surface cannot
pass the existing triangle predicate. Test each chord once before enumerating
triangles. Coplanar segment queries reuse the owner polygon-intersection points,
then restrict them to the finite segment; an infinite supporting line is never
treated as the segment. No tolerances, coordinate changes, or seam waivers.
"""
import itertools,time

def segment_triangle_points(pred,p,q,triangle):
    n=pred._cross(pred._sub(triangle[1],triangle[0]),pred._sub(triangle[2],triangle[0]))
    if not any(n):raise ValueError("degenerate retained triangle")
    dp=pred._dot(n,pred._sub(p,triangle[0]));dq=pred._dot(n,pred._sub(q,triangle[0]))
    if dp==dq==0:
        # Degenerate second polygon is used only to obtain candidate points;
        # restricting to [p,q] removes collinear points beyond its endpoints.
        raw=pred._coplanar_points(triangle,(p,q,p),n)
        return list({x for x in raw if pred._allowed_shared_point(x,{p,q})})
    return pred._segment_triangle(p,q,triangle,n)

def prune_chords(loop,lattice,base_index,pred,*,max_seconds=8.,cache=None,max_witnesses=32):
    started=time.monotonic();loop=list(map(int,loop))
    if len(loop)<3 or len(set(loop))!=len(loop):raise ValueError("chord slots must form one simple virtual or physical cycle")
    physical=getattr(pred,"physical",None)
    def identity(v):return int(physical[v])if physical is not None else int(v)
    cache={}if cache is None else cache
    boundary={tuple(sorted((loop[i],loop[(i+1)%len(loop)])))for i in range(len(loop))}
    blocked=set();witnesses=[];calls=0;hits=0;queries=0
    for u,v in itertools.combinations(loop,2):
        if time.monotonic()-started>max_seconds:raise TimeoutError("exact chord pruning budget exhausted")
        pu,pv=lattice[u],lattice[v];iu,iv=identity(u),identity(v)
        if (iu,pu)>(iv,pv):iu,iv,pu,pv=iv,iu,pv,pu
        key=(id(base_index),iu,iv,pu,pv)
        if key in cache:
            bad,w=cache[key];hits+=1
        elif pu==pv:
            bad,w=True,{"reason":"zero_length_chord","physical_vertices":[iu,iv]};cache[key]=(bad,w)
        else:
            lo=tuple(min(pu[k],pv[k])for k in range(3));hi=tuple(max(pu[k],pv[k])for k in range(3))
            query=((pu,pv,pu),lo,hi,0,(iu,iv,iu));bad=False;w=None
            for _,retained in pred._query_surface_aabb([query],base_index):
                queries+=1;calls+=1
                points=segment_triangle_points(pred,pu,pv,retained[0])
                common_ids={iu,iv}&set(retained[4])
                common={pu if i==iu else pv for i in common_ids}
                illegal=[p for p in points if not pred._allowed_shared_point(p,common)]
                if illegal:
                    bad=True;point=illegal[0]
                    w={"reason":"exact_segment_surface_contact","physical_vertices":[iu,iv],
                       "retained_face_id":int(retained[3]),"shared_vertex_ids":sorted(common_ids),
                       "point_lattice_rationals":[str(x)for x in point],
                       "point_world_m":[float(x)/(1<<149)for x in point]}
                    break
            cache[key]=(bad,w)
        if bad:
            edge=tuple(sorted((u,v)));blocked.add(edge)
            if len(witnesses)<max_witnesses:witnesses.append({"slot_vertices":list(edge),"witness":w})
    blocked_boundary=sorted(blocked&boundary)
    info={"method":"exact_segment_triangle_necessary_chord_constraints","slot_chords":len(loop)*(len(loop)-1)//2,
          "blocked_chords":len(blocked),"blocked_boundary_edges":blocked_boundary,
          "exact_segment_queries":calls,"aabb_candidates_visited":queries,"cache_hits":hits,
          "witnesses":witnesses,"elapsed_s":time.monotonic()-started,"coordinate_changes":0}
    return blocked,info
