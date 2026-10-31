#!/usr/bin/env python3
"""Existing-vertex kernel-cone cavity prepass; proposal only, no producer edits."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np

HERE=Path(__file__).resolve().parent
OWNER_PATH=HERE.parent/"quality-edge-bisection-683-root"/"quality_edge_bisection_683_helper.py"
OWNER_SHA256=hashlib.sha256(OWNER_PATH.read_bytes()).hexdigest()
_spec=importlib.util.spec_from_file_location("kernel_cone_683_owner", OWNER_PATH)
Q=importlib.util.module_from_spec(_spec); sys.modules[_spec.name]=Q; _spec.loader.exec_module(Q)
M=Q.M

def face_key_sign(face):
    a,b,c=map(int,face); inv=int(a>b)+int(a>c)+int(b>c)
    return tuple(sorted((a,b,c))), (-1 if inv&1 else 1)

def face_chain(faces):
    out=Counter()
    for face in np.asarray(faces,np.int64):
        key,sign=face_key_sign(face); out[key]+=sign
    return {k:v for k,v in out.items() if v}

def boundary_topology(faces):
    f=np.asarray(faces,np.int64)
    if f.ndim!=2 or f.shape[1]!=3 or not len(f): return False,"boundary_shape",{}
    if np.any((f[:,0]==f[:,1])|(f[:,1]==f[:,2])|(f[:,2]==f[:,0])):
        return False,"degenerate_boundary_face",{}
    canonical=[tuple(sorted(map(int,row))) for row in f]
    if len(set(canonical))!=len(canonical): return False,"duplicate_boundary_face",{}
    ec=Counter(); es=Counter(); adj=defaultdict(set)
    for row in f:
        a,b,c=map(int,row)
        for u,v in ((a,b),(b,c),(c,a)):
            key=tuple(sorted((u,v))); ec[key]+=1; es[key]+=1 if u<v else -1
            adj[u].add(v); adj[v].add(u)
    if any(n!=2 for n in ec.values()):
        return False,"boundary_not_closed_2_manifold",{"bad_edge_count":sum(n!=2 for n in ec.values())}
    if any(es[e]!=0 for e in ec):
        return False,"boundary_orientation_not_closed",{"bad_oriented_edge_count":sum(es[e]!=0 for e in ec)}
    verts=sorted(adj); seen=set(); todo=[verts[0]]
    while todo:
        v=todo.pop()
        if v in seen: continue
        seen.add(v); todo.extend(adj[v]-seen)
    if len(seen)!=len(verts): return False,"boundary_disconnected",{}
    chi=len(verts)-len(ec)+len(f)
    if chi!=2: return False,"boundary_not_sphere_topology",{"euler_characteristic":chi}
    return True,"PASS",{"boundary_vertex_count":len(verts),"boundary_edge_count":len(ec),
                         "boundary_face_count":len(f),"euler_characteristic":chi}

def tet_face_incidence(tets):
    cnt=Counter(); signs=Counter()
    for face in np.asarray(tets,np.int64)[:,M.PAT].reshape(-1,3):
        key,sign=face_key_sign(face); cnt[key]+=1; signs[key]+=sign
    return cnt,signs

def exact_det_values(points,tets):
    p=np.asarray(points,np.float32); cells=np.asarray(tets,np.int64)
    used=np.unique(cells.ravel()); lookup={int(v):i for i,v in enumerate(used)}
    ip=M.integer_points(p[used])
    return [M.exact_det(ip,[lookup[int(v)] for v in tet]) for tet in cells]

def validate_cavity(points,faces,old_tets,materials):
    p=np.asarray(points,np.float32); f=np.asarray(faces,np.int64)
    old=np.asarray(old_tets,np.int64); mats=np.asarray(materials,np.int32)
    if p.ndim!=2 or p.shape[1]!=3 or not np.all(np.isfinite(p)): return False,"points_shape_or_nonfinite",{}
    if old.ndim!=2 or old.shape[1]!=4 or not len(old) or len(mats)!=len(old): return False,"old_tet_or_material_shape",{}
    if np.any(old<0) or np.any(old>=len(p)) or np.any(np.diff(np.sort(old,axis=1),axis=1)==0): return False,"invalid_old_tets",{}
    if len(set(tuple(sorted(map(int,r))) for r in old))!=len(old): return False,"duplicate_old_tet",{}
    if len(np.unique(mats))!=1: return False,"mixed_material_cavity",{"material_indices":np.unique(mats).tolist()}
    if np.any(f<0) or np.any(f>=len(p)): return False,"boundary_vertex_out_of_range",{}
    ok,why,top=boundary_topology(f)
    if not ok: return False,why,top
    bchain=face_chain(f)
    if M.chain(old)!=bchain: return False,"old_cavity_boundary_chain_mismatch",{}
    cnt,sgn=tet_face_incidence(old)
    if any(n>2 for n in cnt.values()): return False,"old_cavity_nonmanifold",{}
    if any(n==2 and sgn[k]!=0 for k,n in cnt.items()): return False,"old_internal_face_orientation_mismatch",{}
    exact=exact_det_values(p,old); det,q=M.measurements(p,old)
    if min(exact)<=0 or np.any(~np.isfinite(det)) or np.any(det<=0): return False,"old_cavity_not_exact_positive",{}
    return True,"PASS",{**top,"material_index":int(mats[0]),"old_cell_count":len(old),
        "old_bad_count":int(np.count_nonzero(~M.passes(det,q))),
        "old_log_deficit":float(Q._log_deficit(det,q))}

def assess_apex(points,faces,old_tets,materials,apex,merit_policy,base):
    p=np.asarray(points,np.float32); f=np.asarray(faces,np.int64)
    old=np.asarray(old_tets,np.int64); mats=np.asarray(materials,np.int32)
    selected=f[~np.any(f==int(apex),axis=1)]
    if not len(selected): return {"status":"REJECT_EMPTY_CONE","apex_vertex":int(apex)}
    new=np.column_stack((np.full(len(selected),int(apex),np.int64),selected))
    if np.any(np.diff(np.sort(new,axis=1),axis=1)==0): return {"status":"REJECT_DEGENERATE_CONE_TET","apex_vertex":int(apex)}
    if len(set(tuple(sorted(map(int,r))) for r in new))!=len(new):
        return {"status":"REJECT_DUPLICATE_CONE_TET","apex_vertex":int(apex)}
    exact=exact_det_values(p,new)
    bad=[i for i,d in enumerate(exact) if d<=0]
    if bad: return {"status":"REJECT_APEX_OUTSIDE_KERNEL","apex_vertex":int(apex),
                    "nonpositive_exact_face_rows":bad[:8],"nonpositive_exact_face_count":len(bad)}
    bchain=face_chain(f)
    if M.chain(new)!=bchain:
        return {"status":"REJECT_CONE_ORIENTED_CHAIN","apex_vertex":int(apex)}
    cnt,sgn=tet_face_incidence(new)
    if any(n>2 for n in cnt.values()): return {"status":"REJECT_CONE_NONMANIFOLD","apex_vertex":int(apex)}
    if any((n==1)!=(k in bchain) or (n==2 and sgn[k]!=0) for k,n in cnt.items()):
        return {"status":"REJECT_CONE_FACE_INCIDENCE","apex_vertex":int(apex)}
    ok,why=M.certify(p,old,new)
    if not ok: return {"status":"REJECT_EXACT_CAVITY_CERTIFICATE","apex_vertex":int(apex),"reason":why}
    det,q=M.measurements(p,new)
    if np.any(~np.isfinite(det)) or np.any(~np.isfinite(q)) or np.any(det<=0):
        return {"status":"REJECT_FLOAT_MEASUREMENT","apex_vertex":int(apex)}
    new_bad=int(np.count_nonzero(~M.passes(det,q))); old_bad=int(base["old_bad_count"])
    score=float(Q._log_deficit(det,q)); old_score=float(base["old_log_deficit"])
    if new_bad>old_bad: status="REJECT_PER_ROLE_BAD_COUNT_INCREASE"
    elif merit_policy=="strict_bad_count" and new_bad>=old_bad: status="REJECT_NO_STRICT_BAD_COUNT_GAIN"
    elif merit_policy=="strict_lex" and not ((new_bad,score)<(old_bad,old_score)): status="REJECT_NO_STRICT_LEX_GAIN"
    elif merit_policy not in ("nonincrease","strict_bad_count","strict_lex"): status="REJECT_UNKNOWN_MERIT_POLICY"
    else:
        return {"status":"PASS_EXISTING_VERTEX_KERNEL_CONE","apex_vertex":int(apex),
          "material_index":int(mats[0]),"old_bad":old_bad,"new_bad":new_bad,
          "old_log_deficit":old_score,"new_log_deficit":score,"log_deficit_gain":old_score-score,
          "exact_positive_tet_count":len(exact),
          "exact_det6_delta_integer":str(sum(exact)-sum(exact_det_values(p,old))),
          "exact_chain_matches_input":True,"no_duplicate_tetrahedra":True,"manifold_face_incidence":True,
          "replacement_tetrahedra":new,"replacement_material_indices":np.full(len(new),mats[0],np.int32),
          "boundary_face_rows_not_containing_apex":np.flatnonzero(~np.any(f==int(apex),axis=1))}
    return {"status":status,"apex_vertex":int(apex),"old_bad":old_bad,"new_bad":new_bad,
            "old_log_deficit":old_score,"new_log_deficit":score}

def propose_existing_vertex_kernel_cone(points,oriented_boundary_faces,old_tetrahedra,
                                        old_material_indices,*,merit_policy="strict_lex"):
    valid,why,base=validate_cavity(points,oriented_boundary_faces,old_tetrahedra,old_material_indices)
    if not valid: return {"status":"REJECT_INVALID_CAVITY","reason":why,"details":base,
                          "fallback":"existing cavity MILP/solver"}
    f=np.asarray(oriented_boundary_faces,np.int64); candidates=[]; rejects=Counter(); diags=[]
    for apex in sorted(set(map(int,f.ravel()))):
        r=assess_apex(points,f,old_tetrahedra,old_material_indices,apex,merit_policy,base)
        if r["status"]=="PASS_EXISTING_VERTEX_KERNEL_CONE": candidates.append(r)
        else: rejects[r["status"]]+=1; diags.append(r)
    if not candidates:
        return {"status":"NO_VALID_EXISTING_VERTEX_KERNEL_CONE","fallback":"existing cavity MILP/solver",
          "merit_policy":merit_policy,"boundary_vertex_count":len(set(map(int,f.ravel()))),
          "rejections":dict(rejects),"apex_diagnostics":diags,"physical_admission":False,"native_steps":0}
    best=min(candidates,key=lambda r:(r["new_bad"],r["new_log_deficit"],r["apex_vertex"]))
    result=dict(best); result.update({"status":"PASS_EXISTING_VERTEX_KERNEL_CONE_BEST","merit_policy":merit_policy,
      "boundary_vertex_count":len(set(map(int,f.ravel()))),"valid_apex_count":len(candidates),
      "rejections":dict(rejects),"apex_candidates":[{k:v for k,v in r.items() if k!="replacement_tetrahedra"}
       for r in sorted(candidates,key=lambda x:(x["new_bad"],x["new_log_deficit"],x["apex_vertex"]))],
      "physical_admission":False,"native_steps":0})
    return result

def oriented_boundary(old):
    chain=M.chain(np.asarray(old,np.int64))
    return np.asarray([Q._oriented_face(k,v) for k,v in sorted(chain.items())],np.int64)

def convex_octahedron():
    p=np.asarray([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1.0e-5],[0,0,-1.0e-5],[0,0,0]],np.float32)
    faces=np.asarray([(x,y,z) for x in (0,1) for y in (2,3) for z in (4,5)],np.int64)
    old=M.orient(p,np.column_stack((np.full(len(faces),6),faces)))
    return p,oriented_boundary(old),old,np.full(len(old),17,np.int32)

def nonstar_l_prism():
    pts=np.asarray([(x,y,z) for y in range(3) for x in range(3) for z in (0,1)],np.float32)
    ix={(x,y,z):(y*3+x)*2+z for y in range(3) for x in range(3) for z in (0,1)}
    perms=((0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)); tets=[]
    for x,y in ((0,0),(1,0),(0,1)):
        origin=(x,y,0)
        for perm in perms:
            v=list(origin); row=[ix[origin]]
            for axis in perm:
                v=v.copy(); v[axis]+=1; row.append(ix[tuple(v)])
            tets.append(row)
    old=M.orient(pts,np.asarray(tets,np.int64))
    return pts,oriented_boundary(old),old,np.full(len(old),5,np.int32)

def fixture():
    convex=convex_octahedron()
    got=propose_existing_vertex_kernel_cone(*convex,merit_policy="nonincrease")
    assert got["status"]=="PASS_EXISTING_VERTEX_KERNEL_CONE_BEST",got
    assert got["exact_chain_matches_input"] and got["exact_det6_delta_integer"]=="0"
    assert got["no_duplicate_tetrahedra"] and got["manifold_face_incidence"]
    assert np.all(got["replacement_material_indices"]==17)
    strict_count=propose_existing_vertex_kernel_cone(*convex,merit_policy="strict_bad_count")
    strict_lex=propose_existing_vertex_kernel_cone(*convex,merit_policy="strict_lex")
    assert strict_count["status"]=="PASS_EXISTING_VERTEX_KERNEL_CONE_BEST"
    assert strict_lex["status"]=="PASS_EXISTING_VERTEX_KERNEL_CONE_BEST"
    mixed=list(convex); mixed[3]=convex[3].copy(); mixed[3][0]+=1
    mixed_result=propose_existing_vertex_kernel_cone(*mixed,merit_policy="nonincrease")
    assert mixed_result["status"]=="REJECT_INVALID_CAVITY" and mixed_result["reason"]=="mixed_material_cavity"
    nonstar=nonstar_l_prism()
    no=propose_existing_vertex_kernel_cone(*nonstar,merit_policy="nonincrease")
    assert no["status"]=="NO_VALID_EXISTING_VERTEX_KERNEL_CONE",no
    assert no["rejections"].get("REJECT_APEX_OUTSIDE_KERNEL",0)>0
    eps=np.float32(2.0**-20)
    p=np.asarray([[0,0,0],[1,0,0],[0,1,0],[.25,.25,eps]],np.float32)
    assert exact_det_values(p,np.asarray([[0,1,2,3]]))[0]>0
    p[3,2]=-eps
    assert exact_det_values(p,np.asarray([[0,1,2,3]]))[0]<0
    valid=[]; ok,_,base=validate_cavity(*convex)
    for apex in sorted(set(map(int,convex[1].ravel()))):
        r=assess_apex(*convex,apex,"nonincrease",base)
        if r["status"]=="PASS_EXISTING_VERTEX_KERNEL_CONE": valid.append(r)
    assert valid
    expected=min(valid,key=lambda r:(r["new_bad"],r["new_log_deficit"],r["apex_vertex"]))
    assert got["apex_vertex"]==expected["apex_vertex"]
    assert len({round(r["new_log_deficit"],12) for r in valid})>1
    return {"status":"PASS_EXISTING_VERTEX_KERNEL_CONE_CONTROLS",
      "controls":["convex_star_cone","exact_chain_and_det6_volume","dyadic_positive_and_negative_signs",
      "nonconvex_empty_kernel_rejects_all_boundary_apices","no_duplicate_or_nonmanifold_tets",
      "same_material","mixed_material_rejection","nonincrease_strict_bad_count_and_strict_lex_policies",
      "sliver_quality_lexicographic_ranking","MILP_fallback"],
      "convex":{"boundary_faces":len(convex[1]),"valid_apices":got["valid_apex_count"],
       "selected_apex":got["apex_vertex"],"det6_delta":got["exact_det6_delta_integer"],
       "bad_before":got["old_bad"],"bad_after":got["new_bad"],
       "apex_ranking":[{"apex":r["apex_vertex"],"bad":r["new_bad"],
                        "log_deficit":r["new_log_deficit"]} for r in got["apex_candidates"]]},
      "nonstar":{"boundary_faces":len(nonstar[1]),"boundary_vertices":len(set(map(int,nonstar[1].ravel()))),
       "status":no["status"],"rejections":no["rejections"]},
      "owner_path":str(OWNER_PATH),"owner_sha256":OWNER_SHA256,"physical_admission":False,"native_steps":0}

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--fixture",action="store_true"); ap.add_argument("--out",type=Path)
    a=ap.parse_args()
    result=fixture() if a.fixture else {"status":"HELPER_ONLY","owner_sha256":OWNER_SHA256,
                                       "physical_admission":False,"native_steps":0}
    rendered=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if a.out: a.out.write_text(rendered)
    print(rendered,end="")

