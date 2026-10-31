"""Local source-boundary edge flip with exact open-star cavity checks; proposal only."""
from pathlib import Path
from collections import Counter, defaultdict
import importlib.util, sys
import numpy as np
H=Path(__file__).resolve().parent

def _imp(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    if s is None or s.loader is None:raise RuntimeError(f"cannot import {path}")
    m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
_DP=_imp("boundary_flip676_dp",H/"fixed_boundary_volume_polygon_flip_202_root.py")
_J=_imp("boundary_flip676_apex",H/"audit_joint_native_apex_272_root.py")

def _M(owner):
    if all(hasattr(owner,k) for k in ("measurements","passes","chain","integer_points","exact_det")):return owner
    if hasattr(owner,"m") and all(hasattr(owner.m,k) for k in ("measurements","passes","chain","integer_points","exact_det")):return owner.m
    raise TypeError("owner must expose the fixed-boundary volume quality/exact-chain helpers")
def _face_coefficient(face):
    a,b,c=map(int,face);key=tuple(sorted((a,b,c)))
    inv=int(a>b)+int(a>c)+int(b>c)
    return key,(-1 if inv%2 else 1)
def _face_chain(faces):
    out=Counter()
    for f in np.asarray(faces,np.int64):
        key,sign=_face_coefficient(f);out[key]+=sign
    return {k:v for k,v in out.items() if v}
def _edge_chain(faces):
    out=Counter()
    for a,b,c in np.asarray(faces,np.int64):
        for u,v in ((int(a),int(b)),(int(b),int(c)),(int(c),int(a))):out[tuple(sorted((u,v)))]+=1 if u<v else -1
    return {k:v for k,v in out.items() if v}
def _directed_edge_sign(face,a,b):
    f=list(map(int,face))
    for i in range(3):
        u,v=f[i],f[(i+1)%3]
        if (u,v)==(a,b):return 1
        if (u,v)==(b,a):return -1
    return 0
def _third(face,a,b):
    other=[int(x) for x in face if int(x) not in (a,b)]
    if len(other)!=1:raise ValueError("source triangle does not contain exactly one third point")
    return other[0]
def _open_link_path(tets,a,b,start,end):
    edges=[]
    for tet in np.asarray(tets,np.int64):
        rest=tuple(sorted(set(map(int,tet))-{a,b}))
        if len(rest)!=2:return None,"tet_does_not_contain_exact_edge"
        edges.append(rest)
    if len(set(edges))!=len(edges):return None,"duplicate_link_edge"
    adj=defaultdict(list)
    for u,v in edges:adj[u].append(v);adj[v].append(u)
    if start not in adj or end not in adj or start==end:return None,"source_endpoints_missing_or_equal"
    if len([v for v,nbr in adj.items() if len(nbr)==1])!=2:return None,"link_not_open_path"
    if len(adj[start])!=1 or len(adj[end])!=1 or any(len(nbr)!=2 for v,nbr in adj.items() if v not in (start,end)):return None,"link_not_open_path"
    ring=[start];prev=None;cur=start
    while cur!=end:
        nxt=[v for v in adj[cur] if v!=prev]
        if len(nxt)!=1:return None,"link_not_simple"
        prev,cur=cur,nxt[0]
        if cur in ring:return None,"link_cycle_or_repeat"
        ring.append(cur)
        if len(ring)>len(edges)+1:return None,"link_cycle_or_repeat"
    if len(ring)!=len(edges)+1 or len(set(ring))!=len(ring):return None,"link_disconnected"
    return ring,None
def _target_chain(old_chain,old_faces,new_faces):
    out=Counter(old_chain)
    for face in np.asarray(old_faces,np.int64):
        key,sign=_face_coefficient(face)
        if out.get(key,0)!=sign:return None,"old_source_face_not_exact_cavity_boundary"
        out[key]-=sign
        if out[key]==0:del out[key]
    for face in np.asarray(new_faces,np.int64):
        key,sign=_face_coefficient(face);out[key]+=sign
        if out[key]==0:del out[key]
    return {k:v for k,v in out.items() if v},None
def _manifold_boundary(tets):
    count=Counter();sign=Counter()
    pat=np.asarray(((1,2,3),(0,3,2),(0,1,3),(0,2,1)),np.int64)
    for face in np.asarray(tets,np.int64)[:,pat].reshape(-1,3):
        k,s=_face_coefficient(face);count[k]+=1;sign[k]+=s
    return all(n<=2 and (n!=2 or sign[k]==0) for k,n in count.items())
def _min_angles(points,faces):
    tri=np.asarray(points,np.float32)[np.asarray(faces,np.int64)].astype(np.float64)
    a=tri[:,1]-tri[:,0];b=tri[:,2]-tri[:,0];c=tri[:,2]-tri[:,1]
    la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1);lc=np.linalg.norm(c,axis=1)
    cos=np.stack(((la*la+lb*lb-lc*lc)/(2*la*lb),(la*la+lc*lc-lb*lb)/(2*la*lc),(lb*lb+lc*lc-la*la)/(2*lb*lc)),axis=1)
    return np.degrees(np.arccos(np.clip(cos,-1,1))).min(axis=1)
def _log_deficit(det,q):
    ratio=np.minimum(np.minimum(np.asarray(q)/1e-5,np.asarray(det)/(6e-18)),1.)
    return float(-np.log(np.maximum(ratio,1e-300)).sum())
def _exact_det_rows(M,points,tets):
    nodes=np.unique(np.asarray(tets,np.int64).ravel());local={int(v):i for i,v in enumerate(nodes)}
    ip=M.integer_points(np.asarray(points,np.float32)[nodes])
    return [M.exact_det(ip,[local[int(v)] for v in tet]) for tet in np.asarray(tets,np.int64)]
def _chain_payload(chain):
    rows=sorted(chain.items())
    return {"oriented_faces":[list(map(int,k))+[int(v)] for k,v in rows],"face_count":len(rows)}

def propose_boundary_flip(owner,points_f32,local_tets,edge,old_source_faces):
    """Propose one 2-to-2 source-face flip and its material-local open-star replacement.

    `local_tets` must be the complete one-material tetrahedron star of the boundary
    edge. `old_source_faces` must be its two actual oriented cavity-boundary faces,
    indexed in the same local point space. The caller owns whole-mesh star completeness,
    typed ancestry, cell/face duplicate checks, exact global context and collision audits.
    """
    M=_M(owner);p=np.asarray(points_f32,np.float32);t=np.asarray(local_tets,np.int64);sf=np.asarray(old_source_faces,np.int64)
    if p.ndim!=2 or p.shape[1]!=3 or t.ndim!=2 or t.shape[1]!=4 or sf.shape!=(2,3):return {"accepted":False,"reason":"input_shape"}
    a,b=map(int,edge)
    if a==b or min(a,b)<0 or max(a,b)>=len(p) or np.any(t<0) or np.any(t>=len(p)) or np.any(sf<0) or np.any(sf>=len(p)):return {"accepted":False,"reason":"point_row_out_of_range"}
    if np.any(np.diff(np.sort(t,axis=1),axis=1)==0) or len(np.unique(np.sort(t,axis=1),axis=0))!=len(t):return {"accepted":False,"reason":"invalid_or_duplicate_source_cells"}
    # Normalize source-face order to [a,b,c] and [b,a,d] by their shared directed edge.
    dirs=[_directed_edge_sign(f,a,b) for f in sf]
    if sorted(dirs)!=[-1,1]:return {"accepted":False,"reason":"source_faces_do_not_orient_shared_edge_oppositely"}
    i0=dirs.index(1);i1=dirs.index(-1);f0=sf[i0].copy();f1=sf[i1].copy();c=_third(f0,a,b);d=_third(f1,a,b)
    if c==d:return {"accepted":False,"reason":"source_face_third_vertices_equal"}
    for tet in t:
        if a not in tet or b not in tet:return {"accepted":False,"reason":"incomplete_or_wrong_edge_star"}
    ring,reason=_open_link_path(t,a,b,c,d)
    if reason:return {"accepted":False,"reason":reason}
    olddet,oldq=M.measurements(p,t);oldpass=np.asarray(M.passes(olddet,oldq),bool)
    if np.any(olddet<=0):return {"accepted":False,"reason":"source_cavity_nonpositive"}
    new_faces=np.asarray(((c,d,b),(d,c,a)),np.int64)
    if len(set(map(tuple,np.sort(new_faces,axis=1))))!=2:return {"accepted":False,"reason":"duplicate_new_source_face"}
    if _edge_chain(sf)!=_edge_chain(new_faces):return {"accepted":False,"reason":"source_patch_oriented_edge_chain_mismatch"}
    # Keep the existing polygon DP as the connected cavity constructor.
    proposal=_DP.best_triangulation(p,ring,a,b)
    if proposal is None:return {"accepted":False,"reason":"no_positive_ring_triangulation"}
    new_tets=np.asarray(proposal[0],np.int64).reshape(-1,4)
    if not len(new_tets) or len(np.unique(np.sort(new_tets,axis=1),axis=0))!=len(new_tets):return {"accepted":False,"reason":"invalid_or_duplicate_replacement_cells"}
    if not _manifold_boundary(new_tets):return {"accepted":False,"reason":"replacement_not_manifold"}
    old_chain=M.chain(t)
    target,reason=_target_chain(old_chain,sf,new_faces)
    if reason:return {"accepted":False,"reason":reason}
    new_chain=M.chain(new_tets)
    if new_chain!=target:return {"accepted":False,"reason":"exact_target_chain_mismatch"}
    # No changed source triangle may fail the exact necessary joint-apex bound.
    fnodes=np.unique(new_faces);fmap={int(v):i for i,v in enumerate(fnodes)};fip=M.integer_points(p[fnodes])
    for face in new_faces:
        if _J.exact_terms(fip,[fmap[int(v)] for v in face])[0] is not None:return {"accepted":False,"reason":"new_source_face_fails_exact_apex_bound"}
    old_angles=_min_angles(p,sf);new_angles=_min_angles(p,new_faces)
    if not float(new_angles.min())>float(old_angles.min())+1e-12:return {"accepted":False,"reason":"source_min_angle_not_improved"}
    olddet_new,oldq_new=M.measurements(p,t);newdet,newq=M.measurements(p,new_tets);newpass=np.asarray(M.passes(newdet,newq),bool)
    old_bad=int((~oldpass).sum());new_bad=int((~newpass).sum())
    old_score=_log_deficit(olddet_new,oldq_new);new_score=_log_deficit(newdet,newq)
    if new_bad>old_bad:return {"accepted":False,"reason":"native_quality_count_regressed"}
    score_gain=old_score-new_score
    if new_bad==old_bad and score_gain < -max(1e-8,1e-6*old_score):return {"accepted":False,"reason":"native_quality_score_regressed_at_tied_count"}
    old_exact=_exact_det_rows(M,p,t);new_exact=_exact_det_rows(M,p,new_tets)
    if min(old_exact)<=0:return {"accepted":False,"reason":"source_exact_det_not_positive"}
    if min(new_exact)<=0:return {"accepted":False,"reason":"replacement_exact_det_not_positive"}
    delta=sum(new_exact)-sum(old_exact)
    return {"accepted":True,"reason":"PASS_BOUNDARY_EDGE_FLIP_LOCAL_PROPOSAL","edge":[a,b],"open_link_path":list(map(int,ring)),
      "old_rows_local_tets":t.copy(),"new_tets":new_tets,"old_boundary_source_faces":sf.copy(),"new_source_faces":new_faces,
      "exact_target_chain":_chain_payload(target),"source_patch_oriented_edge_chain_preserved":True,"replacement_oriented_chain_matches_exact_target":True,
      "quality_counts":{"native_bad_before":old_bad,"native_bad_after":new_bad,"native_quality_count_nonincrease":new_bad<=old_bad,"new_native_passing_cells":int(newpass.sum())},
      "quality_scores":{"native_log_deficit_before":old_score,"native_log_deficit_after":new_score,"native_log_deficit_gain":score_gain,"local_quality_nonregression":True,"minimum_normalized_det_before":float(np.min(oldq)) if len(oldq) else float("inf"),"minimum_normalized_det_after":float(np.min(newq)) if len(newq) else float("inf")},
      "source_triangle_metrics":{"minimum_angle_deg_before":float(old_angles.min()),"minimum_angle_deg_after":float(new_angles.min()),"angle_gain_deg":float(new_angles.min()-old_angles.min()),"exact_new_apex_bound_pass":True,"new_apex_bound_scores":_J.score(p,new_faces).tolist()},
      "exact_det6":{"old_sum":str(sum(old_exact)),"new_sum":str(sum(new_exact)),"delta":str(delta),"all_old_positive":True,"all_new_positive":True},
      "material_local":True,"coordinates_unchanged":True,"native_thresholds_unchanged":True,"physical_admission":False,"native_steps":0}

def fixture():
    M=_M(_DP.m)
    p=np.asarray(((0,0,0),(1,0,0),(.2,.2,0),(.8,-.2,0),(.5,0,1e-7)),np.float32)
    a,b,c,d,x=0,1,2,3,4
    old=np.asarray(((a,b,c,x),(b,a,d,x)),np.int64)
    # Positive-tet outward boundary orientation; these are the actual shared-edge faces.
    old_faces=np.asarray(((a,c,b),(b,d,a)),np.int64)
    got=propose_boundary_flip(M,p,old,(a,b),old_faces)
    assert got["accepted"],got
    assert got["quality_counts"]["native_bad_after"]<=got["quality_counts"]["native_bad_before"]
    assert got["quality_scores"]["native_log_deficit_gain"]>0
    assert got["source_triangle_metrics"]["angle_gain_deg"]>0
    assert got["exact_det6"]["delta"]=="0"
    assert len(got["new_tets"])==2 and got["replacement_oriented_chain_matches_exact_target"]
    neutral_points=p.copy();neutral_points[x,2]=1e-3
    neutral=propose_boundary_flip(M,neutral_points,old,(a,b),old_faces)
    assert neutral["accepted"],neutral
    assert neutral["quality_counts"]["native_bad_before"]==neutral["quality_counts"]["native_bad_after"]==0
    assert abs(neutral["quality_scores"]["native_log_deficit_gain"])<1e-12
    open_fail=propose_boundary_flip(M,p,old[:1],(a,b),old_faces)
    assert not open_fail["accepted"] and open_fail["reason"]=="source_endpoints_missing_or_equal"
    return {"status":"PASS_BOUNDARY_EDGE_FLIP_LOCAL_FIXTURE","native_thresholds_unchanged":True,"fixture_old_bad":got["quality_counts"]["native_bad_before"],"fixture_new_bad":got["quality_counts"]["native_bad_after"],"source_min_angle_gain_deg":got["source_triangle_metrics"]["angle_gain_deg"],"exact_det6_delta":got["exact_det6"]["delta"],"new_tets":len(got["new_tets"]),"controls":["complete_open_link_path","source_orientation","exact_target_chain","exact_F32_positive_determinants","native_quality_nonincrease_and_score_gain","strict_source_min_angle_improvement","exact_joint_apex_bound","allow_neutral_material_quality","reject_incomplete_star"]}

if __name__=="__main__":
    import argparse,json
    ap=argparse.ArgumentParser();ap.add_argument("--fixture",action="store_true");a=ap.parse_args()
    if a.fixture:print(json.dumps(fixture(),separators=(",",":")))
