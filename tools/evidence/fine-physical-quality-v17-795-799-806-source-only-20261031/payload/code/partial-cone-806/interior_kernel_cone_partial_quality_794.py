#!/usr/bin/env python3
"""Partial-quality interior Steiner cone search; proposal only, exact gates retained."""
from pathlib import Path
import hashlib
import importlib.util
import sys
import time

import numpy as np
from scipy.optimize import linprog, minimize

HERE=Path(__file__).resolve().parent
H=HERE.parent
OLD_PATH=H/"quality-kernel-cone-734-root/kernel_cone_734.py"
OLD_SHA="d9a67e7bba636323196af75ac3eb31e4b2faba6258ec998104788a1ac463e2c5"
OWNER_SHA="650794951cbcf338de9cde691a8f43377b9d023ccea78b876e87c847880243e0"
def sha(p):
    return hashlib.file_digest(Path(p).open("rb"),"sha256").hexdigest()
if sha(OLD_PATH)!=OLD_SHA:
    raise RuntimeError("pinned 734 cavity validator changed")
_spec=importlib.util.spec_from_file_location("partial_quality_steiner_base734",OLD_PATH)
K734=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=K734
_spec.loader.exec_module(K734)
M=K734.M
QUALITY_GATE=1.0e-5
VOLUME_GATE=1.0e-18
POSITIVE_VISIBILITY_FLOOR=128.0*np.finfo(np.float64).eps

def _continuous_metrics(x,n,a,tri,diameter,area):
    d=np.einsum("ij,ij->i",n,x[None,:]-a)
    det=d*area
    delta=x[None,None,:]-tri
    lengths=np.linalg.norm(delta,axis=2)
    maxedge=np.maximum(np.max(lengths,axis=1),np.max(np.stack((
        np.linalg.norm(tri[:,0]-tri[:,1],axis=1),
        np.linalg.norm(tri[:,1]-tri[:,2],axis=1),
        np.linalg.norm(tri[:,2]-tri[:,0],axis=1)),axis=1),axis=1))
    q=det/np.maximum(maxedge**3,np.finfo(float).tiny)
    volume_ratio=det*(diameter**3)/(6.0*VOLUME_GATE)
    quality_ratio=q/QUALITY_GATE
    ratio=np.minimum(1.0,np.minimum(volume_ratio,quality_ratio))
    deficit=float(-np.log(np.maximum(ratio,1.0e-300)).sum())
    return deficit,d,q,volume_ratio,quality_ratio

def _inside_kernel(x,n,a,floor=0.0):
    return bool(np.isfinite(x).all() and np.all(np.einsum("ij,ij->i",n,x[None,:]-a)>floor))

def _candidate_score(points,tets):
    det,q=M.measurements(np.asarray(points,np.float32),np.asarray(tets,np.int64))
    good=np.asarray(M.passes(det,q),bool)
    return int(np.count_nonzero(~good)),float(K734.Q._log_deficit(det,q)),det,q,good

def propose_partial_quality_kernel_cone(points,oriented_boundary_faces,old_tetrahedra,
                                        old_material_indices,*,time_limit_s=2.0,
                                        maxiter=300):
    """Find an exact-positive cone with lexicographically improved (bad count, log deficit).

    The continuous objective minimizes the aggregate native deficit over a strictly
    visible apex. It does not require every child cone tet to meet the native gates.
    Exact F32 positivity, chain, volume, and strict local quality improvement remain
    mandatory after rounding.
    """
    started=time.monotonic()
    p=np.asarray(points)
    f=np.asarray(oriented_boundary_faces,np.int64)
    old=np.asarray(old_tetrahedra,np.int64)
    mats=np.asarray(old_material_indices,np.int32)
    if p.dtype!=np.float32 or p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():
        return {"status":"REJECT_POINTS_NOT_FINITE_F32"}
    valid,why,base=K734.validate_cavity(p,f,old,mats)
    if not valid:return {"status":"REJECT_INVALID_CAVITY","reason":why,"details":base}
    old_det,old_q=M.measurements(p,old)
    old_good=np.asarray(M.passes(old_det,old_q),bool)
    old_bad=int(np.count_nonzero(~old_good))
    old_score=float(K734.Q._log_deficit(old_det,old_q))
    if old_bad<=0 and old_score<=0.0:
        return {"status":"REJECT_CAVITY_HAS_NO_BAD_CELLS","old_bad":old_bad,
                "old_log_deficit":old_score}
    face_vertices=np.asarray(p[f],np.float64)
    used=np.unique(f.reshape(-1))
    boundary_xyz=np.asarray(p[used],np.float64)
    origin=boundary_xyz.mean(axis=0)
    diameter=float(np.linalg.norm(np.ptp(boundary_xyz,axis=0)))
    if not np.isfinite(diameter) or diameter<=0:return {"status":"REJECT_DEGENERATE_CAVITY_SCALE"}
    pn=(np.asarray(p,np.float64)-origin)/diameter
    tri=pn[f]
    a=tri[:,0]
    # For tet order [apex,a,b,c], this normal makes det6(x,a,b,c)=n.(x-a).
    nraw=np.cross(tri[:,2]-tri[:,0],tri[:,1]-tri[:,0])
    area=np.linalg.norm(nraw,axis=1)
    if np.any(~np.isfinite(area)) or np.any(area<=64*np.finfo(float).eps):
        return {"status":"REJECT_DEGENERATE_BOUNDARY_FACE"}
    n=nraw/area[:,None]
    low=tri.reshape(-1,3).min(axis=0);high=tri.reshape(-1,3).max(axis=0)
    A=np.column_stack((-n,np.ones(len(n))))
    b=-np.einsum("ij,ij->i",n,a)
    remaining=float(time_limit_s)-(time.monotonic()-started)
    if not np.isfinite(remaining) or remaining<=0:return {"status":"REJECT_TIME_BUDGET_BEFORE_KERNEL_LP"}
    try:
        lp=linprog(np.asarray((0.,0.,0.,-1.)),A_ub=A,b_ub=b,
                   bounds=[(float(low[i]),float(high[i])) for i in range(3)]+[(None,None)],
                   method="highs",options={"time_limit":max(.02,remaining)})
    except Exception as exc:
        return {"status":"REJECT_KERNEL_LP_EXCEPTION","error":repr(exc)}
    if not lp.success or lp.x is None or not np.isfinite(lp.x).all():
        return {"status":"REJECT_EMPTY_VISIBILITY_KERNEL","lp_status":int(lp.status),"lp_message":str(lp.message)}
    xlp=np.asarray(lp.x[:3],np.float64)
    visibility_height=float(lp.x[3])
    if visibility_height<=POSITIVE_VISIBILITY_FLOOR:
        return {"status":"REJECT_NO_STRICT_KERNEL_INTERIOR","visibility_height_normalized":visibility_height,
                "lp_message":str(lp.message)}

    # Optimize aggregate log deficit over the 3D visibility kernel. Multiple
    # deterministic feasible starts reduce sensitivity to the piecewise-smooth
    # minimum of volume and normalized-edge quality ratios.
    starts=[("KERNEL_MAX_CLEARANCE",xlp)]
    centroid=tri.reshape(-1,3).mean(axis=0)
    if _inside_kernel(centroid,n,a,POSITIVE_VISIBILITY_FLOOR):
        starts.append(("BOUNDARY_VERTEX_MEAN",centroid))
    old_vertices=np.unique(old.reshape(-1))
    internal=np.setdiff1d(old_vertices,used)
    for row in internal:
        xi=(np.asarray(p[int(row)],np.float64)-origin)/diameter
        if _inside_kernel(xi,n,a,POSITIVE_VISIBILITY_FLOOR):
            starts.append(("OLD_INTERIOR_VERTEX_"+str(int(row)),xi))
            break
    # Include a deterministic interior interpolation when it is distinct.
    mid=0.5*(xlp+centroid)
    if _inside_kernel(mid,n,a,POSITIVE_VISIBILITY_FLOOR):
        starts.append(("KERNEL_CENTROID_MIDPOINT",mid))

    constraints=({"type":"ineq",
                  "fun":lambda x:np.einsum("ij,ij->i",n,x[None,:]-a)-POSITIVE_VISIBILITY_FLOOR,
                  "jac":lambda x:n},)
    optimizations=[]
    candidates=[]
    overall_deadline=started+float(time_limit_s)
    for label,x0 in starts:
        remaining=overall_deadline-time.monotonic()
        if remaining<=0:break
        try:
            opt=minimize(lambda x:_continuous_metrics(x,n,a,tri,diameter,area)[0],
                         np.asarray(x0,np.float64),method="SLSQP",
                         bounds=[(float(low[i]),float(high[i])) for i in range(3)],
                         constraints=constraints,
                         options={"maxiter":int(maxiter),"ftol":1e-12,"disp":False})
            optimizations.append({"start":label,"success":bool(opt.success),
                "status":int(opt.status),"message":str(opt.message),
                "iterations":int(getattr(opt,"nit",0)),
                "continuous_score":float(_continuous_metrics(opt.x,n,a,tri,diameter,area)[0])
                    if getattr(opt,"x",None) is not None and np.isfinite(opt.x).all() else None})
            if getattr(opt,"x",None) is not None and np.isfinite(opt.x).all():
                candidates.append((label,np.asarray(opt.x,np.float64)))
        except Exception as exc:
            optimizations.append({"start":label,"error":repr(exc)})
        candidates.append((label+"_START",np.asarray(x0,np.float64)))

    failures=[]
    best=None
    seen32=set()
    for source,x in candidates:
        if not _inside_kernel(x,n,a,POSITIVE_VISIBILITY_FLOOR):
            failures.append({"candidate":source,"reason":"not_strictly_inside_visibility_kernel"})
            continue
        continuous_score,continuous_d,continuous_q,continuous_vr,continuous_qr=_continuous_metrics(x,n,a,tri,diameter,area)
        apex64=origin+diameter*x
        apex32=np.asarray(apex64,np.float32)
        key=apex32.tobytes()
        if key in seen32:continue
        seen32.add(key)
        if not np.isfinite(apex32).all() or any(np.array_equal(apex32,v) for v in p):
            failures.append({"candidate":source,"reason":"rounded_apex_nonfinite_or_duplicate"})
            continue
        paug=np.vstack((p,apex32))
        new=np.column_stack((np.full(len(f),len(p),np.int64),f))
        exact=K734.exact_det_values(paug,new)
        if not exact or min(exact)<=0:
            failures.append({"candidate":source,"reason":"exact_nonpositive_cone_tet",
                             "minimum_exact_det6_integer":str(min(exact)) if exact else None})
            continue
        if M.chain(old)!=M.chain(new):
            failures.append({"candidate":source,"reason":"oriented_chain_mismatch"})
            continue
        ok,why2=M.certify(paug,old,new)
        if not ok:
            failures.append({"candidate":source,"reason":"exact_cavity_certificate","detail":why2})
            continue
        new_bad,new_score,nd,nq,new_good=_candidate_score(paug,new)
        improved=(new_bad<old_bad) or (new_bad==old_bad and new_score<old_score)
        if not improved:
            failures.append({"candidate":source,"reason":"no_strict_lexicographic_quality_improvement",
                "old_bad":old_bad,"new_bad":new_bad,"old_log_deficit":old_score,
                "new_log_deficit":new_score})
            continue
        result={"candidate_source":source,"apex_point_f32":apex32,"replacement_tetrahedra":new,
            "old_bad":old_bad,"new_bad":new_bad,"old_log_deficit":old_score,
            "new_log_deficit":new_score,"continuous_log_deficit":continuous_score,
            "visibility_height_normalized":visibility_height,
            "minimum_visibility_distance_normalized":float(np.min(continuous_d)),
            "slsqp_runs":optimizations,"construction_margin":0.0,
            "normalized_cavity_diameter_m":diameter,
            "minimum_native_quality":float(np.min(nq)),"minimum_det6":float(np.min(nd)),
            "exact_positive_new_cell_count":len(exact),
            "exact_det6_delta_integer":str(sum(exact)-sum(K734.exact_det_values(p,old))),
            "oriented_chain_matches_old":True,"exact_volume_matches_old":True,
            "quality_gate_verified_after_F32_rounding":bool(np.all(new_good)),
            "volume_gate_verified_after_F32_rounding":bool(np.all(nd/6.0>VOLUME_GATE)),
            "partial_quality_improvement":bool(new_bad>0),
            "replacement_tetrahedra_local":new,"elapsed_s":time.monotonic()-started}
        if best is None or (new_bad,new_score,-float(np.min(continuous_d)),source)<(
                best["new_bad"],best["new_log_deficit"],-best["minimum_visibility_distance_normalized"],best["candidate_source"]):
            best=result
    if best is None:
        return {"status":"REJECT_NO_F32_PARTIAL_QUALITY_IMPROVEMENT","old_bad":old_bad,
            "old_log_deficit":old_score,"visibility_height_normalized":visibility_height,
            "slsqp_runs":optimizations,"candidate_rejections":failures,
            "elapsed_s":time.monotonic()-started}
    best["status"]="PASS_INTERIOR_STEINER_PARTIAL_QUALITY_CONE"
    best["candidate_rejections"]=failures
    best["physical_admission"]=False;best["native_steps"]=0
    return best

def partial_sliver_tetra_fixture():
    # A tetrahedral boundary has one permanently skinny face (ABC), but a
    # sufficiently central apex improves the other cone cells. The old apex is
    # deliberately near a boundary corner, giving a 2-to-1 native-bad reduction.
    base=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.5,0.8660254,0.],
                  [0.5,0.28867513,0.81649658]],np.float32)
    p=np.vstack((base,np.asarray([0.5,1.0e-6,0.],np.float32)))
    outer=M.orient(base,np.asarray([[0,1,2,3]],np.int64))
    outer_faces=K734.oriented_boundary(outer)
    faces=[]
    for face in outer_faces:
        if np.all(base[np.asarray(face),2]==0):
            a,b,c=map(int,face); e=len(base)
            faces.extend(((a,b,e),(b,c,e),(c,a,e)))
        else:
            faces.append(tuple(map(int,face)))
    faces=np.asarray(faces,np.int64)
    apex0=np.asarray([0.2747143805,0.4389857948,0.0520610213],np.float32)
    oldpoints=np.vstack((p,apex0))
    old=M.orient(oldpoints,np.column_stack((np.full(len(faces),len(p),np.int64),faces)))
    old_bad,old_score,_,_,_= _candidate_score(oldpoints,old)
    result=propose_partial_quality_kernel_cone(oldpoints,faces,old,np.full(len(old),17,np.int32),time_limit_s=5.0)
    if result["status"]!="PASS_INTERIOR_STEINER_PARTIAL_QUALITY_CONE":
        raise AssertionError("partial cone fixture failed to find a lexicographic improvement: "+repr(result))
    if not (old_bad==2 and result["new_bad"]==1):
        raise AssertionError(f"fixture must demonstrate exact partial 2-to-1 progress, got {old_bad}->{result.get('new_bad')}")
    if result["quality_gate_verified_after_F32_rounding"] is not False:
        raise AssertionError("fixture did not preserve one honest native-bad cone tet")
    paug=np.vstack((oldpoints,result["apex_point_f32"]))
    new=np.asarray(result["replacement_tetrahedra_local"],np.int64)
    exact=K734.exact_det_values(paug,new)
    if not exact or min(exact)<=0:raise AssertionError("partial fixture cone lacks exact-positive tets")
    if M.chain(old)!=M.chain(new) or M.certify(paug,old,new)!=(True,"PASS"):
        raise AssertionError("partial fixture changed chain or exact volume")
    new_bad,new_score,nd,nq,good=_candidate_score(paug,new)
    if new_bad!=1 or not new_score<old_score:
        raise AssertionError("partial fixture did not strictly reduce its log-deficit after bad-count improvement")
    return {"status":"PASS_EXACT_POSITIVE_PARTIAL_CONE_2_TO_1",
        "old_bad":old_bad,"new_bad":new_bad,"old_log_deficit":old_score,"new_log_deficit":new_score,
        "exact_positive_tet_count":len(exact),"minimum_exact_det6_integer":str(min(exact)),
        "oriented_chain_equal":True,"exact_signed_volume_equal":True,
        "fixed_boundary_faces":len(faces),"native_gate_all_new":False,
        "native_admission":False,"physical_admission":False,"native_steps":0,
        "solver":{"candidate_source":result["candidate_source"],
            "visibility_height_normalized":result["visibility_height_normalized"],
            "minimum_native_quality":result["minimum_native_quality"],
            "minimum_det6":result["minimum_det6"],"slsqp_runs":result["slsqp_runs"]}}

if __name__=="__main__":
    import json
    print(json.dumps(partial_sliver_tetra_fixture(),indent=2,sort_keys=True))
