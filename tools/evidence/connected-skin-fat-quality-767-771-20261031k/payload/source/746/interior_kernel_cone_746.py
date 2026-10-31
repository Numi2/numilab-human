#!/usr/bin/env python3
"""Interior Steiner apex from a visibility-kernel LP and quality-margin SLSQP."""
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
_spec=importlib.util.spec_from_file_location("interior_steiner_base734",OLD_PATH)
K734=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=K734;_spec.loader.exec_module(K734)
M=K734.M
QUALITY_GATE=1.0e-5
VOLUME_GATE=1.0e-18
CONSTRUCTION_MARGIN=1.0e-3
QUALITY_PATH=Path(M.__file__).resolve()
if sha(QUALITY_PATH)!=OWNER_SHA:
    raise RuntimeError("pinned 198 quality/exact owner changed")

def _quality_constraints(x,h,n,a,face_len3,tri,vol_floor,qmargin,area):
    d=np.einsum("ij,ij->i",n,x[None,:]-a)
    vis=d-h*area
    face=d-qmargin*face_len3-h*area
    delta=x[None,None,:]-tri
    length=np.linalg.norm(delta,axis=2)
    edge=d[:,None]-qmargin*length**3-h*area[:,None]
    vol=d-vol_floor-h*area
    return np.concatenate((vis,face,edge.reshape(-1),vol))

def _quality_jacobian(x,h,n,a,face_len3,tri,vol_floor,qmargin,area):
    d=len(n);vis=np.column_stack((n,-area))
    face=np.column_stack((n,-area))
    delta=x[None,None,:]-tri
    length=np.linalg.norm(delta,axis=2)
    edge_x=n[:,None,:]-3.0*qmargin*length[:,:,None]*delta
    edge_h=np.broadcast_to(-area[:,None,None],(d,3,1))
    edge=np.concatenate((edge_x,edge_h),axis=2).reshape((-1,4))
    vol=np.column_stack((n,-area))
    return np.vstack((vis,face,edge,vol))

def _normalized_margin(x,n,a,face_len3,tri,vol_floor,qmargin,area):
    d=np.einsum("ij,ij->i",n,x[None,:]-a)
    vals=[d/area,(d-qmargin*face_len3)/area,(d-vol_floor)/area]
    lengths=np.linalg.norm(x[None,None,:]-tri,axis=2)
    vals.append(((d[:,None]-qmargin*lengths**3)/area[:,None]).reshape(-1))
    return float(np.min(np.concatenate(vals)))

def propose_interior_kernel_cone(points,oriented_boundary_faces,old_tetrahedra,
                                 old_material_indices,*,time_limit_s=2.0,
                                 construction_margin=CONSTRUCTION_MARGIN,maxiter=250):
    """Return an appended F32 apex and its exact-positive oriented boundary cone."""
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
    if old_bad<=0:return {"status":"REJECT_CAVITY_HAS_NO_BAD_CELLS","old_bad":old_bad}
    bm=float(construction_margin)
    if not np.isfinite(bm) or bm<0 or bm>0.05:
        return {"status":"REJECT_INVALID_CONSTRUCTION_MARGIN"}
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
    n=np.cross(tri[:,2]-tri[:,0],tri[:,1]-tri[:,0])
    area=np.linalg.norm(n,axis=1)
    if np.any(~np.isfinite(area)) or np.any(area<=64*np.finfo(float).eps):
        return {"status":"REJECT_DEGENERATE_BOUNDARY_FACE"}
    nh=n/area[:,None]
    low=tri.reshape(-1,3).min(axis=0);high=tri.reshape(-1,3).max(axis=0)
    A=np.column_stack((-nh,np.ones(len(nh))))
    b=-np.einsum("ij,ij->i",nh,a)
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
    xlp=np.asarray(lp.x[:3],np.float64);visibility_height=float(lp.x[3])
    if visibility_height<=1e-10:
        return {"status":"REJECT_NO_STRICT_KERNEL_INTERIOR","visibility_height_normalized":visibility_height,
                "lp_message":str(lp.message)}
    qmargin=QUALITY_GATE*(1.0+bm)
    vol_floor=6.0*VOLUME_GATE*(1.0+bm)/(diameter**3)
    face_len=np.stack((np.linalg.norm(tri[:,0]-tri[:,1],axis=1),
                       np.linalg.norm(tri[:,1]-tri[:,2],axis=1),
                       np.linalg.norm(tri[:,2]-tri[:,0],axis=1)),axis=1)
    face_len3=np.max(face_len,axis=1)**3
    h0=_normalized_margin(xlp,n,a,face_len3,tri,vol_floor,qmargin,area)
    z0=np.r_[xlp,h0]
    bounds=[(float(low[i]),float(high[i])) for i in range(3)]+[(None,None)]
    remaining=float(time_limit_s)-(time.monotonic()-started)
    if remaining<=0:return {"status":"REJECT_TIME_BUDGET_BEFORE_QUALITY_SLSQP",
                            "visibility_height_normalized":visibility_height}
    slsqp=None
    try:
        slsqp=minimize(lambda z:-float(z[3]),z0,
            jac=lambda z:np.asarray((0.,0.,0.,-1.)),
            method="SLSQP",
            bounds=bounds,
            constraints=({"type":"ineq",
                "fun":lambda z:_quality_constraints(z[:3],z[3],n,a,face_len3,tri,vol_floor,qmargin,area),
                "jac":lambda z:_quality_jacobian(z[:3],z[3],n,a,face_len3,tri,vol_floor,qmargin,area)},),
            options={"maxiter":int(maxiter),"ftol":1e-12,"disp":False})
    except Exception as exc:
        return {"status":"REJECT_QUALITY_SLSQP_EXCEPTION","error":repr(exc),
                "visibility_height_normalized":visibility_height}
    candidates=[]
    if slsqp is not None and getattr(slsqp,"x",None) is not None and np.isfinite(slsqp.x).all():
        candidates.append(("SLSQP",np.asarray(slsqp.x[:3],np.float64),float(slsqp.x[3])))
    candidates.append(("KERNEL_LP_START",xlp,h0))
    failures=[]
    best=None
    for source,x,h in candidates:
        margin=_normalized_margin(x,n,a,face_len3,tri,vol_floor,qmargin,area)
        if not np.isfinite(margin) or margin<=0:
            failures.append({"candidate":source,"continuous_quality_margin_normalized":margin})
            continue
        apex64=origin+diameter*x
        apex32=np.asarray(apex64,np.float32)
        if not np.isfinite(apex32).all() or any(np.array_equal(apex32,v) for v in p):
            failures.append({"candidate":source,"reason":"rounded_apex_nonfinite_or_duplicate"})
            continue
        paug=np.vstack((p,apex32))
        new=np.column_stack((np.full(len(f),len(p),np.int64),f))
        exact=K734.exact_det_values(paug,new)
        if not exact or min(exact)<=0:
            failures.append({"candidate":source,"reason":"exact_nonpositive_cone_tet"})
            continue
        if M.chain(old)!=M.chain(new):
            failures.append({"candidate":source,"reason":"oriented_chain_mismatch"})
            continue
        ok,why2=M.certify(paug,old,new)
        if not ok:
            failures.append({"candidate":source,"reason":"exact_cavity_certificate","detail":why2})
            continue
        nd,nq=M.measurements(paug,new)
        new_good=np.asarray(M.passes(nd,nq),bool)
        if not np.all(new_good):
            failures.append({"candidate":source,"reason":"rounded_new_cone_fails_native_owner",
                             "bad_new_cells":int(np.count_nonzero(~new_good)),
                             "minimum_normalized_quality":float(np.min(nq)),
                             "minimum_det6":float(np.min(nd))})
            continue
        new_bad=int(np.count_nonzero(~new_good))
        if new_bad>=old_bad:
            failures.append({"candidate":source,"reason":"no_strict_bad_count_drop"})
            continue
        score=float(K734.Q._log_deficit(nd,nq))
        if best is None or (margin,-score,source)<(best["continuous_quality_margin_normalized"],-best["new_log_deficit"],best["candidate_source"]):
            best={"candidate_source":source,"apex_point_f32":apex32,"replacement_tetrahedra":new,
                  "old_bad":old_bad,"new_bad":new_bad,"old_log_deficit":float(base["old_log_deficit"]),
                  "new_log_deficit":score,"visibility_height_normalized":visibility_height,
                  "continuous_quality_margin_normalized":margin,"slsqp_success":bool(slsqp.success),
                  "slsqp_status":int(slsqp.status),"slsqp_message":str(slsqp.message),
                  "slsqp_iterations":int(getattr(slsqp,"nit",0)),
                  "construction_margin":bm,"normalized_cavity_diameter_m":diameter,
                  "minimum_native_quality":float(np.min(nq)),"minimum_det6":float(np.min(nd)),
                  "exact_positive_new_cell_count":len(exact),
                  "exact_det6_delta_integer":str(sum(exact)-sum(K734.exact_det_values(p,old))),
                  "oriented_chain_matches_old":True,"exact_volume_matches_old":True,
                  "quality_gate_verified_after_F32_rounding":True,"volume_gate_verified_after_F32_rounding":True,
                  "replacement_tetrahedra_local":new,"elapsed_s":time.monotonic()-started}
    if best is None:
        return {"status":"REJECT_NO_F32_QUALITY_FEASIBLE_STEINER_APEX",
                "old_bad":old_bad,"visibility_height_normalized":visibility_height,
                "slsqp_success":bool(slsqp.success),"slsqp_status":int(slsqp.status),
                "slsqp_message":str(slsqp.message),"slsqp_iterations":int(getattr(slsqp,"nit",0)),
                "candidate_rejections":failures,"construction_margin":bm,
                "normalized_cavity_diameter_m":diameter,"elapsed_s":time.monotonic()-started}
    best["status"]="PASS_INTERIOR_STEINER_KERNEL_CONE"
    best["candidate_rejections"]=failures
    best["physical_admission"]=False;best["native_steps"]=0
    return best

def sliver_tetra_fixture():
    root3=np.sqrt(3.0);root23=np.sqrt(2.0/3.0)
    p=np.asarray([[0.,0.,0.],[1.,0.,0.],[.5,root3/2,0.],
                  [.5,root3/6,root23]],np.float32)
    outer=M.orient(p,np.asarray([[0,1,2,3]],np.int64))
    faces=K734.oriented_boundary(outer)
    center=np.asarray([.5,root3/6,root23/4],np.float32)
    near=np.asarray([.5,root3/6,1.0e-5],np.float32)
    oldpoints=np.vstack((p,near))
    old=M.orient(oldpoints,np.column_stack((np.full(len(faces),4,np.int64),faces)))
    result=propose_interior_kernel_cone(oldpoints,faces,old,np.full(len(old),17,np.int32),time_limit_s=5.0)
    old_det,old_q=M.measurements(oldpoints,old)
    old_bad=int(np.count_nonzero(~M.passes(old_det,old_q)))
    face_edge=np.stack((np.linalg.norm(p[faces[:,0]]-p[faces[:,1]],axis=1),
                        np.linalg.norm(p[faces[:,1]]-p[faces[:,2]],axis=1),
                        np.linalg.norm(p[faces[:,2]]-p[faces[:,0]],axis=1)),axis=1)
    assert np.all(np.max(face_edge,axis=1)**3 > 0)
    assert old_bad>0
    assert result["status"]=="PASS_INTERIOR_STEINER_KERNEL_CONE",result
    assert result["old_bad"]==old_bad and result["new_bad"]==0
    paug=np.vstack((oldpoints,result["apex_point_f32"]))
    assert M.chain(old)==M.chain(result["replacement_tetrahedra_local"])
    assert M.certify(paug,old,result["replacement_tetrahedra_local"])==(True,"PASS")
    assert np.all(M.passes(*M.measurements(paug,result["replacement_tetrahedra_local"])))
    return {"status":"PASS_INTERIOR_STEINER_BAD_SLIVER_FIXTURE","old_bad":old_bad,
            "new_bad":result["new_bad"],"old_tets":len(old),"new_tets":len(result["replacement_tetrahedra_local"]),
            "fixed_boundary_faces":len(faces),"visibility_height_normalized":result["visibility_height_normalized"],
            "quality_margin_normalized":result["continuous_quality_margin_normalized"],
            "apex_point_f32":result["apex_point_f32"].tolist(),"exact_volume_equal":True,
            "all_new_native_pass":True,"construction_margin":result["construction_margin"],
            "physical_admission":False,"native_steps":0,"solver_result":{k:v for k,v in result.items() if k not in ("replacement_tetrahedra","replacement_tetrahedra_local","apex_point_f32")}}

