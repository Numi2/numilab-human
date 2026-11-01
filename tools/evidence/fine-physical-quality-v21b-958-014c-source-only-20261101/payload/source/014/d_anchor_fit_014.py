#!/usr/bin/env python3
"""014 D-only feasible 3D endpoint-anchor fit; strict gates remain with caller."""
from itertools import product
import numpy as np
from scipy.optimize import minimize

NATIVE_Q=1e-5
MIN_CELL_VOLUME_M3=1e-18
SIGN_ROWS=np.asarray(list(product((-1.0,1.0),repeat=3)),np.float64)

def source_winding_system(source_points,old_faces,new_faces,keep,x0,scale):
    points=np.asarray(source_points,np.float32)
    oldf=np.asarray(old_faces,np.int64);newf=np.asarray(new_faces,np.int64)
    if oldf.ndim!=2 or newf.shape!=oldf.shape or oldf.shape[1]!=3:
        raise ValueError("malformed source-face fit domain")
    if len(oldf)==0: raise ValueError("empty surviving source-face domain")
    xyz0=points[oldf].astype(np.float64)
    oldn=np.cross(xyz0[:,1]-xyz0[:,0],xyz0[:,2]-xyz0[:,0])
    norm=np.linalg.norm(oldn,axis=1)
    if not np.isfinite(norm).all() or np.any(norm<=0):
        raise ValueError("degenerate old source face in anchor fit")
    oldn=oldn/norm[:,None]
    tri=(points[newf].astype(np.float64)-np.asarray(x0,np.float64))/float(scale)
    offset=np.empty(len(newf),np.float64)
    coeff=np.empty((len(newf),3),np.float64)
    for i,row in enumerate(newf):
        moving=np.flatnonzero(row==int(keep))
        if len(moving)!=1:
            raise ValueError("surviving quotient source face must contain exactly one moved keep vertex")
        j=int(moving[0]);base=tri[i].copy();base[j]=0.0
        n0=np.cross(base[1]-base[0],base[2]-base[0])
        offset[i]=float(np.dot(oldn[i],n0))
        for axis in range(3):
            moved=base.copy();moved[j,axis]=1.0
            n1=np.cross(moved[1]-moved[0],moved[2]-moved[0])
            coeff[i,axis]=float(np.dot(oldn[i],n1-n0))
    if not np.isfinite(offset).all() or not np.isfinite(coeff).all():
        raise ValueError("nonfinite source winding constraints")
    return offset,coeff

def l1_constraint_rows(cap_m,scale):
    return np.full(8,float(cap_m)/float(scale),np.float64),SIGN_ROWS.copy()

def fit_single_D_3d(fit_owner,x_keep,x_drop,triangles,old_det6,old_exact_det6,
                    regional_det6,source_points,source_old_faces,source_new_faces,
                    keep_source_row,*,cap_m=250e-6,ulp_radius=2):
    """Minimize anchor displacement subject to the unchanged 795 quality/volume gates.

    Optimization coordinates are normalized displacements from the kept F32 point.
    The eight sign halfspaces encode the exact L1 construction cap; a separate
    sphere constrains the removed endpoint distance. Source-face winding is linear
    in the moving point. Final candidate acceptance uses the exact existing F32
    quality, positivity, source-facet and volume checks.
    """
    engine=fit_owner.E;Q=fit_owner.Q;J=fit_owner.J
    x0=np.asarray(x_keep,np.float32).reshape(3)
    xd=np.asarray(x_drop,np.float32).reshape(3)
    tri0=np.asarray(triangles,np.float32).reshape(-1,3,3)
    cap=float(cap_m)
    src=np.asarray(source_points,np.float32)
    oldf=np.asarray(source_old_faces,np.int64);newf=np.asarray(source_new_faces,np.int64)
    if (not np.isfinite(x0).all() or not np.isfinite(xd).all() or cap<=0 or
        not len(tri0) or not np.isfinite(tri0).all()):
        return {"status":"REJECT_BAD_INPUT"}
    edge=xd.astype(np.float64)-x0.astype(np.float64)
    lengths=np.concatenate([np.linalg.norm(tri0[:,i].astype(np.float64)-tri0[:,j].astype(np.float64),axis=1)
        for i,j in ((0,1),(1,2),(2,0))])
    scale=max(cap,float(np.max(lengths)),
        float(np.max(np.linalg.norm(tri0.astype(np.float64)-x0.astype(np.float64),axis=2))),
        float(np.linalg.norm(edge)),1e-12)
    tri=(tri0.astype(np.float64)-x0.astype(np.float64))/scale
    a=tri[:,0]
    n=np.cross(tri[:,2]-tri[:,0],tri[:,1]-tri[:,0])
    area=np.linalg.norm(n,axis=1)
    if np.any(area<=64*np.finfo(float).eps):return {"status":"REJECT_DEGENERATE_TET_FACE"}
    fl=np.max(np.stack([np.linalg.norm(tri[:,0]-tri[:,1],axis=1),
        np.linalg.norm(tri[:,1]-tri[:,2],axis=1),np.linalg.norm(tri[:,2]-tri[:,0],axis=1)]),axis=0)**3
    vf=6*MIN_CELL_VOLUME_M3/scale**3
    qm=NATIVE_Q*1.01
    h0=Q._normalized_margin(np.zeros(3),n,a,fl,tri,vf,qm,area)
    if not np.isfinite(h0):return {"status":"REJECT_INITIAL_SLACK"}
    volume_budget=min(.01*(float(old_det6)/scale**3),
                      1e-4*(float(regional_det6)/scale**3))
    if volume_budget<=0:return {"status":"REJECT_VOLUME_BUDGET"}
    old_norm=float(old_det6)/scale**3
    grad=np.sum(n,axis=0)
    def volume_delta(z):
        return float(np.sum(np.einsum("ij,ij->i",n,np.asarray(z)[None,:]-a))-old_norm)
    def quality(z):return Q._quality_constraints(np.asarray(z,np.float64),0.0,n,a,fl,tri,vf,qm,area)
    def quality_jac(z):return Q._quality_jacobian(np.asarray(z,np.float64),0.0,n,a,fl,tri,vf,qm,area)[:,:3]
    l1_rhs,l1_rows=l1_constraint_rows(cap,scale)
    drop_offset=edge/scale
    wind_offset,wind_coeff=source_winding_system(
        src,oldf,newf,int(keep_source_row),x0,scale)
    cons=[
      {"type":"ineq","fun":quality,"jac":quality_jac},
      {"type":"ineq","fun":lambda z:volume_budget-volume_delta(z),"jac":lambda z:-grad},
      {"type":"ineq","fun":lambda z:volume_budget+volume_delta(z),"jac":lambda z:grad},
      {"type":"ineq","fun":lambda z:l1_rhs-l1_rows@np.asarray(z),"jac":lambda z:-l1_rows},
      {"type":"ineq","fun":lambda z:(cap/scale)**2-float(np.dot(np.asarray(z)-drop_offset,np.asarray(z)-drop_offset)),
       "jac":lambda z:-2*(np.asarray(z)-drop_offset)},
      {"type":"ineq","fun":lambda z:wind_offset+wind_coeff@np.asarray(z),"jac":lambda z:wind_coeff},
    ]
    # The midpoint is a useful feasible seed when the old edge is longer than
    # the removed-endpoint sphere radius; the solver remains free in 3D.
    z0=.5*drop_offset
    if np.any(quality(z0)<-1e-7):z0=np.zeros(3)
    res=minimize(lambda z:.5*float(np.dot(z,z)),z0,jac=lambda z:np.asarray(z,np.float64),
        method="SLSQP",bounds=[(-cap/scale,cap/scale)]*3,constraints=cons,
        options={"maxiter":500,"ftol":1e-13,"disp":False})
    if res.x is None or not np.isfinite(res.x).all():
        return {"status":"REJECT_SOLVER","message":str(res.message)}
    x64=x0.astype(np.float64)+scale*np.asarray(res.x,np.float64)
    wins=[]
    for cand in fit_owner.neighbors(np.asarray(x64,np.float32),int(ulp_radius)):
        cand=np.asarray(cand,np.float32).reshape(3)
        disp_vec=cand.astype(np.float64)-x0.astype(np.float64)
        l1=float(np.abs(disp_vec).sum())
        removed=float(np.linalg.norm(cand.astype(np.float64)-xd.astype(np.float64)))
        if l1>cap or removed>cap:continue
        if np.any(wind_offset+wind_coeff@((cand.astype(np.float64)-x0.astype(np.float64))/scale)<0):continue
        points=src.copy();points[int(keep_source_row)]=cand
        ox=points[oldf].astype(np.float64);nx=points[newf].astype(np.float64)
        oldn=np.cross(ox[:,1]-ox[:,0],ox[:,2]-ox[:,0])
        newn=np.cross(nx[:,1]-nx[:,0],nx[:,2]-nx[:,0])
        den=np.linalg.norm(oldn,axis=1)*np.linalg.norm(newn,axis=1)
        if np.any(den<=0) or np.any(np.einsum("ij,ij->i",oldn,newn)<.9*den):continue
        try: source_q=np.asarray(J.score(points,newf),np.float64)
        except Exception: continue
        if not np.isfinite(source_q).all() or np.any(source_q<J.GATE):continue
        p=np.vstack((cand,tri0.reshape(-1,3))).astype(np.float32)
        nt=len(tri0)
        t=np.column_stack((np.zeros(nt,np.int64),np.arange(1,3*nt+1,dtype=np.int64).reshape(-1,3)))
        det,q=engine.measurements(p,t)
        if np.any(det<=0) or np.any(det/6<=MIN_CELL_VOLUME_M3):continue
        if np.any(q<qm) or not np.all(engine.M.passes(det,q)):continue
        if not fit_owner.exact_pos(p,t):continue
        exact_delta=fit_owner.exact_sum(p,t)-int(old_exact_det6)
        delta_m3=exact_delta*fit_owner.DET_TO_M3
        if abs(delta_m3)>volume_budget*scale**3/6:continue
        wins.append((float(np.min(q)),-float(np.dot(disp_vec,disp_vec)),cand.copy(),
            l1,removed,float(np.linalg.norm(disp_vec)),int(exact_delta),float(delta_m3),float(np.min(source_q))))
    if not wins:
        return {"status":"REJECT_NO_F32_CANDIDATE","solver_success":bool(res.success),
            "solver_message":str(res.message),"continuous_anchor_world_m":x64.tolist(),
            "continuous_l1_m":float(np.abs(x64-x0.astype(np.float64)).sum()),
            "continuous_removed_endpoint_m":float(np.linalg.norm(x64-xd.astype(np.float64))),
            "minimum_linear_quality_slack":float(np.min(quality(res.x))),
            "minimum_winding_slack":float(np.min(wind_offset+wind_coeff@res.x))}
    best=max(wins,key=lambda v:(v[1],v[0]))
    best_z=(best[2].astype(np.float64)-x0.astype(np.float64))/scale
    return {"status":"PASS_D_3D_FEASIBLE_EDGE_FIT","role":"D",
        "anchor_point_f32":best[2],"anchor_l1_m":best[3],"drop_to_anchor_m":best[4],
        "l1_cap_slack_m":float(cap-best[3]),"removed_endpoint_cap_slack_m":float(cap-best[4]),
        "minimum_linear_winding_slack":float(np.min(wind_offset+wind_coeff@best_z)),
        "anchor_displacement_m":best[5],"exact_det_delta_integer":str(best[6]),
        "delta_volume_m3":best[7],"minimum_new_native_q":best[0],
        "minimum_new_source_face_q":best[8],"solver_success":bool(res.success),
        "solver_message":str(res.message),"construction_q_margin_multiplier":1.01,
        "native_q_gate_unchanged":NATIVE_Q,"minimum_cell_volume_m3_unchanged":MIN_CELL_VOLUME_M3,
        "move_cap_m":cap,"anchor_fit_method":"3D minimum-displacement with L1 sign, removed-endpoint sphere and source-winding constraints",
        "candidate_count":len(wins),"continuous_anchor_world_m":x64.tolist()}

def _csr_rows(offsets,rows,vertex):
    i=int(vertex);return np.asarray(rows[int(offsets[i]):int(offsets[i+1])],np.int64)

def fit_edge(cache,drop,keep,fit_owner,max_l1_m):
    d,k=int(drop),int(keep);xk=cache.sp[k].astype(np.float32);xd=cache.sp[d].astype(np.float32)
    triangles=[];old_det6=0.0;regional_det6=0.0;old_exact=0
    try:
        for vr in cache.targets:
            smap=cache.source_maps[vr];dv=int(smap[d]);kv=int(smap[k])
            if dv<0 or kv<0:raise ValueError(vr+" missing exact source-volume endpoints")
            q=cache.volumes[vr];t=q["tets"];p=q["points"]
            drop_cells=_csr_rows(q["cell_offsets"],q["cell_rows"],dv)
            keep_cells=_csr_rows(q["cell_offsets"],q["cell_rows"],kv)
            star=np.union1d(drop_cells,keep_cells).astype(np.int64)
            old=t[star];mapped=old.copy();mapped[mapped==dv]=kv
            ss=np.sort(mapped,axis=1);valid=~np.any(ss[:,1:]==ss[:,:-1],axis=1);new=mapped[valid]
            if not len(new) or np.any(~np.any(new==kv,axis=1)):
                raise ValueError(vr+" edge quotient does not produce complete keep-anchor fan")
            ordered=fit_owner.anchor_order(new,kv)
            triangles.append(p[ordered[:,1:]])
            old_det6+=float(np.sum(q["det"][star],dtype=np.float64))
            regional_det6+=float(np.sum(q["det"],dtype=np.float64))
            old_exact+=fit_owner.exact_sum(p,old)
        fr=cache._surface_star(d,k);oldall=cache.sf[fr].copy()
        mapped=oldall.copy();mapped[mapped==d]=k
        deg=(mapped[:,0]==mapped[:,1])|(mapped[:,1]==mapped[:,2])|(mapped[:,2]==mapped[:,0])
        oldfaces=oldall[~deg];newfaces=mapped[~deg]
        result=fit_single_D_3d(fit_owner,xk,xd,np.concatenate(triangles,axis=0),
            old_det6,old_exact,regional_det6,cache.sp,oldfaces,newfaces,k,
            cap_m=float(max_l1_m),ulp_radius=2)
        if result.get("status")!="PASS_D_3D_FEASIBLE_EDGE_FIT":
            return result
        result["edge_rows"]=[d,k]
        return result
    except Exception as exc:
        return {"status":"REJECT_D_3D_FIT_PREP","reason":str(exc)[:300]}

