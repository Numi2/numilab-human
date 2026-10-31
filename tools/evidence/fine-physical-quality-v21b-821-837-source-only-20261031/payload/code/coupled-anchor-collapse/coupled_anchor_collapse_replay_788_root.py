#!/usr/bin/env python3
"""Coupled short-edge quotient with an optimized shared I anchor.

Construction only. The CLI runs a synthetic fixture and never loads scene arrays.
A real candidate requires the existing exact 775 changed-vs-complete O/I/D +
wall + all185 audit and witness pruning.
"""
from pathlib import Path
from itertools import combinations, product
from collections import Counter
import argparse, hashlib, importlib.util, json, sys, time
import numpy as np
from scipy.optimize import minimize

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PINS={
 "702":("batch_connected_wide_graph_quality_702_root.py","37f5eff7f7fd8538644bb6c5ec60d2f9df609e4b3d6ba46b2ccee3fbeab3fab5"),
 "470":("transfer_surface_contractions_cross_tissue_470_root.py","e0c0b5ff0f70966431b920f2cb3e469c0bf1d54518068d7ce6e561e5ba16b2e3"),
 "198":("fixed_boundary_volume_flip_198_root.py","650794951cbcf338de9cde691a8f43377b9d023ccea78b876e87c847880243e0"),
 "746":("quality-kernel-cone-746-root/interior_kernel_cone_746.py","d390a196d8b38e480e875414e9bc40e0fb683ecc0115db2b13660909b03520e5"),
 "272":("audit_joint_native_apex_272_root.py","5df00334a92311b4cc0dd24dc7f7d1127f55fd354777ff888eabc607484c8e82"),
 "024":("fixed-boundary-quality-topology-196/prepare_current_earclip_adapter_024.py","21ab4acdabef69a688efcceb568106905c32a014f18e7c307edb50a7cf27ef2c"),
 "197":("fixed-boundary-quality-topology-196/run_fixed_boundary_dp_batch_197.py","d4fcc4a2693201f7ad3934b62b277a65e6f021a6662182f4ab67e85db4d96c15"),
 "775":("combined-surface-diff-audit-270/audit_combined_bounded_point_surface_diff_775_root.py","82627919f81528c47e8269b37dee25099f6c2d29f3d2d8858000c50a14956a85"),
}
NATIVE_Q=1.0e-5
MIN_CELL_VOLUME_M3=1.0e-18
CONSTRUCTION_Q_MARGIN=1.01
PAIR_VOLUME_SCALE=2.0**(-447)/6.0

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def import_owner(name,alias):
    rel,expected=PINS[name];path=ROOT/rel;got=sha(path)
    if got!=expected:raise RuntimeError(f"pinned source {name} changed: {got}")
    spec=importlib.util.spec_from_file_location(alias,path)
    if spec is None or spec.loader is None:raise RuntimeError(f"cannot import {path}")
    mod=importlib.util.module_from_spec(spec);sys.modules[alias]=mod;spec.loader.exec_module(mod);return mod
T=import_owner("470","coupled_anchor_T470");M=T.m
Q=import_owner("746","coupled_anchor_Q746")
J=import_owner("272","coupled_anchor_J272")
A=import_owner("024","coupled_anchor_A024")
B=import_owner("197","coupled_anchor_B197")

def f32key(x):return np.asarray(x,np.float32).reshape(3).view(np.uint32).tobytes()
def cellkey(t):return tuple(sorted(map(int,t)))
def boundary(t):return T.boundary(np.asarray(t,np.int64))
def link(simplices,verts):
    wanted=set(map(int,verts));out=set()
    for simplex in np.asarray(simplices,np.int64):
        ss=set(map(int,simplex))
        if wanted.issubset(ss):
            rest=sorted(ss-wanted)
            for k in range(len(rest)+1):out.update(combinations(rest,k))
    return out
def surface_link(f,keep,drop):
    f=np.asarray(f,np.int64);kr=np.flatnonzero(np.any(f==keep,axis=1));dr=np.flatnonzero(np.any(f==drop,axis=1))
    er=np.intersect1d(kr,dr)
    if len(er)!=2:return False,{"reason":"surface_edge_incidence","rows":er.tolist()}
    lk=link(f[kr],(keep,))&link(f[dr],(drop,));le=link(f[er],(keep,drop))
    return lk==le,{"reason":"PASS" if lk==le else "surface_link","edge_face_rows":er.tolist()}
def volume_link(t,keep,drop,kr,dr):
    er=np.intersect1d(kr,dr)
    return bool(len(er) and (link(t[kr],(keep,))&link(t[dr],(drop,)))==link(t[er],(keep,drop)))
def incident_cache(t,npoints):
    flat=np.asarray(t,np.int64).ravel();order=np.argsort(flat,kind="stable")
    return order,np.r_[0,np.cumsum(np.bincount(flat,minlength=npoints))]
def incident_rows(cache,v):
    order,off=cache;return order[off[int(v)]:off[int(v)+1]]//4
def exact_det_sum(p,t):
    t=np.asarray(t,np.int64)
    if not t.size:return 0
    nodes=np.unique(t.ravel());lk={int(v):i for i,v in enumerate(nodes)}
    ip=M.integer_points(np.asarray(p,np.float32)[nodes])
    return sum(M.exact_det(ip,[lk[int(v)] for v in row]) for row in t)
def exact_positive(p,t):
    t=np.asarray(t,np.int64)
    if not t.size:return True
    nodes=np.unique(t.ravel());lk={int(v):i for i,v in enumerate(nodes)}
    ip=M.integer_points(np.asarray(p,np.float32)[nodes])
    return all(M.exact_det(ip,[lk[int(v)] for v in row])>0 for row in t)
def measurements(p,t):return M.measurements(np.asarray(p,np.float32),np.asarray(t,np.int64))
def anchor_order(tets,anchor):
    out=np.asarray(tets,np.int64).copy()
    for i,t in enumerate(out):
        pos=np.flatnonzero(t==anchor)
        if len(pos)!=1:raise ValueError("replacement tet lacks retained anchor")
        j=int(pos[0])
        if j:
            out[i]=np.r_[t[j:],t[:j]]
            if j&1:out[i,2],out[i,3]=out[i,3],out[i,2]
    return out
def coord_crosswalk(surface_points,volume_points):
    lut={}
    for i,p in enumerate(np.asarray(volume_points,np.float32)):
        k=f32key(p)
        if k in lut:return None
        lut[k]=i
    return np.asarray([lut.get(f32key(p),-1) for p in np.asarray(surface_points,np.float32)],np.int64)
def f32_neighbors(x,radius):
    x=np.asarray(x,np.float32).reshape(3);axes=[]
    for v in x:
        vals=[v];lo=hi=v
        for _ in range(radius):
            lo=np.nextafter(lo,np.float32(-np.inf),dtype=np.float32)
            hi=np.nextafter(hi,np.float32(np.inf),dtype=np.float32)
            vals.extend((lo,hi))
        axes.append(vals)
    return [np.asarray(y,np.float32) for y in product(*axes)]
def source_q(points,faces):
    if not len(faces):return True,1.0
    q=np.asarray(J.score(np.asarray(points,np.float32),np.asarray(faces,np.int64)),np.float64)
    return bool(np.isfinite(q).all() and np.all(q>=J.GATE)),float(np.min(q))
def normals(points,faces):
    x=np.asarray(points,np.float32)[np.asarray(faces,np.int64)].astype(np.float64)
    return np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0])

def fit_common_anchor(x0,fixed_faces,old_det_sums,old_exact_sums,local_det_sums,regional_det_sums,
                      *,move_cap_m,q_margin_multiplier=CONSTRUCTION_Q_MARGIN,paired_volume=True,
                      ulp_radius=2,maxiter=350):
    """Fit one shared anchor over every surviving tet in both endpoint stars.

    fixed_faces[r] is N x 3 x 3, oriented for a positive tet [anchor,a,b,c].
    Signed volume changes are affine. Pair equality and 1%-local/1e-4-regional
    construction budgets are constraints, not native acceptance thresholds.
    """
    x0=np.asarray(x0,np.float32).reshape(3);cap=float(move_cap_m);roles=("Skin","Fat")
    if set(fixed_faces)!=set(roles) or not np.isfinite(x0).all() or cap<=0:return {"status":"REJECT_BAD_INPUT"}
    tris={r:np.asarray(fixed_faces[r],np.float64).reshape(-1,3,3) for r in roles}
    if any(not len(tris[r]) or not np.isfinite(tris[r]).all() for r in roles):return {"status":"REJECT_EMPTY_STAR"}
    lengths=[]
    for r in roles:
        tr=tris[r]
        for i,j in ((0,1),(1,2),(2,0)):lengths.extend(np.linalg.norm(tr[:,i]-tr[:,j],axis=1))
    scale=max(cap,float(np.max(lengths)),float(np.max(np.linalg.norm(np.concatenate(list(tris.values()))-x0,axis=2))),1e-12)
    data={}
    for r in roles:
        tr=(tris[r]-x0.astype(np.float64))/scale;a=tr[:,0]
        n=np.cross(tr[:,2]-tr[:,0],tr[:,1]-tr[:,0]);area=np.linalg.norm(n,axis=1)
        if np.any(area<=64*np.finfo(float).eps):return {"status":"REJECT_DEGENERATE_FACE","role":r}
        e=np.stack([np.linalg.norm(tr[:,0]-tr[:,1],axis=1),np.linalg.norm(tr[:,1]-tr[:,2],axis=1),np.linalg.norm(tr[:,2]-tr[:,0],axis=1)],axis=1)
        data[r]={"tri":tr,"a":a,"n":n,"area":area,"face_len3":np.max(e,axis=1)**3,
                 "oldsum":float(old_det_sums[r])/scale**3,"local":float(local_det_sums[r])/scale**3,
                 "regional":float(regional_det_sums[r])/scale**3,"volfloor":6*MIN_CELL_VOLUME_M3/scale**3}
    qm=NATIVE_Q*float(q_margin_multiplier);x0n=np.zeros(3)
    h0=min(Q._normalized_margin(x0n,data[r]["n"],data[r]["a"],data[r]["face_len3"],data[r]["tri"],data[r]["volfloor"],qm,data[r]["area"]) for r in roles)
    if not np.isfinite(h0):return {"status":"REJECT_INITIAL_SLACK"}
    budgets={r:min(.01*data[r]["local"],1e-4*data[r]["regional"]) for r in roles}
    if any(z<=0 for z in budgets.values()):return {"status":"REJECT_VOLUME_BUDGET"}
    def delta(r,x):
        d=data[r]
        return float(np.sum(np.einsum("ij,ij->i",d["n"],x[None,:]-d["a"]))-d["oldsum"])
    cons=[]
    for r in roles:
        d=data[r];g=np.sum(d["n"],axis=0)
        cons.append({"type":"ineq","fun":lambda z,d=d:Q._quality_constraints(z[:3],z[3],d["n"],d["a"],d["face_len3"],d["tri"],d["volfloor"],qm,d["area"]),
            "jac":lambda z,d=d:Q._quality_jacobian(z[:3],z[3],d["n"],d["a"],d["face_len3"],d["tri"],d["volfloor"],qm,d["area"])})
        cons.append({"type":"ineq","fun":lambda z,r=r:budgets[r]-delta(r,z[:3]),"jac":lambda z,g=g:np.r_[-g,0.]})
        cons.append({"type":"ineq","fun":lambda z,r=r:budgets[r]+delta(r,z[:3]),"jac":lambda z,g=g:np.r_[g,0.]})
    if paired_volume:
        g=np.sum(data["Skin"]["n"],axis=0)+np.sum(data["Fat"]["n"],axis=0)
        residual0=delta("Skin",x0n)+delta("Fat",x0n)
        # A shared planar interface can make paired volume invariant for every
        # anchor, yielding a redundant zero-gradient equality. Skip only that
        # proven-constant equation; exact F32 paired-volume equality remains final.
        if np.linalg.norm(g)<=1e-12:
            if abs(residual0)>1e-12:return {"status":"REJECT_INFEASIBLE_CONSTANT_PAIRED_VOLUME","residual":residual0}
        else:
            cons.append({"type":"eq","fun":lambda z:delta("Skin",z[:3])+delta("Fat",z[:3]),"jac":lambda z,g=g:np.r_[g,0.]})
    capn=cap/scale
    cons.append({"type":"ineq","fun":lambda z:capn**2-float(np.dot(z[:3],z[:3])),"jac":lambda z:np.r_[-2*z[:3],0.]})
    res=minimize(lambda z:-float(z[3]),np.r_[x0n,h0],jac=lambda z:np.array([0.,0.,0.,-1.]),
        method="SLSQP",bounds=[(-capn,capn)]*3+[(None,None)],constraints=cons,
        options={"maxiter":int(maxiter),"ftol":1e-12,"disp":False})
    if res.x is None or not np.isfinite(res.x).all():
        return {"status":"REJECT_SOLVER","solver_success":bool(res.success),"message":str(res.message)}
    x64=x0.astype(np.float64)+scale*np.asarray(res.x[:3],np.float64);winners=[]
    neigh=f32_neighbors(np.asarray(x64,np.float32),ulp_radius)
    for x32 in neigh:
        disp=float(np.linalg.norm(x32.astype(np.float64)-x0.astype(np.float64)))
        if disp>cap:continue
        per={};ok=True;minq=float("inf")
        for r in roles:
            tr=np.asarray(fixed_faces[r],np.float32);n=len(tr)
            p=np.vstack((x32,tr.reshape(-1,3))).astype(np.float32)
            t=np.column_stack((np.zeros(n,np.int64),np.arange(1,3*n+1,dtype=np.int64).reshape(-1,3)))
            det,q=measurements(p,t)
            if np.any(det<=0) or not np.all(M.passes(det,q)) or not exact_positive(p,t):ok=False;break
            delta_exact=int(exact_det_sum(p,t))-int(old_exact_sums[r])
            dm=delta_exact*PAIR_VOLUME_SCALE
            lim=min(.01*float(local_det_sums[r]),1e-4*float(regional_det_sums[r]))/6.0
            if abs(dm)>lim:ok=False;break
            minq=min(minq,float(np.min(q)))
            per[r]={"exact_det_delta_integer":str(delta_exact),"delta_volume_m3":dm,
                    "new_cells":n,"minimum_native_q":float(np.min(q))}
        if not ok:continue
        pair=sum(int(per[r]["exact_det_delta_integer"]) for r in roles)
        if paired_volume and pair!=0:continue
        winners.append((minq,-disp,x32.copy(),per,pair,disp))
    if not winners:
        return {"status":"REJECT_NO_F32_CANDIDATE","solver_success":bool(res.success),"message":str(res.message),
                "continuous_anchor_m":x64.tolist(),"continuous_common_slack_m":float(res.x[3]),"ulp_radius":int(ulp_radius)}
    best=max(winners,key=lambda z:(z[0],z[1]))
    return {"status":"PASS","anchor_point_f32":best[2],"anchor_point_world_m":best[2].tolist(),
        "anchor_displacement_m":best[5],"continuous_anchor_world_m":x64.tolist(),
        "continuous_common_slack_m":float(res.x[3]),"solver_success":bool(res.success),
        "solver_message":str(res.message),"minimum_new_native_q":best[0],"exact_volume_by_role":best[3],
        "exact_paired_volume_delta_integer":str(best[4]),"move_cap_m":cap,
        "construction_q_margin_multiplier":float(q_margin_multiplier),"native_q_gate_unchanged":NATIVE_Q,
        "min_cell_volume_gate_m3_unchanged":MIN_CELL_VOLUME_M3,"tested_f32_candidates":len(neigh),
        "all185_775":"REQUIRED_NOT_RUN"}

def cache_volume(v):
    p=np.asarray(v["points_world_m"],np.float32);t=np.asarray(v["tetrahedra"],np.int64);mat=np.asarray(v["material_indices"],np.int32)
    det,q=measurements(p,t)
    if np.any(~np.isfinite(det)) or np.any(det<=0):raise ValueError("baseline has nonpositive cells")
    lut={}
    for i,x in enumerate(p):
        k=f32key(x)
        if k in lut:raise ValueError("ambiguous volume coordinate map")
        lut[k]=i
    return {"arrays":v,"p":p,"t":t,"mat":mat,"det":det,"q":q,"bad":~M.passes(det,q),
            "inc":incident_cache(t,len(p)),"lookup":lut,"desired":np.asarray(v["desired_boundary_faces"],np.int64)}

def prepare_operation(surface,volumes,keep,drop,*,protected_surface_rows=(),route_anchor_rows=(),move_cap_m=.0005):
    sp=np.asarray(surface["points_world_m_f32"],np.float32);sf=np.asarray(surface["faces"],np.int64);keep=int(keep);drop=int(drop)
    if keep==drop or min(keep,drop)<0 or max(keep,drop)>=len(sp):return {"status":"REJECT_ENDPOINTS"}
    ok,why=surface_link(sf,keep,drop)
    if not ok:return {"status":"REJECT_SURFACE_LINK","witness":why}
    fullf=np.union1d(np.flatnonzero(np.any(sf==keep,axis=1)),np.flatnonzero(np.any(sf==drop,axis=1)))
    verts=set(map(int,sf[fullf].ravel()))
    if verts&set(map(int,protected_surface_rows)):return {"status":"REJECT_PROTECTED_BACK"}
    if {keep,drop}&set(map(int,route_anchor_rows)):return {"status":"REJECT_ROUTE_ANCHOR"}
    stars={};fixed={};oldf={};olde={};loc={};reg={}
    for role in ("Skin","Fat"):
        v=volumes[role];u=v["lookup"].get(f32key(sp[keep]));w=v["lookup"].get(f32key(sp[drop]))
        if u is None or w is None:return {"status":"REJECT_ENDPOINT_CROSSWALK","role":role}
        kr=incident_rows(v["inc"],u);dr=incident_rows(v["inc"],w)
        if not volume_link(v["t"],u,w,kr,dr):return {"status":"REJECT_VOLUME_LINK","role":role}
        rows=np.union1d(kr,dr);old=v["t"][rows];mapped=np.where(old==w,u,old)
        deleted=np.any(np.diff(np.sort(mapped,axis=1),axis=1)==0,axis=1);new=mapped[~deleted];parents=rows[~deleted]
        if not len(new) or not np.all(np.any(new==u,axis=1)):return {"status":"REJECT_EMPTY_NEW_STAR","role":role}
        if len({cellkey(x) for x in new})!=len(new):return {"status":"REJECT_DUPLICATE_NEW_CELL","role":role}
        ordered=anchor_order(new,u)
        if np.any(measurements(v["p"],ordered)[0]<=0):return {"status":"REJECT_ORIENTATION","role":role}
        olddet=v["det"][rows];fixed[role]=v["p"][ordered[:,1:]]
        oldf[role]=float(np.sum(olddet));olde[role]=int(exact_det_sum(v["p"],old))
        loc[role]=float(np.sum(olddet));reg[role]=float(np.sum(v["det"]))
        stars[role]={"keep":u,"drop":w,"rows":rows,"old":old,"new":new,"parents":parents,
            "materials":v["mat"][parents],"old_bad":int(np.count_nonzero(v["bad"][rows])),
            "point_rows":np.unique(old.ravel())}
    fit=fit_common_anchor(sp[keep],fixed,oldf,olde,loc,reg,move_cap_m=move_cap_m)
    if fit["status"]!="PASS":return {"status":"REJECT_ANCHOR_FIT","fit":fit}
    x=np.asarray(fit["anchor_point_f32"],np.float32);spnew=sp.copy();spnew[keep]=x
    qmap=np.arange(len(sp),dtype=np.int64);qmap[drop]=keep
    fq=qmap[sf];fkeep=~np.any(np.diff(np.sort(fq,axis=1),axis=1)==0,axis=1)
    fparents=np.flatnonzero(fkeep);fout=fq[fkeep]
    if len({tuple(sorted(map(int,f))) for f in fout})!=len(fout):return {"status":"REJECT_DUPLICATE_SURFACE_FACE"}
    changed=np.isin(fparents,fullf);oldn=normals(sp,sf[fullf]);mapped_full=qmap[sf[fullf]]
    valid=~np.any(np.diff(np.sort(mapped_full,axis=1),axis=1)==0,axis=1);newn=normals(spnew,mapped_full[valid]);oldn=oldn[valid]
    den=np.linalg.norm(oldn,axis=1)*np.linalg.norm(newn,axis=1)
    if np.any(den<=0) or np.any(np.einsum("ij,ij->i",oldn,newn)<.9*den):return {"status":"REJECT_SOURCE_NORMAL"}
    qok,minq=source_q(spnew,mapped_full[valid])
    if not qok:return {"status":"REJECT_SOURCE_APEX","minimum":minq}
    vr={};gain=0
    for role in ("Skin","Fat"):
        v=volumes[role];s=stars[role];pnew=v["p"].copy();pnew[s["keep"]]=x
        nd,nq=measurements(pnew,s["new"])
        if np.any(nd<=0) or not np.all(M.passes(nd,nq)) or not exact_positive(pnew,s["new"]):return {"status":"REJECT_NEW_TET_GATE","role":role}
        delta=int(exact_det_sum(pnew,s["new"]))-olde[role];dm=delta*PAIR_VOLUME_SCALE
        budget=min(.01*loc[role],1e-4*reg[role])/6.0
        if abs(dm)>budget:return {"status":"REJECT_VOLUME_BUDGET","role":role,"delta_m3":dm,"budget_m3":budget}
        gain+=s["old_bad"]
        vr[role]={"points":pnew,"new_tets":s["new"],"parent_rows":s["parents"],"materials":s["materials"],"keep_volume_node":s["keep"],"drop_volume_node":s["drop"],
            "old_rows":s["rows"],"old_bad":s["old_bad"],"exact_delta":delta,"delta_m3":dm,
            "point_rows":s["point_rows"]}
    if sum(int(vr[r]["exact_delta"]) for r in ("Skin","Fat"))!=0:return {"status":"REJECT_PAIRED_EXACT_VOLUME"}
    if gain<=0:return {"status":"REJECT_NO_STRICT_QUALITY_GAIN"}
    return {"status":"PASS","keep":keep,"drop":drop,"anchor_f32":x,"fit":fit,
        "surface_points":spnew,"surface_faces":fout,"surface_parent_rows":fparents,
        "surface_full_endpoint_face_rows":fullf,"surface_retired_face_rows":np.setdiff1d(fullf,fparents),
        "surface_changed_output_face_rows":np.flatnonzero(changed),"surface_point_quotient":qmap,
        "source_min_face_apex_bound":minq,"volumes":vr,"native_bad_gain_by_role":{r:stars[r]["old_bad"] for r in stars},
        "surface_star_vertices":np.asarray(sorted(verts),np.int64),
        "volume_star_points":{r:stars[r]["point_rows"] for r in stars},
        "volume_star_cells":{r:stars[r]["rows"] for r in stars},
        "exact_paired_delta":"0","requires_775_all185":True,"physical_admission":False,"native_steps":0}

def typed_source_output(parent,ops,points,faces,faceparents):
    data={k:np.array(v,copy=True) for k,v in parent.items()};nface=len(parent["faces"]);rowop={}
    unions=[];ancestry=[];oprows=[]
    for oi,op in enumerate(ops):
        affected=list(map(int,op["surface_full_endpoint_face_rows"]))
        for row in affected:rowop[row]=oi
    cached={}
    for oi,op in enumerate(ops):
        rows=op["surface_full_endpoint_face_rows"]
        u=B.union_tuples(parent,rows);a=B.ancestry_union(parent,rows)
        if not u or not a:raise ValueError("missing endpoint source ancestry")
        cached[oi]=(u,a)
    for row in map(int,faceparents):
        oi=rowop.get(row,-1)
        if oi>=0:un,an=cached[oi];unions.append(un);ancestry.append(an);oprows.append(oi)
        else:unions.append(A.rows_union(parent,row));ancestry.append(A.rows_anc(parent,row));oprows.append(-1)
    uo,ud=A.csr(unions,7);ao,ad=A.csr(ancestry,3)
    for k in list(data):
        if k.startswith("source_") and k.endswith("_operation_rows"):data.pop(k)
    data["points_world_m_f32"]=np.asarray(points,np.float32)
    for k,v in list(data.items()):
        a=np.asarray(v)
        if k!="points_world_m_f32" and ("points_world_m" in k or k=="points") and a.shape==np.asarray(points).shape and np.issubdtype(a.dtype,np.floating):
            data[k]=np.asarray(points,dtype=a.dtype)
    data["faces"]=np.asarray(faces,np.int64)
    fid=np.asarray(parent.get("quality_star_face_output_to_input_row",np.arange(nface)),np.int64)[faceparents].copy()
    fid[np.asarray(oprows)>=0]=-1
    data["quality_star_face_output_to_input_row"]=fid;data["quality_star_patch_union_offsets"]=uo;data["face_ancestry_offsets"]=ao
    data["source_input_face_rows"]=np.asarray(faceparents,np.int64)
    q=np.arange(len(points),dtype=np.int64)
    for op in ops:q[op["drop"]]=op["keep"]
    data["source_input_vertex_quotient"]=q;data["source_collapse_operation_rows"]=np.asarray(oprows,np.int64)
    for j,k in enumerate(A.T):data[k]=np.asarray([u[j] for u in unions],np.int64)
    for j,k in enumerate(A.U):data[k]=ud[:,j]
    for j,k in enumerate(A.A):data[k]=ad[:,j]
    data["coupled_anchor_operations_json"]=np.asarray([json.dumps({
        "role":"I","keep_parent_point_row":op["keep"],"drop_parent_point_row":op["drop"],
        "moved_anchor_world_m":op["anchor_f32"].tolist(),"move_cap_m":op["fit"]["move_cap_m"],
        "complete_parent_face_rows":op["surface_full_endpoint_face_rows"].tolist(),
        "retired_parent_face_rows":op["surface_retired_face_rows"].tolist(),
        "physical_admission":False,"native_steps":0},sort_keys=True) for op in ops])
    return data

def run_batch(surface,volumes,candidate_edges,*,protected_surface_rows=(),route_anchor_rows=(),
              move_cap_m=.0005,max_operations=32,time_limit_s=90,output_dir=None):
    """Bounded disjoint I-edge candidate batch over both materials."""
    start=time.monotonic();out=Path(output_dir) if output_dir else None
    if out is not None:out.mkdir(parents=True,exist_ok=False)
    cache={r:cache_volume(volumes[r]) for r in ("Skin","Fat")}
    for r in cache:
        if not np.array_equal(T.boundary_identity(boundary(cache[r]["t"])),T.boundary_identity(cache[r]["desired"])):
            return {"status":"REJECT_BASELINE_TARGET_MISMATCH","role":r}
    accepted=[];reject=Counter();usedsv=set();usedcp={r:set() for r in cache};usedcc={r:set() for r in cache}
    for keep,drop in candidate_edges:
        if time.monotonic()-start>=time_limit_s or len(accepted)>=max_operations:break
        op=prepare_operation(surface,cache,keep,drop,protected_surface_rows=protected_surface_rows,
                             route_anchor_rows=route_anchor_rows,move_cap_m=move_cap_m)
        if op["status"]!="PASS":reject[op["status"]]+=1;continue
        if set(op["surface_star_vertices"])&usedsv or any(
            set(op["volume_star_points"][r])&usedcp[r] or set(op["volume_star_cells"][r])&usedcc[r] for r in cache):
            reject["overlapping_complete_endpoint_stars"]+=1;continue
        accepted.append(op);usedsv.update(map(int,op["surface_star_vertices"]))
        for r in cache:usedcp[r].update(map(int,op["volume_star_points"][r]));usedcc[r].update(map(int,op["volume_star_cells"][r]))
    if not accepted:return {"status":"NO_VALID_COUPLED_ANCHOR_COLLAPSE","accepted_operations":0,"rejections":dict(reject)}
    sp=np.asarray(surface["points_world_m_f32"],np.float32).copy();sq=np.arange(len(sp),dtype=np.int64)
    for op in accepted:sq[op["drop"]]=op["keep"];sp[op["keep"]]=op["anchor_f32"]
    sfq=sq[np.asarray(surface["faces"],np.int64)];fk=~np.any(np.diff(np.sort(sfq,axis=1),axis=1)==0,axis=1)
    fparents=np.flatnonzero(fk);sfout=sfq[fk]
    if len({tuple(sorted(map(int,f))) for f in sfout})!=len(sfout):return {"status":"REJECT_DUPLICATE_SURFACE"}
    typed=typed_source_output(surface,accepted,sp,sfout,fparents)
    volumeout={};quality={}
    for r,v in cache.items():
        p=v["p"].copy();qmap=np.arange(len(p),dtype=np.int64);retire=np.zeros(len(v["t"]),bool)
        newt=[];newm=[];newparents=[];parent_unions=[];replacement_offsets=[0]
        for op in accepted:
            st=op["volumes"][r];vm=coord_crosswalk(surface["points_world_m_f32"],v["p"])
            if vm is None or vm[op["keep"]]<0 or vm[op["drop"]]<0:return {"status":"REJECT_AMBIGUOUS_CROSSWALK","role":r}
            u,w=map(int,(vm[op["keep"]],vm[op["drop"]]));p[u]=op["anchor_f32"];qmap[w]=u
            rr=np.asarray(st["old_rows"],np.int64);retire[rr]=True
            newt.append(st["new_tets"]);newm.append(st["materials"]);newparents.append(st["parent_rows"])
            star=np.asarray(op["volume_star_cells"][r],np.int64)
            for _ in range(len(st["new_tets"])):
                parent_unions.extend(star.tolist());replacement_offsets.append(len(parent_unions))
        retained=np.flatnonzero(~retire);replacement=np.concatenate(newt);repmat=np.concatenate(newm);repparent=np.concatenate(newparents)
        unionparents=np.asarray(parent_unions,np.int64);unionoffsets=np.asarray(replacement_offsets,np.int64)
        t=np.vstack((v["t"][retained],replacement));mat=np.r_[v["mat"][retained],repmat]
        if len({cellkey(x) for x in t})!=len(t):return {"status":"REJECT_DUPLICATE_GLOBAL_CELL","role":r}
        det,q=measurements(p,t)
        if np.any(det<=0) or not exact_positive(p,t):return {"status":"REJECT_GLOBAL_NONPOSITIVE","role":r}
        bad=int(np.count_nonzero(~M.passes(det,q)));target=qmap[v["desired"]]
        tf=~np.any(np.diff(np.sort(target,axis=1),axis=1)==0,axis=1);target=target[tf];bd=boundary(t)
        if not np.array_equal(T.boundary_identity(bd),T.boundary_identity(target)):return {"status":"REJECT_GLOBAL_TARGET_BOUNDARY","role":r}
        imap=np.full(len(v["t"]),-1,np.int64);imap[retained]=np.arange(len(retained))
        parentmap=np.r_[retained,repparent]
        a={k:np.array(z,copy=True) for k,z in v["arrays"].items()}
        for key,val in list(a.items()):
            vv=np.asarray(val)
            if key!="points_world_m" and ("points_world_m" in key or key=="points") and vv.shape==p.shape and np.issubdtype(vv.dtype,np.floating):
                a[key]=p.astype(vv.dtype)
        a.update(points_world_m=p,tetrahedra=t,material_indices=mat,boundary_faces=bd,desired_boundary_faces=target,
          output_cell_to_input_cell_rows=parentmap,input_cell_to_output_cell_rows=imap,retained_input_cell_rows=retained,
          replacement_parent_cell_rows=unionparents,replacement_parent_offsets=unionoffsets,
          replacement_tetrahedra=replacement,input_node_to_output_node_rows=qmap,
          output_node_to_input_node_rows=np.arange(len(p),dtype=np.int64),output_point_to_input_point_rows=np.arange(len(p),dtype=np.int64),
          output_point_parent_offsets=np.zeros(len(p)+1,np.int64),output_point_parent_input_cell_rows=np.empty(0,np.int64))
        quality[r]={"cells":len(t),"native_bad_before":int(np.count_nonzero(v["bad"])),"native_bad_after":bad,
                    "native_bad_removed":int(np.count_nonzero(v["bad"]))-bad,"nonpositive":0}
        volumeout[r]=a
    exact_pair=sum(int(o["exact_paired_delta"]) for o in accepted)
    if exact_pair:return {"status":"REJECT_PAIRED_VOLUME","exact_delta":str(exact_pair)}
    report={"schema":"h706.coupled-anchor-collapse-batch.v1","status":"CANDIDATE_PENDING_775_ALL185",
        "accepted_operations":len(accepted),"rejections":dict(reject),"quality":quality,
        "exact_paired_volume_delta_integer":"0","move_cap_m":float(move_cap_m),
        "native_q_gate_unchanged":NATIVE_Q,"minimum_cell_volume_gate_m3_unchanged":MIN_CELL_VOLUME_M3,
        "exact_complete_target_boundary":True,"exact_positive_f32":True,"all_new_cells_pass_native":True,
        "complete_typed_parent_face_unions":True,"all185_775":"REQUIRED_NOT_RUN",
        "physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING",
        "operation_records":[{"keep":o["keep"],"drop":o["drop"],"anchor_f32":o["anchor_f32"].tolist(),
          "displacement_m":o["fit"]["anchor_displacement_m"],"surface_parent_faces":o["surface_full_endpoint_face_rows"].tolist(),
          "surface_changed_output_face_rows":o["surface_changed_output_face_rows"].tolist(),
          "retired_surface_faces":o["surface_retired_face_rows"].tolist(),
          "volume_endpoints":{r:[int(o["volumes"][r]["keep_volume_node"]),int(o["volumes"][r]["drop_volume_node"])] for r in cache},
          "full_star_cells":{r:o["volume_star_cells"][r].tolist() for r in cache},
          "exact_delta_by_role":{r:str(o["volumes"][r]["exact_delta"]) for r in cache},
          "native_bad_gain_by_role":o["native_bad_gain_by_role"]} for o in accepted]}
    if out is not None:
        np.savez_compressed(out/"candidate-I-typed-current.npz",**typed)
        for r,a in volumeout.items():np.savez_compressed(out/(r.lower()+"-coupled-anchor-volume-proposal.npz"),**a)
        report["producer"]={"path":str(Path(__file__)),"sha256":sha(Path(__file__))}
        (out/"state.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def load_pinned_npz(entry,label):
    if not isinstance(entry,dict) or "path" not in entry or "sha256" not in entry:
        raise ValueError(f"{label}: expected path and sha256 pin")
    p=Path(entry["path"]).resolve()
    got=sha(p)
    if got!=entry["sha256"]:raise ValueError(f"{label}: SHA mismatch {got}")
    with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}

def pinned_path(entry,label):
    if not isinstance(entry,dict) or "path" not in entry or "sha256" not in entry:
        raise ValueError(f"{label}: expected path and sha256 pin")
    p=Path(entry["path"]).resolve()
    if sha(p)!=entry["sha256"]:raise ValueError(f"{label}: SHA mismatch")
    return p

def load_config(path):
    p=Path(path).resolve();cfg=json.loads(p.read_text())
    if cfg.get("schema")!="h706.coupled-anchor-collapse-run-config.v1":
        raise ValueError("wrong coupled-anchor config schema")
    surface_ref=cfg["surface_I"]
    surface=load_pinned_npz(surface_ref,"surface_I")
    baserefs=cfg["baseline_surfaces"]
    if set(baserefs)!={"O","I","D"}:raise ValueError("baseline_surfaces must pin O/I/D")
    baseline={r:load_pinned_npz(baserefs[r],"baseline "+r) for r in ("O","I","D")}
    for key in ("faces","points_world_m_f32"):
        if key not in baseline["I"] or key not in surface or not np.array_equal(baseline["I"][key],surface[key]):
            raise ValueError("surface_I must be byte-value identical to baseline_surfaces.I")
    volumes={r:load_pinned_npz(cfg["volumes"][r],r+" input volume") for r in ("Skin","Fat")}
    edge_entry=cfg["candidate_edges"]
    if isinstance(edge_entry,list):
        edges=np.asarray(edge_entry,np.int64)
    else:
        if "values" in edge_entry:
            edges=np.asarray(edge_entry["values"],np.int64)
        else:
            ee=load_pinned_npz(edge_entry,"candidate edge roster")
            ek=edge_entry.get("key","candidate_edges")
            if ek not in ee:raise ValueError(f"edge roster lacks key {ek}")
            edges=np.asarray(ee[ek],np.int64)
    if edges.ndim!=2 or edges.shape[1]!=2:raise ValueError("candidate edge roster must be Nx2")
    def values(ref,label):
        if ref is None:return np.empty(0,np.int64)
        if isinstance(ref,list):return np.asarray(ref,np.int64)
        if "values" in ref:return np.asarray(ref["values"],np.int64)
        z=load_pinned_npz(ref,label);key=ref["key"]
        if key not in z:raise ValueError(f"{label} lacks key {key}")
        return np.asarray(z[key],np.int64)
    protected=values(cfg.get("protected_surface_rows",[]),"protected surface rows")
    routes=values(cfg.get("route_anchor_rows",[]),"route anchor rows")
    return {"path":p,"cfg":cfg,"surface":surface,"baseline":baseline,"volumes":volumes,
            "edges":edges,"protected":protected,"routes":routes}

def _write_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+"\n")
    return p

def make_775_config(run_dir,loaded,report,out_path):
    run_dir=Path(run_dir).resolve()
    ip=run_dir/"candidate-I-typed-current.npz"
    if not ip.is_file():raise ValueError("run has no candidate I output")
    ientry={"path":str(ip),"sha256":sha(ip)}
    base={r:{"path":str(pinned_path(loaded["cfg"]["baseline_surfaces"][r],"baseline "+r)),
             "sha256":loaded["cfg"]["baseline_surfaces"][r]["sha256"]} for r in ("O","I","D")}
    candidates={"O":base["O"],"I":ientry,"D":base["D"]}
    move=float(loaded["cfg"].get("move_cap_m",.0005))
    context={"schema":"h706.combined-current-exact-surface-diff-config.v1",
             "baseline_candidates":base,"candidate_surfaces":candidates,"changed_roles":["I"],
             "bounded_point_repair":{"O":{"maximum_displacement_m":0.0},
                 "I":{"maximum_displacement_m":move},"D":{"maximum_displacement_m":0.0}},
             "source_candidate_run":{"operation_count":int(report.get("accepted_operations",0)),
             "surface_output":ientry},
             "physical_admission":False,"native_steps":0}
    out=Path(out_path).resolve()
    _write_json(out,context)
    return {"path":str(out),"sha256":sha(out)}

def run_from_config(config_path,out_dir,edges_override=None,prune_receipt=None):
    x=load_config(config_path);out=Path(out_dir).resolve()
    use_edges=x["edges"] if edges_override is None else np.asarray(edges_override,np.int64)
    report=run_batch(x["surface"],x["volumes"],use_edges,
        protected_surface_rows=x["protected"],route_anchor_rows=x["routes"],
        move_cap_m=float(x["cfg"].get("move_cap_m",.0005)),
        max_operations=int(x["cfg"].get("max_operations",32)),
        time_limit_s=float(x["cfg"].get("time_limit_s",90)),output_dir=out)
    report["run_config"]={"path":str(x["path"]),"sha256":sha(x["path"])}
    report["source_inputs"]={"surface_I":x["cfg"]["surface_I"],
        "baseline_surfaces":x["cfg"]["baseline_surfaces"],"volumes":x["cfg"]["volumes"],
        "candidate_edges":x["cfg"]["candidate_edges"]}
    used_edges_path=out/"effective-candidate-edges.npy"
    np.save(used_edges_path,np.asarray(use_edges,np.int64))
    report["effective_candidate_edges"]={"path":str(used_edges_path),"sha256":sha(used_edges_path),"count":int(len(use_edges))}
    report["candidate_edge_count"]=int(len(use_edges))
    report["protected_surface_row_count"]=int(len(x["protected"]))
    report["route_anchor_row_count"]=int(len(x["routes"]))
    if prune_receipt is not None:report["775_prune_parent"]=prune_receipt
    if report.get("status")=="CANDIDATE_PENDING_775_ALL185":
        oprecords=report["operation_records"]
        retained={};cursor={}
        for role in ("Skin","Fat"):
            v=x["volumes"][role];t=np.asarray(v["tetrahedra"],np.int64)
            retired=set();direct_by_op=[];new_by_op=[]
            for op in oprecords:
                u,w=map(int,op["volume_endpoints"][role]);star=np.asarray(op["full_star_cells"][role],np.int64)
                if np.any(star<0) or np.any(star>=len(t)):raise ValueError("operation cell row out of range")
                if not np.array_equal(np.unique(star),np.flatnonzero(np.any((t==u)|(t==w),axis=1))):
                    raise ValueError("operation full-star rows do not match source cells")
                mapped=np.where(t[star]==w,u,t[star]);dead=np.any(np.diff(np.sort(mapped,axis=1),axis=1)==0,axis=1)
                retired.update(map(int,star));direct=star[~dead];news=mapped[~dead]
                direct_by_op.append(direct.tolist());new_by_op.append(news)
            keep=np.asarray([i for i in range(len(t)) if i not in retired],np.int64)
            c=len(keep);cursor[role]=c
            for oi,op in enumerate(oprecords):
                n=len(new_by_op[oi]);op.setdefault("retired_source_cell_rows",{})[role]=op["full_star_cells"][role]
                op.setdefault("replacement_parent_rows",{})[role]=direct_by_op[oi]
                op.setdefault("replacement_output_cell_rows",{})[role]=list(range(cursor[role],cursor[role]+n))
                cursor[role]+=n
        for name in ("candidate-I-typed-current.npz","skin-coupled-anchor-volume-proposal.npz","fat-coupled-anchor-volume-proposal.npz"):
            p=out/name
            if p.is_file():report.setdefault("outputs",{})[name]={"path":str(p),"sha256":sha(p)}
        context=make_775_config(out,x,report,out/"775-context-config.json")
        report["775_context_config"]=context
    _write_json(out/"state.json",report)
    return report

def _array_equal(a,b):
    aa,bb=np.asarray(a),np.asarray(b)
    return aa.shape==bb.shape and aa.dtype==bb.dtype and np.ascontiguousarray(aa).tobytes()==np.ascontiguousarray(bb).tobytes()

def independent_replay(config_path,run_dir,out_path=None):
    x=load_config(config_path);rd=Path(run_dir).resolve();rp=rd/"state.json";report=json.loads(rp.read_text())
    if report.get("status")!="CANDIDATE_PENDING_775_ALL185":raise ValueError("run is not a candidate")
    if report.get("run_config",{}).get("sha256")!=sha(x["path"]):raise ValueError("run config pin mismatch")
    surf=x["surface"];basefaces=np.asarray(surf["faces"],np.int64);basepoints=np.asarray(surf["points_world_m_f32"],np.float32)
    ops=report["operation_records"];q=np.arange(len(basepoints),dtype=np.int64);sp=basepoints.copy()
    faceop={}
    for oi,op in enumerate(ops):
        k,d=int(op["keep"]),int(op["drop"])
        if q[d]!=d or q[k]!=k:raise ValueError("overlapping surface point quotient")
        q[d]=k;sp[k]=np.asarray(op["anchor_f32"],np.float32)
        expected=np.union1d(np.flatnonzero(np.any(basefaces==k,axis=1)),np.flatnonzero(np.any(basefaces==d,axis=1)))
        if not np.array_equal(expected,np.asarray(op["surface_parent_faces"],np.int64)):
            raise ValueError("surface endpoint face star mismatch")
        for fr in expected:
            if int(fr) in faceop:raise ValueError("overlapping surface endpoint face stars")
            faceop[int(fr)]=oi
    mapped=q[basefaces];valid=~np.any(np.diff(np.sort(mapped,axis=1),axis=1)==0,axis=1)
    fparents=np.flatnonzero(valid);expectedfaces=mapped[valid]
    sout=load_pinned_npz({"path":str(rd/"candidate-I-typed-current.npz"),
                          "sha256":report["outputs"]["candidate-I-typed-current.npz"]["sha256"]},"candidate I")
    if not _array_equal(sout["points_world_m_f32"],sp) or not _array_equal(sout["faces"],expectedfaces):
        raise ValueError("surface candidate differs from exact endpoint quotient")
    for key in surf:
        pv=np.asarray(surf[key])
        if key!="points_world_m_f32" and ("points_world_m" in key or key=="points") and pv.shape==sp.shape and np.issubdtype(pv.dtype,np.floating) and key in sout:
            expected_alias=pv.copy()
            for op in ops:
                expected_alias[int(op["keep"])]=np.asarray(op["anchor_f32"],dtype=pv.dtype)
            if not _array_equal(sout[key],expected_alias):raise ValueError("surface coordinate alias mismatch "+key)
    if not np.array_equal(np.asarray(sout["source_input_face_rows"],np.int64),fparents):
        raise ValueError("surface immediate parent face map mismatch")
    expected_op=np.asarray([faceop.get(int(fr),-1) for fr in fparents],np.int64)
    if not np.array_equal(np.asarray(sout["source_collapse_operation_rows"],np.int64),expected_op):
        raise ValueError("surface operation ancestry rows mismatch")
    qexpect=np.arange(len(basepoints),dtype=np.int64)
    for op in ops:qexpect[int(op["drop"])]=int(op["keep"])
    if not np.array_equal(np.asarray(sout["source_input_vertex_quotient"],np.int64),qexpect):
        raise ValueError("surface immediate point quotient mismatch")
    rowop={int(fr):oi for oi,op in enumerate(ops) for fr in op["surface_parent_faces"]}
    from_surface_unions=[];from_surface_anc=[]
    for fr in fparents:
        oi=rowop.get(int(fr),-1)
        if oi>=0:
            rows=ops[oi]["surface_parent_faces"]
            from_surface_unions.append(B.union_tuples(surf,rows));from_surface_anc.append(B.ancestry_union(surf,rows))
        else:
            from_surface_unions.append(A.rows_union(surf,int(fr)));from_surface_anc.append(A.rows_anc(surf,int(fr)))
    uoff,ud=A.csr(from_surface_unions,7);aoff,ad=A.csr(from_surface_anc,3)
    if not np.array_equal(np.asarray(sout["quality_star_patch_union_offsets"]),uoff):raise ValueError("surface typed union offsets mismatch")
    if not np.array_equal(np.asarray(sout["face_ancestry_offsets"]),aoff):raise ValueError("surface ancestry offsets mismatch")
    for j,key in enumerate(A.U):
        if not _array_equal(sout[key],ud[:,j]):raise ValueError("surface typed union lane mismatch "+key)
    for j,key in enumerate(A.A):
        if not _array_equal(sout[key],ad[:,j]):raise ValueError("surface ancestry lane mismatch "+key)
    for j,key in enumerate(A.T):
        if not _array_equal(sout[key],np.asarray([u[0][j] for u in from_surface_unions],np.int64)):
            raise ValueError("surface direct source lane mismatch "+key)
    expected_face_map=np.asarray(surf.get("quality_star_face_output_to_input_row",np.arange(len(basefaces))),np.int64)[fparents].copy()
    expected_face_map[expected_op>=0]=-1
    if not _array_equal(sout["quality_star_face_output_to_input_row"],expected_face_map):
        raise ValueError("surface immediate source-face row mapping mismatch")
    candidate_row_by_parent={int(fr):i for i,fr in enumerate(fparents)}
    for oi,op in enumerate(ops):
        oldrows=np.asarray(op["surface_parent_faces"],np.int64)
        keeprows=[int(fr) for fr in oldrows if int(fr) in candidate_row_by_parent]
        outrows=np.asarray([candidate_row_by_parent[fr] for fr in keeprows],np.int64)
        oldn=normals(basepoints,basefaces[np.asarray(keeprows,np.int64)])
        newn=normals(sp,expectedfaces[outrows])
        den=np.linalg.norm(oldn,axis=1)*np.linalg.norm(newn,axis=1)
        if np.any(den<=0) or np.any(np.einsum("ij,ij->i",oldn,newn)<.9*den):
            raise ValueError("surface source-normal gate failed during replay")
        qok,minq=source_q(sp,expectedfaces[outrows])
        if not qok:raise ValueError("surface source-apex gate failed during replay")
    quality={}
    volume_outputs={}
    paired_op_deltas=[{} for _ in ops]
    for role in ("Skin","Fat"):
        parent=x["volumes"][role];pt=np.asarray(parent["points_world_m"],np.float32)
        bt=np.asarray(parent["tetrahedra"],np.int64);bm=np.asarray(parent["material_indices"],np.int32)
        old_global_det,old_global_q=measurements(pt,bt)
        vmq=np.arange(len(pt),dtype=np.int64);vp=pt.copy();stars=[];newparents=[];newt=[];newmat=[];deltas=[];newcounts=[];oldbadcounts=[]
        for op in ops:
            u,w=map(int,op["volume_endpoints"][role]);anchor=np.asarray(op["anchor_f32"],np.float32)
            vp[u]=anchor;vmq[w]=u
            expectedstar=np.flatnonzero(np.any((bt==u)|(bt==w),axis=1))
            star=np.asarray(op["full_star_cells"][role],np.int64)
            if not np.array_equal(star,expectedstar):raise ValueError(role+" full volume endpoint star mismatch")
            kr=np.flatnonzero(np.any(bt==u,axis=1));dr=np.flatnonzero(np.any(bt==w,axis=1))
            if not volume_link(bt,u,w,kr,dr):raise ValueError(role+" volume link invalid at replay")
            mappedtet=np.where(bt[star]==w,u,bt[star]);dead=np.any(np.diff(np.sort(mappedtet,axis=1),axis=1)==0,axis=1)
            direct=star[~dead];news=mappedtet[~dead]
            if len({cellkey(t) for t in news})!=len(news):raise ValueError(role+" duplicate local replacement cell")
            stars.append(star);newcounts.append(len(news));newparents.extend(direct.tolist());newt.extend(news.tolist());newmat.extend(bm[direct].tolist())
            old_exact=exact_det_sum(pt,bt[star]);new_exact=exact_det_sum(vp,news);delta=new_exact-old_exact
            if str(delta)!=op["exact_delta_by_role"][role]:raise ValueError(role+" per-operation exact delta mismatch")
            delta_m=delta*PAIR_VOLUME_SCALE
            budget=min(.01*float(np.sum(old_global_det[star])),1e-4*float(np.sum(old_global_det)))/6.0
            if abs(delta_m)>budget:raise ValueError(role+" local/regional exact-volume budget failure")
            oldbad=int(np.count_nonzero(~M.passes(old_global_det[star],old_global_q[star])))
            if oldbad!=int(op["native_bad_gain_by_role"][role]):raise ValueError(role+" local native-gain source count mismatch")
            oldbadcounts.append(oldbad)
            if not exact_positive(vp,news):raise ValueError(role+" replacement exact orientation failure")
            nd,nq=measurements(vp,news)
            if not np.all(M.passes(nd,nq)):raise ValueError(role+" replacement native gate failure")
            deltas.append(delta)
            paired_op_deltas[len(deltas)-1][role]=delta
        retired=np.unique(np.concatenate(stars)) if stars else np.empty(0,np.int64)
        retained=np.setdiff1d(np.arange(len(bt),dtype=np.int64),retired,assume_unique=True)
        nt=np.asarray(newt,np.int64).reshape((-1,4));nm=np.asarray(newmat,np.int32)
        expectedt=np.vstack((bt[retained],nt));expectedm=np.r_[bm[retained],nm]
        outname=role.lower()+"-coupled-anchor-volume-proposal.npz"
        vo=load_pinned_npz({"path":str(rd/outname),"sha256":report["outputs"][outname]["sha256"]},role+" candidate volume")
        if not _array_equal(vo["points_world_m"],vp):raise ValueError(role+" changed point coordinates mismatch")
        for key in parent:
            pv=np.asarray(parent[key])
            if key!="points_world_m" and ("points_world_m" in key or key=="points") and pv.shape==vp.shape and np.issubdtype(pv.dtype,np.floating) and key in vo:
                expected_alias=pv.copy()
                for op in ops:
                    u=int(op["volume_endpoints"][role][0]);expected_alias[u]=np.asarray(op["anchor_f32"],dtype=pv.dtype)
                if not _array_equal(vo[key],expected_alias):raise ValueError(role+" coordinate alias mismatch "+key)
        for key in ("tetrahedra","material_indices","retained_input_cell_rows","replacement_tetrahedra","replacement_parent_cell_rows","replacement_parent_offsets","input_node_to_output_node_rows","output_node_to_input_node_rows","output_point_to_input_point_rows","output_point_parent_offsets","output_point_parent_input_cell_rows","output_cell_to_input_cell_rows","input_cell_to_output_cell_rows"):
            if key not in vo:raise ValueError(role+" missing emitted map "+key)
        if not _array_equal(vo["tetrahedra"],expectedt) or not _array_equal(vo["material_indices"],expectedm):
            raise ValueError(role+" full volume output differs from reconstructed source operation")
        if not _array_equal(vo["replacement_tetrahedra"],nt):raise ValueError(role+" replacement tet array mismatch")
        if not _array_equal(vo["replacement_parent_cell_rows"],np.asarray(newparents,np.int64)):raise ValueError(role+" direct replacement parents mismatch")
        if not _array_equal(vo["retained_input_cell_rows"],retained):raise ValueError(role+" retained cell map mismatch")
        wantimap=np.full(len(bt),-1,np.int64);wantimap[retained]=np.arange(len(retained))
        if not _array_equal(vo["input_cell_to_output_cell_rows"],wantimap):raise ValueError(role+" immediate input-cell map mismatch")
        wantcmap=np.r_[retained,np.asarray(newparents,np.int64)]
        if not _array_equal(vo["output_cell_to_input_cell_rows"],wantcmap):raise ValueError(role+" output cell parent map mismatch")
        wantq=np.arange(len(pt),dtype=np.int64)
        for op in ops:wantq[int(op["volume_endpoints"][role][1])]=int(op["volume_endpoints"][role][0])
        if not _array_equal(vo["input_node_to_output_node_rows"],wantq):raise ValueError(role+" volume point quotient mismatch")
        ident=np.arange(len(pt),dtype=np.int64)
        for key in ("output_node_to_input_node_rows","output_point_to_input_point_rows"):
            if not _array_equal(vo[key],ident):raise ValueError(role+" output point identity map mismatch")
        if not _array_equal(vo["output_point_parent_offsets"],np.zeros(len(pt)+1,np.int64)) or len(vo["output_point_parent_input_cell_rows"]):
            raise ValueError(role+" output point parent CSR must be empty for unchanged point domain")
        # Independently verify complete parent-star CSR for every replacement.
        poff=np.asarray(vo["replacement_parent_offsets"],np.int64)
        counts=[len(star) for star,nnew in zip(stars,newcounts) for _ in range(nnew)]
        wantoff=np.r_[0,np.cumsum(np.asarray(counts,np.int64))]
        wantflat=np.asarray([int(row) for star,nnew in zip(stars,newcounts) for _ in range(nnew) for row in star],np.int64)
        if not _array_equal(poff,wantoff) or not _array_equal(vo["replacement_parent_cell_rows"],wantflat):
            raise ValueError(role+" complete replacement parent-star union mismatch")
        det,qv=measurements(vp,expectedt)
        if np.any(det<=0) or not exact_positive(vp,expectedt):raise ValueError(role+" global positivity failure")
        bad=int(np.count_nonzero(~M.passes(det,qv)))
        if not np.array_equal(T.boundary_identity(boundary(expectedt)),T.boundary_identity(vo["desired_boundary_faces"])):
            raise ValueError(role+" full oriented boundary differs from current target")
        if not np.array_equal(T.boundary_identity(boundary(expectedt)),T.boundary_identity(vo["boundary_faces"])):
            raise ValueError(role+" boundary_faces metadata differs from actual")
        target=np.asarray(parent["desired_boundary_faces"],np.int64)
        target=vmq[target];ok=~np.any(np.diff(np.sort(target,axis=1),axis=1)==0,axis=1);target=target[ok]
        if not np.array_equal(T.boundary_identity(target),T.boundary_identity(vo["desired_boundary_faces"])):
            raise ValueError(role+" current desired boundary quotient mismatch")
        total_delta=exact_det_sum(vp,expectedt)-exact_det_sum(pt,bt)
        if total_delta!=sum(deltas):raise ValueError(role+" total exact volume delta mismatch")
        bad_before=int(np.count_nonzero(~M.passes(old_global_det,old_global_q)))
        if bad!=bad_before-sum(oldbadcounts):raise ValueError(role+" quality count delta differs from replayed stars")
        if report.get("quality",{}).get(role,{}).get("native_bad_before")!=bad_before or report.get("quality",{}).get(role,{}).get("native_bad_after")!=bad:
            raise ValueError(role+" run quality summary mismatch")
        quality[role]={"cells":len(expectedt),"native_bad_before":bad_before,
            "native_bad_after":bad,"native_bad_removed":bad_before-bad,"exact_det_delta_integer":str(total_delta),"replacement_cells":len(nt)}
        volume_outputs[role]={"path":str(rd/outname),"sha256":sha(rd/outname)}
    if sum(int(quality[r]["exact_det_delta_integer"]) for r in ("Skin","Fat"))!=0:
        raise ValueError("paired exact total volume delta is not zero")
    if any(sum(v.values())!=0 for v in paired_op_deltas):
        raise ValueError("a coupled operation has nonzero exact paired volume delta")
    if sum(int(quality[r]["exact_det_delta_integer"]) for r in ("Skin","Fat"))!=0:
        raise ValueError("paired exact total volume delta is not zero")
    result={"schema":"h706.coupled-anchor-collapse-independent-replay.v1",
        "status":"PASS_INDEPENDENT_SOURCE_REPLAY","run_state":{"path":str(rp),"sha256":sha(rp)},
        "run_config":{"path":str(x["path"]),"sha256":sha(x["path"])},"source_I":{"path":x["cfg"]["surface_I"]["path"],"sha256":x["cfg"]["surface_I"]["sha256"]},
        "surface_output":{"path":str(rd/"candidate-I-typed-current.npz"),"sha256":sha(rd/"candidate-I-typed-current.npz")},
        "volume_outputs":volume_outputs,"accepted_operations":len(ops),"quality":quality,
        "exact_complete_oriented_volume_boundary":"PASS","complete_typed_face_unions_and_ancestry":"PASS",
        "native_q_gate_unchanged":NATIVE_Q,"minimum_cell_volume_gate_m3_unchanged":MIN_CELL_VOLUME_M3,
        "all185_775":"REQUIRED_NOT_RUN","physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"}
    if out_path:_write_json(out_path,result)
    return result

def _audit_face_id_map(audit,loaded):
    roles=audit.get("roles",{})
    # The 775 owner uses positive role namespaces plus IDs 900000..901089 for its two 90-face walls.
    # All-185 IDs are negative. Its changed query records start at max(positive IDs)+1,000,000.
    facebase={"D":0,"O":300000,"I":600000}
    maxid=901089
    for r in ("D","O","I"):
        nf=len(loaded["baseline"][r]["faces"])
        maxid=max(maxid,facebase[r]+nf-1)
    nextid=maxid+1_000_000
    row_to_id={}
    for r in ("D","O","I"):
        rr=np.asarray(roles[r].get("changed_candidate_face_rows",[]),np.int64)
        start=nextid
        for i,row in enumerate(rr):row_to_id[(r,int(row))]=start+i
        nextid+=len(rr)
    return row_to_id

def prune_from_775(config_path,run_dir,audit_path,out_dir):
    x=load_config(config_path);rd=Path(run_dir).resolve();state_path=rd/"state.json"
    state=json.loads(state_path.read_text());auditp=Path(audit_path).resolve();audit=json.loads(auditp.read_text())
    if audit.get("config",{}).get("sha256") is None:raise ValueError("775 report does not pin its config")
    if audit.get("schema")!="h706.combined-current-exact-surface-diff-audit.v1":
        raise ValueError("unexpected 775 report schema")
    for r in ("O","I","D"):
        base=x["cfg"]["baseline_surfaces"][r];ar=audit["roles"][r]
        if ar["baseline"]["sha256"]!=base["sha256"]:raise ValueError("775 baseline is stale for "+r)
    ip=rd/"candidate-I-typed-current.npz"
    if audit["roles"]["I"]["candidate"]["path"]!=str(ip) or audit["roles"]["I"]["candidate"]["sha256"]!=sha(ip):
        raise ValueError("775 candidate I pin does not match this run")
    if audit["roles"]["O"]["candidate"]["sha256"]!=x["cfg"]["baseline_surfaces"]["O"]["sha256"] or audit["roles"]["D"]["candidate"]["sha256"]!=x["cfg"]["baseline_surfaces"]["D"]["sha256"]:
        raise ValueError("775 unchanged O/D candidates are not the current baseline")
    row_to_id=_audit_face_id_map(audit,x);id_to_op={}
    for oi,op in enumerate(state["operation_records"]):
        for row in op["surface_changed_output_face_rows"]:
            fid=row_to_id.get(("I",int(row)))
            if fid is not None:
                if fid in id_to_op and id_to_op[fid]!=oi:raise ValueError("changed face has multiple operation owners")
                id_to_op[fid]=oi
    badids=set();paircounts={}
    for key in ("changed_vs_complete_retained_OID_both_walls_all185","changed_OID_joint_self_pairs"):
        g=audit.get(key,{})
        pairs=g.get("triangle_pairs",[])
        if int(g.get("count",-1))!=len(pairs):raise ValueError("775 witness list is incomplete; refusing pruning")
        paircounts[key]=len(pairs)
        for pair in pairs:
            if len(pair)!=2:raise ValueError("malformed 775 triangle pair")
            badids.update(map(int,pair))
    badops=sorted({id_to_op[fid] for fid in badids if fid in id_to_op})
    all_rows=set(range(len(state["operation_records"])))
    if badids and not badops:raise ValueError("775 witnesses do not map to an accepted operation")
    keepops=[op for i,op in enumerate(state["operation_records"]) if i not in set(badops)]
    edges=np.asarray([[int(o["keep"]),int(o["drop"])] for o in keepops],np.int64).reshape((-1,2))
    receipt={"schema":"h706.coupled-anchor-collapse-775-prune.v1","input_run_state":{"path":str(state_path),"sha256":sha(state_path)},
        "775_report":{"path":str(auditp),"sha256":sha(auditp)},"775_config":audit["config"],
        "witness_pair_counts":paircounts,"witness_face_ids":sorted(badids),
        "pruned_operation_indices":badops,"pruned_operations":[{"keep":state["operation_records"][i]["keep"],"drop":state["operation_records"][i]["drop"]} for i in badops],
        "kept_operation_count":len(keepops),"physical_admission":False,"native_steps":0}
    result=run_from_config(config_path,out_dir,edges_override=edges,prune_receipt=receipt)
    return result

def promote_candidate(run_dir,replay_path,audit_path,receipt_path,out_path):
    rd=Path(run_dir).resolve();statep=rd/"state.json";state=json.loads(statep.read_text())
    rp=Path(replay_path).resolve();replay=json.loads(rp.read_text());ap=Path(audit_path).resolve();audit=json.loads(ap.read_text())
    lp=Path(receipt_path).resolve();loader=json.loads(lp.read_text())
    if replay.get("status")!="PASS_INDEPENDENT_SOURCE_REPLAY":raise ValueError("independent replay did not pass")
    if replay.get("run_state",{}).get("sha256")!=sha(statep):raise ValueError("independent replay is stale")
    if audit.get("status")!="PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185":raise ValueError("775 exact all185 audit did not pass")
    if not str(loader.get("status","")).startswith("PASS_ACTUAL_697_217_TYPED_"):raise ValueError("actual 697/217 typed I loader did not pass")
    expected_surface=state.get("outputs",{}).get("candidate-I-typed-current.npz",{})
    if loader.get("typed_source_sha256")!=expected_surface.get("sha256"):raise ValueError("actual 697 receipt is for a different typed I candidate")
    if loader.get("typed_source_path")!=expected_surface.get("path"):raise ValueError("actual 697 receipt path pin mismatch")
    for k in ("changed_vs_complete_retained_OID_both_walls_all185","changed_OID_joint_self_pairs"):
        if audit.get(k,{}).get("count")!=0:raise ValueError("775 exact collision/self count is nonzero")
    context=state.get("775_context_config",{})
    if audit.get("config",{}).get("sha256")!=context.get("sha256"):raise ValueError("775 audit is not for this run context")
    if replay.get("surface_output",{}).get("sha256")!=state.get("outputs",{}).get("candidate-I-typed-current.npz",{}).get("sha256"):
        raise ValueError("independent replay surface candidate mismatch")
    entry={"schema":"h706.coupled-anchor-collapse-geometry-increment.v1","status":"READY_FOR_PARENT_BRANCH_REVIEW",
        "run_state":{"path":str(statep),"sha256":sha(statep)},"independent_replay":{"path":str(rp),"sha256":sha(rp)},
        "exact_775_all185":{"path":str(ap),"sha256":sha(ap)},
        "actual_697_217":{"path":str(lp),"sha256":sha(lp),"status":loader.get("status")},
        "outputs":state.get("outputs"),"accepted_operations":state.get("accepted_operations"),
        "native_q_gate_unchanged":NATIVE_Q,"minimum_cell_volume_gate_m3_unchanged":MIN_CELL_VOLUME_M3,
        "all185":"PASS","mass_reconciliation":"PENDING","physical_admission":False,"native_steps":0,
        "note":"Construction candidate only; no central checkpoint mutation or physical/native admission."}
    _write_json(out_path,entry);return entry

def topology_fixture():
    p=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]],np.float32)
    old=M.orient(p,np.asarray([[0,1,2,3],[1,2,3,4]],np.int64));f=boundary(old)
    ok,w=surface_link(f,0,1);assert ok,w
    mapped=np.where(old==1,0,old);dead=np.any(np.diff(np.sort(mapped,axis=1),axis=1)==0,axis=1);new=mapped[~dead]
    expected=np.where(f==1,0,f);fk=~np.any(np.diff(np.sort(expected,axis=1),axis=1)==0,axis=1);expected=expected[fk]
    assert len(new)==1 and np.array_equal(T.boundary_identity(boundary(new)),T.boundary_identity(expected))
    return {"status":"PASS","old_cells":len(old),"new_cells":len(new),"retired_edge_cells":int(dead.sum()),
            "surface_link":"PASS","exact_oriented_boundary_after_quotient":"PASS"}

def fixture():
    topo=topology_fixture();side=np.float32(3e-4);height=np.float32(.9e-8);H=float(side)*np.sqrt(3)/2
    x0=np.asarray([4.5e-4,0,0],np.float32);fixed={};oldf={};olde={};local={};region={}
    for role,sign in (("Skin",-1),("Fat",1)):
        z=np.float32(sign*height);tri=np.asarray([[0,0,z],[side,0,z],[side/2,H,z]],np.float32)
        p=np.vstack((x0,tri)).astype(np.float32);tt=M.orient(p,np.asarray([[0,1,2,3]],np.int64))
        ordered=anchor_order(tt,0);fixed[role]=p[ordered[:,1:]]
        d,_=measurements(p,tt);oldf[role]=float(np.sum(d));olde[role]=int(exact_det_sum(p,tt))
        local[role]=oldf[role];region[role]=oldf[role]
    q0={r:float(measurements(np.vstack((x0,fixed[r].reshape(-1,3))),
        np.column_stack((np.zeros(1,np.int64),np.arange(1,4,dtype=np.int64).reshape(1,3))))[1][0]) for r in fixed}
    assert max(q0.values())<NATIVE_Q,q0
    fit=fit_common_anchor(x0,fixed,oldf,olde,local,region,move_cap_m=.0005)
    assert fit["status"]=="PASS",fit
    assert fit["minimum_new_native_q"]>=NATIVE_Q and fit["exact_paired_volume_delta_integer"]=="0"
    assert fit["anchor_displacement_m"]<=.0005
    return {"status":"PASS_COUPLED_ANCHOR_COLLAPSE_FIXTURE","topology":topo,
        "endpoint_only_q_by_role":q0,"optimized_anchor_world_m":fit["anchor_point_world_m"],
        "optimized_displacement_m":fit["anchor_displacement_m"],"minimum_new_native_q":fit["minimum_new_native_q"],
        "exact_delta_integer_by_role":{r:v["exact_det_delta_integer"] for r,v in fit["exact_volume_by_role"].items()},
        "exact_paired_volume_delta_integer":fit["exact_paired_volume_delta_integer"],
        "move_cap_m":fit["move_cap_m"],"native_thresholds_unchanged":True,
        "all185_775":"NOT RUN; mandatory for scene candidates","physical_admission":False,"native_steps":0}




BACK_SCOPE_PATH=ROOT/"current-back412-scope-597-root/run003/output-001/protected-back412-aabb-current-O558-I50-D264-rows.npz"
BACK_SCOPE_SHA="1abbf3c5887cfa507c303e845dafbb78d247e6b7a4893722a5d81ed202fd2676"
BACK_REFERENCE_I=ROOT/"paired-I-local-cavity-527-root/run052-current-I-batch-typed-replay/typed-replay-006/candidate-I-typed-current.npz"
BACK_REFERENCE_I_SHA="e7d76ae905c15ae3ad5c5dda7603cfdd18399934c12721b942fc4903571970a6"

def _f32_coord_row_map(points):
    out={}
    for i,p in enumerate(np.asarray(points,np.float32)):
        k=f32key(p)
        if k in out:raise ValueError("duplicate F32 surface coordinates")
        out[k]=i
    return out

def state_config(state_path,out_dir,*,move_cap_m=.0005,max_operations=32,time_limit_s=90,
                 max_edge_length_m=.0005,max_edges=100000):
    state_path=Path(state_path).resolve();out=Path(out_dir).resolve()
    state=json.loads(state_path.read_text())
    for key in ("source_surfaces","Skin_current","Fat_current","exact_context"):
        if key not in state:raise ValueError("verified state is missing "+key)
    if state.get("physical_admission") is not False or state.get("native_steps",0)!=0:
        raise ValueError("state wrapper accepts only construction-only verified states")
    if state.get("source_audit_status") not in (None,"PASS_SOURCE_ONLY_EXACT_SURFACE_AND_VOLUME_BOUNDARY"):
        raise ValueError("state source audit is not passing")
    context=state["exact_context"];cp=Path(context["path"]).resolve()
    if sha(cp)!=context["sha256"]:raise ValueError("state exact-context receipt SHA mismatch")
    cr=json.loads(cp.read_text())
    if not str(cr.get("status","")).startswith("PASS_EXACT_"):raise ValueError("state all185 context is not passing")
    for k in ("changed_vs_complete_retained_OID_both_walls_all185","changed_OID_joint_self_pairs"):
        if cr.get(k,{}).get("count")!=0:raise ValueError("state current surfaces have nonzero exact-context pairs")
    surfaces={}
    for r in ("O","I","D"):surfaces[r]=state["source_surfaces"][r]
    # Pinned back-scope rows are reused only after proving their reference coordinates
    # remain exact in the selected current I row domain.
    if sha(BACK_SCOPE_PATH)!=BACK_SCOPE_SHA or sha(BACK_REFERENCE_I)!=BACK_REFERENCE_I_SHA:
        raise ValueError("back-scope or exact reference-I pin changed")
    back=load_pinned_npz({"path":str(BACK_SCOPE_PATH),"sha256":BACK_SCOPE_SHA},"back412 scope")
    refi=load_pinned_npz({"path":str(BACK_REFERENCE_I),"sha256":BACK_REFERENCE_I_SHA},"back412 reference I")
    irow=np.asarray(back["I_vertex_rows"],np.int64)
    currentI=load_pinned_npz(surfaces["I"],"current I")
    ip=np.asarray(currentI["points_world_m_f32"],np.float32)
    rp=np.asarray(refi["points_world_m_f32"],np.float32)
    if len(ip)!=len(rp) or np.any(irow<0) or np.any(irow>=len(ip)):
        raise ValueError("current I point-row domain differs from authenticated back scope")
    if not np.array_equal(ip[irow],rp[irow]):
        bad=irow[np.any(ip[irow]!=rp[irow],axis=1)]
        raise ValueError(f"back412 protected I coordinates moved at rows {bad[:16].tolist()}")
    # Use the pinned registered routes and map their exact F32 coordinates to this I.
    routes=json.loads(A.PROT.read_text())["surface_routes"]
    coordmap=_f32_coord_row_map(ip);route_rows=[]
    for rec in routes:
        if rec.get("role")!="I":continue
        key=f32key(np.asarray(rec["target_world_m_f32"],np.float32))
        if key not in coordmap:raise ValueError("registered I route anchor absent from current state")
        route_rows.append(coordmap[key])
    route_rows=np.asarray(sorted(set(route_rows)),np.int64)
    protected=np.asarray(sorted(set(map(int,irow))|set(map(int,route_rows))),np.int64)
    # Compile candidate I surface edges that are true interior edges and occur in
    # at least one currently native-bad Skin or Fat tetrahedron.
    faces=np.asarray(currentI["faces"],np.int64)
    es=np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1)
    edges,edgecounts=np.unique(es,axis=0,return_counts=True)
    surface_edges=edges[edgecounts==2]
    score=np.zeros(len(surface_edges),np.int64)
    edge_lookup={tuple(map(int,e)):i for i,e in enumerate(surface_edges)}
    role_recs={}
    for role in ("Skin","Fat"):
        ent=state[role+"_current"];vol=load_pinned_npz(ent,"current "+role+" volume")
        vp=np.asarray(vol["points_world_m"],np.float32);tt=np.asarray(vol["tetrahedra"],np.int64)
        det,qv=measurements(vp,tt);bad=~M.passes(det,qv)
        cross=coord_crosswalk(ip,vp)
        if cross is None or np.any(cross<0):raise ValueError(role+" surface/volume F32 crosswalk is not complete and unique")
        inv=np.full(len(vp),-1,np.int64);inv[cross]=np.arange(len(ip),dtype=np.int64)
        counts=Counter()
        for pair in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
            rr=tt[bad][:,pair];a=inv[rr[:,0]];b=inv[rr[:,1]];ok=(a>=0)&(b>=0)&(a!=b)
            for x,y in np.sort(np.column_stack((a[ok],b[ok])),axis=1):
                ei=edge_lookup.get((int(x),int(y)))
                if ei is not None:counts[ei]+=1
        for ei,n in counts.items():score[ei]+=int(n)
        role_recs[role]={"path":ent["path"],"sha256":ent["sha256"],
            "cells":int(len(tt)),"native_bad":int(np.count_nonzero(bad)),
            "surface_volume_point_crosswalk":"PASS"}
    plen=np.linalg.norm(ip[surface_edges[:,0]].astype(np.float64)-ip[surface_edges[:,1]].astype(np.float64),axis=1)
    valid=(score>0)&(plen<=float(max_edge_length_m))
    # Exclude full surface edge stars intersecting authenticated back or route anchors,
    # matching the 702 source protection predicate before scheduling.
    protectset=set(map(int,protected));eligible=[]
    for ei in np.flatnonzero(valid):
        k,d=map(int,surface_edges[ei])
        fr=np.flatnonzero(np.any((faces==k)|(faces==d),axis=1))
        if protectset & set(map(int,faces[fr].ravel())):continue
        eligible.append((int(ei),k,d))
    eligible.sort(key=lambda z:(-int(score[z[0]]),float(plen[z[0]]),z[1],z[2]))
    eligible=eligible[:int(max_edges)]
    ordered=[]
    for ei,k,d in eligible:ordered.extend(((k,d),(d,k)))
    edges_entry_dir=out/"inputs";edges_entry_dir.mkdir(parents=True,exist_ok=False)
    edgepath=edges_entry_dir/"candidate-I-edges.npz"
    np.savez_compressed(edgepath,candidate_edges=np.asarray(ordered,np.int64).reshape((-1,2)),
        edge_length_m=np.asarray([plen[ei] for ei,_,_ in eligible],np.float64),
        bad_tet_incidence=np.asarray([score[ei] for ei,_,_ in eligible],np.int64),
        undirected_edges=np.asarray([[k,d] for _,k,d in eligible],np.int64).reshape((-1,2)))
    protpath=edges_entry_dir/"protected-I-rows.npz";np.savez_compressed(protpath,
        protected_surface_rows=protected,route_anchor_rows=route_rows,back412_rows=irow)
    cfg={"schema":"h706.coupled-anchor-collapse-run-config.v1","surface_I":surfaces["I"],
        "baseline_surfaces":surfaces,
        "volumes":{"Skin":state["Skin_current"],"Fat":state["Fat_current"]},
        "candidate_edges":{"path":str(edgepath),"sha256":sha(edgepath),"key":"candidate_edges"},
        "protected_surface_rows":{"path":str(protpath),"sha256":sha(protpath),"key":"protected_surface_rows"},
        "route_anchor_rows":{"path":str(protpath),"sha256":sha(protpath),"key":"route_anchor_rows"},
        "move_cap_m":float(move_cap_m),"max_operations":int(max_operations),"time_limit_s":float(time_limit_s)}
    cfgpath=edges_entry_dir/"run-config.json";_write_json(cfgpath,cfg)
    prep={"schema":"h706.coupled-anchor-collapse-state-preparation.v1","state":{"path":str(state_path),"sha256":sha(state_path)},
        "current_exact_context":{"path":str(cp),"sha256":sha(cp),"status":cr["status"]},
        "current_surface_pins":surfaces,"volume_inputs":role_recs,
        "back_scope":{"path":str(BACK_SCOPE_PATH),"sha256":sha(BACK_SCOPE_PATH),"back412_I_rows":len(irow),
            "coordinates_bitwise_unchanged":True},
        "registered_route_anchor_rows":len(route_rows),"protected_surface_rows":len(protected),
        "candidate_surface_I_interior_edges":int(len(surface_edges)),"bad_incident_eligible_edges_before_cap":int(np.count_nonzero(valid)),
        "scheduled_undirected_edges":len(eligible),"scheduled_orientations":len(ordered),
        "edge_length_cap_m":float(max_edge_length_m),"max_edges":int(max_edges),
        "edge_roster":{"path":str(edgepath),"sha256":sha(edgepath)},
        "run_config":{"path":str(cfgpath),"sha256":sha(cfgpath)},
        "native_q_gate_unchanged":NATIVE_Q,"physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"}
    _write_json(edges_entry_dir/"state-preparation.json",prep)
    return cfgpath,prep

def run_verified_state(state_path,out_dir,*,move_cap_m=.0005,max_operations=32,time_limit_s=90,
                       max_edge_length_m=.0005,max_edges=100000):
    out=Path(out_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    cfgpath,prep=state_config(state_path,out,move_cap_m=move_cap_m,max_operations=max_operations,
        time_limit_s=time_limit_s,max_edge_length_m=max_edge_length_m,max_edges=max_edges)
    proposal=out/"proposal"
    report=run_from_config(cfgpath,proposal)
    wrapper={"schema":"h706.coupled-anchor-collapse-state-run.v1","preparation":prep,
        "candidate_run":{"path":str(proposal/"state.json"),"sha256":sha(proposal/"state.json"),
            "status":report.get("status"),"accepted_operations":report.get("accepted_operations",0)},
        "all185_775":"REQUIRED_NOT_RUN","actual_697_217":"REQUIRED_NOT_RUN",
        "physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"}
    _write_json(out/"state-wrapper.json",wrapper)
    return wrapper


def main():
    ap=argparse.ArgumentParser(description="Coupled I-anchor edge-collapse candidate, replay, 775 context, witness pruning and local promotion manifest.")
    ap.add_argument("--mode",choices=("fixture","run","context","replay","prune-775","promote"),default="fixture")
    ap.add_argument("--config",type=Path);ap.add_argument("--state",type=Path);ap.add_argument("--run-dir",type=Path);ap.add_argument("--out","--output",dest="out",type=Path)
    ap.add_argument("--move-cap-m",type=float,default=.0005);ap.add_argument("--max-operations",type=int,default=32)
    ap.add_argument("--seconds",type=float,default=90);ap.add_argument("--max-edge-length-m",type=float,default=.0005);ap.add_argument("--max-edges",type=int,default=100000)
    ap.add_argument("--775-report",type=Path);ap.add_argument("--replay-report",type=Path);ap.add_argument("--697-receipt",type=Path)
    a=ap.parse_args()
    if a.state is not None:
        if a.out is None:ap.error("--state requires --output")
        r=run_verified_state(a.state,a.out,move_cap_m=a.move_cap_m,max_operations=a.max_operations,
            time_limit_s=a.seconds,max_edge_length_m=a.max_edge_length_m,max_edges=a.max_edges)
        print(json.dumps(r,indent=2));return
    if a.mode=="fixture":
        r=fixture()
        if a.out:
            a.out.mkdir(parents=True,exist_ok=False);p=a.out/"fixture-report.json";_write_json(p,r)
            r["report_path"]=str(p);r["report_sha256"]=sha(p)
        print(json.dumps(r,indent=2));return
    if a.mode=="run":
        if not a.config or not a.out:ap.error("run requires --config and --out")
        r=run_from_config(a.config,a.out);print(json.dumps({k:v for k,v in r.items() if k!="operation_records"},indent=2));return
    if a.mode=="context":
        if not a.config or not a.run_dir or not a.out:ap.error("context requires --config --run-dir --out")
        x=load_config(a.config);r=json.loads((a.run_dir/"state.json").read_text())
        q=make_775_config(a.run_dir,x,r,a.out);print(json.dumps(q,indent=2));return
    if a.mode=="replay":
        if not a.config or not a.run_dir or not a.out:ap.error("replay requires --config --run-dir --out")
        r=independent_replay(a.config,a.run_dir,a.out);print(json.dumps(r,indent=2));return
    if a.mode=="prune-775":
        if not a.config or not a.run_dir or not a.out or not a.__dict__.get("775_report"):ap.error("prune-775 requires --config --run-dir --775-report --out")
        r=prune_from_775(a.config,a.run_dir,a.__dict__["775_report"],a.out)
        print(json.dumps({k:v for k,v in r.items() if k!="operation_records"},indent=2));return
    if a.mode=="promote":
        if not a.run_dir or not a.replay_report or not a.__dict__.get("775_report") or not a.__dict__.get("697_receipt") or not a.out:
            ap.error("promote requires --run-dir --replay-report --775-report --697-receipt --out")
        r=promote_candidate(a.run_dir,a.replay_report,a.__dict__["775_report"],a.__dict__["697_receipt"],a.out);print(json.dumps(r,indent=2));return

if __name__=="__main__":
    if "--mode" not in sys.argv or sys.argv[sys.argv.index("--mode")+1]!="replay":
        raise SystemExit("788 root is restricted to independent replay")
    main()
