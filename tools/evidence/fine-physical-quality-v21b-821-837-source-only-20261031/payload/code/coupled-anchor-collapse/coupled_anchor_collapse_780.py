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
        vr[role]={"points":pnew,"new_tets":s["new"],"parent_rows":s["parents"],"materials":s["materials"],
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
        newt=[];newm=[];newparents=[]
        for op in accepted:
            st=op["volumes"][r];vm=coord_crosswalk(surface["points_world_m_f32"],v["p"])
            if vm is None or vm[op["keep"]]<0 or vm[op["drop"]]<0:return {"status":"REJECT_AMBIGUOUS_CROSSWALK","role":r}
            u,w=map(int,(vm[op["keep"]],vm[op["drop"]]));p[u]=op["anchor_f32"];qmap[w]=u
            rr=np.asarray(st["old_rows"],np.int64);retire[rr]=True
            newt.append(st["new_tets"]);newm.append(st["materials"]);newparents.append(st["parent_rows"])
        retained=np.flatnonzero(~retire);replacement=np.concatenate(newt);repmat=np.concatenate(newm);repparent=np.concatenate(newparents)
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
        a.update(points_world_m=p,tetrahedra=t,material_indices=mat,boundary_faces=bd,desired_boundary_faces=target,
          output_cell_to_input_cell_rows=parentmap,input_cell_to_output_cell_rows=imap,retained_input_cell_rows=retained,
          replacement_parent_cell_rows=repparent,replacement_parent_offsets=np.arange(len(repparent)+1,dtype=np.int64),
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
          "retired_surface_faces":o["surface_retired_face_rows"].tolist(),
          "full_star_cells":{r:o["volume_star_cells"][r].tolist() for r in cache},
          "exact_delta_by_role":{r:str(o["volumes"][r]["exact_delta"]) for r in cache},
          "native_bad_gain_by_role":o["native_bad_gain_by_role"]} for o in accepted]}
    if out is not None:
        np.savez_compressed(out/"candidate-I-typed-current.npz",**typed)
        for r,a in volumeout.items():np.savez_compressed(out/(r.lower()+"-coupled-anchor-volume-proposal.npz"),**a)
        report["producer"]={"path":str(Path(__file__)),"sha256":sha(Path(__file__))}
        (out/"state.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def prune_775_witnesses(operations,report):
    """Prune exactly operations owning face IDs reported by 775."""
    bad=set()
    for key in ("changed_vs_complete_retained_OID_both_walls_all185","changed_OID_joint_self_pairs"):
        g=report.get(key,{})
        for a,b in g.get("triangle_pairs",[]):bad.update((int(a),int(b)))
    badops={i for i,o in enumerate(operations) if bad&set(map(int,o.get("all185_global_changed_face_ids",[])))}
    return [o for i,o in enumerate(operations) if i not in badops],[{"operation_index":i,"reason":"775 exact changed-face witness"} for i in sorted(badops)]

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

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--fixture",action="store_true");ap.add_argument("--output",type=Path);a=ap.parse_args()
    if not a.fixture:ap.error("bounded entrypoint only runs the synthetic fixture; scene arrays are not loaded")
    r=fixture()
    if a.output:
        a.output.mkdir(parents=True,exist_ok=False);p=a.output/"fixture-report.json";p.write_text(json.dumps(r,indent=2)+"\n")
        r["report_path"]=str(p);r["report_sha256"]=sha(p)
    print(json.dumps(r,indent=2))
if __name__=="__main__":main()
