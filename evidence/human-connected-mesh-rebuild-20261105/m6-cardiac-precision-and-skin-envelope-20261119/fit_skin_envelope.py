"""Fit a single exterior envelope to exact tissue witnesses at three retained body poses."""
from collections import Counter
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, eye, hstack, vstack

from fit_cardiac_volume_grid import HERE, PREVIOUS, hash_file
sys.path.insert(0,str(PREVIOUS))
from fit_skin_source import BASE, PRIOR, bindings, capture, decode, compose, normal
from rebind_arm_witnesses import axes

def main():
    source_path=HERE.parent/"resting-clearance-and-atrium-002/skin-rebound.nhskin"
    model=decode(source_path.read_bytes())
    source=model["verts"][:,:3]
    faces=model["inds"]
    referenced=np.unique(faces)
    normal_src=np.zeros_like(source)
    normal_src[referenced]=bindings.audit.clearance._area_weighted_vertex_normals(
        source[referenced],np.searchsorted(referenced,faces))
    raw_edges=np.sort(faces[:,[0,1,1,2,2,0]].reshape(-1,2),axis=1)
    edges,degree=np.unique(raw_edges,axis=0,return_counts=True)
    pinned=np.zeros(len(source),bool)
    pinned[edges[degree!=2].ravel()]=True
    pinned[np.load(PREVIOUS/"bed-section-pins.npz")["vertices"]]=True
    pinned[bindings.audit.clearance._exact_coordinate_tied_source_vertex_ids(source)]=True
    snapshots=[
        (BASE,HERE.parent/"resting-surface-clearance-001/initial.json"),
        (PRIOR/"wider-release-variable-4/native/accepted-geometry/step-2000.mrvpack",PRIOR/"wider-release-variable-4-exact.json"),
        (PRIOR/"wider-release-control-16/native/accepted-geometry/step-8000.mrvpack",PRIOR/"wider-release-control-16-exact.json")]
    fit_data=[]
    active=np.zeros(len(source),bool)
    for raw,report in snapshots:
        proof,p,mesh=capture(raw)
        assert np.array_equal(mesh,faces)
        audit=json.loads(report.read_text())
        target,side,gap=compose(raw,audit,p,faces)
        eligible=np.any(~pinned[faces[target]],axis=1)
        target,side,gap=(obj[eligible] for obj in (target,side,gap))
        active[faces[target].ravel()]=True
        jac=np.einsum("vb,bdc->vdc",model["weights"],axes(model,proof)[0])
        outward=np.einsum("vij,vj->vi",jac,normal_src)
        fit_data.append((p,outward,target,side,gap))
        print("prepared",proof["accepted_step"],len(target),flush=True)
    for _ in range(3):
        active[edges[active[edges].any(axis=1)].ravel()]=True
    idx=np.flatnonzero(active & ~pinned)
    inverse=np.full(len(source),-1,dtype=int)
    inverse[idx]=np.arange(len(idx))
    own=edges[np.any(inverse[edges]>=0,axis=1)]
    nv=len(idx)
    n_contact=sum(len(t) for _,_,t,_,_ in fit_data)
    shape_face=faces[np.any(inverse[faces]>=0,axis=1)]
    rows=[]
    cols=[]
    vals=[]
    rhs=[]
    def add(nodes, coeff, cap, extra=()):
        row=len(rhs)
        for v,c in zip(nodes,coeff):
            if inverse[v]>=0 and c!=0:
                rows.append(row)
                cols.append(inverse[v])
                vals.append(c)
        for index,c in extra:
            rows.append(row)
            cols.append(index)
            vals.append(c)
        rhs.append(cap)

    slab=0
    for p,outward,target,side,gap in fit_data:
        for face,n,limits in zip(faces[target],side,gap):
            projection=outward[face] @ n
            for v,d,h in zip(face,projection,limits):
                add((v,),(-d,),h,((nv+slab,-1.),))
            slab+=1
        tri=p[shape_face].astype(float)*1000
        fn,area=normal(tri)
        d=np.cross(np.roll(tri,-1,axis=1)-np.roll(tri,-2,axis=1),fn[:,None])
        fd=np.einsum("fid,fid->fi",d,outward[shape_face])
        for f,c,b in zip(shape_face,fd,area):
            add(f,-c,.28*b)
    assert slab==n_contact
    lengths=1000*np.linalg.norm(source[own[:,0]]-source[own[:,1]],axis=1)
    for k,(a,b) in enumerate(own):
        off=nv+n_contact+k
        cap=.1+.42*lengths[k]
        add((a,b),(1.,-1.),0.,((off,-1.),))
        add((a,b),(-1.,1.),0.,((off,-1.),))
        add((a,b),(1.,-1.),cap)
        add((a,b),(-1.,1.),cap)
    cost=np.r_[np.full(nv,.12),np.ones(n_contact),np.full(len(own),1.2)]
    A=coo_matrix((vals,(rows,cols)),shape=(len(rhs),len(cost))).tocsr()
    print("solve",nv,n_contact,len(rhs),A.nnz,flush=True)
    result=linprog(cost,A_ub=A,b_ub=rhs,bounds=[(0,6)]*nv+[(0,None)]*(len(cost)-nv),
                    method="highs",options={"time_limit":150.0})
    print(result.message,flush=True)
    summary=dict(result=result.message,solved=bool(result.success),skin_source_sha256=hash_file(source_path),
                 free_rows=nv,pinned_rows=int(pinned.sum()),crossing_constraints=n_contact)
    if result.success:
        scalar=np.zeros(len(source))
        scalar[idx]=result.x[:nv]*.001
        delta=normal_src*scalar[:,None]
        summary.update(nonzero_rows=int(np.count_nonzero(scalar)),max_displacement_m=float(scalar.max()),
                       positive_slacks=int(np.count_nonzero(result.x[nv:nv+n_contact]>1.e-6)))
        fit=[]
        for p,outward,*_ in fit_data:
            before,areas=normal(p[shape_face].astype(float))
            predicted=(p+outward*scalar[:,None]).astype("<f4")
            after,areas2=normal(predicted[shape_face].astype(float))
            fit.append(dict(min_area_ratio=float(np.min(np.sum(before*after,axis=1)*areas2/areas))))
        summary["fitted_pose_orientation"]=fit
        np.savez_compressed(HERE/"skin-envelope-proposal.npz",displacement_m=delta,
             initial_delta=fit_data[0][1]*scalar[:,None],early_delta=fit_data[1][1]*scalar[:,None],
             late_delta=fit_data[2][1]*scalar[:,None])
    (HERE/"skin-envelope-fit.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(summary,flush=True)

if __name__=="__main__":
    main()
