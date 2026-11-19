"""Fit a source-space skin correction to typed witnesses on the two retained poses."""
import importlib.util
import json
import mmap
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/"resting-clearance-and-atrium-003"
sys.path.insert(0,str(PRIOR))
from prepare_skin import BASE, bindings, capture, decode, full_self
from rebind_arm_witnesses import axes
from correct_cardiac_field_soft import Inequalities

POSES=(("wider-release-variable-4",2000),("wider-release-control-16",8000))
MAX_COMPONENT_MM=2.0
MARGIN_MM=.02


def normal(tri):
    vector=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    area=np.linalg.norm(vector,axis=1)
    assert np.isfinite(area).all() and np.all(area>0)
    return vector/area[:,None],area


def witness_axes(skin, target):
    ns,_=normal(skin)
    nt,_=normal(target)
    se=np.roll(skin,-1,axis=1)-skin
    te=np.roll(target,-1,axis=1)-target
    cross=np.cross(se[:,:,None],te[:,None,:]).reshape(-1,9,3)
    basis=np.concatenate([ns[:,None],nt[:,None],-nt[:,None],cross,-cross],axis=1)
    norms=np.linalg.norm(basis,axis=2)
    basis/=np.maximum(norms[:,:,None],1.e-10)
    alignment=np.einsum("pbd,pd->pb",basis,ns)
    origin=skin[:,:1]
    gap=np.einsum("pvd,pbd->pvb",target-origin,basis).max(axis=1)
    gap-=np.einsum("pvd,pbd->pvb",skin-origin,basis).min(axis=1)
    score=np.where((norms>1e-10)&(alignment>.55),gap/(alignment*alignment+.001),np.inf)
    return basis[np.arange(len(skin)),np.argmin(score,axis=1)]


def compose(path, report, p, faces):
    records={(r["semantic"],r["stable_id"]):r for r in report["surfaces"] if r["count"]}
    receipt=json.loads(path.with_suffix(".receipt.json").read_text())
    assert report["accepted_pack_sha256"]==receipt["pack_file_sha256"]
    bodies={x["body_index"]:x for x in receipt["accepted_body_poses"]}
    data=[]
    with tempfile.TemporaryDirectory(dir=ROOT) as temp:
        pack=Path(temp)/"capture.mrvpack"
        with pack.open("wb") as file:
            subprocess.run(["/opt/homebrew/bin/zstd","-dc",str(path)+".zst"],stdout=file,check=True)
        assert bindings.audit.sha(pack)==receipt["pack_file_sha256"]
        with pack.open("rb") as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as raw:
            for key,_id,v,f in bindings.audit.primitives(raw,bodies):
                if key in records:
                    pairs=np.asarray(records.pop(key)["triangle_pairs"])
                    skin=1000*p[faces[pairs[:,0]]].astype(float)
                    target=1000*v[f[pairs[:,1]]].astype(float)
                    n=witness_axes(skin,target)
                    cap=np.einsum("pvd,pd->pv",target-skin[:,:1],n).max(axis=1)+MARGIN_MM
                    gap=np.einsum("pvd,pd->pv",skin-skin[:,:1],n)-cap[:,None]
                    data.append((pairs[:,0],n,gap))
    assert not records
    return tuple(np.concatenate(columns) for columns in zip(*data))


def main():
    model=decode((PRIOR.parent/"resting-clearance-and-atrium-002/skin-rebound.nhskin").read_bytes())
    _,reference,faces=capture(BASE)
    source=model["verts"][:,:3]
    protected=reference[:,2]<=.020
    protected[bindings.audit.clearance._exact_coordinate_tied_source_vertex_ids(source)]=True
    rows=np.sort(faces[:,[0,1,1,2,2,0]].reshape(-1,2),axis=1)
    edges,counts=np.unique(rows,axis=0,return_counts=True)
    protected[edges[counts!=2].ravel()]=True
    cases=[]
    change=np.zeros(len(source),bool)
    for name,step in POSES:
        path=PRIOR/name/f"native/accepted-geometry/step-{step}.mrvpack"
        receipt,p,f=capture(path)
        assert np.array_equal(f,faces)
        report=json.loads((PRIOR/f"{name}-exact.json").read_text())
        j=np.einsum("vb,bdc->vdc",model["weights"],axes(model,receipt)[0])
        skin_ids,n,gap=compose(path,report,p,faces)
        cases.append((p,j,skin_ids,n,gap,report))
        protected|=p[:,2]<=.020
        change[faces[skin_ids].ravel()]=True
        print(f"Prepared accepted step {step} with {len(skin_ids)} typed intersections.",flush=True)
    for _ in range(2):
        change[edges[change[edges].any(axis=1)].ravel()]=True
    index=np.full(len(source),-1,dtype=int)
    local=np.flatnonzero(change & ~protected)
    index[local]=np.arange(len(local))
    incident=faces[np.any(index[faces]>=0,axis=1)]
    edges=edges[np.any(index[edges]>=0,axis=1)]
    nd=3*len(local)
    gaps=sum(len(x[2]) for x in cases)
    L=Inequalities()
    def entries(v,vector):
        ix=index[v]
        if ix>=0:
            yield from ((3*ix+k,c) for k,c in enumerate(vector))

    constraint=0
    for p,j,skin,n,gap,_ in cases:
        for local_ids,u,cap in zip(faces[skin],n,gap):
            slack=2*nd+constraint
            for v,delta in zip(local_ids,cap):
                L.upper([*entries(v,-(u @ j[v])),(slack,-1.)], delta)
            constraint+=1
        triangles=1000*p[incident].astype(float)
        fn,area=normal(triangles)
        derivatives=np.cross(np.roll(triangles,-1,axis=1)-np.roll(triangles,-2,axis=1),fn[:,None])
        for f,d,lower in zip(incident,derivatives,area):
            L.upper([term for v,g in zip(f,d) for term in entries(v,-(g @ j[v]))],0.20*lower)
    assert constraint==gaps
    for k in range(nd):
        L.upper(((k,1.),(k+nd,-1.)),0.)
        L.upper(((k,-1.),(k+nd,-1.)),0.)
    for k,(a,b) in enumerate(edges):
        for h,v in enumerate(np.eye(3)):
            edge=2*nd+gaps+3*k+h
            L.upper([*entries(a,v),*entries(b,-v),(edge,-1.)],0.)
            L.upper([*entries(a,-v),*entries(b,v),(edge,-1.)],0.)
    cost=np.r_[np.zeros(nd),np.full(nd,.25),np.ones(gaps),np.full(3*len(edges),1.8)]
    print(f"Solving {len(local)} local source rows, {len(L.rhs)} inequalities.",flush=True)
    solution=linprog(cost,A_ub=L.matrix(len(cost)),b_ub=L.rhs,
          bounds=[(-MAX_COMPONENT_MM,MAX_COMPONENT_MM)]*nd+[(0.,None)]*(len(cost)-nd),
          method="highs",options={"time_limit":95.0})
    print(solution.message,flush=True)
    if not solution.success:
        return
    d=np.zeros_like(source)
    d[local]=solution.x[:nd].reshape(-1,3)*.001
    trials={}
    for scale in (0.25,0.5,0.75,1.):
        poses=[]
        for p,j,*_ in cases:
            moved=(p+scale*np.einsum("vij,vj->vi",j,d)).astype("<f4")
            pre,area0=normal(p[incident].astype(float))
            post,area1=normal(moved[incident].astype(float))
            poses.append(float(np.min(np.sum(pre*post,axis=1)*area1/area0)))
        trials[str(scale)]=poses
    print(trials,flush=True)
    np.savez_compressed(ROOT/"separating-plane-skin-fit.npz", displacement_m=d,
       early=(cases[0][0]+.5*np.einsum("vij,vj->vi",cases[0][1],d)).astype("<f4"),
       late=(cases[1][0]+.5*np.einsum("vij,vj->vi",cases[1][1],d)).astype("<f4"))
    report={"projected_area_ratios":trials,"reference_max_delta_mm":float(1000*np.linalg.norm(d,axis=1).max()),
            "source_rows_free":len(local),"source_rows_changed":int(np.count_nonzero(np.any(d,axis=1))),
            "fitted_crossings":gaps,"protected_rows":int(protected.sum())}
    (ROOT/"separating-plane-skin-fit.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":
    main()
