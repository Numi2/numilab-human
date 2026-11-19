"""Jointly separate the remaining self-contact witnesses on the retained skin poses."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from fit_cardiac_volume_grid import HERE, PREVIOUS
sys.path.insert(0, str(PREVIOUS))
from fit_skin_source import BASE, PRIOR, bindings, capture, decode, normal
from correct_cardiac_field_trust import Inequalities, best_axis, gap_pairs
from rebind_arm_witnesses import axes

def main():
    model = decode((HERE/"skin-envelope-native.nhskin").read_bytes())
    source = model["verts"][:, :3].astype(float)
    faces = model["inds"]
    edge_faces = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    edges, degree = np.unique(edge_faces, axis=0, return_counts=True)
    protected=np.unique(np.r_[np.load(PREVIOUS/"bed-section-pins.npz")["vertices"],
        edges[degree!=2].ravel(),
        bindings.audit.clearance._exact_coordinate_tied_source_vertex_ids(source.astype("<f4"))])
    delta = np.load(HERE/"skin-envelope-rest-and-motion.npz")["displacement_m"]
    samples = [
        (BASE, HERE.parent/"resting-surface-clearance-001/initial.json"),
        (PRIOR/"wider-release-variable-4/native/accepted-geometry/step-2000.mrvpack",
         PRIOR/"wider-release-variable-4-exact.json"),
        (PRIOR/"wider-release-control-16/native/accepted-geometry/step-8000.mrvpack",
         PRIOR/"wider-release-control-16-exact.json")]
    poses=[]
    free=np.zeros(len(source),bool)
    for pack, report in samples:
        receipt, p, f = capture(pack)
        baseline=json.loads(report.read_text())
        assert np.array_equal(f,faces) and receipt["pack_file_sha256"]==baseline["accepted_pack_sha256"]
        derivative=np.einsum("vb,bdc->vdc",model["weights"],axes(model,receipt)[0])
        current=p+np.einsum("vij,vj->vi",derivative,delta)
        self_pairs=np.asarray(baseline["self_intersections"]["triangle_pairs"],dtype=np.int64).reshape(-1,2)
        free[faces[self_pairs].reshape(-1)]=True
        poses.append((current*1000.,derivative,self_pairs))
    for _ in range(2):
        free[edges[np.any(free[edges],axis=1)].ravel()]=True
    free[protected]=False
    movable=np.flatnonzero(free)
    reference=poses[0][0]
    cases=[(source*1000.,None,np.empty((0,2),dtype=np.int64)),*poses]
    neighbors=edges[np.any(free[edges],axis=1)]
    affected=faces[np.any(free[faces],axis=1)]
    index=np.full(len(source),-1,dtype=int)
    index[movable]=np.arange(len(movable))
    offsets=3*len(movable)
    L=Inequalities()
    def entries(v,d,j=None):
        k=index[v]
        if k>=0:
            vec=d if j is None else d @ j[v]
            yield from ((3*k+i,c) for i,c in enumerate(vec))
    witnesses=[]
    blocked=[]
    for points,j,records in cases:
        for a,b in records:
            x,y=faces[a],faces[b]
            if not np.any(free[np.r_[x,y]]) or len(np.intersect1d(x,y))==2:
                blocked.append([int(a),int(b)])
                continue
            pairs=gap_pairs(x,y)
            g=best_axis(x,y,points,reference,pairs)
            slack=2*offsets+len(witnesses)
            witnesses.append([int(a),int(b)])
            for u,v in pairs:
                L.upper([*entries(u,g,j),*entries(v,-g,j),(slack,-1.)],
                        float((points[v]-points[u])@g)-.02)
        tri=points[affected]
        n,areas=normal(tri)
        gradient=np.cross(np.roll(tri,-1,axis=1)-np.roll(tri,-2,axis=1),n[:,None])
        for f,d,a in zip(affected,gradient,areas):
            L.upper([c for v,fd in zip(f,d) for c in entries(v,-fd,j)],.22*a)
    for k in range(offsets):
        L.upper(((k,1.),(offsets+k,-1.)),0)
        L.upper(((k,-1.),(offsets+k,-1.)),0)
    variation=2*offsets+len(witnesses)
    for k,(v,w) in enumerate(neighbors):
        cap=.11+.22*np.linalg.norm(source[v]-source[w])*1000
        for i,n in enumerate(np.eye(3)):
            s=variation+3*k+i
            L.upper([*entries(v,n),*entries(w,-n),(s,-1.)],0)
            L.upper([*entries(v,-n),*entries(w,n),(s,-1.)],0)
            L.upper([*entries(v,n),*entries(w,-n)],cap)
            L.upper([*entries(v,-n),*entries(w,n)],cap)
    cost=np.r_[np.zeros(offsets),np.full(offsets,.25),np.full(len(witnesses),30.),
               np.full(3*len(neighbors),1.1)]
    print(f"{len(witnesses)} witness constraints, {len(blocked)} blocked, {len(movable)} movable vertices.",flush=True)
    fit=linprog(cost,A_ub=L.matrix(len(cost)),b_ub=L.rhs,
                bounds=[(-2.1,2.1)]*offsets+[(0,None)]*(len(cost)-offsets),
                method="highs",options={"time_limit":120})
    if not fit.success:
        raise RuntimeError(fit.message)
    patch=np.zeros_like(source)
    patch[movable]=fit.x[:offsets].reshape(-1,3)*.001
    outputs={"source_delta":patch,"baseline_source_delta":delta}
    for name,(_,j,_) in zip(("initial","early","late"),poses):
        outputs[name+"_delta"]=np.einsum("vij,vj->vi",j,patch)
    np.savez_compressed(HERE/"skin-self-fit.npz",**outputs)
    proof=dict(blocked=blocked,witnesses=len(witnesses),moved=int(np.any(patch!=0,axis=1).sum()),
               residuals=int(np.count_nonzero(fit.x[2*offsets:variation]>1.e-7)),
               max_movement_m=float(np.linalg.norm(patch,axis=1).max()))
    (HERE/"skin-self-fit.json").write_text(json.dumps(proof,indent=2)+"\n")
    print(proof,flush=True)

if __name__=="__main__":
    main()
