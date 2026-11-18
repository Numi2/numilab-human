"""Fit one bounded skin enclosure to the retained release and settled witnesses."""
import json
import mmap
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, eye, hstack, vstack

from prepare_skin import BASE, HERE, ROOT, bindings, capture, decode, full_self
from rebind_arm_witnesses import axes

END = HERE/"rebound-enabled-four/native/accepted-geometry/step-2000.mrvpack"

def face_triangles(path, audit):
    """Load only triangles for already reported pairs; bank/row identities are explicit."""
    receipt=json.loads(path.with_suffix(".receipt.json").read_text())
    assert audit["accepted_pack_sha256"]==receipt["pack_file_sha256"]
    bodies={v["body_index"]:v for v in receipt["accepted_body_poses"]}
    needed={(r["semantic"],r["stable_id"]):r for r in audit["surfaces"] if r["count"]}
    with tempfile.TemporaryDirectory(dir=HERE) as temp:
        unpacked=Path(temp)/"capture.mrvpack"
        with unpacked.open("wb") as f:
            subprocess.run(["/opt/homebrew/bin/zstd","-dc",str(path)+".zst"],stdout=f,check=True)
        assert bindings.audit.sha(unpacked) == receipt["pack_file_sha256"]
        with unpacked.open("rb") as stream, mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as raw:
            for key,_id,v,f in bindings.audit.primitives(raw,bodies):
                if key not in needed:
                    continue
                record=needed.pop(key)
                ids=np.asarray(record["triangle_pairs"])
                yield ids[:,0],v[f[ids[:,1]]].astype(float)
    assert not needed

def csr(rows,cols,values,shape):
    return coo_matrix((values,(rows,cols)),shape=shape).tocsr()

def face_orientation(p,f):
    xyz=p[f].astype(float)
    return np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])

def main():
    packed=HERE/"skin-rebound.nhskin"
    model=decode(packed.read_bytes())
    source=model["verts"][:,:3].astype(float)
    weights=model["weights"].astype(float)
    fsrc=model["inds"].astype(np.int64)
    raw_audits=[ROOT/"resting-surface-clearance-001/initial.json",HERE/"rebound-four-exact.json"]
    captures=[capture(path) for path in (BASE,END)]
    audits=[json.loads(f.read_text()) for f in raw_audits]
    f=captures[0][2]
    for rec,p,fp in captures:
        assert np.array_equal(f,fp)
    referenced=np.unique(f)
    source_n=bindings.audit.clearance._area_weighted_vertex_normals(source[referenced],np.searchsorted(referenced,f))
    normals=np.zeros_like(source)
    normals[referenced]=source_n
    # Preserve source boundaries, coincident seams, and all near-bed points.
    protected=np.any([p[:,2]<=.020 for _,p,_ in captures],axis=0)
    protected[bindings.audit.clearance._exact_coordinate_tied_source_vertex_ids(source.astype("<f4"))]=True
    edges,counts=np.unique(np.sort(f[:,[0,1,1,2,2,0]].reshape(-1,2),axis=1),axis=0,return_counts=True)
    protected[edges[counts!=2].reshape(-1)]=True

    samples=[]
    seeds=[]
    for path,a,(rec,p,_) in zip((BASE,END),audits,captures):
        jac=np.einsum("vi,ijk->vjk",weights,axes(model,rec)[0])
        direction=np.einsum("vij,vj->vi",jac,normals)
        fn=face_orientation(p,f)
        fn/=np.linalg.norm(fn,axis=1)[:,None]
        need=np.zeros(len(f))
        records=0
        for skin_ids,target in face_triangles(path,a):
            # Gap to the entire target face along the skin-facing plane.
            nn=fn[skin_ids]
            target_end=np.einsum("pct,pt->pc",target,nn).max(axis=1)
            own=np.einsum("pt,pt->p",p[f[skin_ids,0]],nn)
            np.maximum.at(need,skin_ids,np.maximum(0,target_end-own)+0.00025)
            records+=len(skin_ids)
        active=np.flatnonzero(need>0)
        assert records==sum(x["count"] for x in a["surfaces"])
        print("prepared",rec["accepted_step"],records,len(active),flush=True)
        samples.append((p,fn,direction,need,active))
        seeds.extend(f[active].reshape(-1))
    halo=np.isin(np.arange(len(source)),np.unique(seeds))
    for _ in range(3):
        halo[edges[np.any(halo[edges],axis=1)].reshape(-1)]=True
    ids=np.flatnonzero(halo & ~protected)
    inv=np.full(len(source),-1,dtype=int)
    inv[ids]=np.arange(len(ids))
    active_edges=edges[np.any(inv[edges]>=0,axis=1)]
    m=sum(len(sample[-1]) for sample in samples)
    var=len(ids)
    rc=[]; cc=[]; dd=[]; targets=[]; si=[]; ri=[]
    row=0
    slack=0
    for p,fn,direction,need,active in samples:
        for face in active:
            for node in f[face]:
                if inv[node]>=0:
                    rc.append(row); cc.append(inv[node])
                    dd.append(-float(direction[node]@fn[face]))
                si.append(slack); ri.append(row); targets.append(-1000*need[face])
                row+=1
            slack+=1
    L=csr(rc,cc,dd,(row,var))
    S=csr(ri,si,-np.ones(row),(row,m))
    topology=inv[active_edges]
    erows=np.repeat(np.arange(len(topology)),2)
    ecols=topology.reshape(-1)
    ecoeff=np.tile([1.,-1.],len(topology))
    keep=ecols>=0
    E=csr(erows[keep],ecols[keep],ecoeff[keep],(len(topology),var))
    NE=eye(len(topology),format="csr")
    sparse=lambda rows,cols: coo_matrix((rows,cols))
    A=vstack((hstack((L,S,sparse(row,len(topology)))),
              hstack((E,sparse(len(topology),m),-NE)),
              hstack((-E,sparse(len(topology),m),-NE))),format="csr")
    objective=np.r_[np.full(var,0.09),np.ones(m),np.full(len(topology),0.18)]
    solution=linprog(objective,A_ub=A,b_ub=np.r_[targets,np.zeros(2*len(topology))],
                     bounds=[(0.0,4.0)]*var+[(0.0,None)]*(m+len(topology)),method="highs",
                     options={"time_limit":45.0})
    assert solution.success,solution.message
    delta=np.zeros(len(source))
    delta[ids]=solution.x[:var]/1000
    attempts=[]
    for scale in (1.,.5,.25):
        volumes=[]
        for p,fn,direction,_,_ in samples:
            trial=(p+scale*delta[:,None]*direction).astype("<f4")
            old=face_orientation(p,f)
            new=face_orientation(trial,f)
            areas=np.sum(old*new,axis=1)/np.sum(old*old,axis=1)
            status={"failing_projection_area_rows":int((areas<.25).sum()),
                    "minimum_projection_area_ratio":float(areas.min())}
            if status["failing_projection_area_rows"]==0:
                exact=full_self(trial,f)
                status["skin_self_pairs"]=exact["count"]
                status["self_pair_rows"]=exact["triangle_pairs"]
            volumes.append(status)
        attempts.append({"scale":scale,"pose_checks":volumes})
        print(json.dumps(attempts[-1]),flush=True)
    np.savez_compressed(HERE/"enclosure-proposal.npz",source=source,normals=normals,delta=delta)
    report={"candidate":"inferred normal-direction source correction",
            "input_skin_sha256":bindings.audit.sha(packed),"vertices_in_patch":len(ids),
            "total_resolving_faces":m,"max_displacement_mm":float(1000*delta.max()),
            "unmet_plane_slack_sum_mm":float(solution.x[var:var+m].sum()),
            "initial_witness_counts":[0,33],"attempts":attempts,"native_eligible":False}
    (HERE/"enclosure-experiment.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":
    main()
