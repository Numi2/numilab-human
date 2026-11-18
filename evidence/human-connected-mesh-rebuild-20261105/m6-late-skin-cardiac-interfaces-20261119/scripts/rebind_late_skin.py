"""Improve collared skin blending on accepted early and late motion witnesses."""
from functools import partial
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from prepare_skin import HERE, BASE, bindings, capture, decode, full_self
from rebind_arm_witnesses import axes, incident, projection, rebind

def hits(audit):
    return {tuple(x) for x in audit["triangle_pairs"]}

def main():
    predicates=bindings.audit.clearance
    inter=bindings.audit.intersections
    prior=HERE.parent/"resting-clearance-and-atrium-002"
    source_file=prior/"skin-rebound.nhskin"
    model=decode(source_file.read_bytes())
    original=model["weights"]
    vertex=model["verts"][:,:3]
    start,p0,faces=capture(BASE)
    reference=axes(model,start)
    ref=projection(original,vertex,reference)
    known=np.unique(faces)
    protected=(p0[:,2]<.015)
    protected[predicates._exact_coordinate_tied_source_vertex_ids(vertex)]=True
    protected|=np.linalg.norm(ref-p0,axis=1)>.002

    states=[]
    for run,step in (("wider-release-variable-4",2000),("wider-release-control-16",8000)):
        pack=HERE/run/f"native/accepted-geometry/step-{step}.mrvpack"
        receipt,points,surfaces=capture(pack)
        assert np.array_equal(faces,surfaces)
        exact=json.loads((HERE/f"{run}-exact.json").read_text())
        assert receipt["pack_file_sha256"]==exact["accepted_pack_sha256"]
        basis=axes(model,receipt)
        protected|=points[:,2]<.015
        states.append(dict(points=points, current=points.copy(), sha=receipt["pack_file_sha256"],
                           basis=basis, projected=projection(original,vertex,basis),
                           audit=exact["self_intersections"]))
    edge=np.sort(faces[:,[0,1,1,2,2,0]].reshape(-1,2),axis=1)
    edge,counts=np.unique(edge,axis=0,return_counts=True)
    protected[edge[counts!=2].reshape(-1)]=True
    adj=coo_matrix((np.ones(2*len(edge)),(np.r_[edge[:,0],edge[:,1]],np.r_[edge[:,1],edge[:,0]])),
                   shape=(len(original),len(original))).tocsr()
    final=states[-1]
    pairs=np.asarray(sorted(hits(final["audit"])),dtype=np.int64)
    centers=final["points"][faces[pairs]].mean(axis=(1,2))
    _,groups=connected_components(coo_matrix(np.linalg.norm(centers[:,None]-centers[None,:],axis=2)<.014))
    order=sorted(np.unique(groups),key=lambda i:-int((groups==i).sum()))
    weights=original.copy()
    changed=np.zeros(len(weights),dtype=bool)
    for state in states:
        state["baseline_count"]=state["audit"]["count"]
    accepted=[]
    def judge(record,pos):
        return predicates.audit_incremental_skin_self_intersections(
            baseline_world_positions=record["current"],candidate_world_positions=pos,
            baseline_faces=faces,candidate_faces=faces,baseline_self_audit=record["audit"],
            expected_baseline_world_f32_sha256=predicates._float32_xyz_sha256(record["current"]),
            expected_face_index_sha256=predicates._face_index_sha256(faces),
            expected_baseline_self_pair_table_sha256=predicates._baseline_self_pair_table_sha256(record["audit"],len(faces)))
    for group in order:
        witnessed=pairs[groups==group]
        face_ids=np.unique(witnessed)
        seeds=np.unique(faces[face_ids])
        distance=np.full(len(weights),-1)
        distance[seeds]=0
        for k in range(1,5):
            frontier=np.flatnonzero(distance==k-1)
            ring=np.unique(np.concatenate([adj[i].indices for i in frontier]))
            distance[ring[distance[ring]<0]]=k
        patch=np.flatnonzero((distance>=0)&(distance<4)&~protected&~changed)
        if not len(patch):
            continue
        affected=incident(faces,patch)
        fraction=1.0-distance[patch]*.25
        collar=(fraction*fraction*(3.0-2.0*fraction))[:,None]
        candidates=[]
        local=np.mean(weights[seeds],axis=0)
        directions=[("mean",local)]
        for lane in np.argsort(-local)[:5]:
            end=np.zeros(weights.shape[1])
            end[lane]=1
            directions.append((str(model["binds"]["body"][lane]),end))
        for name,goal in directions:
            for fraction in (.125,.25,.5,.75,1):
                new=(weights[patch]*(1-collar*fraction)+goal*collar*fraction).astype("<f4")
                a=np.einsum("vb,bij->vij",new,reference[0])
                b=new @ reference[1]
                xyz=np.linalg.solve(a,(ref[patch]-b)[...,None])[...,0].astype("<f4")
                positions=[]
                magnitude=0.0
                for state in states:
                    p=state["current"].copy()
                    p[patch]=(state["points"][patch].astype(float)+projection(new,xyz,state["basis"])-
                        state["projected"][patch]).astype("<f4")
                    old=state["current"][faces[affected]].astype(float)
                    trial=p[faces[affected]].astype(float)
                    n0=np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0])
                    n1=np.cross(trial[:,1]-trial[:,0],trial[:,2]-trial[:,0])
                    if np.any((n0*n1).sum(axis=1)<.25*(n0*n0).sum(axis=1)):
                        break
                    step=float(np.linalg.norm(p[patch]-state["current"][patch],axis=1).max())
                    if step>.004:
                        break
                    magnitude=max(magnitude,step)
                    positions.append(p)
                if len(positions)!=len(states):
                    continue
                records=predicates._exact_surface_records(positions[-1],faces[face_ids])
                selection=np.searchsorted(face_ids,witnessed)
                count=inter._audit_record_pairs(((records[i],records[j]) for i,j in selection),same_surface=True)["count"]
                if count<len(witnessed):
                    candidates.append((count,magnitude,name,fraction,new,positions))
        candidates.sort(key=lambda x:(x[0],x[1]))
        for _,move,name,amount,values,out in candidates[:6]:
            completed=[]
            for record,pos in zip(states,out):
                result=judge(record,pos)
                if not hits(result).issubset(hits(record["audit"])):
                    break
                completed.append(result)
            if len(completed)!=len(states) or completed[-1]["count"]>=final["audit"]["count"]:
                continue
            event={"cluster":int(group),"blend_target":name,"blend_fraction":amount,
                   "maximum_displacement_mm":1000*move,"patch_vertices":len(patch),
                   "before":[r["audit"]["count"] for r in states],"after":[r["count"] for r in completed]}
            for r,p,audit in zip(states,out,completed):
                r["current"]=p
                r["audit"]=audit
            weights[patch]=values
            changed[patch]=True
            accepted.append(event)
            print(event,flush=True)
            break
    for record in states:
        assert full_self(record["current"],faces)["triangle_pairs"]==record["audit"]["triangle_pairs"]
    receipt={"input_skin":bindings.audit.sha(source_file),"selection":accepted,
        "pose_pack_sha256":[r["sha"] for r in states],
        "start":[r["baseline_count"] for r in states],"end":[r["audit"]["count"] for r in states],
        "scope":"Frozen accepted Float32 replay. Native and internal tissue qualification required."}
    if accepted:
        payload,report=rebind(model,weights,*reference)
        path=HERE/"skin-rebound-late.nhskin"
        path.write_bytes(payload)
        receipt["candidate_skin"]=bindings.audit.sha(path)
        receipt["reference_parity"]=report
    (HERE/"skin-rebound-late.json").write_text(json.dumps(receipt,indent=2)+"\n")
    np.savez_compressed(HERE/"late-skin-frozen-replay.npz",early=states[0]["current"],late=states[1]["current"])
    print(receipt,flush=True)

if __name__=="__main__":
    main()
