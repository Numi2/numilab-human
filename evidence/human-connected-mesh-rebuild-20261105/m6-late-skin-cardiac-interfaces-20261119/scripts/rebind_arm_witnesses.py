"""Test collared registration changes on frozen whole-body native captures."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from prepare_skin import HERE, ROOT, RUN, BASE, END, bindings, capture, decode, full_self, rotation
from rebind_human_skin_reference import rebind

def incident(faces, vertices):
    return np.flatnonzero(np.any(np.isin(faces,vertices),axis=1))

def axes(model, receipt):
    body = {b["body_index"]: b for b in receipt["accepted_body_poses"]}
    rows = [body[int(i)] for i in model["binds"]["body"]]
    r = np.array([rotation(b["quaternion_xyzw"]) for b in rows])
    translations = np.array([b["position_m"] for b in rows])
    local = model["binds"]["v"]
    matrix = np.array([rotation(b[3:7])*b[7] for b in local])
    return r @ matrix, translations + np.einsum("bij,bj->bi",r,local[:,:3])

def projection(w, xyz, ax):
    a, t = ax
    return np.einsum("vb,bij,vj->vi",w,a,xyz,optimize=True) + w @ t

def main():
    c = bindings.audit.clearance
    inter = bindings.audit.intersections
    launch = json.loads((RUN/"combined-12s-no-export/launch.json").read_text())
    argv = launch["argv"]
    file = Path(argv[argv.index("--skin-payload")+1])
    source_sha = bindings.audit.sha(file)
    assert source_sha == launch["pins"][str(file)]
    model = decode(file.read_bytes())
    rest,p0,faces = capture(BASE)
    settled,p1,f1 = capture(END)
    assert np.array_equal(faces,f1)
    audit = json.loads((ROOT/"coupled-crossing-repair-001/thumb-exact.json").read_text())
    assert audit["accepted_pack_sha256"] == settled["pack_file_sha256"]
    pairs = np.asarray(audit["self_intersections"]["triangle_pairs"],dtype=np.int64)
    w = model["weights"]
    xyz = model["verts"][:,:3]
    reference, motion = axes(model,rest), axes(model,settled)
    ref = projection(w,xyz,reference)
    replay = projection(w,xyz,motion)
    referenced = np.unique(faces)
    # The layered bed envelope and respiratory displacement remain in the frozen
    # captures. The candidate changes only the static rigid weight registration.
    protected = (p0[:,2] < .015) | (p1[:,2] < .015)
    protected[c._exact_coordinate_tied_source_vertex_ids(xyz)] = True
    reference_error = np.linalg.norm(ref-p0,axis=1)
    protected |= reference_error > .002

    all_edges = np.sort(faces[:,[0,1,1,2,2,0]].reshape(-1,2),axis=1)
    edges,counts=np.unique(all_edges,axis=0,return_counts=True)
    protected[edges[counts!=2].reshape(-1)] = True
    adj=coo_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),
                   shape=(len(w),len(w))).tocsr()
    centers=p1[faces[pairs].reshape(len(pairs),-1)].mean(axis=1)
    _,groups=connected_components(coo_matrix(np.linalg.norm(centers[:,None]-centers[None,:],axis=-1)<.014))

    # Refresh exact pairs touching every changed face, leaving unrelated,
    # hash-bound baseline witnesses intact.
    def exact(next_pose, previous_pose, previous_pairs):
        return c.audit_incremental_skin_self_intersections(
            baseline_world_positions=previous_pose,candidate_world_positions=next_pose,
            baseline_faces=faces,candidate_faces=faces,baseline_self_audit=previous_pairs,
            expected_baseline_world_f32_sha256=c._float32_xyz_sha256(previous_pose),
            expected_face_index_sha256=c._face_index_sha256(faces),
            expected_baseline_self_pair_table_sha256=c._baseline_self_pair_table_sha256(previous_pairs,len(faces)),
        )

    current=w.copy()
    pose=p1.copy()
    current_audit=audit["self_intersections"]
    selected=[]
    for cluster in np.unique(groups):
        old_pairs=pairs[groups==cluster]
        old_faces=np.unique(old_pairs)
        seeds=np.unique(faces[old_faces])
        patch_ring=np.zeros(len(w),dtype=int)-1
        patch_ring[seeds]=0
        for index in range(1,5):
            front=np.flatnonzero(patch_ring==index-1)
            next_ring=np.unique(np.concatenate([adj[i].indices for i in front]))
            patch_ring[next_ring[patch_ring[next_ring]<0]]=index
        patch=np.flatnonzero((patch_ring>=0) & (patch_ring<4) & ~protected)
        if not len(patch):
            continue
        affected=incident(faces,patch)
        # Anchored collar. The outer ring retains the original weights.
        s=1.0-patch_ring[patch]*.25
        gate=(s*s*(3-2*s))[:,None]
        old_xyz = pose[faces[affected]].astype(float)
        a = np.cross(old_xyz[:,1]-old_xyz[:,0],old_xyz[:,2]-old_xyz[:,0])
        rev = np.sum(a*a,axis=1)
        def evaluate(values):
            matrices=np.einsum("vb,bij->vij",values,reference[0])
            offsets=values @ reference[1]
            proposal=np.linalg.solve(matrices,(ref[patch]-offsets)[...,None])[...,0].astype("<f4")
            out=pose.copy()
            out[patch]=(p1[patch].astype(float) + projection(values,proposal,motion)-replay[patch]).astype("<f4")
            x=out[faces[affected]].astype(float)
            b=np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0])
            if np.min(np.sum(a*b,axis=1)/rev) < .25:
                return None
            max_change=float(np.linalg.norm(out[patch]-pose[patch],axis=1).max())
            if max_change > .008:
                return None
            records = c._exact_surface_records(out,faces[old_faces])
            original = {k:i for i,k in enumerate(old_faces)}
            tested = [tuple(original[k] for k in pair) for pair in old_pairs]
            hits = inter._audit_record_pairs(((records[i],records[j]) for i,j in tested),same_surface=True)
            return hits["count"], max_change, out

        basis = []
        lane_scores = np.mean(current[seeds],axis=0)
        for lane in np.argsort(-lane_scores)[:6]:
            unit = np.zeros(model["nb"])
            unit[lane]=1.0
            basis.append((f"body {int(model['binds']['body'][lane])}",unit))
        basis.append(("local weight average",lane_scores))
        proposals=[]
        for label,target in basis:
            for f in (.125,.25,.5,.75,1.):
                new_weights=(current[patch]*(1-gate*f)+target*(gate*f)).astype("<f4")
                tested=evaluate(new_weights)
                if tested is not None and tested[0] < len(old_pairs):
                    proposals.append((tested[0],tested[1],label,f,new_weights,tested[2]))
        proposals.sort(key=lambda row:(row[0],row[1]))
        start_count=current_audit["count"]
        for _,move,label,amount,values,updated in proposals[:6]:
            result=exact(updated,pose,current_audit)
            retained={tuple(x) for x in current_audit["triangle_pairs"]}
            changed={tuple(x) for x in result["triangle_pairs"]}
            if len(changed) < start_count and changed.issubset(retained):
                current[patch]=values
                pose=updated
                current_audit=result
                selected.append(dict(cluster=int(cluster),owner=label,blend=amount,max_move_mm=1000*move,
                                     before=start_count,after=len(changed),patch_vertices=len(patch)))
                print(json.dumps(selected[-1]),flush=True)
                break
    full=full_self(pose,faces)
    assert full["triangle_pairs"] == current_audit["triangle_pairs"]
    result={"baseline":len(pairs),"candidate":full["count"],"selection":selected,
            "basis_max_error_before_layers_mm":1000*float(reference_error[referenced].max()),
            "rest_pose":rest["pack_file_sha256"],"settled_pose":settled["pack_file_sha256"],
            "baseline_skin":source_sha}
    if full["count"] < len(pairs):
        raw, transfer = rebind(model,current,*reference)
        output=HERE/"skin-rebound.nhskin"
        output.write_bytes(raw)
        result["binding_replay"]=transfer
        result["candidate_skin"]=bindings.audit.sha(output)
    (HERE/"skin-rebound.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="binding_replay"}),flush=True)

if __name__=="__main__":
    main()
