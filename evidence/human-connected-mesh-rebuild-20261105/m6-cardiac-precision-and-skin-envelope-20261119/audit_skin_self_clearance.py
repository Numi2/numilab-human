"""Audit and guard the three-pose skin envelope using exact typed geometry."""
import json
import sys

import numpy as np

from fit_cardiac_volume_grid import HERE, PREVIOUS
sys.path.insert(0,str(PREVIOUS))
from fit_skin_source import BASE, PRIOR, capture, decode
from trim_prior_and_self import Pose, projected_guard, positioned_records, exact

def audit_source(old,new,faces,active,baseline_tree,baseline_cache):
    changed=np.flatnonzero(np.any(active[faces],axis=1))
    if not len(changed):
        return set(),0
    rows=positioned_records(np.where(active[:,None],new,old),faces,np.arange(len(faces)))
    candidate=exact._prepare_surface_aabb(rows)
    def pairs(tree):
        records=tree.records
        seen=set()
        for left,right in exact._query_surface_aabb((records[i] for i in changed),tree):
            key=tuple(sorted((left[3],right[3])))
            if key[0]==key[1] or key in seen:
                continue
            seen.add(key)
            yield records[key[0]],records[key[1]]
    if not baseline_cache:
        baseline_cache.add(None)
        baseline_cache.update(map(tuple,exact._audit_record_pairs(pairs(baseline_tree),same_surface=True)["triangle_pairs"]))
    after=set(map(tuple,exact._audit_record_pairs(pairs(candidate),same_surface=True)["triangle_pairs"]))
    bad=after-baseline_cache
    return {i for pair in bad for i in pair},len(bad)

def main():
    from rebind_arm_witnesses import axes
    base_fit=np.load(HERE/"skin-envelope-rest-and-motion.npz")["displacement_m"]
    perturbation=np.load(HERE/"skin-self-fit.npz")["source_delta"]
    previous=base_fit+perturbation
    plan={"displacement_m":previous}
    source=decode((HERE.parent/"resting-clearance-and-atrium-002/skin-rebound.nhskin").read_bytes())
    old=source["verts"][:,:3]
    new=(old.astype(float)+plan["displacement_m"].astype(float)).astype("<f4")
    faces=source["inds"]
    baseline_tree=exact._prepare_surface_aabb(positioned_records(old,faces,np.arange(len(faces))))
    baseline_cache=set()
    delta=plan["displacement_m"]
    active=np.any(previous!=0,axis=1)
    cases=[
        (BASE,HERE.parent/"resting-surface-clearance-001/initial.json","initial_delta"),
        (PRIOR/"wider-release-variable-4/native/accepted-geometry/step-2000.mrvpack",PRIOR/"wider-release-variable-4-exact.json","early_delta"),
        (PRIOR/"wider-release-control-16/native/accepted-geometry/step-8000.mrvpack",PRIOR/"wider-release-control-16-exact.json","late_delta")]
    poses=[]
    for path,receipt,key in cases:
        state,p,_=capture(path)
        skin_map=np.einsum("vb,bdc->vdc",source["weights"],axes(source,state)[0])
        plan[key]=np.einsum("vij,vj->vi",skin_map,previous)
        spec=json.loads(receipt.read_text())
        poses.append(Pose(path,spec,(p+plan[key]).astype("<f4"),active,
                          extra=(p+np.einsum("vij,vj->vi",skin_map,base_fit)).astype("<f4")))
    history=[]
    for i in range(18):
        bad,n_source=audit_source(old,new,faces,active,baseline_tree,baseline_cache)
        reports=[]
        for pose in poses:
            row,witness=pose.audit(active,check_self=True)
            area=projected_guard(pose,active)
            row["orientation_bad_triangles"]=len(area)
            bad.update(witness|area)
            reports.append(row)
        status=dict(iteration=i,changed_rows=int(active.sum()),bad_faces=len(bad),source_self_added=n_source,poses=reports)
        history.append(status)
        print(json.dumps(status),flush=True)
        if not bad:
            choice=np.where(active[:,None],delta,0)
            np.savez_compressed(HERE/"skin-self-guarded.npz",displacement_m=choice)
            break
        exposed=np.unique(poses[0].faces[sorted(bad)].ravel())
        # Undo the self-clearance perturbation first. Preserve the prior
        # envelope position and repeat with its actual forward map.
        moved=exposed[np.any(perturbation[exposed]!=0,axis=1)]
        if not len(moved):
            raise RuntimeError("new witnesses remain after reverting local self-clearance edits")
        perturbation[moved]=0
        delta=base_fit+perturbation
        new=(old.astype(float)+delta).astype("<f4")
        active=np.any(delta!=0,axis=1)
        for pose,(path,report,key) in zip(poses,cases):
            skin_map=np.einsum("vb,bdc->vdc",source["weights"],axes(source,pose.receipt)[0])
            pose.new=(pose.old+np.einsum("vij,vj->vi",skin_map,delta)).astype("<f4")
    (HERE/"skin-self-exact.json").write_text(json.dumps(history,indent=2)+"\n")

if __name__=="__main__":
    main()
