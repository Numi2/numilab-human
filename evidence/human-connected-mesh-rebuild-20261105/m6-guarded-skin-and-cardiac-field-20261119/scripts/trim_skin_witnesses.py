"""Retain a subset of the proposed displacement with zero newly sampled pairs."""
import json
import mmap
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from fit_skin_source import ROOT, PRIOR, POSES, bindings, capture

clear = bindings.audit.clearance
exact = bindings.audit.intersections


def positioned_records(points, faces, ids):
    if len(ids)==0:
        return []
    triangles = faces[ids]
    vertices,local = np.unique(triangles, return_inverse=True)
    records = clear._exact_surface_records(points[vertices], local.reshape(-1,3))
    return [(tri,lo,hi,int(i),tuple(map(int,f))) for (tri,lo,hi,_,_),i,f in zip(records,ids,triangles)]


class Pose:
    def __init__(self, path, reference, proposal, active):
        receipt, old, faces = capture(path)
        assert reference["accepted_pack_sha256"] == receipt["pack_file_sha256"]
        self.receipt = receipt
        self.old = old
        self.new = proposal
        self.faces = faces
        affected = np.flatnonzero(np.any(active[faces], axis=1))
        original = positioned_records(old, faces, affected)
        endpoint = positioned_records(proposal, faces, affected)
        boxes=[]
        for a,b in zip(original, endpoint):
            assert a[3] == b[3]
            boxes.append((None,tuple(map(min,zip(a[1],b[1]))),
                          tuple(map(max,zip(a[2],b[2]))),a[3],()))
        envelope=clear._float32_skin_aabb_index(boxes)
        self.targets, self.baseline = self.prepare_targets(path, reference, envelope)
        self.skin_before={tuple(p) for p in reference["self_intersections"]["triangle_pairs"]}
        baseline=self.audit_targets(old,affected)
        moved=set(affected.tolist())
        expected={pair for pair in self.baseline if pair[0] in moved}
        assert baseline==expected,(len(baseline),len(expected))
        print(f"Prepared {receipt['accepted_step']} with {len(self.targets.records)} relevant typed faces.",flush=True)

    def prepare_targets(self,path,reference,envelope):
        banks={(a["semantic"],a["stable_id"]):a for a in reference["surfaces"]}
        bodies={p["body_index"]:p for p in self.receipt["accepted_body_poses"]}
        all_records=[]
        baseline=set()
        offset=0
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            raw=Path(d)/"capture.mrvpack"
            with raw.open("wb") as f:
                subprocess.run(["/opt/homebrew/bin/zstd","-dc",str(path)+".zst"],stdout=f,check=True)
            assert bindings.audit.sha(raw)==self.receipt["pack_file_sha256"]
            with raw.open("rb") as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as pack:
                for key,_id,xyz,faces in bindings.audit.primitives(pack,bodies):
                    if key==bindings.audit.SKIN:
                        continue
                    ids,_=clear._candidate_target_rows_for_skin_aabb(faces,xyz,envelope)
                    rows=positioned_records(xyz,faces,ids)
                    all_records.extend((t,lo,hi,offset+i,v) for t,lo,hi,i,v in rows)
                    baseline.update((a,offset+b) for a,b in banks.pop(key)["triangle_pairs"])
                    offset+=len(faces)
        assert not banks
        return exact._prepare_surface_aabb(all_records), baseline

    def audit_targets(self, points, changed_faces):
        query=positioned_records(points,self.faces,changed_faces)
        pairs=((a,b) for a,b in exact._query_surface_aabb(query,self.targets))
        return {tuple(p) for p in exact._audit_record_pairs(pairs,same_surface=False)["triangle_pairs"]}

    def audit(self, active, check_self=False):
        changed=np.flatnonzero(np.any(active[self.faces],axis=1))
        current=np.where(active[:,None],self.new,self.old)
        hits=self.audit_targets(current,changed)
        moved=set(changed.tolist())
        retained={p for p in self.baseline if p[0] not in moved}
        target_pairs=hits|retained
        added=target_pairs-self.baseline
        removed=self.baseline-target_pairs
        result={"step":self.receipt["accepted_step"],"before":len(self.baseline),"after":len(target_pairs),
                "added":len(added),"removed":len(removed)}
        bad=set(p[0] for p in added)
        if check_self:
            rec=positioned_records(current,self.faces,np.arange(len(self.faces)))
            after=exact._audit_pair_prepared_first(exact._prepare_surface_aabb(rec),rec,same_surface=True)
            self_after={tuple(p) for p in after["triangle_pairs"]}
            bad.update(k for pair in (self_after-self.skin_before) for k in pair)
            result["self_before"]=len(self.skin_before)
            result["self_after"]=len(self_after)
            result["self_added"]=len(self_after-self.skin_before)
        return result,bad


def projected_guard(pose, active):
    p=pose.old
    trial=np.where(active[:,None],pose.new,pose.old)
    ids=np.flatnonzero(np.any(active[pose.faces],axis=1))
    tri,now=(x[pose.faces[ids]].astype(float) for x in (p,trial))
    cross=lambda a:np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0])
    old,new=cross(tri),cross(now)
    bad=ids[np.sum(old*new,axis=1)<.6*np.sum(old*old,axis=1)]
    return set(bad.tolist())


def main():
    proposal=np.load(ROOT/"separating-plane-skin-fit.npz")
    d=proposal["displacement_m"]
    active=np.any(d!=0,axis=1)
    poses=[]
    for (run,step),key in zip(POSES,("early","late")):
        path=PRIOR/run/f"native/accepted-geometry/step-{step}.mrvpack"
        reference=json.loads((PRIOR/f"{run}-exact.json").read_text())
        pose=Pose(path,reference,proposal[key],active)
        if poses:
            assert np.array_equal(poses[0].faces,pose.faces)
        poses.append(pose)
    history=[]
    accepted=False
    for n in range(12):
        bad=set()
        evaluations=[]
        for pose in poses:
            bad.update(projected_guard(pose,active))
            if bad:
                break
        if not bad:
            for pose in poses:
                report,witnesses=pose.audit(active)
                bad.update(witnesses)
                evaluations.append(report)
            if not bad:
                evaluations=[]
                for pose in poses:
                    report,witnesses=pose.audit(active,check_self=True)
                    bad.update(witnesses)
                    evaluations.append(report)
        entry={"iteration":n,"moves":int(active.sum()),"evaluations":evaluations,"retire_faces":len(bad)}
        history.append(entry)
        print(entry,flush=True)
        if not bad:
            accepted=all(r["removed"]>0 for r in evaluations)
            break
        retired=poses[0].faces[np.asarray(sorted(bad))].ravel()
        assert np.any(active[retired])
        active[retired]=False
    report={"passed_sampled_geometry_checks":accepted,"candidate_scale":0.5,
             "history":history,"no_default_promotion":True}
    (ROOT/"trim-skin-witnesses.json").write_text(json.dumps(report,indent=2)+"\n")
    if accepted:
        np.savez_compressed(ROOT/"trim-skin-proposal.npz",displacement_m=np.where(active[:,None],.5*d,0),
                            early=np.where(active[:,None],poses[0].new,poses[0].old),
                            late=np.where(active[:,None],poses[1].new,poses[1].old))

if __name__=="__main__":
    main()
