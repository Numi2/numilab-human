"""Audit and guard the three-pose skin envelope using exact typed geometry."""
import json
import sys

import numpy as np

from fit_cardiac_volume_grid import HERE, PREVIOUS
sys.path.insert(0,str(PREVIOUS))
from fit_skin_source import BASE, PRIOR, capture
from trim_three_poses import Pose, projected_guard

def main():
    plan=np.load(HERE/"skin-envelope-proposal.npz")
    delta=plan["displacement_m"]
    active=np.any(delta!=0,axis=1)
    cases=[
        (BASE,HERE.parent/"resting-surface-clearance-001/initial.json","initial_delta"),
        (PRIOR/"wider-release-variable-4/native/accepted-geometry/step-2000.mrvpack",PRIOR/"wider-release-variable-4-exact.json","early_delta"),
        (PRIOR/"wider-release-control-16/native/accepted-geometry/step-8000.mrvpack",PRIOR/"wider-release-control-16-exact.json","late_delta")]
    poses=[]
    for path,receipt,key in cases:
        _,p,_=capture(path)
        spec=json.loads(receipt.read_text())
        poses.append(Pose(path,spec,(p+plan[key]).astype("<f4"),active))
    history=[]
    for i in range(18):
        bad=set()
        reports=[]
        for pose in poses:
            row,witness=pose.audit(active,check_self=True)
            area=projected_guard(pose,active)
            row["orientation_bad_triangles"]=len(area)
            bad.update(witness|area)
            reports.append(row)
        status=dict(iteration=i,changed_rows=int(active.sum()),bad_faces=len(bad),poses=reports)
        history.append(status)
        print(json.dumps(status),flush=True)
        if not bad:
            choice=np.where(active[:,None],delta,0)
            np.savez_compressed(HERE/"skin-envelope-guarded.npz",displacement_m=choice)
            break
        exposed=poses[0].faces[sorted(bad)].ravel()
        if not np.any(active[exposed]):
            raise RuntimeError("new exact witnesses are outside the source edit region")
        active[exposed]=False
    (HERE/"skin-envelope-audit.json").write_text(json.dumps(history,indent=2)+"\n")

if __name__=="__main__":
    main()
