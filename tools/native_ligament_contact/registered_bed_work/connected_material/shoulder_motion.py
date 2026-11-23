"""Split the native shoulder's vertical descent into translation, rotation and joint motion."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def diagnose(case):
    proof=json.loads((case/"terminal.receipt.json").read_text())
    terms=np.load(case/"shoulder-terms.npz")
    local=terms["local"].astype(float);ids=terms["body"].astype(int);w=terms["weight"].astype(float)
    if abs(w.sum()-1)>1e-5:raise ValueError("Unbalanced source weights")
    def sample(state):
        by_id={b["body_index"]:b for b in state}
        return np.array([by_id[int(i)]["position_m"] for i in ids]),Rotation.from_quat([by_id[int(i)]["quaternion_xyzw"] for i in ids]),by_id
    a,ar,initial=sample(proof["initial_anatomical_registration"]["body_poses"]);b,br,current=sample(proof["accepted_body_poses"])
    x=ar.apply(local)+a;y=br.apply(local)+b
    start=initial[0];end=current[0];t0=np.array(start["position_m"]);t1=np.array(end["position_m"])
    rot=Rotation.from_quat(end["quaternion_xyzw"])*Rotation.from_quat(start["quaternion_xyzw"]).inv()
    after_turn=rot.apply(x-t0)+t0
    residual=(y-(after_turn+t1-t0))[:,2]*w
    rotation=float((after_turn-x)[:,2]@w);translation=float((t1-t0)[2]*w.sum());actual=float((y-x)[:,2]@w)
    if abs(actual-(rotation+translation+sum(residual)))>1e-10:raise ValueError("Incomplete motion decomposition")
    grouped=np.bincount(ids,weights=residual,minlength=proof["accepted_body_pose_count"])
    bnd=np.bincount(ids,weights=w,minlength=len(grouped))
    rows=sorted((dict(body=int(i),weight=float(bnd[i]),relative_z_m=float(grouped[i])) for i in np.unique(ids)),key=lambda j:-abs(j["relative_z_m"]))
    with (case/"registered-bed-columns.csv").open() as f:
        row,=(r for r in csv.DictReader(f) if r["vertex"]=="17764")
    if abs(x[:,2]@w-float(row["z_m"]))>5e-7:raise ValueError("Incorrect native vertex")
    traction=[]
    with (case/"resting-com-support-impulses.csv").open() as f:
        for r in csv.DictReader(f):traction.append(r)
    with (case/"resting-com-momentum-diagnostic.csv").open() as f:
        summary={int(r["accepted_step"]):r for r in csv.DictReader(f)}
    support={}
    for r in traction:
        step=int(r["accepted_step"])
        support[step]=support.get(step,0.)+float(r["impulse_world_z_ns"])
    observations=[]
    for step,impulse in sorted(support.items()):
        row=summary[step]
        dt=float(row["time_s"])/step
        force=float(row["normal_force_last_physical_step_n"])
        if abs(impulse-float(row["normal_impulse_last_physical_step_ns"]))>1e-5 or abs(impulse/dt-force)>.01:
            raise ValueError("Support rows disagree with the last physical step")
        observations.append(dict(step=step,time_s=float(row["time_s"]),normal_impulse_last_physical_step_Ns=impulse,
                                 normal_force_last_physical_step_N=force))
    result=dict(step=proof["accepted_step"],time_s=proof["accepted_time_s"],vertex=17764,
        pelvis_translation_m=translation,pelvis_rotation_contribution_m=rotation,relative_to_pelvis_m=float(residual.sum()),
        total_z_m=actual,body_contributions=rows[:10],positive_skin_donors=int(sum(bnd>0)),sampled_last_step_support=observations,
        scope="Forward-kinematic decomposition of the same native material point. Terms include contact and internal forces acting on all coordinates; a displacement term alone does not identify a force cause.")
    (case/"shoulder-motion.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


if __name__=="__main__":
    res=diagnose(Path(sys.argv[1]));print({k:v for k,v in res.items() if k.endswith('_m')})
