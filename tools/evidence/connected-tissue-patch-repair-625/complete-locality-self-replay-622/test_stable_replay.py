from pathlib import Path
import json,time
from stable_face_self_replay import *
from run_pinned_current025 import owner
p=owner();t=time.monotonic()
x=[(0.,0.,0.),(2.,0.,0.),(0.,2.,0.),(.25,.25,-1.),(.25,.25,1.),(1.,1.,0.),(5.,0.,0.),(6.,0.,0.),(5.,1.,0.)]
f0=[(0,1,2),(3,4,5),(6,7,8)]
r0,_=p.float32_surface_records(x,f0)
base,full=p._prepare_complete_self_pair_baseline(r0)
checks=[]
for label,faces in [
 ("reorder",list(reversed(f0))),
 ("remove_crossing",[f0[2],f0[0]]),
 ("new_crossing",[f0[0],f0[2],(3,4,1)]),
 ("same_shared_edge",[(0,1,2),(1,0,8),f0[1]]),
 ("orientation_change",[(0,2,1),f0[1],f0[2]]),
 ("empty",[])]:
 r,_=p.float32_surface_records(x,faces);ids=stable_face_ids(f0,faces)
 upd,a=p._update_complete_self_pair_baseline(base,persistent_records(r,ids))
 exact=p._audit_self_pairs_touching_faces(r,set(range(len(r))))
 got=report_pairs_to_rows(a["triangle_pairs"],ids)
 assert got==exact["triangle_pairs"],(label,got,exact)
 checks.append({"case":label,"exact_pair_count":len(got),"pass":True})
for label,old,new in [("duplicate_base",[f0[0],f0[0]],[f0[1]]),("duplicate_new",f0,[f0[1],f0[1]])]:
 try:stable_face_ids(old,new)
 except ValueError:checks.append({"case":label,"pass":True})
 else:raise AssertionError(label)
out={"status":"PASS","cases":checks,"elapsed_s":time.monotonic()-t}
Path(__file__).with_name("regression.json").write_text(json.dumps(out,indent=2));print(json.dumps(out))
