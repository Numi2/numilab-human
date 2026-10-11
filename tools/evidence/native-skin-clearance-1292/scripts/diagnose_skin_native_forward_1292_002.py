from pathlib import Path
import sys,importlib.util,json,hashlib,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"skin-native-integration-1292-001/native-forward-diagnostic-002";O.mkdir(exist_ok=False)
H=A/"skin-candidate-30-pose-preparation-001/prepare_skin_candidate_30pose_005.py";sp=importlib.util.spec_from_file_location("p",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sys.path.insert(0,str(h.CA.parent.parent))
from numilab_human import common_atlas_skin_clearance as ca
_,skin=h.load_skin(h.BASE_SKIN);fit=np.load(h.FIT_NPY);ref=np.unique(skin["faces"])
fmsp=importlib.util.spec_from_file_location("fm",h.FORWARD);fm=importlib.util.module_from_spec(fmsp);fmsp.loader.exec_module(fm)
N=A/"skin-native-integration-1292-001/baseline/native-run/accepted-geometry";B=A/"thirty-surface-native-composition-001/baseline/native-run/accepted-geometry"
receipts=[];captured=[];native=[]
for step in h.STEPS:
 rc=json.loads((B/f"step-{step}.receipt.json").read_text());receipts.append(rc)
 for root,out in ((B,captured),(N,native)):
  xyz,ss,_=ca._pack_surfaces(root/f"step-{step}.mrvpack",{(51007,1)});f=np.asarray(ss[(51007,1)]["faces"],int);off=int(f.min());assert np.array_equal(f-off,skin["faces"]);out.append(xyz[off:off+skin["nv"]][ref].copy())
states=[{"step":s,"body_poses":r["accepted_registered_body_poses"],"respiratory_motion":r["accepted_respiratory_motion"]} for s,r in zip(h.STEPS,receipts)]
forward,mr,ids=fm.build_forward(skin=skin,source_positions=skin["pos"],referenced_ids=ref,captured_by_pose=np.asarray(captured).astype(float),state_receipts=states,initial_body_poses=receipts[0]["initial_anatomical_registration"]["body_poses"],map_receipt=receipts[0],anatomy_parameters_path=receipts[0]["skin_source_mapping"]["anatomy_parameters"]["path"],respiration_source=h.RESP,clearance_module=ca)
base=np.asarray(forward(skin["pos"])["world_positions_by_pose"],np.float32);assert np.array_equal(base,captured)
pred=np.asarray(forward(fit)["world_positions_by_pose"],np.float32)
rows=[]
for step,expected,actual in zip(h.STEPS,pred,native):
 diff=np.linalg.norm(actual.astype(float)-expected.astype(float),axis=1);ix=np.flatnonzero(diff>0)
 np.savez(O/f"step-{step}.npz",expected=expected,actual=actual,source_referenced_ids=ref)
 rows.append({"step":step,"nchanged":len(ix),"max_m":float(diff.max()),"rms_m":float(np.sqrt(np.mean(diff**2))),"changed_ids":ref[ix].tolist(),"expected_sha256":ca._float32_xyz_sha256(expected),"native_sha256":ca._float32_xyz_sha256(actual)})
(O/"report.json").write_text(json.dumps({"rows":rows,"map_report":mr},indent=2)+"\n")
print(json.dumps([{k:v for k,v in x.items() if k!="changed_ids"} for x in rows],indent=2))

