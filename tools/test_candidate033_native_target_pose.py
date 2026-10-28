#!/usr/bin/env python3
"""Actual native target-placement regression; no mesh or physics admission."""
import copy,json,tempfile
from pathlib import Path
import numpy as np
import prepare_candidate033_d37 as prep

P=Path("/Users/numi/Work/numi-human-connected-surface-069")
old=P/"D/connected-obstacle-207/physical-transfer-209/volume-worker/skin-source-033/prepared-007-kind-aware-canonical-compact/receipt.json"
r=json.loads(old.read_text());inputs=r["inputs"]
_,_,binds,_,_,_,_=prep.load_skin(inputs["source-payload"]["path"])
_,source=prep.load_pose(inputs["source-poses"]["path"],binds)
target=P/"D/connected-obstacle-207/physical-transfer-209/P/native-kind-aware-source-preflight-226/run-006-native-emission/registered-skin-poses-100.json"
raw=json.loads(target.read_text());_,actual=prep.load_target_pose(target,binds,source)
ids=[b["body"] for b in binds]
maxdelta=max(np.linalg.norm(actual[i]["current_p"].astype(float)-source[i]["current_p"].astype(float)) for i in ids)
assert 0<maxdelta<1e-5
with tempfile.TemporaryDirectory(prefix="candidate033-target-pose-") as t:
 p=Path(t)/"pose.json"
 def rejects(value,token):
  p.write_text(json.dumps(value))
  try:prep.load_target_pose(p,binds,source)
  except ValueError as e:assert token in str(e),str(e)
  else:raise AssertionError("corruption unexpectedly passed")
 bad=copy.deepcopy(raw);bad["bodies"].append(copy.deepcopy(bad["bodies"][0]));rejects(bad,"duplicate")
 row=next(i for i,b in enumerate(raw["bodies"]) if b["body_index"]==ids[0])
 bad=copy.deepcopy(raw);bad["bodies"][row]["rest"]["position_world_m"][0]+=.001;rejects(bad,"rest/binding")
 bad=copy.deepcopy(raw);bad["bodies"][row]["current"]["orientation_world_xyzw"]=[0,0,0,0];rejects(bad,"quaternion")
 bad=copy.deepcopy(raw);bad["bodies"][row]["current"]["position_world_m"][0]=float("nan");rejects(bad,"invalid")
 bad=copy.deepcopy(raw);bad["bodies"]=[b for b in bad["bodies"] if b["body_index"]!=ids[0]];rejects(bad,"omit")
print(json.dumps({"native_target_pose":"PASS","binding_bodies":len(ids),"maximum_source_to_target_translation_m":float(maxdelta),"negative_cases":5,"physical_steps":0,"changed_volume_admitted":False},indent=2))
