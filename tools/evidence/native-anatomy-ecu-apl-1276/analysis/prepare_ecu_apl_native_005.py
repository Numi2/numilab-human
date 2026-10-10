from pathlib import Path
import sys,json,hashlib,argparse,shutil
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276"
O=A/"ecu-apl-native-composition-005";O.mkdir(exist_ok=False)
SRC=Path("/Users/n/numi-human-anatomy-completion-1276/src")
sys.path.insert(0,str(SRC))
from numilab_human import passive_attachment_composition as pc,resting_run as rr
from numilab_human.resting_lung_edge_repair import collapse_midpoint_edge
from numilab_human.resting_anatomy_interface_patch import normals
T=R/"source-seam-connectivity-1247/fhl-current-7b23-count-reconciliation-1258/attempt-002/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
B=R/"skin-resting-multipose-clearance-1218/package-preparation-005/composed-candidate-001"
P=R/"skin-resting-multipose-clearance-1218/native-coupled-pair-preparation-011"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
auditpath=A/"ecu-apl-allpose-interface-audit-002/report.json";audit=json.loads(auditpath.read_text())
assert audit["complete"] and audit["inputs_unchanged"] and len(audit["poses"])==16
assert all(x["self_count"]==x["new_triangle_pair_count"]==0 and x["external_targets_scanned"]==859 for p in audit["poses"] for x in p["rows"])
d=pc._read_nhtiss4(T);replacements=[];bound={}
for name in ["ecu-source-resolution-001","apl-tfl-source-resolution-001"]:
 rp=A/name/"report.json";report=json.loads(rp.read_text())
 assert report["complete"] and report["inputs_unchanged"]
 for row in report["rows"]:
  if not row["source_and_sampled_pose_self_clear"]:continue
  sid=row["stable_id"];npz=Path(row["candidate_path"]);assert sha(npz)==row["candidate_sha256"]
  src=pc._biceps_row_arrays(pc._row_slices(d,next(r for r in d["records"] if int(r[6])==sid)))
  u,first,inv=np.unique(src["vertices6"][:,:3],axis=0,return_index=True,return_inverse=True)
  used=np.unique(inv[src["faces"]]);mapping=np.full(len(u),-1);mapping[used]=np.arange(len(used))
  f=mapping[inv[src["faces"]]];ref=u[used];v=np.column_stack((ref,normals(ref.astype(float),f))).astype(np.float32)
  ancestor=[{int(j)} for j in first[used]];origin=np.arange(len(f));reference=src["vertices6"][:,:3]
  for op in row["operations"]:
   remove=op["removed_vertex_before_compaction"];keep=op["retained_vertex_before_compaction"]
   nv,nf,no,check=collapse_midpoint_edge(v,f,origin,remove_vertex=remove,keep_vertex=keep,max_endpoint_displacement_m=.00025,max_abs_volume_delta_m3=1e-10)
   assert check==op
   replaced=f.copy();replaced[replaced==remove]=keep
   valid=(replaced[:,0]!=replaced[:,1])&(replaced[:,1]!=replaced[:,2])&(replaced[:,2]!=replaced[:,0])
   survived=np.unique(replaced[valid]);newa=[ancestor[int(j)] for j in survived]
   newa[op["retained_vertex_after_compaction"]]=ancestor[keep]|ancestor[remove]
   v,f,origin,ancestor=nv,nf,no,newa
  z=np.load(npz);assert np.array_equal(v,z["vertices6"]) and np.array_equal(f,z["faces"]) and np.array_equal(origin,z["face_origins"])
  maxmove=max(float(np.linalg.norm(v[i,:3].astype(float)-reference[list(anc)].astype(float),axis=1).max()) for i,anc in enumerate(ancestor))
  pure=set(np.flatnonzero(src["weights"].max(1)>=.999999)) & set(first[used])
  touched=set().union(*(a for a in ancestor if len(a)>1))
  assert maxmove<=.00025 and not (pure&touched)
  bound[sid]={"maximum_original_ancestry_displacement_m":maxmove,"pure_attachment_proxy_count":len(pure),"pure_attachment_proxy_changed":False,"source_reconstruction_matches_candidate":True}
  replacements.append((sid,npz,rp))
assert sorted(x[0] for x in replacements)==[115,116,147,148]
payload=O/"assets"/T.name
composition=pc.compose(T,payload.parent,replacements,reference_surface_rows=(115,116,147,148))
receipt=O/"resting-anatomy-receipt.json"
binding=pc.bind_anatomy_receipt(B/"composed-anatomy/resting-anatomy-receipt.json",payload,receipt)
(O/"composition-result.json").write_text(json.dumps(composition,indent=2,sort_keys=True)+"\n")
(O/"binding-result.json").write_text(json.dumps(binding,indent=2,sort_keys=True)+"\n")
(O/"source-resolution-bounds.json").write_text(json.dumps(bound,indent=2,sort_keys=True)+"\n")
base=json.loads((P/"baseline/run-declaration.json").read_text())
argsv=base["argv"][base["argv"].index("human-resting")+1:]
for flag,value in [("--anatomy-receipt",str(receipt)),("--output",str(O/"baseline/native-run")),("--seconds","16.0")]:
 argsv[argsv.index(flag)+1]=value
argsv+=["--lab","/Users/n/numi-lab-resting-window-vector-trace-1271","--build","/Users/n/numi-lab-resting-window-vector-trace-build-1271-001"]
ap=argparse.ArgumentParser();rr.add_arguments(ap);args=ap.parse_args(argsv);native,assets=rr.command(args)
env={}
for item in base["argv"][2:base["argv"].index("human-resting")-1]:
 key,value=item.split("=",1);env[key]=value
env["NUMI_HUMAN_ROOT"]=str(SRC.parent)
env["PYTHONPATH"]=str(SRC)
env["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"]="0,4767,5023,5599,6207,6815,7423,8000"
assert all(n in (0,8000) or (n+1)%32==0 for n in map(int,env["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"].split(",")))
env["NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_FIRST_STEP"]="5001"
env["NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP"]="8000"
env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(args.output/"common-field-failure.json")
env.update(DYLD_LIBRARY_PATH=str(args.build/"lib")+":"+str(args.build/"matter"),DYLD_PRINT_LIBRARIES="1",NUMI_HUMAN_RESTING_INSPECTION_TOUR="1",NUMI_HUMAN_RESTING_INSPECTION_PERIOD_SECONDS=str(args.inspection_period_seconds))
argv=["/usr/bin/env","-i"]+[k+"="+v for k,v in sorted(env.items())]+["/Users/n/numi-human-prep-venv-20261005/bin/python3.13","-m","numilab_human.resting_run"]+argsv
launch=O/"launch_arm.py"
launch.write_text((P/"launch_arm.py").read_text().replace("Launch one frozen 310-second coupled-physiology arm; refuses existing outputs.","Launch one frozen coupled-physiology arm; refuses existing outputs.").replace('"310-second matched coupled-physiology arm; known passive anatomy defects remain, so not whole-anatomy qualification"', 'declaration["qualification"]'))
pins={**assets,**{str(p):sha(p) for p in [Path(__file__),Path(pc.__file__),Path(rr.__file__),auditpath,launch,O/"source-resolution-bounds.json"]}}
for _,npz,rp in replacements:pins[str(npz)]=sha(npz);pins[str(rp)]=sha(rp)
decl={"schema":"numi.human.native-smoke.run-declaration.v1","seconds":16.0,"accepted_steps":8000,"capture_steps":[0,4767,5023,5599,6207,6815,7423,8000],"immutable_assets":pins,"argv":argv,"launch_adapter":{"path":str(launch),"sha256":sha(launch)},"owner_cli_preview":{"native_argv":native,"asset_sha256":assets,"recorded_invocation_environment":rr.invocation_environment(env)},"qualification":"Four passive muscle repairs integrated into existing resting coupled human. Short anatomy verification only, not five-minute or whole-body qualification."}
(O/"baseline").mkdir()
dp=O/"baseline/run-declaration.json";dp.write_text(json.dumps(decl,indent=2,sort_keys=True)+"\n")
guard={"launcher_sha256":sha(launch),"arms":{"baseline":{"run_declaration_sha256":sha(dp),"native_output":str(args.output)}}}
(O/"launch-guard.json").write_text(json.dumps(guard,indent=2,sort_keys=True)+"\n")
print(json.dumps({"output":str(O),"tissue_sha256":sha(payload),"input_pins":len(pins),"rows":bound,"launch":["/usr/bin/python3",str(launch),"baseline"]},indent=2))
