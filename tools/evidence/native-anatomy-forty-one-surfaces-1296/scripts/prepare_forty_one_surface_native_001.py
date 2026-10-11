from pathlib import Path
import sys,json,hashlib,argparse,numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276"
O=A/"forty-one-surface-native-composition-001";O.mkdir(exist_ok=False)
SRC=Path("/Users/n/numi-human-positive-winding-1279/src");sys.path.insert(0,str(SRC))
from numilab_human import passive_attachment_composition as pc,resting_run as rr
from numilab_human.resting_anatomy_interface_patch import normals,signed_volume
from numilab_human.cardiac_cavity_geometry import analyze_topology
B=A/"forty-surface-native-composition-001"
SK=A/"skin-candidate-30-pose-preparation-001/result-005"
SN=B
T=B/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
P=R/"skin-resting-multipose-clearance-1218/native-coupled-pair-preparation-011"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=pc._read_nhtiss4(T);replacements=[];bounds={};proofpaths=[];union_inputs=[]
for stem,proof in [("foot25-current-reference-selection-013","foot25-current-interface-audit-012")]:
 rp=A/stem/"report.json";ip=A/proof/"report.json"
 report=json.loads(rp.read_text());inter=json.loads(ip.read_text())
 for doc in (report,inter):
  assert doc["complete"] and doc["inputs_unchanged"]
  for path,digest in doc["pins"].items():assert sha(path)==digest
  union_inputs.extend(Path(path) for path in doc["pins"])
 assert len(inter["poses"])==8
 assert all(r["self_count"]==r["new_triangle_pair_count"]==0 and r["external_targets_scanned"]==859 for p in inter["poses"] for r in p["rows"])
 for row in report["rows"]:
  sid=row["stable_id"];assert sid in (25,)
  assert row["source_and_sampled_pose_self_clear"] and not row["changed_or_missing_pure_attachment_proxies"]
  cp=Path(row["candidate_path"]);assert sha(cp)==row["candidate_sha256"]
  replacements.append((sid,cp,rp))
  bounds[sid]={"operation":"Existing positive-winding reference boundary with source-face binding lift, micrometre junction separation, restored original source attachment proxies and bounded generated-point correction across24K0poses; exact derivation in selection pins.","source_area_m2":row["source_area_m2"],"candidate_area_m2":row["candidate_area_m2"],"source_signed_volume_m3":row["source_signed_volume_m3"],"candidate_signed_volume_m3":row["candidate_signed_volume_m3"],"pure_attachment_proxy_count":row["pure_attachment_proxy_count"],"pure_attachment_proxies_preserved":True,"native_verification_pending":True,"retained_pose_scope":"24retainedK0poses; K1 stiffness sensitivity remains3selfpairs and is explicitly unqualified"}
 proofpaths.extend([rp,ip])
assert {sid for sid,_,_ in replacements}=={25}
parent_report=SN/"native-verification-001/report.json";parent=json.loads(parent_report.read_text())
assert parent["all_simultaneously_repaired_pairs_pass"]
assert parent["complete"] and parent["inputs_unchanged"] and parent["all_native_pair_checks_pass"] and parent["all_prior38_and_skin_geometry_exact"] and parent["all_coupled_trace_rows_exact_parent"]
proofpaths.append(parent_report)
payload=O/"assets"/T.name
composition=pc.compose(T,payload.parent,replacements,reference_surface_rows=(25,))
receipt=O/"resting-anatomy-receipt.json";pc.bind_anatomy_receipt(B/"resting-anatomy-receipt.json",payload,receipt)
(O/"composition-result.json").write_text(json.dumps(composition,indent=2,sort_keys=True)+"\n")
(O/"source-resolution-bounds.json").write_text(json.dumps(bounds,indent=2,sort_keys=True)+"\n")
base=json.loads((P/"baseline/run-declaration.json").read_text());argsv=base["argv"][base["argv"].index("human-resting")+1:]
for flag,value in [("--body-scene",str(SK/"resting-supine-scene.manifest.json")),("--anatomy-receipt",str(receipt)),("--output",str(O/"baseline/native-run")),("--seconds","16.0")]:argsv[argsv.index(flag)+1]=value
argsv+=["--lab","/Users/n/numi-lab-complete-body-capture-1276","--build","/Users/n/numi-lab-complete-body-capture-build-1276-001"]
parser=argparse.ArgumentParser();rr.add_arguments(parser);args=parser.parse_args(argsv);native,assets=rr.command(args)
env={}
for item in base["argv"][2:base["argv"].index("human-resting")-1]:
 key,value=item.split("=",1);env[key]=value
env["NUMI_HUMAN_ROOT"]=str(SRC.parent);env["PYTHONPATH"]=str(SRC)
env["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"]="0,4767,5023,5599,6207,6815,7423,8000"
assert all(n in (0,8000) or (n+1)%32==0 for n in map(int,env["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"].split(",")))
# Baseline011's Q observer is inactive throughout its first16s.
# A window beyond this short run is rejected; disable the observer and omit bounds.
env["NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT"]="0"
env.pop("NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_FIRST_STEP",None)
env.pop("NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP",None)
env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(args.output/"common-field-failure.json")
env.update(DYLD_LIBRARY_PATH=str(args.build/"lib")+":"+str(args.build/"matter"),DYLD_PRINT_LIBRARIES="1",NUMI_HUMAN_RESTING_INSPECTION_TOUR="1",NUMI_HUMAN_RESTING_INSPECTION_PERIOD_SECONDS=str(args.inspection_period_seconds))
argv=["/usr/bin/env","-i"]+[k+"="+v for k,v in sorted(env.items())]+["/Users/n/numi-human-prep-venv-20261005/bin/python3.13","-m","numilab_human.resting_run"]+argsv
launch=O/"launch_arm.py";launch.write_text((A/"ecu-apl-native-composition-005/launch_arm.py").read_text().replace("6000 * 1024 * 1024", "4500 * 1024 * 1024"))
pins={**assets,**{str(p):sha(p) for p in [Path(__file__),Path(pc.__file__),Path(rr.__file__),launch,O/"source-resolution-bounds.json",*proofpaths,*union_inputs]}}
for _,npz,rp in replacements:pins[str(npz)]=sha(npz);pins[str(rp)]=sha(rp)
decl={"schema":"numi.human.native-smoke.run-declaration.v1","seconds":16.0,"accepted_steps":8000,"capture_steps":[0,4767,5023,5599,6207,6815,7423,8000],"immutable_assets":pins,"argv":argv,"launch_adapter":{"path":str(launch),"sha256":sha(launch)},"owner_cli_preview":{"native_argv":native,"asset_sha256":assets,"recorded_invocation_environment":rr.invocation_environment(env)},"qualification":"Existing native40-surface repairs and corrected skin plus right flexor digitorum longus repair. Source and24retained K0poses selfclear;8early poses no new changed-star pairs against859 targets. K1 alternate stiffness pose remains3selfpairs and is unqualified. Native actual geometry and candidate-candidate pair verification pending. Not whole-body or five-minute qualification; performance deferred."}
(O/"baseline").mkdir();dp=O/"baseline/run-declaration.json";dp.write_text(json.dumps(decl,indent=2,sort_keys=True)+"\n")
(O/"launch-guard.json").write_text(json.dumps({"launcher_sha256":sha(launch),"arms":{"baseline":{"run_declaration_sha256":sha(dp),"native_output":str(args.output)}}},indent=2,sort_keys=True)+"\n")
print(json.dumps({"output":str(O),"payload_sha256":sha(payload),"pins":len(pins),"bounds":bounds},indent=2))

