from pathlib import Path
import sys,json,hashlib,argparse,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");B=A/"left-scm-masseters-outward-native-001";D=A/"sartorius50-native-precision-repair-002";O=A/"sartorius50-precision-native-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import resting_run as rr,passive_attachment_composition as pc
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rp=D/"pose-target-audit-001/report.json";proof=json.loads(rp.read_text());assert proof["complete"] and proof["inputs_unchanged"] and proof["all30_self_closed_clear"] and proof["no_new_parent_target_pairs"]
T=B/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";C=D/"stable-50-unadmitted.npz"
composition=pc.compose(T,O/"assets",[(50,C,rp)],reference_surface_rows=(50,));payload=O/"assets"/T.name;receipt=O/"resting-anatomy-receipt.json";bound=pc.bind_anatomy_receipt(B/"resting-anatomy-receipt.json",payload,receipt)
assert bound["mass_geometry_accounting"]==json.loads((B/"resting-anatomy-receipt.json").read_text())["mass_geometry_accounting"]
parent=pc._read_nhtiss4(T);child=pc._read_nhtiss4(payload)
for pr,cr in zip(parent["records"],child["records"]):
 if int(pr[6])!=50:assert pc._row_local_equal(pc._row_slices(parent,pr),pc._row_slices(child,cr))
(O/"composition-result.json").write_text(json.dumps(composition,indent=2,sort_keys=True)+"\n")
(O/"source-resolution-bounds.json").write_text(json.dumps({"50":{"repair":json.loads((D/"report.json").read_text()),"pose_target_audit":str(rp),"pose_target_audit_sha256":sha(rp),"prior153_rows_exact":True,"mass_accounting_exact":True}},indent=2,sort_keys=True)+"\n")
bp=B/"baseline/run-declaration.json";base=json.loads(bp.read_text());idx=base["argv"].index("numilab_human.resting_run");av=base["argv"][idx+1:]
for flag,value in (("--output",str(O/"baseline/native-run")),("--anatomy-receipt",str(receipt))):av[av.index(flag)+1]=value
parser=argparse.ArgumentParser();rr.add_arguments(parser);args=parser.parse_args(av);native,assets=rr.command(args)
env=dict(x.split("=",1) for x in base["argv"][2:idx-2]);env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(args.output/"common-field-failure.json")
argv=["/usr/bin/env","-i"]+[k+"="+v for k,v in sorted(env.items())]+["/Users/n/numi-human-prep-venv-20261005/bin/python3.13","-m","numilab_human.resting_run"]+av
launch=O/"launch_arm.py";launch.write_text((B/"launch_arm.py").read_text());pins={**assets,**{str(p):sha(p) for p in (Path(__file__),bp,launch,C,rp,D/"report.json",A/"repair_sartorius50_native_precision_002.py",Path(rr.__file__),Path(pc.__file__))}}
decl={"schema":"numi.human.native-smoke.run-declaration.v1","seconds":16.0,"accepted_steps":8000,"capture_steps":[0,4767,5023,5599,6207,6815,7423,8000],"immutable_assets":pins,"argv":argv,"launch_adapter":{"path":str(launch),"sha256":sha(launch)},"owner_cli_preview":{"native_argv":native,"asset_sha256":assets,"recorded_invocation_environment":rr.invocation_environment(env)},"qualification":"Native validation of precision endpoint collapse in passive left sartorius50;0.133micrometre sourceedge removed with allthree closedcomponents preserved. Current MTP0 configuration unchanged. No new force/support/physics; allother153 anatomyrows exact. Exactpose/target checks pending. Wholebody completion and final five-minute pair stilloutstanding; performance deferred."}
(O/"baseline").mkdir();dp=O/"baseline/run-declaration.json";dp.write_text(json.dumps(decl,indent=2,sort_keys=True)+"\n");(O/"launch-guard.json").write_text(json.dumps({"launcher_sha256":sha(launch),"arms":{"baseline":{"run_declaration_sha256":sha(dp),"native_output":str(args.output)}}},indent=2,sort_keys=True)+"\n")
print(json.dumps({"path":str(O),"tissue_sha256":sha(payload),"pins":len(pins)}))
