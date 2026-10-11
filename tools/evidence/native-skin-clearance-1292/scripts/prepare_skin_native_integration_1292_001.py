from pathlib import Path
import json,hashlib,sys,argparse
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");P=A/"skin-candidate-30-pose-preparation-001/result-005";O=A/"skin-native-integration-1292-001";O.mkdir(exist_ok=False);(O/"baseline").mkdir()
sys.path.insert(0,"/Users/n/numi-human-positive-winding-1279/src")
from numilab_human import resting_run as rr
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rp=P/"preparation-report.json";r=json.loads(rp.read_text());assert r["all_declared_inputs_unchanged"] and not r["candidate_admitted"]
preview=P/"run-declaration-preview.json";d=json.loads(preview.read_text())
for p,v in d["immutable_assets"].items():assert sha(p)==v,(p,"input drift")
old=d["candidate_preview"]["output_path_reserved_not_created"];new=str(O/"baseline/native-run");assert not Path(old).exists()
argv=[v.replace(old,new) for v in d["argv"]];start=argv.index("numilab_human.resting_run")+1
parser=argparse.ArgumentParser();rr.add_arguments(parser);args=parser.parse_args(argv[start:]);native,assets=rr.command(args)
env=dict(x.split("=",1) for x in argv[2:argv.index("/Users/n/numi-human-prep-venv-20261005/bin/python3.13")])
launch=O/"launch_arm.py";launch.write_bytes((A/"thirty-surface-native-composition-001/launch_arm.py").read_bytes())
d.update(argv=argv,launch_adapter={"path":str(launch),"sha256":sha(launch)},owner_cli_preview={"native_argv":native,"asset_sha256":assets,"recorded_invocation_environment":rr.invocation_environment(env)},qualification="Native16s geometry-only integration of reference-derived skin clearance on current30 muscle scene. SCM151/152 are not added; their vessel/interface qualification remains independent. Skin displacement <=2.781mm, exactly fixed32contact witnesses/ocular boundaries/bindings. No full-body or five-minute final-anatomy admission. Compare complete coupled trace/accepted poses/unchanged anatomy to native30; bind fresh skin captures to eight-pose offline proof.")
d["immutable_assets"].update(assets);d["immutable_assets"].update({str(p):sha(p) for p in (Path(__file__),preview,rp,launch)})
d["candidate_preview"]={"status":"prepared_for_fresh_native_verification","parent_preview_path":str(preview),"parent_preview_sha256":sha(preview),"not_a_combined_SCM_admission":True}
dp=O/"baseline/run-declaration.json";dp.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
(O/"launch-guard.json").write_text(json.dumps({"launcher_sha256":sha(launch),"arms":{"baseline":{"run_declaration_sha256":sha(dp),"native_output":new}}},indent=2)+"\n")
print(json.dumps({"output":str(O),"pins":len(d["immutable_assets"]),"skin_sha256":r["candidate_skin"]["sha256"]}))

