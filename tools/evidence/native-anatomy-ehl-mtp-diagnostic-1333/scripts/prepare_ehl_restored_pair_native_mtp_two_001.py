from pathlib import Path
import sys,json,hashlib,argparse,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");B=A/"current-anatomy-mtp-two-native-002";D=A/"ehl-k2-refinement-interface-identity-004";O=A/"ehl-restored-pair-native-mtp-two-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import resting_run as rr,passive_attachment_composition as pc
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rp=D/"report.json";proof=json.loads(rp.read_text());assert proof["complete"] and proof["inputs_unchanged"] and proof["all_original_crossing_triangles_and_target_pairs_exact"]
ap=A/"ehl-current-anatomy-mtp-two-full-audit-003/report.json";audit=json.loads(ap.read_text())
assert audit["complete"] and audit["inputs_unchanged"] and len(audit["poses"])==8
assert all(not r["self_pairs"] and r["topology"]["closed_oriented_manifold_candidate"] and r["signed_volume_m3"]>0 for e in audit["poses"] for r in e["rows"] if r["variant"]=="refined")
T=B/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
replacements=[(sid,A/f"ehl-restored-boundary-batch-001/stable-{sid}/posed-conforming-refinement-001/stable-{sid}-unadmitted-row-patch.npz",rp) for sid in (23,24)]
composition=pc.compose(T,O/"assets",replacements,reference_surface_rows=(23,24));payload=O/"assets"/T.name;receipt=O/"resting-anatomy-receipt.json";bound=pc.bind_anatomy_receipt(B/"resting-anatomy-receipt.json",payload,receipt)
assert bound["mass_geometry_accounting"]==json.loads((B/"resting-anatomy-receipt.json").read_text())["mass_geometry_accounting"]
parent=pc._read_nhtiss4(T);child=pc._read_nhtiss4(payload)
for pr,cr in zip(parent["records"],child["records"]):
 if int(pr[6]) not in (23,24):assert pc._row_local_equal(pc._row_slices(parent,pr),pc._row_slices(child,cr))
(O/"composition-result.json").write_text(json.dumps(composition,indent=2,sort_keys=True)+"\n")
(O/"source-resolution-bounds.json").write_text(json.dumps({"rows":[23,24],"refinement_audit":str(ap),"refinement_audit_sha256":sha(ap),"unchanged_crossing_triangle_proof":proof,"all153_other_rows_exact":True,"mass_accounting_exact":True,"geometry_scope":"Source-restored and conformingly refined EHL boundaries; preexisting neighbor interfaces unchanged and unresolved. Current K2 diagnostic operating point retained, not final-qualified.","active_muscle_routes_and_tendons_unchanged":True},indent=2,sort_keys=True)+"\n")
bp=B/"baseline/run-declaration.json";base=json.loads(bp.read_text());idx=base["argv"].index("numilab_human.resting_run");av=base["argv"][idx+1:]
for flag,value in (("--output",str(O/"baseline/native-run")),("--anatomy-receipt",str(receipt))):av[av.index(flag)+1]=value
parser=argparse.ArgumentParser();rr.add_arguments(parser);args=parser.parse_args(av);native,assets=rr.command(args)
env=dict(x.split("=",1) for x in base["argv"][2:idx-2]);env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(args.output/"common-field-failure.json")
argv=["/usr/bin/env","-i"]+[k+"="+v for k,v in sorted(env.items())]+["/Users/n/numi-human-prep-venv-20261005/bin/python3.13","-m","numilab_human.resting_run"]+av
launch=O/"launch_arm.py";launch.write_text((B/"launch_arm.py").read_text());pins={**assets,**{str(p):sha(p) for p in (Path(__file__),bp,launch,rp,ap,A/"verify_ehl_k2_refinement_interface_identity_004.py",A/"refine_ehl_pose_triangles_001.py",Path(rr.__file__),Path(pc.__file__))},**{str(cp):sha(cp) for sid,cp,_ in replacements}}
for sid,cp,_ in replacements:
 proofpath=cp.parent/"report.json";pins[str(proofpath)]=sha(proofpath)

decl={"schema":"numi.human.native-smoke.run-declaration.v1","seconds":16.0,"accepted_steps":8000,"capture_steps":[0,4767,5023,5599,6207,6815,7423,8000],"immutable_assets":pins,"argv":argv,"launch_adapter":{"path":str(launch),"sha256":sha(launch)},"owner_cli_preview":{"native_argv":native,"asset_sha256":assets,"recorded_invocation_environment":rr.invocation_environment(env)},"qualification":"Native anatomical diagnostic: restored and conformingly refined EHL23/24 under existing K2 passive reference joint configuration. Original source vertices and maps preserved; source boundaries restored, generated edges interpolated. Offline eightposes closed/positive/selfclear, all crossing triangles against other surfaces exact to source-restored parent; preexisting anatomical interfaces remain unresolved. No new force, support or physiology. Other153 anatomy rows/mass/routes exact. K2 parameter and final five-minute wholebody qualification remain pending; performance deferred."}

(O/"baseline").mkdir();dp=O/"baseline/run-declaration.json";dp.write_text(json.dumps(decl,indent=2,sort_keys=True)+"\n");(O/"launch-guard.json").write_text(json.dumps({"launcher_sha256":sha(launch),"arms":{"baseline":{"run_declaration_sha256":sha(dp),"native_output":str(args.output)}}},indent=2,sort_keys=True)+"\n")
print(json.dumps({"path":str(O),"tissue_sha256":sha(payload),"pins":len(pins)}))
