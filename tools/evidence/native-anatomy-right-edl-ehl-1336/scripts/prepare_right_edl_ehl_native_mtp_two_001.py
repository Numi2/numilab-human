from pathlib import Path
import sys,json,hashlib,argparse,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
B=A/"right-deep-masseter-native-003";K=A/"ehl-restored-pair-native-mtp-two-001";O=A/"right-edl-ehl-native-mtp-two-001"
P=Path("/Users/n/numi-human-complete-reference-inventory-1335")
sys.path.insert(0,str(P/"src"))
from numilab_human import resting_run as rr,passive_attachment_composition as pc
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
audits={21:A/"edl-right-local-binding-target-audit-030/report.json"}
extra=A/"edl-right-refined-ehl-target-supplement-035/report.json"
for path in [*audits.values(),extra]:
 a=json.loads(path.read_text())
 assert a["complete"] and a["inputs_unchanged"] and a["all_pose_self_clear"] and a["no_new_target_pairs"],path
 assert len(a["poses"])==8
 assert all(e["rows"][1]["topology"]["closed_oriented_manifold_candidate"] and e["rows"][1]["signed_volume_m3"]>0 for e in a["poses"])
 for p,d in a["pins"].items():assert sha(p)==d,p
ep=A/"ehl-k2-refinement-interface-identity-004/report.json";eh=json.loads(ep.read_text())
assert eh["complete"] and eh["inputs_unchanged"] and eh["all_original_crossing_triangles_and_target_pairs_exact"]
ev=K/"native-verification-001/report.json";vr=json.loads(ev.read_text())
assert vr["all_native_geometry_checks_pass"] and vr["all_native_interface_pairs_exact_offline"] and vr["inputs_unchanged"]
replacements=[
 (21,A/"edl-right-local-binding-sensitivity-029/stable-21-unadmitted-iteration-3.npz",audits[21]),
 *[(sid,A/f"ehl-restored-boundary-batch-001/stable-{sid}/posed-conforming-refinement-001/stable-{sid}-unadmitted-row-patch.npz",ep) for sid in [23,24]]
]
O.mkdir(exist_ok=False)
T=B/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
composition=pc.compose(T,O/"assets",replacements,reference_surface_rows=(21,23,24))
payload=O/"assets"/T.name;receipt=O/"resting-anatomy-receipt.json"
bound=pc.bind_anatomy_receipt(B/"resting-anatomy-receipt.json",payload,receipt)
assert bound["mass_geometry_accounting"]==json.loads((B/"resting-anatomy-receipt.json").read_text())["mass_geometry_accounting"]
parent=pc._read_nhtiss4(T);child=pc._read_nhtiss4(payload);assert parent["surface_count"]==child["surface_count"]==156
for pr,cr in zip(parent["records"],child["records"],strict=True):
 if int(pr[6]) not in [21,23,24]:assert pc._row_local_equal(pc._row_slices(parent,pr),pc._row_slices(child,cr))
(O/"composition-result.json").write_text(json.dumps(composition,indent=2,sort_keys=True)+"\n")
bounds={"changed_rows":[21,23,24],"all153_other_rows_exact":True,"mass_accounting_exact":True,"active_routes_tendons_and_support_transforms_unchanged":True,"inferred_display_weight_changes":{"21":[146,917,1301,1628]},"reference_geometry_scope":"EDL positive-union source boundaries with local inferred mixed display weights; original one-hot proxy weights and source coordinates retained. EHL restored/conforming boundaries preserve source coordinates/maps. Existing neighboring anatomical intersections remain unresolved; these refinements are not final anatomical qualification.","operating_point":"Explicit existing2 Nm/rad aggregate passive five-ray MTP diagnostic; supported flat-plane contact. Native physiology derives from unchanged coupled GPU owner."}
(O/"source-resolution-bounds.json").write_text(json.dumps(bounds,indent=2,sort_keys=True)+"\n")
bp=K/"baseline/run-declaration.json";base=json.loads(bp.read_text());idx=base["argv"].index("numilab_human.resting_run");av=base["argv"][idx+1:]
for flag,value in [("--output",str(O/"baseline/native-run")),("--anatomy-receipt",str(receipt))]:av[av.index(flag)+1]=value
parser=argparse.ArgumentParser();rr.add_arguments(parser);args=parser.parse_args(av);native,assets=rr.command(args)
env=dict(x.split("=",1) for x in base["argv"][2:idx-2]);env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(args.output/"common-field-failure.json")
argv=["/usr/bin/env","-i"]+[k+"="+v for k,v in sorted(env.items())]+["/Users/n/numi-human-prep-venv-20261005/bin/python3.13","-m","numilab_human.resting_run"]+av
launch=O/"launch_arm.py";launch.write_text((K/"launch_arm.py").read_text())
paths=[Path(__file__),bp,launch,*audits.values(),extra,ep,ev,Path(rr.__file__),Path(pc.__file__),O/"source-resolution-bounds.json",
 A/"try_edl_right_local_binding_sensitivity_029.py",
 A/"audit_edl_right_local_binding_targets_030.py",
 A/"audit_edl_right_refined_ehl_targets_035.py",A/"edl-right-local-binding-sensitivity-029/report.json"]
pins={**assets,**{str(p):sha(p) for p in paths},**{str(cp):sha(cp) for _,cp,_ in replacements}}
decl={"schema":"numi.human.native-smoke.run-declaration.v1","seconds":16.0,"accepted_steps":8000,"capture_steps":[0,4767,5023,5599,6207,6815,7423,8000],"immutable_assets":pins,"argv":argv,"launch_adapter":{"path":str(launch),"sha256":sha(launch)},"owner_cli_preview":{"native_argv":native,"asset_sha256":assets,"recorded_invocation_environment":rr.invocation_environment(env)},"qualification":"Integrated native anatomical diagnostic: right EDL21 source boundary and local inferred binding correction, plus source-restored conformingly refined EHL23/24, together with both admitted deep masseters and all existing anatomy. Eight K2 offline poses are closed/positive/selfclear and EDL target tables have no new pairs versus original union maps. Existing target interfaces are unresolved; no final anatomy or five-minute qualification claim. Other153 rows, mass, physical muscle routes and GPU physiology unchanged. Flat support contact; performance deferred."}
(O/"baseline").mkdir();dp=O/"baseline/run-declaration.json";dp.write_text(json.dumps(decl,indent=2,sort_keys=True)+"\n")
(O/"launch-guard.json").write_text(json.dumps({"launcher_sha256":sha(launch),"arms":{"baseline":{"run_declaration_sha256":sha(dp),"native_output":str(args.output)}}},indent=2,sort_keys=True)+"\n")
print(json.dumps({"path":str(O),"tissue_sha256":sha(payload),"pins":len(pins)}))
