from pathlib import Path
import sys,json,hashlib,argparse,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
B=A/"right-edl-ehl-native-mtp-two-001";O=A/"bilateral-edl-ehl-native-mtp-two-001";P=Path("/Users/n/numi-human-complete-reference-inventory-1335")
sys.path.insert(0,str(P/"src"))
from numilab_human import resting_run as rr,passive_attachment_composition as pc
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
AP=A/"edl-left-posed-union-current-target-audit-054/report.json";SP=A/"edl-left-posed-union-ancestry-map-sensitivity-053/report.json";CP=A/"edl-left-generated-source-characterization-056/report.json"
for path in [AP,SP,CP]:
 d=json.loads(path.read_text());assert d["complete"] and d["inputs_unchanged"],path
 for p,v in d["pins"].items():assert sha(p)==v,p
a=json.loads(AP.read_text());assert a["all_source_and_pose_self_clear"] and a["no_strict_new_parent_pairs"] and len(a["poses"])==8
assert all(e["rows"][1]["closed"] and e["rows"][1]["components"]==14 and e["rows"][1]["signed_volume_m3"]>0 and e["rows"][1]["target_count"]==865 for e in a["poses"])
c=json.loads(CP.read_text());assert c["source_closed"] and all(r["signed_volume_m3"]>0 for r in c["component_volumes"]) and c["original_vertices_preserved"]==2289 and c["generated_vertex_count"]==3
assert abs(a["source_relative_area_change"])<.001 and abs(a["source_relative_volume_change"])<.001
C=A/"edl-left-posed-union-ancestry-map-sensitivity-053/stable-22-unadmitted-alpha-0.75-0.75-0.75.npz"
s=json.loads(SP.read_text());trial=next(t for t in s["trials"] if t["alphas"]==[.75,.75,.75]);assert trial["sha256"]==sha(C) and len(trial["audits"])==9 and not trial["initial_bone21_strict_added"]
assert all(not x["pairs"] and not x["degenerate"] for x in trial["audits"])
ev=B/"native-verification-003/report.json";vr=json.loads(ev.read_text());assert vr["all_native_geometry_checks_pass"] and vr["inputs_unchanged"]
O.mkdir(exist_ok=False);T=B/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
composition=pc.compose(T,O/"assets",[(22,C,AP)],reference_surface_rows=(22,))
payload=O/"assets"/T.name;receipt=O/"resting-anatomy-receipt.json"
bound=pc.bind_anatomy_receipt(B/"resting-anatomy-receipt.json",payload,receipt)
assert bound["mass_geometry_accounting"]==json.loads((B/"resting-anatomy-receipt.json").read_text())["mass_geometry_accounting"]
parent=pc._read_nhtiss4(T);child=pc._read_nhtiss4(payload);assert parent["surface_count"]==child["surface_count"]==156
for pr,cr in zip(parent["records"],child["records"],strict=True):
 if int(pr[6])!=22:assert pc._row_local_equal(pc._row_slices(parent,pr),pc._row_slices(child,cr))
(O/"composition-result.json").write_text(json.dumps(composition,indent=2,sort_keys=True)+"\n")
bounds={"changed_rows":[22],"all155_other_rows_exact":True,"mass_accounting_exact":True,"active_routes_tendons_and_support_transforms_unchanged":True,"original_union_source_vertices_preserved":2289,"original_visual_one_hot_proxy_maps_preserved":True,"existing_mixed_display_map_changes":{"986":{"local_body":0,"blend":.05},"1371":{"local_body":0,"blend":.02}},"generated_reference_vertices":3,"generated_reference_faces_net_added":6,"generated_map_ancestry_blend":.75,"scope":"Inferred reference boundary reconstruction of left EDL22 via the existing exact positive-winding boundary owner in a retained pose, then source lifting through the existing affine skin-binding owner. All original union source coordinates remain fixed. Existing physical muscle/tendon routes, supports and physiology unchanged. This is passive anatomical display geometry, not measured entheses or a new tissue solver.","generated_point_bound_disposition":"Attempt050 rejected distance from each generated point to every individual ancestry barycentric point above0.5mm. That failed diagnostic is retained. This native diagnostic uses an explicitly inferred boundary rather than claiming that check passed: current maximum individual ancestry-point distance1.093mm, maximum new-point distance to the actual original source surface0.0178mm, and maximum sampled changed-face distance to that surface0.116mm. The sampled distances are not a full Hausdorff bound. All2289 existing source points have zero displacement. Source and eight poses are closed/self-clear, with positive components and no new exact parent interfaces among865 neighbors. Full anatomical and five-minute qualification remain open.","source_relative_area_change":a["source_relative_area_change"],"source_relative_volume_change":a["source_relative_volume_change"],"operating_point":"Unchanged existing2Nm/rad aggregate passive MTP diagnostic, flat-plane contact, coupled GPU physiology."}
(O/"source-resolution-bounds.json").write_text(json.dumps(bounds,indent=2,sort_keys=True)+"\n")
bp=B/"baseline/run-declaration.json";base=json.loads(bp.read_text());idx=base["argv"].index("numilab_human.resting_run");av=base["argv"][idx+1:]
for flag,value in [("--output",str(O/"baseline/native-run")),("--anatomy-receipt",str(receipt))]:av[av.index(flag)+1]=value
parser=argparse.ArgumentParser();rr.add_arguments(parser);args=parser.parse_args(av);native,assets=rr.command(args)
env=dict(x.split("=",1) for x in base["argv"][2:idx-2]);env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(args.output/"common-field-failure.json")
argv=["/usr/bin/env","-i"]+[k+"="+v for k,v in sorted(env.items())]+["/Users/n/numi-human-prep-venv-20261005/bin/python3.13","-m","numilab_human.resting_run"]+av
launch=O/"launch_arm.py";launch.write_text((B/"launch_arm.py").read_text())
paths=[Path(__file__),bp,launch,AP,SP,CP,C,ev,Path(rr.__file__),Path(pc.__file__),O/"source-resolution-bounds.json",A/"try_edl_left_posed_union_ancestry_maps_053.py",A/"audit_edl_left_posed_union_current_targets_054.py",A/"characterize_edl_left_generated_source_056.py",A/"edl-left-posed-positive-union-049/stable-22-posed-exact-union.json",A/"edl-left-posed-union-source-lift-050/report.json"]
pins={**assets,**{str(p):sha(p) for p in paths}}
decl={"schema":"numi.human.native-smoke.run-declaration.v1","seconds":16.0,"accepted_steps":8000,"capture_steps":[0,4767,5023,5599,6207,6815,7423,8000],"immutable_assets":pins,"argv":argv,"launch_adapter":{"path":str(launch),"sha256":sha(launch)},"owner_cli_preview":{"native_argv":native,"asset_sha256":assets,"recorded_invocation_environment":rr.invocation_environment(env)},"qualification":"Integrated native left EDL22 inferred reference-boundary diagnostic, adding to the current bilateral EHL/right EDL and deep masseters. Three new source points and six net faces; all original source points retained; two preexisting mixed display weights locally inferred. Existing target intersections remain unresolved. Other155 rows, mass, physical muscle/tendon routes and GPU physiology unchanged. No full anatomy or five-minute claim; performance deferred."}
(O/"baseline").mkdir();dp=O/"baseline/run-declaration.json";dp.write_text(json.dumps(decl,indent=2,sort_keys=True)+"\n")
(O/"launch-guard.json").write_text(json.dumps({"launcher_sha256":sha(launch),"arms":{"baseline":{"run_declaration_sha256":sha(dp),"native_output":str(args.output)}}},indent=2,sort_keys=True)+"\n")
print(json.dumps({"path":str(O),"tissue_sha256":sha(payload),"pins":len(pins)}))
