from pathlib import Path
import sys,json,hashlib,argparse,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");B=A/"bilateral-edl-ehl-native-mtp-two-001";O=A/"left-pq-retriangulated-native-mtp-two-002";P=Path("/Users/n/numi-human-complete-reference-inventory-1335")
sys.path.insert(0,str(P/"src"))
from numilab_human import resting_run as rr,passive_attachment_composition as pc
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
AP=A/"pq-left-retriangulated-native-target-audit-029/report.json";LP=A/"pq-left-retriangulated-all25-target-audit-030/report.json";CP=A/"pq-left-retriangulated-source-characterization-032/report.json";SP=A/"pq-left-native-local-retriangulation-028/report.json"
for path in [AP,LP,CP,SP]:
 d=json.loads(path.read_text());assert d["complete"] and d["inputs_unchanged"],path
 for p,v in d["pins"].items():assert sha(p)==v,p
a=json.loads(AP.read_text());l=json.loads(LP.read_text());c=json.loads(CP.read_text());s=json.loads(SP.read_text())
assert a["all_source_and_pose_self_clear"] and a["no_strict_new_parent_pairs"] and len(a["poses"])==8
assert l["source_and_all25_sampled_pose_self_clear"] and l["all25_retained_targets_no_added_pairs"] and len(l["poses"])==25
assert c["all_components_positive"] and c["maps_and_raw_parent_lineage_exact_source_reference"] and c["onehot_source_coordinates_exact_reference"]
assert c["max_source_correction_from_micrometre_boundary_m"]<.0005 and max(abs(c[k]) for k in ["source_relative_area_change","source_relative_volume_change"])<.001
C=Path(s["candidate_path"]);assert sha(C)==s["candidate_sha256"]
ev=B/"native-verification-001/report.json";vr=json.loads(ev.read_text());assert vr["all_native_geometry_checks_pass"] and vr["all_native_interface_pairs_exact_offline"] and vr["inputs_unchanged"]
O.mkdir(exist_ok=False);T=B/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
composition=pc.compose(T,O/"assets",[(130,C,LP)],reference_surface_rows=(130,))
payload=O/"assets"/T.name;receipt=O/"resting-anatomy-receipt.json";bound=pc.bind_anatomy_receipt(B/"resting-anatomy-receipt.json",payload,receipt)
assert bound["mass_geometry_accounting"]==json.loads((B/"resting-anatomy-receipt.json").read_text())["mass_geometry_accounting"]
parent=pc._read_nhtiss4(T);child=pc._read_nhtiss4(payload);assert parent["surface_count"]==child["surface_count"]==156
for pr,cr in zip(parent["records"],child["records"],strict=True):
 if int(pr[6])!=130:assert pc._row_local_equal(pc._row_slices(parent,pr),pc._row_slices(child,cr))
(O/"composition-result.json").write_text(json.dumps(composition,indent=2,sort_keys=True)+"\n")
bounds={"changed_rows":[130],"all155_other_rows_exact":True,"mass_accounting_exact":True,"physical_routes_tendons_support_transforms_and_controller_unchanged":True,"reference_surface_scope":"Existing source-boundary reconstruction of left pronator quadratus plus bounded source-coordinate self/interface correction. All source maps and raw-parent lineage remain exact the micrometre-boundary reference. Two triangle rows sharing raw parent1474 are retriangulated by the existing position-preserving owner; the failed native parent capture with four self pairs at5599 is retained. Coordinates are unchanged by this precision repair. Its one pure visual proxy remains fixed; this is not measured enthesis geometry or active forearm function.","max_total_source_correction_m":c["max_source_correction_from_micrometre_boundary_m"],"max_latest_source_increment_m":c["max_latest_source_increment_m"],"source_relative_area_change":c["source_relative_area_change"],"source_relative_volume_change":c["source_relative_volume_change"],"component_count":6,"all_components_positive":True,"qualification_limit":"Source and25 sampled poses clear of self intersections; no new exact source-parent neighbor pair identities against all current865 or older859 targets. Existing external interfaces remain unresolved. No continuous self-intersection proof or completed anatomy claim.","operating_point":"Unchanged provisional2Nm/rad aggregate passive MTP, real flat-plane contact, coupled GPU physiology; no performance changes."}
(O/"source-resolution-bounds.json").write_text(json.dumps(bounds,indent=2,sort_keys=True)+"\n")
bp=B/"baseline/run-declaration.json";base=json.loads(bp.read_text());idx=base["argv"].index("numilab_human.resting_run");av=base["argv"][idx+1:]
for flag,value in [("--output",str(O/"baseline/native-run")),("--anatomy-receipt",str(receipt))]:av[av.index(flag)+1]=value
parser=argparse.ArgumentParser();rr.add_arguments(parser);args=parser.parse_args(av);native,assets=rr.command(args)
env=dict(x.split("=",1) for x in base["argv"][2:idx-2]);env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(args.output/"common-field-failure.json")
argv=["/usr/bin/env","-i"]+[k+"="+v for k,v in sorted(env.items())]+["/Users/n/numi-human-prep-venv-20261005/bin/python3.13","-m","numilab_human.resting_run"]+av
launch=O/"launch_arm.py";launch.write_text((B/"launch_arm.py").read_text())
paths=[Path(__file__),bp,launch,AP,LP,CP,SP,C,ev,Path(rr.__file__),Path(pc.__file__),O/"source-resolution-bounds.json",A/"try_pq_left_native_local_retriangulation_028.py",A/"audit_pq_left_retriangulated_native_targets_029.py",A/"audit_pq_left_retriangulated_all25_targets_030.py",A/"characterize_pq_left_retriangulated_source_032.py",A/"remaining-original-boundary-batch-001/stable-132/joint-target-separation-013/attempt-023/owner-source-snapshot-1321/common_atlas_skin_clearance.py"]
pins={**assets,**{str(p):sha(p) for p in paths}}
decl={"schema":"numi.human.native-smoke.run-declaration.v1","seconds":16.0,"accepted_steps":8000,"capture_steps":[0,4767,5023,5599,6207,6815,7423,8000],"immutable_assets":pins,"argv":argv,"launch_adapter":{"path":str(launch),"sha256":sha(launch)},"owner_cli_preview":{"native_argv":native,"asset_sha256":assets,"recorded_invocation_environment":rr.invocation_environment(env)},"qualification":"Integrated native left pronator quadratus130 inferred boundary diagnostic. Other155 tissue rows, physical mass/routes/tendons/support and GPU physiology unchanged. Source and25 prior poses self-clear with no new exact parent interfaces; existing interfaces remain unresolved. Current native geometry still requires verification. No five-minute/full-anatomy claim; performance deferred."}
(O/"baseline").mkdir();dp=O/"baseline/run-declaration.json";dp.write_text(json.dumps(decl,indent=2,sort_keys=True)+"\n")
(O/"launch-guard.json").write_text(json.dumps({"launcher_sha256":sha(launch),"arms":{"baseline":{"run_declaration_sha256":sha(dp),"native_output":str(args.output)}}},indent=2,sort_keys=True)+"\n")
print(json.dumps({"path":str(O),"tissue_sha256":sha(payload),"pins":len(pins)}))
