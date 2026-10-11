from pathlib import Path
import sys,json,hashlib,argparse,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");B=A/"left-pq-retriangulated-native-mtp-two-002";O=A/"right-pq-native-mtp-two-001";P=Path("/Users/n/numi-human-complete-reference-inventory-1335")
sys.path.insert(0,str(P/"src"))
from numilab_human import resting_run as rr,passive_attachment_composition as pc
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
LP=A/"pq-right-final-all25-target-audit-050/report.json";CP=A/"pq-right-final-source-characterization-051/report.json";SP=A/"pq-right-localized-legacy-binding-049/report.json"
for path in [LP,CP,SP]:
 d=json.loads(path.read_text());assert d["complete"] and d["inputs_unchanged"],path
 for p,v in d["pins"].items():assert sha(p)==v,p
l=json.loads(LP.read_text());c=json.loads(CP.read_text());s=json.loads(SP.read_text())
assert l["source_and_all25_sampled_pose_self_clear"] and l["all25_retained_targets_no_added_pairs"] and len(l["poses"])==25
assert c["all_components_positive"] and c["faces_and_raw_parent_lineage_exact_reference"] and c["onehot_proxy_coordinates_and_maps_exact_reference"] and c["map_support_and_indices_exact_reference"]
assert c["max_coordinate_increment_from_reference_m"]<.0005 and max(abs(c[k]) for k in ["source_relative_area_change","source_relative_volume_change"])<.001
assert {x["vertex_id"] for x in c["binding_changes"]}=={686,881,886,894}
C=Path(l["candidate_path"]);assert sha(C)==l["candidate_sha256"]==s["accepted_updates"][-1]["candidate_sha256"]
ev=B/"native-verification-001/report.json";vr=json.loads(ev.read_text());assert vr["all_native_geometry_checks_pass"] and vr["all_native_interface_pairs_exact_offline"] and vr["inputs_unchanged"]
O.mkdir(exist_ok=False);T=B/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
composition=pc.compose(T,O/"assets",[(129,C,LP)],reference_surface_rows=(129,))
payload=O/"assets"/T.name;receipt=O/"resting-anatomy-receipt.json";bound=pc.bind_anatomy_receipt(B/"resting-anatomy-receipt.json",payload,receipt)
assert bound["mass_geometry_accounting"]==json.loads((B/"resting-anatomy-receipt.json").read_text())["mass_geometry_accounting"]
parent=pc._read_nhtiss4(T);child=pc._read_nhtiss4(payload);assert parent["surface_count"]==child["surface_count"]==156
for pr,cr in zip(parent["records"],child["records"],strict=True):
 if int(pr[6])!=129:assert pc._row_local_equal(pc._row_slices(parent,pr),pc._row_slices(child,cr))
(O/"composition-result.json").write_text(json.dumps(composition,indent=2,sort_keys=True)+"\n")
bounds={"changed_rows":[129],"all155_other_rows_exact":True,"mass_accounting_exact":True,"physical_routes_tendons_support_transforms_and_controller_unchanged":True,"reference_surface_scope":"Existing source-boundary reconstruction of right pronator quadratus followed by bounded source-coordinate correction and four explicitly inferred mixed visual binding weights. Source topology, raw-parent ancestry, active body support and pure visual attachment proxy remain fixed. These are not measured entheses or active forearm function.","max_total_source_correction_m":c["max_coordinate_increment_from_reference_m"],"binding_changes":c["binding_changes"],"source_relative_area_change":c["source_relative_area_change"],"source_relative_volume_change":c["source_relative_volume_change"],"component_count":6,"all_components_positive":True,"qualification_limit":"Source and25 sampled poses clear of self intersections; no new exact source-parent neighbor pair identities against all current865 or older859 targets. Existing external interfaces remain unresolved. No continuous self-intersection proof or completed anatomy claim.","operating_point":"Unchanged provisional2Nm/rad aggregate passive MTP, real flat-plane contact, coupled GPU physiology; no performance changes."}
(O/"source-resolution-bounds.json").write_text(json.dumps(bounds,indent=2,sort_keys=True)+"\n")
bp=B/"baseline/run-declaration.json";base=json.loads(bp.read_text());idx=base["argv"].index("numilab_human.resting_run");av=base["argv"][idx+1:]
for flag,value in [("--output",str(O/"baseline/native-run")),("--anatomy-receipt",str(receipt))]:av[av.index(flag)+1]=value
parser=argparse.ArgumentParser();rr.add_arguments(parser);args=parser.parse_args(av);native,assets=rr.command(args)
env=dict(x.split("=",1) for x in base["argv"][2:idx-2]);env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(args.output/"common-field-failure.json")
argv=["/usr/bin/env","-i"]+[k+"="+v for k,v in sorted(env.items())]+["/Users/n/numi-human-prep-venv-20261005/bin/python3.13","-m","numilab_human.resting_run"]+av
launch=O/"launch_arm.py";launch.write_text((B/"launch_arm.py").read_text())
paths=[Path(__file__),bp,launch,LP,CP,SP,C,ev,Path(rr.__file__),Path(pc.__file__),O/"source-resolution-bounds.json",A/"try_pq_right_localized_legacy_binding_049.py",A/"audit_pq_right_final_all25_targets_050.py",A/"characterize_pq_right_final_source_051.py",A/"pq-right-legacy-binding-sensitivity-047/stable-129-unadmitted-v881-delta--5e-05.npz",A/"pq-right-fine-binding-sensitivity-041/stable-129-unadmitted-weight--0.0004.npz",A/"remaining-original-boundary-batch-001/stable-129/post-correction-lift-006/stable-129-geometric-reference-row-patch.npz"]
pins={**assets,**{str(p):sha(p) for p in paths}}
decl={"schema":"numi.human.native-smoke.run-declaration.v1","seconds":16.0,"accepted_steps":8000,"capture_steps":[0,4767,5023,5599,6207,6815,7423,8000],"immutable_assets":pins,"argv":argv,"launch_adapter":{"path":str(launch),"sha256":sha(launch)},"owner_cli_preview":{"native_argv":native,"asset_sha256":assets,"recorded_invocation_environment":rr.invocation_environment(env)},"qualification":"Integrated native right pronator quadratus129 inferred boundary diagnostic. Other155 tissue rows, physical mass/routes/tendons/support and GPU physiology unchanged. Source and25 prior poses self-clear with no new exact parent interfaces; existing interfaces remain unresolved. Current native geometry still requires verification. No five-minute/full-anatomy claim; performance deferred."}
(O/"baseline").mkdir();dp=O/"baseline/run-declaration.json";dp.write_text(json.dumps(decl,indent=2,sort_keys=True)+"\n")
(O/"launch-guard.json").write_text(json.dumps({"launcher_sha256":sha(launch),"arms":{"baseline":{"run_declaration_sha256":sha(dp),"native_output":str(args.output)}}},indent=2,sort_keys=True)+"\n")
print(json.dumps({"path":str(O),"tissue_sha256":sha(payload),"pins":len(pins)}))
