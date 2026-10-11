from pathlib import Path
import sys,json,hashlib,struct,argparse,copy,subprocess,numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276";B=A/"left-deep-masseter-native-001";O=A/"right-deep-masseter-native-003";O.mkdir(exist_ok=False)
P=Path("/Users/n/numi-human-anatomy-provenance-publication-1332")
D=R/"head-muscle-coverage-1291/right-deep-161-z-candidate-audit-001/shell-subtraction-001/replay-recovery-002";C=D/"output-001/result-001/source-lift-f32.npz";RP=C.parent/"report.json";EP=D/"volume-erratum-001/result-001/report.json"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(C)=="ce5a229e715aec9a6ee0cc6b6f143a77967366c8c9fe91afd09fd806345c0345"
assert sha(RP)=="e1a8fe0068741f52dd1528409a4528157d5812f05185a5417d3ec5b012831fdf"
assert sha(EP)=="43516505a755c5b114fef905305ae801d6511eaca9868afd3eaf6e8e6eeb6405"
proof=json.loads(RP.read_text());assert proof["inputs_unchanged"] and proof["all_pose_self_and_topology_pass"] and proof["target_coverage"]["all_targets_clear"]
assert proof["target_coverage"]["non_bed_target_count_per_pose"]==865
for p,d in proof["inputs_sha256"].items():assert sha(p)==d,p
sys.path.insert(0,str(P/"src"));from numilab_human import passive_attachment_composition as pc,resting_run as rr
original=B/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";parent=pc._read_nhtiss4(original);assert parent["surface_count"]==155
pm=json.loads(original.with_suffix(".manifest.json").read_text())
fix=P/"tools/evidence/anatomy-passive-provenance-1332/left-deep-masseter.corrected.manifest.json";correct=json.loads(fix.read_text())["source"]["surfaces"][0]
old=next(r for r in pm["source"]["surfaces"] if r["stable_id"]==162);assert old["reference_surface_reconstruction"]["stable_id"]==160
assert {k:v for k,v in old.items() if k!="reference_surface_reconstruction"}=={k:v for k,v in correct.items() if k!="reference_surface_reconstruction"}
# Replay the exact existing row through the existing reference composition owner.
# This corrects provenance while keeping the complete payload byte-identical.
same_row=next(r for r in parent["records"] if int(r[6])==162)
same_fields=pc._biceps_row_arrays(pc._row_slices(parent,same_row))
patch=O/"left162-byte-identical-row.npz";np.savez(patch,**same_fields)
erratum=P/"tools/evidence/anatomy-passive-provenance-1332/report.json"
pc.compose(original,O/"parent-input",[(162,patch,erratum)],reference_surface_rows=(162,))
T=O/"parent-input"/original.name;assert sha(T)==sha(original)
parent_receipt=O/"parent-input/resting-anatomy-receipt.json"
pc.bind_anatomy_receipt(B/"resting-anatomy-receipt.json",T,parent_receipt)
pm=json.loads(T.with_suffix(".manifest.json").read_text())
assert next(r for r in pm["source"]["surfaces"] if r["stable_id"]==162)["reference_surface_reconstruction"]["stable_id"]==162
row=next(r for r in parent["records"] if int(r[6])==159);local=pc._row_slices(parent,row);assert local["binding_count"]==1 and struct.unpack_from("<I",local["binding_bytes"])[0]==23
z=np.load(C,allow_pickle=False);p=np.asarray(z["positions_m"],dtype="<f4");f=np.asarray(z["faces"],dtype="<u4")
assert np.all(z["weights"][:,0]==1) and np.all(z["weights"][:,1:]==0) and np.all(z["binding_indices"][:,0]==0)
tri=p.astype(float)[f];acc=np.zeros_like(p,dtype=float);n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
for j in range(3):np.add.at(acc,f[:,j],n)
length=np.linalg.norm(acc,axis=1);assert np.all(length>1e-20);norm=np.asarray(acc/length[:,None],dtype="<f4")
assert np.sum(np.einsum("ij,ij->i",tri[:,0],np.cross(tri[:,1],tri[:,2])))/6>0
vb=b"".join(struct.pack("<6f4I4f",*p[i],*norm[i],0,0xffffffff,0xffffffff,0xffffffff,1,0,0,0) for i in range(len(p)))
raw=struct.pack("<8s6I32s",b"NHTISS4\0",5,1,1,len(p),f.size,parent["fingerprint"],parent["source_digest"])+struct.pack("<8I",0,1,0,len(p),0,f.size,161,1)+local["binding_bytes"]+vb+f.tobytes()
sub=O/"subset";sub.mkdir();payload=sub/"right-deep-masseter.nhtissue";payload.write_bytes(raw)
meta=copy.deepcopy(next(r for r in pm["source"]["surfaces"] if r["stable_id"]==159))
meta.update(stable_id=161,member_id="FMA49004",source_name="right deep masseter",vertex_count=len(p),triangle_count=len(f))
meta.pop("source_geometry_bounds",None);meta.pop("reference_surface_reconstruction",None)
ext=meta["external_source"]
for k in list(ext):
 if k.startswith("harmonic_source_") or k=="identity_mapping":ext.pop(k)
ext.update(source_object_name="Deep part of masseter.r",source_member_id="FMA49004",source_correction="Registered reference boundary reconciliation against temporal bone30, mandible35 and right superficial masseter159 with inferred10 micrometre reserve. Four source vertices receive bounded Float32 replay correction, max0.164082 micrometre. Not measured anatomy or certified entheses.",clearance_report_path=str(RP),clearance_report_sha256=sha(RP),candidate_npz_path=str(C),candidate_npz_sha256=sha(C),volume_erratum_path=str(EP),volume_erratum_sha256=sha(EP))
meta["reference_surface_reconstruction"]={"stable_id":161,"candidate":str(C),"candidate_sha256":sha(C),"report":str(RP),"report_sha256":sha(RP),"vertex_count_after":len(p),"triangle_count_after":len(f),"scope":"Source-bound registered right deep masseter, passive head support only."}
meta["source_geometry_bounds"]={"face_ancestry_counts":proof["face_ancestry_counts"],"source_signed_volume_m3":proof["source_signed_volume_m3"],"source_lift_volume_erratum":str(EP),"source_lift_volume_erratum_sha256":sha(EP),"generic_attachment_markers_are_proxies_not_certified_footprints":True}
sm={"schema":pm["schema"],"payload":{"file":payload.name,"sha256":sha(payload),"bytes":len(raw),"surface_count":1,"binding_count":1,"vertex_count":len(p),"index_count":f.size,"registration_fingerprint32":f'{parent["fingerprint"]:08x}'},"source":{k:copy.deepcopy(pm["source"][k]) for k in ("bodyparts","myosim_source_archive_sha256","registration","myosim_manifest")},"coverage":{"configured_surface_count":1,"emitted_surface_count":1,"muscle_surface_count":1,"tendon_surface_count":0},"runtime_binding":"Existing single head core23 passive support; no active jaw mechanics.","evidence_boundary":"Registered mixed-source reference anatomy, not a measured individual."}
sm["source"]["surfaces"]=[meta];payload.with_suffix(".manifest.json").write_text(json.dumps(sm,indent=2,sort_keys=True)+"\n")
composition=pc.append_passive_surfaces(T,payload,O/"assets",stable_ids=(161,));child=O/"assets"/T.name
receipt=O/"resting-anatomy-receipt.json";bound=pc.bind_anatomy_receipt(parent_receipt,child,receipt);assert bound["mass_geometry_accounting"]==json.loads((B/receipt.name).read_text())["mass_geometry_accounting"]
(O/"composition-result.json").write_text(json.dumps(composition,indent=2,sort_keys=True)+"\n")
(O/"source-resolution-bounds.json").write_text(json.dumps({"161":meta,"all155_parent_payload_rows_exact":True,"physical_mass_unchanged":True,"parent162_provenance_corrected_only":True,"original_parent_payload_sha256":sha(original),"corrected_parent_payload_sha256":sha(T),"original_parent_manifest_sha256":sha(original.with_suffix(".manifest.json")),"corrected_parent_manifest_sha256":sha(T.with_suffix(".manifest.json"))},indent=2,sort_keys=True)+"\n")
bp=B/"baseline/run-declaration.json";base=json.loads(bp.read_text());idx=base["argv"].index("numilab_human.resting_run");av=base["argv"][idx+1:]
for flag,val in (("--output",str(O/"baseline/native-run")),("--anatomy-receipt",str(receipt))):av[av.index(flag)+1]=val
parser=argparse.ArgumentParser();rr.add_arguments(parser);args=parser.parse_args(av);native,assets=rr.command(args)
env=dict(x.split("=",1) for x in base["argv"][2:idx-2]);env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(args.output/"common-field-failure.json")
argv=["/usr/bin/env","-i"]+[k+"="+v for k,v in sorted(env.items())]+["/Users/n/numi-human-prep-venv-20261005/bin/python3.13","-m","numilab_human.resting_run"]+av
launch=O/"launch_arm.py";launch.write_text((B/"launch_arm.py").read_text())
pins={**assets,**proof["inputs_sha256"],**{str(x):sha(x) for x in (Path(__file__),Path(pc.__file__),Path(rr.__file__),bp,launch,C,RP,EP,fix,patch,erratum,parent_receipt,payload,payload.with_suffix(".manifest.json"),O/"source-resolution-bounds.json",original.with_suffix(".manifest.json"),T,T.with_suffix(".manifest.json"))}}
decl={"schema":"numi.human.native-smoke.run-declaration.v1","seconds":16.0,"accepted_steps":8000,"capture_steps":[0,4767,5023,5599,6207,6815,7423,8000],"immutable_assets":pins,"argv":argv,"launch_adapter":{"path":str(launch),"sha256":sha(launch)},"owner_cli_preview":{"native_argv":native,"asset_sha256":assets,"recorded_invocation_environment":rr.invocation_environment(env)},"qualification":"Right deep masseter161 added to155-row MTP0 scene. Left162 copied reconstruction provenance corrected without changing payload. All prior surfaces/mass/routes/physics unchanged. Passive head reference, native geometry validation pending. Whole-body completion and final five-minute qualification outstanding. Performance deferred."}
(O/"baseline").mkdir();dp=O/"baseline/run-declaration.json";dp.write_text(json.dumps(decl,indent=2,sort_keys=True)+"\n")
(O/"launch-guard.json").write_text(json.dumps({"launcher_sha256":sha(launch),"arms":{"baseline":{"run_declaration_sha256":sha(dp),"native_output":str(args.output)}}},indent=2,sort_keys=True)+"\n")
print(json.dumps({"output":str(O),"tissue_sha256":sha(child),"rows":156,"canonical_id":161,"pins":len(pins)}))
