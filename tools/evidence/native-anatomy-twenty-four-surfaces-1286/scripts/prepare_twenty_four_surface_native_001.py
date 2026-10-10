from pathlib import Path
import sys,json,hashlib,argparse,numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276"
O=A/"twenty-four-surface-native-composition-001";O.mkdir(exist_ok=False)
SRC=Path("/Users/n/numi-human-positive-winding-1279/src");sys.path.insert(0,str(SRC))
from numilab_human import passive_attachment_composition as pc,resting_run as rr
from numilab_human.resting_anatomy_interface_patch import normals,signed_volume
from numilab_human.cardiac_cavity_geometry import analyze_topology
B=A/"twenty-two-surface-native-composition-002"
T=B/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
P=R/"skin-resting-multipose-clearance-1218/native-coupled-pair-preparation-011"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=pc._read_nhtiss4(T);replacements=[];bounds={};proofpaths=[];union_inputs=[]
def canon(ix,w,bc):
 out=np.zeros(bc,float)
 for i,x in zip(ix,w):
  if x>0:out[int(i)]+=float(x)
 return out
for sid,attempt,external in [(67,3,1),(68,4,2)]:
 rp=A/f"external-oblique-local-reference-{attempt:03}/report.json"
 ip=A/f"external-oblique-local-interface-audit-{external:03}/report.json"
 report=json.loads(rp.read_text());inter=json.loads(ip.read_text())
 for doc in (report,inter):
  assert doc["complete"] and doc["inputs_unchanged"]
  for p,digest in doc["pins"].items():assert sha(p)==digest
  union_inputs.extend(Path(p) for p in doc["pins"])
 assert all(x["new_triangle_pair_count"]==0 and x["external_targets_scanned"]==859 for x in inter["poses"])
 row=report["rows"][0];assert row["stable_id"]==sid and row["source_quotient_closed"]
 trial=next(x for x in row["trials"] if x["alpha"]==.1)
 assert trial["source"]["count"]==0 and len(trial["poses"])==2 and all(x["count"]==0 for x in trial["poses"])
 npz=Path(trial["candidate_path"]);assert sha(npz)==trial["candidate_sha256"]
 z=np.load(npz)
 raw=pc._biceps_row_arrays(pc._row_slices(d,next(r for r in d["records"] if int(r[6])==sid)))
 u,first,iv=np.unique(raw["vertices6"][:,:3],axis=0,return_index=True,return_inverse=True)
 used=np.unique(iv[raw["faces"]]);mapping=np.full(len(u),-1);mapping[used]=np.arange(len(used))
 f=mapping[iv[raw["faces"]]];p=u[used]
 assert np.array_equal(f,z["faces"]) and np.array_equal(z["face_origins"],np.arange(len(f)))
 idx=raw["binding_indices"][first[used]];w=raw["weights"][first[used]]
 assert np.array_equal(idx,z["binding_indices"]) and np.array_equal(w,z["weights"])
 bc=int(max(idx[w>0]))+1
 for j in np.unique(raw["faces"]):
  assert np.array_equal(canon(raw["binding_indices"][j],raw["weights"][j],bc),canon(idx[mapping[iv[j]]],w[mapping[iv[j]]],bc))
 pairs=row["source"]["pairs"]+sum((b["pairs"] for b in row["baseline_poses"].values()),[])
 seeds=np.unique(f[np.unique(np.asarray(pairs,int))])
 neighbors={int(v):set() for v in seeds}
 for tri in f:
  for v in tri:
   if int(v) in neighbors:neighbors[int(v)].update(int(x) for x in tri if x!=v)
 expected=p.copy()
 for v in seeds:expected[v]=(p[v].astype(float)+.1*(p[sorted(neighbors[int(v)])].astype(float).mean(0)-p[v])).astype(np.float32)
 assert np.array_equal(expected,z["vertices6"][:,:3])
 assert np.array_equal(normals(expected.astype(float),f).astype(np.float32),z["vertices6"][:,3:])
 displacement=np.linalg.norm(expected.astype(float)-p.astype(float),axis=1)
 maximum=float(displacement.max());assert maximum<.00006
 assert not np.any((displacement>0)&(w.max(1)>=.999999))
 topo=analyze_topology(expected.tolist(),f.tolist());assert topo["closed_oriented_manifold_candidate"]
 before=signed_volume(p.astype(float),f);after=signed_volume(expected.astype(float),f)
 assert before>0 and after>0 and abs(after-before)<abs(before)*.001
 bounds[sid]={"operation":"Canonical exact-duplicate quotient and explicit local reference offset on seven vertices; unchanged source face order, deformation weights and physical routes","maximum_original_source_displacement_m":maximum,"changed_vertex_count":int(np.count_nonzero(displacement)),"pure_attachment_proxy_changed":False,"source_volume_before_m3":before,"source_volume_after_m3":after,"source_volume_relative_change":(after-before)/before,"closed_oriented_manifold_candidate":True,"native_verification_pending":True}
 replacements.append((sid,npz,rp));proofpaths.extend([rp,ip])
payload=O/"assets"/T.name
composition=pc.compose(T,payload.parent,replacements,reference_surface_rows=(67,68))
receipt=O/"resting-anatomy-receipt.json";pc.bind_anatomy_receipt(B/"resting-anatomy-receipt.json",payload,receipt)
(O/"composition-result.json").write_text(json.dumps(composition,indent=2,sort_keys=True)+"\n")
(O/"source-resolution-bounds.json").write_text(json.dumps(bounds,indent=2,sort_keys=True)+"\n")
base=json.loads((P/"baseline/run-declaration.json").read_text());argsv=base["argv"][base["argv"].index("human-resting")+1:]
for flag,value in [("--anatomy-receipt",str(receipt)),("--output",str(O/"baseline/native-run")),("--seconds","16.0")]:argsv[argsv.index(flag)+1]=value
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
launch=O/"launch_arm.py";launch.write_bytes((A/"ecu-apl-native-composition-005/launch_arm.py").read_bytes())
pins={**assets,**{str(p):sha(p) for p in [Path(__file__),Path(pc.__file__),Path(rr.__file__),launch,O/"source-resolution-bounds.json",*proofpaths,*union_inputs]}}
for _,npz,rp in replacements:pins[str(npz)]=sha(npz);pins[str(rp)]=sha(rp)
decl={"schema":"numi.human.native-smoke.run-declaration.v1","seconds":16.0,"accepted_steps":8000,"capture_steps":[0,4767,5023,5599,6207,6815,7423,8000],"immutable_assets":pins,"argv":argv,"launch_adapter":{"path":str(launch),"sha256":sha(launch)},"owner_cli_preview":{"native_argv":native,"asset_sha256":assets,"recorded_invocation_environment":rr.invocation_environment(env)},"qualification":"Existing qualified twenty-two-surface resting human plus two bounded external-oblique reference corrections. Source face identity and all binding weights preserved, original source offsets below 0.06 mm. Both source and predicted endpoint self-intersections clear, no new external pairs versus all859 neighbouring targets at endpoints. Native full-cycle checks pending. Short anatomy verification only, not whole-body or five-minute qualification."}
(O/"baseline").mkdir();dp=O/"baseline/run-declaration.json";dp.write_text(json.dumps(decl,indent=2,sort_keys=True)+"\n")
(O/"launch-guard.json").write_text(json.dumps({"launcher_sha256":sha(launch),"arms":{"baseline":{"run_declaration_sha256":sha(dp),"native_output":str(args.output)}}},indent=2,sort_keys=True)+"\n")
print(json.dumps({"output":str(O),"payload_sha256":sha(payload),"pins":len(pins),"bounds":bounds},indent=2))

