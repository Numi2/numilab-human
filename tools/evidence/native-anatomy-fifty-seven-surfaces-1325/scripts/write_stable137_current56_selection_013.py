from pathlib import Path
import json,hashlib,importlib.util,sys,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");D=A/"remaining-original-boundary-batch-001/stable-137";O=D/"current56-selection-013";O.mkdir(exist_ok=False)
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
T=A/"fifty-six-surface-native-composition-002/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";M=T.with_suffix(".manifest.json");t=h.load_tissue(T);raw=h.row_data(t,137)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rp=D/"current56-target-audit-012/report.json";audit=json.loads(rp.read_text())
assert audit["complete"] and audit["inputs_unchanged"] and audit["source_and_all25_sampled_pose_self_clear"] and audit["all25_retained_targets_no_added_pairs"] and audit["all_baseline_forward_capture_replays_within_6e-7m"]
C=Path(audit["candidate_path"]);assert sha(C)==audit["candidate_sha256"];z=np.load(C);p=z["vertices6"][:,:3];f=z["faces"]
lr=D/"source-local-translation-002/report.json";loc=json.loads(lr.read_text());fr=D/"first-boundary-intermediate-001/report.json";first=json.loads(fr.read_text())
basez=np.load(Path(loc["candidate"]["path"]));assert all(np.array_equal(z[k],basez[k]) for k in ["faces","face_origins","binding_indices","weights"]);fixed=np.unique(f[np.isin(z["face_origins"],[2217,2311,2309,2371])]);assert np.array_equal(p[fixed],basez["vertices6"][:,:3][fixed])
partp=D/"source-separation/rejected-partition.json";part=json.loads(partp.read_text());exact=np.array([[int(c[0],0)/int(c[1],0) for c in v] for v in part["mesh"]["vertices_m"]]);ancestry=np.asarray(first["partition_vertex_ancestry"],int)
bound=float(np.linalg.norm(p.astype(float)-exact[ancestry],axis=1).max())
assert bound<1e-3
pure=np.flatnonzero(raw["weights"].max(1)>=.999999);missing=[]
for i in pure:
 matches=np.flatnonzero(np.all(p==raw["positions"][i],axis=1))
 if not any(np.array_equal(z["weights"][j],raw["weights"][i]) and np.array_equal(z["binding_indices"][j],raw["local"][i]) for j in matches):missing.append(int(i))
assert not missing
def av(v,faces):
 tri=v.astype(float)[faces];return float(.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()),float(np.einsum("ij,ij->i",tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)
sa,sv=av(raw["positions"],raw["faces"]);ca,cv=av(p,f)
m=next(r for r in json.loads(M.read_text())["source"]["surfaces"] if r["stable_id"]==137)
pins={}
stage_reports=[D/"posed-protected-bone-interface-010/report.json",D/"source-after-protected-bone-011/report.json"]
for doc in [audit,loc,first,json.loads((D/"source-separation/report.json").read_text()),*[json.loads(q.read_text()) for q in stage_reports]]:
 for path,digest in doc["pins"].items():
  assert sha(path)==digest,(path,digest);pins[path]=digest
for pp in [Path(__file__),rp,lr,fr,partp,C,H,T,M,D/"source-separation/report.json",*stage_reports]:pins[str(pp)]=sha(pp)
row=dict(stable_id=137,label=m["label"],member_id=m["member_id"],member_sha256=m["member_sha256"],candidate_path=str(C),candidate_sha256=sha(C),source_and_sampled_pose_self_clear=True,all25_actual_retained_target_contexts_no_new_pairs=True,changed_or_missing_pure_attachment_proxies=missing,pure_attachment_proxy_count=len(pure),source_area_m2=sa,candidate_area_m2=ca,source_signed_volume_m3=sv,candidate_signed_volume_m3=cv,maximum_displacement_from_union_reference_m=bound,cumulative_source_displacement_bound_m=bound,construction="Positive-winding source boundary with declared point-fan regularization, local source-face translation and two existing bounded self-separation corrections. Four named wrist-bone interface parent faces remain fixed. Original topology, face ancestry and canonical binding maps are preserved after source regularization. Bound measures displacement from the rational boundary; it is not source measurement accuracy.")
out=dict(scope="Offline source and25 retained-pose selection of right extensor digitorum, all859 external targets per context. No native admission yet.",complete=True,inputs_unchanged=all(sha(k)==v for k,v in pins.items()),pins=pins,all_simultaneously_repaired_pairs_pass=True,simultaneous_pair_count=0,rows=[row])
(O/"report.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(row))
