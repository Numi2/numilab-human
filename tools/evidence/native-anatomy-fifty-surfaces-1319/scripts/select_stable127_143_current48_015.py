from pathlib import Path
from fractions import Fraction
import json,numpy as np,importlib.util,hashlib
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
O=A/"stable-127-143-current48-selection-015";O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
T=A/"forty-eight-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";M=T.with_suffix(".manifest.json");t=h.load_tissue(T);man=json.loads(M.read_text())
pins={str(p):sha(p) for p in [Path(__file__),H,T,M]}
def absorb(path):
 pins[str(path)]=sha(path);doc=json.loads(path.read_text())
 assert doc.get("complete",True) and doc.get("inputs_unchanged",True)
 for p,d in doc.get("pins",{}).items():
  assert sha(p)==d,("changed",p)
  assert p not in pins or pins[p]==d
  pins[p]=d
 return doc
def av(p,f):
 tri=p[f].astype(float)
 return float(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2),float(np.einsum("ij,ij->i",tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)
out={"schema":"numi.human.reference-surface-selection.v1","scope":"Two passive source-derived anatomical surface repairs. Exact individual topology/self and all859 actual neighboring surfaces audited in25 retained contexts; simultaneous candidate interface tested separately here. Engineering reconstruction bounds are not source measurement accuracy or measured-person anatomy. No native or whole-body admission.","pins":pins,"rows":[],"complete":False,"candidate_admitted":False,"native_admitted":False}
candidates={}
for sid,ad,ld in [(127,"stable-127-current48-independent-010","stable-127-original-split-lift-009"),(143,"stable-143-current48-independent-013","epl-geometric-point-lift-003-050um")]:
 audit=absorb(A/ad/"report.json");lift=absorb(A/ld/"report.json")["rows"][0]
 assert audit["all25_retained_targets_no_added_pairs"] and audit["no_added_parent_pairs_against_current48_targets"] and audit["source_and_all25_sampled_pose_self_clear"]
 assert audit["all_baseline_forward_capture_replays_within_6e-7m"] and len(audit["all25_retained_target_audits"])==25
 cp=Path(audit["candidate_path"]);assert sha(cp)==audit["candidate_sha256"];pins[str(cp)]=sha(cp)
 z=np.load(cp);raw=h.row_data(t,sid);candidates[sid]=(z,raw)
 rowman=next(x for x in man["source"]["surfaces"] if x["stable_id"]==sid)
 used=np.unique(raw["faces"]);pure=used[raw["weights"][used].max(1)>=.999999];missing=[]
 for j in pure:
  hits=np.flatnonzero(np.all(z["vertices6"][:,:3]==raw["positions"][j],axis=1))
  if not any(np.array_equal(z["binding_indices"][i],raw["local"][j]) and np.array_equal(z["weights"][i],raw["weights"][j]) for i in hits):missing.append(int(j))
 assert not missing
 if sid==127:
  sep=absorb(A/"stable-127-original-point-separation-008/report.json")
  pdpath=A/"stable-127-original-point-separation-008/rejected-partition.json"
  ancestry=absorb(A/"stable-127-original-quotient-parent-004/report.json")
  parent=np.load(ancestry["candidate_path"]);pins[ancestry["candidate_path"]]=sha(ancestry["candidate_path"])
  parent_bounds=np.zeros(len(parent["vertices6"]))
  assert set(map(tuple,parent["vertices6"][:,:3])).issubset(set(map(tuple,raw["positions"])))
  construction="Original coordinate quotient; positive-winding boundary with isolated multi-fan point contacts separated by1micrometre, followed by unchanged canonical ancestry binding lift. No midpoint source-edge collapse retained."
  for p in [A/"separate_stable127_original_pinches_008.py",A/"lift_stable127_original_split_009.py",A/"audit_stable127_original_split_010.py"]:pins[str(p)]=sha(p)
 else:
  pr=absorb(A/"bounded-halfmm-source-resolution-001/report.json")
  ops=[x for x in next(x for x in pr["rows"] if x["stable_id"]==143)["attempts"] if x.get("accepted_for_next_offline_iteration")]
  pp=A/"bounded-halfmm-source-resolution-001/stable-143-reference-resolution-row-patch.npz";pins[str(pp)]=sha(pp);parent=np.load(pp)
  pdpath=A/"epl-geometric-point-separation-003/rejected-partition.json"
  parent_bounds=np.zeros(len(parent["vertices6"]));original=set(map(tuple,raw["positions"]))
  for i,p in enumerate(parent["vertices6"][:,:3]):
   if tuple(p) in original:continue
   hit=[x for x in ops if np.array_equal(p,x["collapse_report"]["collapse_position_f32_m"])]
   assert len(hit)==1;parent_bounds[i]=hit[0]["cumulative_ancestry_displacement_m"]
  for path in [A/"epl-geometric-point-separation-003/report.json",A/"epl-local-self-separation-008/report.json",A/"epl-active-external-clearance-005/constraint-report.json",A/"stable-143-parent-gap-correction-012/report.json"]:
   # Prior proposals remain unadmitted; pin their full derivation without interpreting a diagnostic success as anatomy qualification.
   pins[str(path)]=sha(path)
  for p in [A/"correct_stable143_parent_gaps_012.py",A/"audit_stable143_corrected_013.py"]:pins[str(p)]=sha(p)
  construction="Two bounded source-edge collapses, positive-winding boundary with geometric point-fan separation, canonical binding lift, source/posed fold separation and neighbor clearance correction. Final per-vertex ancestry bounds account for the actual final position, not the sum of mutually reversing trial displacements."
 pins[str(pdpath)]=sha(pdpath);pd=json.loads(pdpath.read_text())
 pv=np.asarray([[float(Fraction(int(c[0],16),int(c[1],16))) for c in v] for v in pd["mesh"]["vertices_m"]])
 mapping=np.asarray(lift["partition_vertex_ancestry"],int);attributes=lift["canonical_attribute_lift"]
 assert len(attributes)==len(pv) and [x["candidate_vertex"] for x in attributes]==list(range(len(pv)))
 local_bounds=[]
 for a in attributes:
  if a["kind"]=="source_coordinate_preserved":b=parent_bounds[a["source_representative"]]
  else:b=parent_bounds[np.unique(parent["faces"][a["parent_faces"]])].max()
  local_bounds.append(b)
 direct=np.linalg.norm(z["vertices6"][:,:3].astype(float)-pv[mapping],axis=1)
 total=np.asarray(local_bounds)[mapping]+direct+1e-12
 # Each child is on a parent triangle: its endpoint ancestry bound is bounded by
 # the largest incident parent support bound. Final Euclidean displacement from
 # that exact partition point includes both branch split and subsequent edits.
 assert float(total.max())<=.0005
 np.savez(O/f"stable-{sid}-per-vertex-source-bounds.npz",partition_vertex=mapping,parent_support_bound_m=np.asarray(local_bounds)[mapping],final_partition_displacement_m=direct,total_bound_m=total)
 bp=O/f"stable-{sid}-per-vertex-source-bounds.npz";pins[str(bp)]=sha(bp)
 a,v=av(raw["positions"],raw["faces"]);na,nv=av(z["vertices6"][:,:3],z["faces"])
 assert abs(nv-v)<=abs(v)*.001,("reference volume change exceeds declared0.1percent engineering sensitivity",sid,v,nv)
 out["rows"].append({"stable_id":sid,"label":rowman["label"],"member_id":rowman["member_id"],"member_sha256":rowman["member_sha256"],"candidate_path":str(cp),"candidate_sha256":sha(cp),"source_and_sampled_pose_self_clear":True,"changed_or_missing_pure_attachment_proxies":missing,"pure_attachment_proxy_count":len(pure),"pure_attachment_proxy_meaning":"Exact onehot proxies only; zero does not certify anatomical attachment footprint accuracy.","source_area_m2":a,"candidate_area_m2":na,"source_signed_volume_m3":v,"candidate_signed_volume_m3":nv,"volume_relative_change":(nv-v)/v,"maximum_displacement_from_union_reference_m":float(direct.max()),"cumulative_source_displacement_bound_m":float(total.max()),"construction":construction,"bound_interpretation":"Per-child maximum incident parent endpoint ancestry bound plus actual final Euclidean displacement from its exact rational partition coordinate and1picometre arithmetic reserve. Piecewise convex parent support; not source measurement accuracy.","per_vertex_bound_path":str(bp),"all25_actual_retained_target_contexts_no_new_pairs":True,"current48_capture_pose_audits":audit["current48_target_audits"],"physical_route_force_mass_state_unchanged":True})
# Direct simultaneous old-old/new-new exact interface audit in all25 actual pose maps.
import sys
sys.path.insert(0,"/Users/n/numi-human-anatomy-completion-1276/src")
from numilab_human import cardiac_cavity_intersections as ci
pins[ci.__file__]=sha(ci.__file__)
sim=[]
receipts=[Path(p) for p in pins if p.endswith(".receipt.json") and "/accepted-geometry/" in p]
# Selection includes older lift receipts. Only audit the25 contexts from independent current48 audit pins.
audit=json.loads((A/"stable-127-current48-independent-010/report.json").read_text())
receipts=[Path(p) for p in audit["pins"] if p.endswith(".receipt.json") and "/accepted-geometry/" in p]
assert len(receipts)==25
for rp in receipts:
 r=json.loads(rp.read_text());pm={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in r.get("accepted_body_poses",r.get("accepted_registered_body_poses"))}
 records={};origins={}
 for sid,(z,raw) in candidates.items():
  world=h.forward(z["vertices6"][:,:3],z["binding_indices"],z["weights"],raw,t,pm);rec,deg=h.exact_rows(world,z["faces"],ci);assert not deg;records[sid]=rec;origins[sid]=z["face_origins"]
 ar=ci._audit_pair(records[127],records[143],same_surface=False)
 old=[]
 for sid in (127,143):
  raw=candidates[sid][1];world=h.forward(raw["positions"],raw["local"],raw["weights"],raw,t,pm);rec,deg=h.exact_rows(world,raw["faces"],ci);assert not deg;old.append(rec)
 br=ci._audit_pair(*old,same_surface=False);before=set(map(tuple,br["triangle_pairs"]));after={(int(origins[127][a]),int(origins[143][b])) for a,b in ar["triangle_pairs"]}
 added=sorted(after-before);assert not added,(rp,added)
 sim.append({"receipt":str(rp),"baseline_pairs":len(before),"candidate_pairs":len(after),"added_parent_pairs":added})
out.update(simultaneous_pair_audits=sim,all_simultaneously_repaired_pairs_pass=True,complete=True,inputs_unchanged=all(sha(p)==v for p,v in pins.items()))
assert out["inputs_unchanged"]
(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report_sha256":sha(O/"report.json"),"rows":[{k:r[k] for k in ("stable_id","candidate_sha256","cumulative_source_displacement_bound_m","volume_relative_change")} for r in out["rows"]],"simultaneous_contexts":len(sim),"pins":len(pins)}))
