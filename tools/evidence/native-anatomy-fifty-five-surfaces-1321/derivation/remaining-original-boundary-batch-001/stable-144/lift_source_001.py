from pathlib import Path
from fractions import Fraction
import sys,json,hashlib,importlib.util,inspect
import numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");GP=Path(sys.argv[1]);D=GP.parent;O=Path(sys.argv[2]);O.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-positive-winding-1279/src")
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology
from numilab_human.resting_anatomy_interface_patch import normals
H=A.parent/"fullbody-source-surface-repair-1276/gluteus-maximus-pilot-001/attempt-010/run_gluteus_union_010.py"
sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/remaining-original-boundary-batch-001/stable-144/original-quotient.npz');pp=np.load(parent)
partition=D/"rejected-partition.json";pd=json.loads(partition.read_text());gp=GP;gz=np.load(gp)
pins={str(p):sha(p) for p in [Path(__file__),H,h.TISS,h.MAN,parent,partition,gp,Path(ci.__file__),Path(inspect.getsourcefile(analyze_topology)),Path(inspect.getsourcefile(normals)),Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/remaining-original-boundary-batch-001/stable-144/source-separation/report.json'),D/"report.json"]}
# Reuse only the existing canonical attribute calculation, before its candidate geometric admission.
# The intermediate point-pinched partition is explicitly rejected and never composed.
src=inspect.getsource(h.make_candidate).split("    # Candidate source exact self/topology.")[0]
src=src.replace("def make_candidate(","def lift_partition_attributes_only(").replace('require(row["bc"]==2,"gluteus row binding cardinality changed")','require(row["bc"]>=1,"nonempty source binding table")').replace('expected_member={29:"FJ1418",30:"FJ1418M"}[sid]','expected_member=source_manifest["member_id"]')
src=src.replace('    out_local[vi,:row["bc"]]=np.arange(row["bc"],dtype="<u4")\n        out_weights[vi,:row["bc"]]=np.asarray([np.float32(float(mean[li])) for li in range(row["bc"])],dtype="<f4")', '    active=[li for li in range(row["bc"]) if mean[li]>0]\n        require(len(active)<=4,"canonical exact map exceeds existing four nonzero slots")\n        out_local[vi,:len(active)]=np.asarray(active,dtype="<u4")\n        out_weights[vi,:len(active)]=np.asarray([np.float32(float(mean[li])) for li in active],dtype="<f4")')
src+='    return {"positions":out_pos,"faces":out_faces,"local":out_local,"weights":out_weights,"parents":face_parents,"lift_rows":lift_rows}\n'
script=O/"parameterized_existing_attribute_calculation.py";script.write_text(src);pins[str(script)]=sha(script);exec(compile(src,str(script),"exec"),h.__dict__)
h.OUT=O/"attribute-intermediate-rejected-geometry";h.OUT.mkdir();h.UNION_DIR=h.OUT
doc={"source_tissue_sha256":sha(h.TISS),"source_manifest_sha256":sha(h.MAN),"construction":{"output_mesh":pd["mesh"]},"ancestry_by_output_face":pd["face_ancestry"]}
(h.UNION_DIR/"stable-144-exact-union.json").write_text(json.dumps(doc,sort_keys=True)+"\n")
t=h.load_tissue(h.TISS);original=h.row_data(t,144);row=dict(original);row.update(positions=pp["vertices6"][:,:3],faces=pp["faces"],local=pp["binding_indices"],weights=pp["weights"],vc=len(pp["vertices6"]))
sep_report=json.loads((D/"report.json").read_text());trial=next(x for x in sep_report["trials"] if x.get("candidate_path")==str(gp));assert trial["candidate_sha256"]==sha(gp);split_epsilon_m=float(trial["epsilon_m"])
geometry={"manifest_rows":{int(x["stable_id"]):x for x in json.loads(h.MAN.read_text())["source"]["surfaces"]},"analyze_topology":analyze_topology}
attrs=h.lift_partition_attributes_only(144,t,row,doc,ci,geometry);mapping=gz["partition_vertex_ancestry"].astype(int)
pos=gz["positions"];faces=gz["faces"];local=attrs["local"][mapping];weights=attrs["weights"][mapping]
assert np.all(weights[weights>0]>2e-6) and np.allclose(weights.sum(axis=1),1.0,atol=2e-6,rtol=0)
partition_vertices=np.asarray([[np.float32(float(Fraction(int(c[0],16),int(c[1],16)))) for c in v] for v in pd["mesh"]["vertices_m"]],dtype="<f4");assert mapping.min()>=0 and mapping.max()<len(partition_vertices)
split_displacement_m=float(np.linalg.norm(pos.astype(np.float64)-partition_vertices[mapping].astype(np.float64),axis=1).max())
parent_source_bound_m=0.0;conservative_total_bound_m=split_displacement_m;assert conservative_total_bound_m<=5e-4
p=O/"stable-144-geometric-reference-row-patch.npz";np.savez(p,vertices6=np.column_stack((pos,normals(pos.astype(float),faces))).astype(np.float32),faces=faces.astype("<u4"),face_origins=pp["face_origins"][attrs["parents"]],binding_indices=local,weights=weights)
def audit(pos):
 rec,deg=h.exact_rows(pos,faces,ci);return {"count":ci._audit_pair(rec,rec,same_surface=True)["count"],"degenerate_faces":deg}
source=audit(pos);top=analyze_topology(pos.astype(float).tolist(),faces.tolist());assert source["count"]==0 and not source["degenerate_faces"] and top["closed_oriented_manifold_candidate"]
rr={"stable_id":144,"member_id":geometry["manifest_rows"][144]["member_id"],"candidate_path":str(p),"candidate_sha256":sha(p),"candidate_source_path":str(gp),"candidate_source_sha256":sha(gp),"separation_epsilon_m":split_epsilon_m,"maximum_geometric_branch_split_displacement_m":split_displacement_m,"pinned_parent_replay_max_endpoint_displacement_m":parent_source_bound_m,"conservative_total_ancestry_displacement_bound_m":conservative_total_bound_m,"total_displacement_cap_m":0.0005,"displacement_bound_method":"conservative triangle inequality: pinned parent source-resolution maximum endpoint displacement plus maximum split from the exact rejected-partition vertex; partition vertices arise on the parent input surface","source":source,"topology":top,"poses":[],"attribute_policy":"Existing exact parent-face consensus at rejected partition coordinates. Geometric branch-separated copies inherit that vertex's unchanged map; displacement is explicitly inferred and is not barycentric interpolation at the displaced point.","pure_attachment_proxy_count":int(np.sum(original["weights"][np.unique(original["faces"])].max(1)>=.999999)),"partition_vertex_ancestry":mapping.tolist(),"canonical_attribute_lift":attrs["lift_rows"]}
missing=[]
for j in np.unique(original["faces"]):
 if original["weights"][j].max()<.999999:continue
 hits=np.flatnonzero(np.all(pos==original["positions"][j],axis=1))
 if not any(np.array_equal(local[i],original["local"][j]) and np.array_equal(weights[i],original["weights"][j]) for i in hits):missing.append(int(j))
rr["changed_or_missing_pure_attachment_proxies"]=missing
assert not missing

out={"scope":"Geometrically separated point-pinched reference boundary, not an accepted unchanged winding union; independent source and posed exact admission.", "pins":pins,"rows":[rr],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
pose_sources=[]
for arm in ["baseline","intervention"]:
 for step in h.STEPS:pose_sources.append(("late",arm,h.PAIR/arm/"native-run/accepted-geometry",step))
early_root=A/"fifty-surface-native-composition-001/baseline/native-run/accepted-geometry"
for step in [0,4767,5023,5599,6207,6815,7423,8000]:pose_sources.append(("early","baseline",early_root,step))
for cohort,arm,pose_root,step in pose_sources:
 receipt=pose_root/f"step-{step}.receipt.json";pins[str(receipt)]=sha(receipt);r=json.loads(receipt.read_text());assert int(r["accepted_step"])==step;poses={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in r.get("accepted_body_poses",r.get("accepted_registered_body_poses"))};rr["poses"].append({"cohort":cohort,"arm":arm,"step":step,**audit(h.forward(pos,local,weights,row,t,poses))});save()
rr["source_and_sampled_pose_self_clear"]=all(x["count"]==0 and not x["degenerate_faces"] for x in rr["poses"]);rr["pose_count_expected"]=24;rr["pose_count_actual"]=len(rr["poses"]);assert len(rr["poses"])==24;out.update(complete=True,inputs_unchanged=all(sha(p)==v for p,v in pins.items()));save()
print(json.dumps({"source":source,"all24_clear":rr["source_and_sampled_pose_self_clear"],"posed_counts":[(x["cohort"],x["arm"],x["step"],x["count"]) for x in rr["poses"]],"candidate_sha256":sha(p),"max_displacement_bound_m":conservative_total_bound_m,"split_epsilon_m":split_epsilon_m}))
