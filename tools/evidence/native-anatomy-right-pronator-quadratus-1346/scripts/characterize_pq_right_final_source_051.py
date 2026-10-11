from pathlib import Path
import json,sys,importlib.util,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"pq-right-final-source-characterization-051";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-anatomy-publication-1308/src")
from numilab_human import cardiac_cavity_geometry as cg
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
C=A/"pq-right-localized-legacy-binding-049/stable-129-unadmitted-v894-delta-0.0002.npz";B=A/"remaining-original-boundary-batch-001/stable-129/post-correction-lift-006/stable-129-geometric-reference-row-patch.npz";P=A/"pq-right-preserved-joint-source-clearance-024/stable-129-unadmitted-joint-source.npz"
N=A/"left-pq-retriangulated-native-mtp-two-002";T=N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
pins={str(x):h.sha(x) for x in [Path(__file__),H,C,B,P,T,Path(cg.__file__)]}
z=dict(np.load(C));b=dict(np.load(B));prev=dict(np.load(P));p=z["vertices6"][:,:3];f=z["faces"];w=z["weights"];local=z["binding_indices"]
for k in ["faces","binding_indices","face_origins"]:assert np.array_equal(z[k],b[k]) and np.array_equal(z[k],prev[k]),k
assert np.array_equal(p,prev["vertices6"][:,:3]) and np.array_equal(b["weights"],prev["weights"])
changed=np.flatnonzero(np.any(w!=b["weights"],axis=1));assert {686,881,886}.issubset(set(changed.tolist())) and set(changed.tolist()).issubset({686,881,886,887,894,925,890})
assert np.array_equal(w>0,b["weights"]>0)
fixed=np.flatnonzero(b["weights"].max(1)>=.999999);assert np.array_equal(p[fixed],b["vertices6"][fixed,:3]) and np.array_equal(w[fixed],b["weights"][fixed])
assert np.max(np.abs(w.sum(1)-1))<2e-7
def metrics(v):
 top=cg.analyze_topology(v.astype(float).tolist(),f.tolist());vol=[]
 for comp in top["face_components"]:
  tr=v[f[comp]].astype(float);tr-=v[np.unique(f[comp])].mean(0,dtype=float);vol.append(float(np.einsum("ij,ij->i",tr[:,0],np.cross(tr[:,1],tr[:,2])).sum()/6))
 tr=v[f].astype(float);area=float(np.linalg.norm(np.cross(tr[:,1]-tr[:,0],tr[:,2]-tr[:,0]),axis=1).sum()/2)
 return {"closed":top["closed_oriented_manifold_candidate"],"component_volumes_m3":vol,"volume_m3":sum(vol),"area_m2":area}
t=h.load_tissue(T);row=h.row_data(t,129);before=metrics(b["vertices6"][:,:3]);after=metrics(p)
out={"scope":"Passive right PQ129 reference-boundary source and binding characterization. Earlier006 reconstruction remains the explicitly declared source reference; source-coordinate corrections and bounded inferred display-binding changes are inferred, not measurements. Physical mass/routes/tendons are unchanged.","pins":pins,"complete":False,"reference_patch_sha256":h.sha(B),"source_reference":before,"source_candidate":after,"faces_and_raw_parent_lineage_exact_reference":True,"onehot_proxy_coordinates_and_maps_exact_reference":True,"onehot_proxy_count":len(fixed),"map_support_and_indices_exact_reference":True,"max_coordinate_increment_from_reference_m":float(np.linalg.norm(p.astype(float)-b["vertices6"][:,:3],axis=1).max()),"source_coordinates_unchanged_by_latest_binding_step":True,"source_relative_area_change":after["area_m2"]/before["area_m2"]-1,"source_relative_volume_change":after["volume_m3"]/before["volume_m3"]-1,"binding_changes":[],"poses":[]}
for vid in changed:
 out["binding_changes"].append({"vertex_id":int(vid),"local_indices":local[vid].tolist(),"reference_weights":b["weights"][vid].tolist(),"candidate_weights":w[vid].tolist(),"active_registered_body_indices":[int(t["bindings"][row["fb"]+int(li)]["core"]) for li in local[vid][w[vid]>0]]})
for step in [0,4767,5023,5599,6207,6815,7423,8000]:
 rp=N/f"baseline/native-run/accepted-geometry/step-{step}.receipt.json";pins[str(rp)]=h.sha(rp);rc=json.loads(rp.read_text());pm={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc["accepted_body_poses"]}
 v=h.forward(p,local,w,row,t,pm);old=h.forward(p,local,b["weights"],row,t,pm)
 out["poses"].append({"step":step,"max_latest_binding_world_change_m":float(np.linalg.norm(v.astype(float)-old,axis=1).max()),**metrics(v)})
out.update(complete=True,inputs_unchanged=all(h.sha(x)==v for x,v in pins.items()),all_components_positive=all(v>0 for m in [after,*out["poses"]] for v in m["component_volumes_m3"]))
(O/"report.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({k:v for k,v in out.items() if k not in ["pins","poses","source_candidate","source_reference"]}),flush=True)
