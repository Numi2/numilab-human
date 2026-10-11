from pathlib import Path
import sys,json,importlib.util,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"pq-left-retriangulated-source-characterization-032";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-anatomy-publication-1308/src")
from numilab_human import cardiac_cavity_geometry as cg
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
N=A/"bilateral-edl-ehl-native-mtp-two-001";T=N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";C=A/"pq-left-native-local-retriangulation-028/stable-130-unadmitted-native-retriangulated.npz";B=A/"remaining-original-boundary-batch-001/stable-130/micrometre-replay-003/stable-130-unadmitted-row-patch.npz";P=A/"remaining-original-boundary-batch-001/stable-130/posed-local-separation-004/stable-130-unadmitted-local-candidate.npz"
pins={str(p):h.sha(p) for p in [Path(__file__),H,T,C,B,P,Path(cg.__file__)]}
z=dict(np.load(C));b=dict(np.load(B));old=dict(np.load(P));p=z["vertices6"][:,:3];f=z["faces"];local=z["binding_indices"];w=z["weights"];t=h.load_tissue(T);row=h.row_data(t,130)
for k in ["binding_indices","weights","face_origins"]:assert np.array_equal(z[k],b[k]) and np.array_equal(z[k],old[k]),k
fixed=np.flatnonzero(w.max(1)>=.999999);assert np.array_equal(p[fixed],b["vertices6"][fixed,:3])
def metrics(v,f=f):
 top=cg.analyze_topology(v.astype(float).tolist(),f.tolist());vs=[]
 for comp in top["face_components"]:
  ids=np.asarray(comp,int);tri=v[f[ids]].astype(float);tri-=v[np.unique(f[ids])].mean(0,dtype=float);vs.append(float(np.einsum("ij,ij->i",tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6))
 tr=v[f].astype(float);area=float(np.linalg.norm(np.cross(tr[:,1]-tr[:,0],tr[:,2]-tr[:,0]),axis=1).sum()/2)
 return {"closed":top["closed_oriented_manifold_candidate"],"component_volumes_m3":vs,"area_m2":area,"volume_m3":sum(vs)}
before=metrics(b["vertices6"][:,:3],b["faces"]);after=metrics(p)
changed=np.flatnonzero(np.any(f!=b["faces"],axis=1));assert changed.tolist()==[1865,1866] and len(set(z["face_origins"][changed].tolist()))==1
report={"scope":"Source and current-pose component/accounting characterization of PQ130 inferred boundary correction; passive anatomy changes no physical mass or active muscle route. One-hot visualization proxies are not measured entheses.","pins":pins,"maps_and_raw_parent_lineage_exact_source_reference":True,"changed_face_rows":changed.tolist(),"both_changed_rows_raw_parent":int(z["face_origins"][changed[0]]),"onehot_source_proxy_count":len(fixed),"onehot_source_coordinates_exact_reference":True,"max_source_correction_from_micrometre_boundary_m":float(np.linalg.norm(p.astype(float)-b["vertices6"][:,:3],axis=1).max()),"max_latest_source_increment_m":float(np.linalg.norm(p.astype(float)-old["vertices6"][:,:3],axis=1).max()),"source_reference":before,"source_candidate":after,"source_relative_area_change":after["area_m2"]/before["area_m2"]-1,"source_relative_volume_change":after["volume_m3"]/before["volume_m3"]-1,"poses":[]}
for step in [0,4767,5023,5599,6207,6815,7423,8000]:
 rp=N/f"baseline/native-run/accepted-geometry/step-{step}.receipt.json";pins[str(rp)]=h.sha(rp);rc=json.loads(rp.read_text());pm={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc["accepted_body_poses"]}
 report["poses"].append({"step":step,**metrics(h.forward(p,local,w,row,t,pm))})
report.update(complete=True,inputs_unchanged=all(h.sha(p)==v for p,v in pins.items()),all_components_positive=all(x>0 for m in [after,*report["poses"]] for x in m["component_volumes_m3"]))
(O/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({k:v for k,v in report.items() if k not in ["pins","poses"]},indent=2))
