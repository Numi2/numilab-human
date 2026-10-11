from pathlib import Path
import sys,json,hashlib,numpy as np,importlib.util
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"sartorius50-native-precision-repair-002";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-anatomy-publication-1308/src")
from numilab_human import resting_lung_edge_repair as er,cardiac_cavity_intersections as ci,cardiac_cavity_geometry as cg
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
T=A/"left-scm-masseters-outward-native-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T);r=h.row_data(t,50)
C=A/"sartorius-tibialis-posed-local-reference-007/stable-50-alpha-0.1-bound-0.0001.npz";z=np.load(C);assert np.array_equal(z["vertices6"][:,:3],r["positions"]) and np.array_equal(z["faces"],r["faces"]) and np.array_equal(z["binding_indices"],r["local"]) and np.array_equal(z["weights"],r["weights"])
failure=A/"current-anatomy-mtp-two-native-001/baseline/native-run/surface-audit-failure-step-6591.json"
pins={str(p):h.sha(p) for p in (Path(__file__),H,T,C,failure,Path(er.__file__),Path(ci.__file__),Path(cg.__file__))}
keep=1694;drop=1695;assert np.array_equal(z["binding_indices"][keep],z["binding_indices"][drop])
report={"scope":"Unadmitted precision repair for current passive left sartorius50/FJ1434M after actual native step6591 under diagnostic K2 MTP produced a zero-area triangle. Existing topology-preserving endpoint-collapse owner; no joint, support, force or physiological state changes. Removes a 0.133-micrometre edge, not a biological component. No final anatomy/adoption claim.","pins":pins,"keep":keep,"drop":drop,"source_endpoint_distance_m":float(np.linalg.norm(r["positions"][keep].astype(float)-r["positions"][drop])),"max_binding_weight_difference":float(np.max(np.abs(r["weights"][keep].astype(float)-r["weights"][drop]))),"complete":False}
def save():(O/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
save()
try:
 from scipy.sparse import coo_matrix
 from scipy.sparse.csgraph import connected_components
 faces=z["faces"];edges=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]));nc,labels=connected_components(coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(z["vertices6"]),)*2),directed=False)
 component_face_ids=np.flatnonzero(labels[faces[:,0]]==labels[keep]);component_vertices=np.unique(faces[component_face_ids]);cf=np.searchsorted(component_vertices,faces[component_face_ids]);assert labels[drop]==labels[keep]
 cv,cf2,co,proof=er.collapse_midpoint_edge(z["vertices6"][component_vertices],cf,component_face_ids,remove_vertex=int(np.searchsorted(component_vertices,drop)),keep_vertex=int(np.searchsorted(component_vertices,keep)),position_policy="keep",max_endpoint_displacement_m=2e-7,max_abs_volume_delta_m3=1e-12)
 replaced=faces.copy();replaced[replaced==drop]=keep;mask=np.all(np.diff(np.sort(replaced,axis=1),axis=1)!=0,axis=1);used=np.unique(replaced[mask]);origin=np.flatnonzero(mask);f=np.searchsorted(used,replaced[mask]);v=z["vertices6"][used].copy()
 component_replaced=faces[component_face_ids].copy();component_replaced[component_replaced==drop]=keep;cm=np.all(np.diff(np.sort(component_replaced,axis=1),axis=1)!=0,axis=1);component_used=np.unique(component_replaced[cm]);assert np.array_equal(cv[:,:3],z["vertices6"][component_used,:3]);v[np.searchsorted(used,component_used)]=cv
 report["source_component_count"]=int(nc);report["component_face_ids"]=component_face_ids.tolist();report["component_vertices"]=component_vertices.tolist();report["all_other_components_byte_exact"]=bool(np.array_equal(v[np.isin(used,component_used,invert=True)],z["vertices6"][used[np.isin(used,component_used,invert=True)]]))
 replaced=z["faces"].copy();replaced[replaced==drop]=keep;mask=np.all(np.diff(np.sort(replaced,axis=1),axis=1)!=0,axis=1);used=np.unique(replaced[mask]);assert np.array_equal(v[:,:3],z["vertices6"][used,:3])
 rec,dg=h.exact_rows(v[:,:3],f,ci);assert not dg;pairs=ci._audit_pair(rec,rec,same_surface=True)["triangle_pairs"];top=cg.analyze_topology(v[:,:3].astype(float).tolist(),f.tolist())
 cp=O/"stable-50-unadmitted.npz";data=dict(vertices6=v,faces=f.astype("<u4"),face_origins=origin,binding_indices=z["binding_indices"][used],weights=z["weights"][used]);np.savez(cp,**data)
 report.update(collapse=proof,source_self_pairs=pairs,source_topology=top,candidate=str(cp),candidate_sha256=h.sha(cp),parent_vertex_ids=used.tolist(),complete=True,inputs_unchanged=all(h.sha(p)==d for p,d in pins.items()));save()
 print(json.dumps({"complete":True,"self":len(pairs),"closed":top["closed_oriented_manifold_candidate"],"candidate":str(cp),"collapse":proof}),flush=True)
except Exception as e:
 report["failure"]=type(e).__name__+": "+str(e);save();raise
