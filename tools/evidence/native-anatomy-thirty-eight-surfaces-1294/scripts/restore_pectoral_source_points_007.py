from pathlib import Path
import sys,importlib.util,json,hashlib,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"pectoral-source-point-restoration-007";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-touching-loop-arrangement-1290/src")
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
T=A/"thirty-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T);row=h.row_data(t,73)
cp=A/"pectoral-73-fixed-source-junction-reference-002/stable-73-junction-1-microns.npz";z0=np.load(cp);z={k:z0[k].copy() for k in z0.files};p=z["vertices6"][:,:3].copy();f=z["faces"]
rp=A/"pectoral-source-defect-diagnosis-006/report.json";d=json.loads(rp.read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
out={"scope":"Restore two original source points excluded by strictly-positive winding extraction, retaining original attachment maps. Explicit reference surface proposal; full source and posed gates remain mandatory.","pins":{str(p):sha(p) for p in [Path(__file__),H,T,cp,rp,Path(ci.__file__),Path(sys.modules[analyze_topology.__module__].__file__)]},"edits":[]}
seen=set()
for x in d["missing_source_proxies"]:
 j=x["vertex"];key=row["positions"][j].tobytes()
 if key in seen:continue
 seen.add(key)
 dist=np.linalg.norm(p.astype(float)-row["positions"][j],axis=1)
 k=int(np.argmin(dist));assert dist[k]<.0002
 out["edits"].append({"source_vertex":j,"candidate_vertex":k,"distance_m":float(dist[k]),"old_position":p[k].tolist(),"restored_position":row["positions"][j].tolist(),"old_weights":z["weights"][k].tolist(),"old_local":z["binding_indices"][k].tolist(),"source_weights":row["weights"][j].tolist(),"source_local":row["local"][j].tolist()})
 p[k]=row["positions"][j];z["weights"][k]=row["weights"][j];z["binding_indices"][k]=row["local"][j]
u,iv=np.unique(p,axis=0,return_inverse=True);top=analyze_topology(u.astype(float).tolist(),iv[f].tolist());rec,deg=h.exact_rows(p,f,ci);a=ci._audit_pair(rec,rec,same_surface=True) if not deg else {"count":None}
out.update(topology=top,degenerate=deg,self_audit=a)
if top["closed_oriented_manifold_candidate"] and not deg and a["count"]==0:
 z["vertices6"]=np.concatenate([p,h.vertex_normals(p,f)],axis=1).astype("<f4");dest=O/"stable-73-source-points-restored.npz";np.savez(dest,**z);out.update(candidate_path=str(dest),candidate_sha256=sha(dest))
out["inputs_unchanged"]=all(sha(p)==v for p,v in out["pins"].items());(O/"report.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({k:v for k,v in out.items() if k not in ("pins","topology")}))
