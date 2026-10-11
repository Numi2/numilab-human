from pathlib import Path
import sys,json,importlib.util,numpy as np,time
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"pq-left-native-local-retriangulation-028";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-touching-loop-arrangement-1290/src")
from numilab_human import surface_precision_retriangulation as sr,common_atlas_skin_clearance as ca,cardiac_cavity_intersections as ci,cardiac_cavity_geometry as cg
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
C=A/"pq-left-joint-source-clearance-018/stable-130-unadmitted-joint-source.npz";N=A/"left-pq-native-mtp-two-001"
z=dict(np.load(C));p=z["vertices6"][:,:3];f=z["faces"];orig=z["face_origins"];worlds={"source":p}
pins={str(x):h.sha(x) for x in [Path(__file__),H,C,Path(sr.__file__),Path(ca.__file__),Path(ci.__file__),Path(cg.__file__)]}
for step in [0,4767,5023,5599,6207,6815,7423,8000]:
 pack=N/f"baseline/native-run/accepted-geometry/step-{step}.mrvpack";rp=pack.with_suffix(".receipt.json");pins[str(pack)]=h.sha(pack);pins[str(rp)]=h.sha(rp);assert json.loads(rp.read_text())["pack_file_sha256"]==pins[str(pack)]
 pos,surfs,_=ca._pack_surfaces(pack,{(51005,130)});sf=np.asarray(surfs[(51005,130)]["faces"]);off=sf-f;assert np.all(off==off.flat[0]);worlds[str(step)]=pos[int(off.flat[0]):int(off.flat[0])+len(p)].copy()
def audit(v,ff):
 rec,d=h.exact_rows(v,ff,ci)
 return {"degenerate":d,"pairs":ci._audit_pair(rec,rec,same_surface=True)["triangle_pairs"] if not d else None}
out={"scope":"Unadmitted source-position-preserving precision retriangulation. Actual native captured vertex coordinates, not CPU forward approximation, are tested in all eight retained poses. Each flip must have the same raw parent face identity on both old rows; topology and all source/map bytes retained.","pins":pins,"trials":[],"iterations":[],"baseline":{k:audit(v,f) for k,v in worlds.items()},"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2)+"\n")
cur=f.copy();save()
for iteration in range(4):
 checks={k:audit(v,cur) for k,v in worlds.items()};bad={i for a in checks.values() for pair in (a["pairs"] or []) for i in pair};score=sum(len(a["pairs"] or []) for a in checks.values())
 if not bad:break
 adjacency=sr.build_surface_adjacency(cur);edges=sorted({tuple(sorted((int(a),int(b)))) for i in bad for a,b in zip(cur[i],np.roll(cur[i],-1))});best=None
 for edge in edges:
  rows=adjacency.edge_faces.get(edge,())
  if len(rows)!=2:continue
  record={"iteration":iteration,"edge":edge,"rows":list(rows),"raw_parents":[int(orig[i]) for i in rows]}
  if len(set(record["raw_parents"]))!=1:record["error"]="Two different raw parent identities; this bounded trial does not merge ancestry.";out["trials"].append(record);continue
  try:
   ff,receipt=sr.flip_interior_edge(p,cur,edge=edge,face_indices=rows,minimum_altitude_m=1e-12,adjacency=adjacency,face_origins=orig)
   source=audit(p,ff);native=audit(worlds["5599"],ff);record.update(receipt=receipt,source=source,native5599=native)
   if not source["degenerate"] and not source["pairs"] and not native["degenerate"] and len(native["pairs"])<len(checks["5599"]["pairs"]):
    allchecks={k:audit(v,ff) for k,v in worlds.items()};total=sum(len(a["pairs"] or []) for a in allchecks.values());record["checks"]=allchecks;record["total_pairs"]=total
    if all(not a["degenerate"] for a in allchecks.values()) and total<score and (best is None or total<best[0]):best=(total,ff,record)
  except ValueError as e:record["error"]=str(e)
  out["trials"].append(record);save();print(json.dumps({k:v for k,v in record.items() if k in ["iteration","edge","error","total_pairs"]}),flush=True)
 if best is None:break
 cur=best[1];out["iterations"].append(best[2]);save()
checks={k:audit(v,cur) for k,v in worlds.items()};out["checks"]=checks
out["all_source_and_native_pose_self_clear"]=all(not a["degenerate"] and not a["pairs"] for a in checks.values())
out["topologies"]={}
for k,v in worlds.items():
 top=cg.analyze_topology(v.astype(float).tolist(),cur.tolist());vol=[]
 for comp in top["face_components"]:
  tr=v[cur[comp]].astype(float);tr-=v[np.unique(cur[comp])].mean(0,dtype=float);vol.append(float(np.einsum("ij,ij->i",tr[:,0],np.cross(tr[:,1],tr[:,2])).sum()/6))
 out["topologies"][k]={"closed":top["closed_oriented_manifold_candidate"],"components":top["face_component_count"],"component_volumes_m3":vol}
if out["all_source_and_native_pose_self_clear"]:
 z["faces"]=cur.astype(np.uint32);z["vertices6"]=np.column_stack([p,h.vertex_normals(p,cur)]).astype(np.float32);dest=O/"stable-130-unadmitted-native-retriangulated.npz";np.savez(dest,**z);out.update(candidate_path=str(dest),candidate_sha256=h.sha(dest),source_positions_and_maps_unchanged=True)
out.update(complete=True,inputs_unchanged=all(h.sha(x)==v for x,v in pins.items()));save();print(json.dumps({k:v for k,v in out.items() if k in ["complete","all_source_and_native_pose_self_clear","candidate_path"]}),flush=True)
