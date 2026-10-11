from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"stable-125-restored-edges-parent-012";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-anatomy-completion-1276/src")
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_intersections as ci,cardiac_cavity_geometry as cg
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
C=A/"bounded-halfmm-source-resolution-001/stable-125-reference-resolution-row-patch.npz";T=A/"forty-six-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(p):sha(p) for p in [Path(__file__),H,C,T,Path(ca.__file__),Path(ci.__file__),Path(cg.__file__),A/"parent-positive-union-lift-001/report.json"]}
assert C.is_file()
z=np.load(C);p=z["vertices6"][:,:3];f=z["faces"];w=z["weights"];local=z["binding_indices"];orig=z["face_origins"];t=h.load_tissue(T);row=h.row_data(t,125)
assert orig.min()>=0 and orig.max()<len(row["faces"])
def exact(p,f):
 ids=np.unique(f);rec,deg=h.exact_rows(p[ids],np.searchsorted(ids,f),ci);assert not deg;return rec
def selfaudit(p):
 rec=exact(p,f);return ci._audit_pair(rec,rec,same_surface=True)
assert cg.analyze_topology(p.astype(float).tolist(),f.tolist())["closed_oriented_manifold_candidate"]




prpath=A/"bounded-halfmm-source-resolution-001/report.json";pins[str(prpath)]=sha(prpath)
pr=next(a for a in json.loads(prpath.read_text())["rows"] if a["stable_id"]==125)
raw=row["positions"];rf=row["faces"];xx=p.copy();ll=local.copy();ww=w.copy();ff=f.copy();fo=orig.copy();operations=[]
for operation in (0,1,4):
 op=next(a for a in pr["attempts"] if a.get("accepted_for_next_offline_iteration") and a["iteration"]==operation)["collapse_report"]
 cp=np.asarray(op["collapse_position_f32_m"],np.float32)
 hits=np.flatnonzero(np.all(xx==cp,axis=1));assert len(hits)==1;vi=int(hits[0]);dv=op["deleted_face_origins"]
 assert len(dv)==2 and not np.any(np.isin(fo,dv))
 common=set(map(tuple,raw[rf[dv[0]]]))&set(map(tuple,raw[rf[dv[1]]]))
 assert len(common)==2
 ends=sorted(common);end_indices=[int(np.flatnonzero(np.all(raw==e,axis=1))[0]) for e in ends]
 oldn=len(xx);xx=np.vstack([xx,np.asarray(ends,dtype=np.float32)]);ll=np.vstack([ll,row["local"][end_indices]]);ww=np.vstack([ww,row["weights"][end_indices]])
 changes=[]
 for i in np.flatnonzero(np.any(ff==vi,axis=1)):
  old=raw[rf[int(fo[i])]];hits=[j for j,e in enumerate(ends) if np.any(np.all(old==e,axis=1))]
  assert len(hits)==1,(operation,i,int(fo[i]),hits)
  ff[i,ff[i]==vi]=oldn+hits[0];changes.append(int(i))
 for origin in dv:
  tri=[]
  for pt in raw[rf[origin]]:
   hits=np.flatnonzero(np.all(xx==pt,axis=1));assert len(hits)==1,(operation,origin,pt,hits)
   tri.append(int(hits[0]))
  ff=np.vstack([ff,np.asarray(tri,dtype=ff.dtype)]);fo=np.append(fo,origin)
 used=np.unique(ff);xx=xx[used];ll=ll[used];ww=ww[used];ff=np.searchsorted(used,ff)
 top=cg.analyze_topology(xx.astype(float).tolist(),ff.tolist());assert top["closed_oriented_manifold_candidate"]
 operations.append({"reversed_original_operation":operation,"endpoint_original_indices":end_indices,"restored_face_origins":dv,"changed_face_rows":changes})
rec,deg=h.exact_rows(xx,ff,ci);assert not deg;audit=ci._audit_pair(rec,rec,same_surface=True)
dest=O/"stable-125-restored-edges-parent.npz";np.savez(dest,vertices6=np.column_stack([xx,h.vertex_normals(xx,ff)]).astype("<f4"),faces=ff,face_origins=fo,binding_indices=ll,weights=ww)
report={"scope":"Restore three original source edges and endpoint bindings that affect external interfaces. Two other parent midpoint repairs retained. Explicit original face ancestry; source-only unadmitted parent.","pins":pins,"operations":operations,"source_self_pairs":audit["triangle_pairs"],"source_self_count":audit["count"],"closed":top["closed_oriented_manifold_candidate"],"candidate_path":str(dest),"candidate_sha256":sha(dest),"inputs_unchanged":all(sha(k)==v for k,v in pins.items())}
(O/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))
