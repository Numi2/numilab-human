from pathlib import Path
import sys,json,hashlib,importlib.util,inspect,time
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276";O=A/"existing-body-60-pinched-reference-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-touching-loop-arrangement-1290/src")
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology
from numilab_human.resting_anatomy import _split_disconnected_vertex_fans
H=R/"fullbody-source-surface-repair-1276/gluteus-maximus-pilot-001/attempt-010/run_gluteus_union_010.py"
sp=importlib.util.spec_from_file_location("prior",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
h.TISS=A/"twenty-six-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";h.MAN=h.TISS.with_suffix(".manifest.json")
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
h.EXPECTED_TISS_SHA=sha(h.TISS);h.EXPECTED_MAN_SHA=sha(h.MAN)
W=A/"existing-body-60-rejected-boundary-004/rejected-boundary-witness.json";w=json.loads(W.read_text())
pins={str(p):sha(p) for p in [Path(__file__),H,h.TISS,h.MAN,W,Path(ci.__file__),Path(sys.modules[analyze_topology.__module__].__file__),Path(sys.modules[_split_disconnected_vertex_fans.__module__].__file__)]}
src=inspect.getsource(h.make_candidate).replace('require(row["bc"]==2,"gluteus row binding cardinality changed")','require(1<=row["bc"]<=4,"row exceeds four-slot existing binding format")').replace('expected_member={29:"FJ1418",30:"FJ1418M"}[sid]','expected_member=source_manifest["member_id"]')
lp=O/"parameterized_existing_lift.py";lp.write_text(src);pins[str(lp)]=sha(lp);exec(compile(src,str(lp),"exec"),h.__dict__);h.OUT=O;h.UNION_DIR=O
u={"status":"REJECTED_boundary_for_attribute_diagnosis_only","source_tissue_sha256":h.EXPECTED_TISS_SHA,"source_manifest_sha256":h.EXPECTED_MAN_SHA,"construction":{"output_mesh":w["output_mesh"]},"ancestry_by_output_face":w["output_face_ancestry"]}
up=O/"stable-60-exact-union.json";up.write_text(json.dumps(u));pins[str(up)]=sha(up)
t=h.load_tissue(h.TISS);row=h.row_data(t,60);man=json.loads(h.MAN.read_text())
geom={"manifest_rows":{int(x["stable_id"]):x for x in man["source"]["surfaces"]},"analyze_topology":analyze_topology}
try:
 h.make_candidate(60,t,row,u,ci,geom)
 raise AssertionError("expected retained boundary manifold rejection")
except Exception as e:
 assert str(e)=="candidate union topology not closed/oriented manifold",str(e)
 failed=str(e)
z=np.load(O/"stable-60-reference-union-row-patch.npz");p=z["vertices6"][:,:3];f=z["faces"];weights=z["weights"];local=z["binding_indices"]
top=analyze_topology(p.astype(float).tolist(),f.tolist())
vv,ff,edits=_split_disconnected_vertex_fans(p.tolist(),f.tolist(),top["vertex_manifold_defect_ids"])
p2=np.asarray(vv,np.float32);f2=np.asarray(ff,int);assert np.array_equal(p2[f2],p[f])
orig=list(range(len(p)));changed=[]
for e in edits:
 changed.append(e["source_vertex_id"])
 for j in e["copied_vertex_ids"]:
  assert j==len(orig);orig.append(e["source_vertex_id"]);changed.append(j)
ww=weights[orig];ii=local[orig]
assert np.array_equal(ww[f2],weights[f]) and np.array_equal(ii[f2],local[f])
directions=np.zeros_like(p2,dtype=float);locked=[]
for j in changed:
 if ww[j].max()>=.999999:locked.append(j);continue
 nbr=np.unique(f2[np.any(f2==j,axis=1)]);nbr=nbr[nbr!=j]
 d=p2[nbr].astype(float).mean(0)-p2[j]
 assert np.linalg.norm(d)>0
 directions[j]=d/np.linalg.norm(d)
out={"scope":"Bounded inferred reference separation of rejected winding-boundary point junctions. No tolerance waiver. Source geometry only, not native or anatomical admission.","pins":pins,"rejected_lift_error":failed,"fan_edits":edits,"locked_pure_attachment_proxy_vertices":locked,"trials":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
for microns in (.1,1,10):
 pp=(p2.astype(float)+directions*microns*1e-6).astype(np.float32)
 u,inv=np.unique(pp,axis=0,return_inverse=True);qf=inv[f2]
 tt=analyze_topology(u.astype(float).tolist(),qf.tolist())
 rec,deg=h.exact_rows(u,qf,ci);audit=ci._audit_pair(rec,rec,same_surface=True) if not deg else {"count":None,"triangle_pairs":[]}
 trial={"microns":microns,"max_original_boundary_offset_m":float(np.linalg.norm(pp.astype(float)-p2.astype(float),axis=1).max()),"closed":tt["closed_oriented_manifold_candidate"],"degenerate":deg,"self_count":audit["count"],"pairs":audit["triangle_pairs"]}
 if trial["closed"] and not deg and trial["self_count"]==0:
  dest=O/f"stable-60-junction-{microns}-microns.npz"
  np.savez(dest,vertices6=np.concatenate([pp,h.vertex_normals(pp,f2)],axis=1).astype("<f4"),faces=f2.astype("<u4"),binding_indices=ii,weights=ww,face_origins=np.asarray([int(x["source_face"]) for x in w["output_face_ancestry"]],int))
  trial.update(candidate_path=str(dest),candidate_sha256=sha(dest))
 out["trials"].append(trial);save();print(json.dumps({k:v for k,v in trial.items() if k not in ("pairs","degenerate")}),flush=True)
out["complete"]=True;out["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());save()

