from pathlib import Path
import sys,json,hashlib,importlib.util,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");sid=int(sys.argv[1]);D=A/f"remaining-original-boundary-batch-001/stable-{sid}";O=D/"source-local-translation-002";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-positive-winding-1279/src")
from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca
from numilab_human.resting_anatomy_interface_patch import normals
from numilab_human.cardiac_cavity_geometry import analyze_topology
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
C=D/f"first-boundary-intermediate-001/stable-{sid}-unadmitted-row-patch.npz";z=np.load(C);p=z["vertices6"][:,:3].copy();base=p.copy();f=z["faces"];pure=set(np.flatnonzero(z["weights"].max(1)>=.999999));n0=np.cross(p[f[:,1]].astype(float)-p[f[:,0]],p[f[:,2]].astype(float)-p[f[:,0]])
pins={str(x):sha(x) for x in [Path(__file__),C,H,Path(ci.__file__),Path(ca.__file__)]};out=dict(scope="Existing exact continuous SAT triangle translation for source float32 branch separation; at most5micrometres from prior unadmitted geometric boundary; original onehot coordinates/maps protected. All source positions, ancestry and maps retained except explicit moved vertices. Source-only, no pose/external/native admission.",pins=pins,trials=[],accepted=[],complete=False,candidate=None)
def save():(O/"report.json").write_text(json.dumps(out,indent=2)+"\n")
def audit(p):
 rec,dg=h.exact_rows(p,f,ci);return None if dg else ci._audit_pair(rec,rec,same_surface=True)["triangle_pairs"],dg
for iteration in range(4):
 pairs,dg=audit(p);assert not dg
 if not pairs:break
 counts={}
 for pair in pairs:
  for fi in pair:counts[fi]=counts.get(fi,0)+1
 accepted=False
 for fi,_ in sorted(counts.items(),key=lambda x:(-x[1],x[0])):
  ids=f[fi];movable=[int(i) for i in ids if int(i) not in pure]
  if len(movable)!=3:continue
  tri=p[ids].astype(float);normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=np.linalg.norm(normal)
  opponents=[b if a==fi else a for a,b in pairs if a==fi or b==fi]
  for sign in [-1,1]:
   direction=normal*sign
   distances=[ca._triangle_normal_translation_to_separate(tri,p[f[j]].astype(float),direction) for j in opponents]
   distance=max(distances)+2e-7
   for factor in [1.,2.,4.]:
    x=p.copy();x[ids]=(x[ids].astype(float)+direction*distance*factor).astype(np.float32);disp=float(np.linalg.norm(x.astype(float)-base,axis=1).max());tr=dict(face=int(fi),sign=sign,distance_m=distance,factor=factor,maximum_displacement_m=disp,accepted=False);out["trials"].append(tr)
    if disp>5e-6:tr["rejection"]="displacement bound";continue
    n=np.cross(x[f[:,1]].astype(float)-x[f[:,0]],x[f[:,2]].astype(float)-x[f[:,0]])
    if np.any(np.einsum("ij,ij->i",n,n0)<=0):tr["rejection"]="source face reversal";continue
    ap,ad=audit(x);tr.update(pairs=ap,degenerate=ad)
    if not ad and len(ap)<len(pairs):
     p=x;tr["accepted"]=True;out["accepted"].append(tr);accepted=True;break
   if accepted:break
  if accepted:break
 save()
 if not accepted:break
pairs,dg=audit(p);out.update(source_self_pairs=pairs,degenerate=dg,complete=True)
if not pairs and not dg:
 top=analyze_topology(p.astype(float).tolist(),f.tolist());assert top["closed_oriented_manifold_candidate"]
 data={k:z[k] for k in z.files};data["vertices6"]=np.column_stack((p,normals(p.astype(float),f))).astype(np.float32);dst=O/f"stable-{sid}-geometric-reference-row-patch.npz";np.savez(dst,**data);out.update(candidate=dict(path=str(dst),sha256=sha(dst)),topology=top)
out["inputs_unchanged"]=all(sha(p)==s for p,s in pins.items());save();print(json.dumps({k:out[k] for k in ["candidate","source_self_pairs","complete"]}),flush=True)
