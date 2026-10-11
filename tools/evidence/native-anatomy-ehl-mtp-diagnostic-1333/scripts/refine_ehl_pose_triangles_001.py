from pathlib import Path
import numpy as np,json,hashlib,importlib.util,sys,time,inspect
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");sid=int(sys.argv[1]);D=A/f"ehl-restored-boundary-batch-001/stable-{sid}";O=D/"posed-conforming-refinement-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-positive-winding-1279/src")
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.resting_anatomy_conforming_refinement import refine_surface_edges
from numilab_human.resting_anatomy_interface_patch import normals,signed_volume
from numilab_human.cardiac_cavity_geometry import analyze_topology
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
C=D/f"micrometre-resolution-003/stable-{sid}-geometric-reference-row-patch.npz";z=np.load(C);v=z["vertices6"].copy();f=z["faces"].copy();local=z["binding_indices"].copy();w=z["weights"].copy();orig=z["face_origins"].copy();v0=v.copy();local0=local.copy();w0=w.copy()
T=A.parent/"source-seam-connectivity-1247/regen-4rows-001/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T);row=h.row_data(t,sid)
ar=json.loads((A/"remaining-original-boundary-batch-001/stable-121/current50-target-audit-001/report.json").read_text());receipts=[Path(x) for x in ar["pins"] if x.endswith(".receipt.json") and "/accepted-geometry/" in x];assert len(receipts)==25
pins={str(x):sha(x) for x in [Path(__file__),C,T,H,Path(ci.__file__),Path(inspect.getsourcefile(refine_surface_edges)),*receipts]}
contexts=[]
for rp in receipts:
 d=json.loads(rp.read_text());pm={int(x["body_index"]):(np.array(x["position_m"],np.float32),np.array(x["quaternion_xyzw"],np.float32)) for x in d.get("accepted_body_poses",d.get("accepted_registered_body_poses"))};contexts.append((str(rp),pm))
def checks(v,f,local,w):
 result=[]
 for name,pm in [("source",None),*contexts]:
  p=v[:,:3] if pm is None else h.forward(v[:,:3],local,w,row,t,pm);rec,dg=h.exact_rows(p,f,ci);pairs=None if dg else ci._audit_pair(rec,rec,same_surface=True)["triangle_pairs"];result.append(dict(context=name,degenerate=dg,pairs=pairs))
 return result
out=dict(scope="Existing conforming source-edge refinement for long triangles crossing under nonlinear blended posing. Original coordinates and canonical maps unchanged; new maps are endpoint linear interpolation. Unadmitted until complete source, pose, external, and native checks.",pins=pins,iterations=[],complete=False,candidate=None)
def save():(O/"report.json").write_text(json.dumps(out,indent=2)+"\n")
for iteration in range(5):
 ck=checks(v,f,local,w);out["iterations"].append(dict(iteration=iteration,vertices=len(v),faces=len(f),checks=ck));save();print(json.dumps(dict(iteration=iteration,counts=[None if x["pairs"] is None else len(x["pairs"]) for x in ck])),flush=True)
 if ck[0]["degenerate"] or ck[0]["pairs"]:break
 if all(x["pairs"]==[] and not x["degenerate"] for x in ck):break
 if iteration==4:break
 badfaces=set(int(i) for c in ck if c["pairs"] for pair in c["pairs"] for i in pair)
 edges=set()
 for fi in badfaces:
  face=f[fi]
  for a,b in [(face[0],face[1]),(face[1],face[2]),(face[2],face[0])]:
   if np.linalg.norm(v[a,:3].astype(float)-v[b,:3])>0.002:edges.add(tuple(sorted((int(a),int(b)))))
 if not edges:break
 nv,nf,mapping,report=refine_surface_edges(v,f,edges)
 nl=np.full((len(nv),4),2**32-1,dtype="<u4");nw=np.zeros((len(nv),4),dtype="<f4");nl[:len(v)]=local;nw[:len(v)]=w
 for ni,(a,b) in enumerate(sorted(edges),start=len(v)):
  mean={}
  for vi in [a,b]:
   for k,weight in zip(local[vi],w[vi]):
    if weight>0:mean[int(k)]=mean.get(int(k),0)+float(weight)*.5
  assert len(mean)<=4
  for j,(k,value) in enumerate(sorted(mean.items())):nl[ni,j]=k;nw[ni,j]=np.float32(value)
 no=np.empty(len(nf),orig.dtype)
 for parent,children in mapping.items():no[children]=orig[parent]
 out["iterations"][-1]["refinement"]=report
 out["iterations"][-1]["source_signed_volume_delta_m3"]=signed_volume(nv[:,:3],nf)-signed_volume(v[:,:3],f)
 v,f,local,w,orig=nv,nf,nl,nw,no
 assert np.array_equal(v[:len(v0),:3],v0[:,:3]) and np.array_equal(local[:len(v0)],local0) and np.array_equal(w[:len(v0)],w0)
 assert np.allclose(w.sum(1),1,rtol=0,atol=2e-6)
 save()
dst=O/f"stable-{sid}-unadmitted-row-patch.npz";np.savez(dst,vertices6=np.column_stack((v[:,:3],normals(v[:,:3].astype(float),f))).astype("<f4"),faces=f,face_origins=orig,binding_indices=local,weights=w)
out.update(complete=True,candidate=dict(path=str(dst),sha256=sha(dst)),source_all25_self_clear=all(x["pairs"]==[] and not x["degenerate"] for x in ck),topology=analyze_topology(v[:,:3].astype(float).tolist(),f.tolist()),original_coordinates_maps_unchanged=True,inputs_unchanged=all(sha(p)==s for p,s in pins.items()));save()
