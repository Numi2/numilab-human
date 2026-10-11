from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"stable-126-parent-gap-correction-020";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-anatomy-completion-1276/src")
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_intersections as ci,cardiac_cavity_geometry as cg
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
C=A/"stable-126-restored-edges-lift-014/stable-126-reference-union-row-patch.npz";T=A/"forty-seven-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(p):sha(p) for p in [Path(__file__),H,C,T,Path(ca.__file__),Path(ci.__file__),Path(cg.__file__),A/"parent-positive-union-lift-001/report.json"]}
assert sha(C)=="6d3af85294b488a71b14d2adfe6bb9f0023cd2a32950ee1f94b2bd5fb5828180"
z=np.load(C);p=z["vertices6"][:,:3];f=z["faces"];w=z["weights"];local=z["binding_indices"];orig=z["face_origins"];t=h.load_tissue(T);row=h.row_data(t,126)
assert orig.min()>=0 and orig.max()<len(row["faces"])
def exact(p,f):
 ids=np.unique(f);rec,deg=h.exact_rows(p[ids],np.searchsorted(ids,f),ci);assert not deg;return rec
def selfaudit(p):
 rec=exact(p,f);return ci._audit_pair(rec,rec,same_surface=True)
assert cg.analyze_topology(p.astype(float).tolist(),f.tolist())["closed_oriented_manifold_candidate"]
assert selfaudit(p)["count"]==0
poses=[];worlds=[];targets={};labels=[];native_baselines=[];target_triangles={}
for arm,steps,base in [("early",[0,4767,5023,5599,6207,6815,7423,8000],A/"forty-seven-surface-native-composition-001/baseline/native-run"),("baseline",h.STEPS,h.PAIR/"baseline/native-run"),("intervention",h.STEPS,h.PAIR/"intervention/native-run"),("K1",[8000],A/"mtp-passive-native-sensitivity-one-001/baseline/native-run")]:
 for step in steps:
  rp=base/"accepted-geometry"/f"step-{step}.receipt.json";pins[str(rp)]=sha(rp);rc=json.loads(rp.read_text());pm={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc.get("accepted_body_poses",rc.get("accepted_registered_body_poses"))}
  i=len(poses);poses.append(pm);labels.append({"arm":arm,"step":step});world=h.forward(p,local,w,row,t,pm);worlds.append(world);audit=selfaudit(world);assert audit["count"]==0,(arm,step,audit)
  if True:
   pack=rp.with_name(f"step-{step}.mrvpack");pins[str(pack)]=sha(pack);assert rc["pack_file_sha256"]==pins[str(pack)]
   target_keys={(51005,128),(51005,136)}
   pos,surfs,_=ca._pack_surfaces(pack,target_keys|{(51005,126)})
   sf=np.asarray(surfs[(51005,126)]["faces"]);off=sf-row["faces"];assert np.all(off==off.flat[0]);actual=pos[int(off.flat[0]):int(off.flat[0])+row["vc"]]
   native_baselines.append(actual.copy());target_triangles[i]={}
   br=exact(actual,row["faces"]);ar=exact(world,f);targets[i]=[]
   for key in sorted(target_keys):
    tf=np.asarray(surfs[key]["faces"]);target_triangles[i][key]=pos[tf].copy();target=exact(pos,tf)
    bp=set(tuple(x) for x in ci._audit_pair(br,target,same_surface=False)["triangle_pairs"])
    ap={(int(orig[aa]),int(bb)) for aa,bb in ci._audit_pair(ar,target,same_surface=False)["triangle_pairs"]}
    targets[i].append((key,target,bp))
    if ap-bp:print(json.dumps({"arm":arm,"step":step,"target":key,"added_pairs":sorted(ap-bp)}),flush=True)


from scipy.optimize import linprog
prpath=A/"bounded-halfmm-source-resolution-001/report.json";pins[str(prpath)]=sha(prpath)
pr=next(a for a in json.loads(prpath.read_text())["rows"] if a["stable_id"]==126)
prior=max(a["cumulative_ancestry_displacement_m"] for a in pr["attempts"] if a.get("accepted_for_next_offline_iteration") and a["iteration"] in (2,3))+2.0647654623614278e-7
bound=.0005-prior
report={"scope":"Offline source-position proposal restoring positive separating support of original accepted parent triangles against two selected anatomical neighbors. No geometry admission; full independent 859-target and native verification required.","pins":pins,"pose_labels":labels,"prior_ancestry_bound_m":prior,"maximum_additional_L2_m":bound,"iterations":[],"selected":None}
def save():(O/"report.json").write_text(json.dumps(report,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+"\n")
def rot(q):
 q=np.asarray(q,float);v=q[:3];cross=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
 return np.eye(3)+2*q[3]*cross+2*cross@cross
jacs=[]
for pm in poses:
 mats=[]
 for vi in range(len(p)):
  j=np.zeros((3,3))
  for b,wt in zip(local[vi],w[vi]):
   if wt<=0:continue
   br=t["bindings"][row["fb"]+int(b)];val=br["value"]
   j+=float(wt)*float(val[7])*rot(pm[int(br["core"])][1])@rot(val[3:7])
  mats.append(j)
 jacs.append(np.asarray(mats))
def sat(a,b):
 a=a.astype(float);b=b.astype(float);ea=np.roll(a,-1,axis=0)-a;eb=np.roll(b,-1,axis=0)-b
 axes=[np.cross(ea[0],ea[1]),np.cross(eb[0],eb[1])]+[np.cross(u,v) for u in ea for v in eb]
 best=None
 for u in axes:
  n=np.linalg.norm(u)
  if n<1e-18:continue
  u=u/n
  for nrm in (u,-u):
   gap=float((a@nrm).min()-(b@nrm).max())
   if best is None or gap>best[0]:best=(gap,nrm)
 return best
def volume(x):return float(np.einsum("ij,ij->i",x[f[:,0]].astype(float),np.cross(x[f[:,1]].astype(float),x[f[:,2]].astype(float))).sum()/6)
def area(x):return float(np.linalg.norm(np.cross(x[f[:,1]].astype(float)-x[f[:,0]],x[f[:,2]].astype(float)-x[f[:,0]]),axis=1).sum()/2)
rawp=row["positions"];rawf=row["faces"];rawvolume=float(np.einsum("ij,ij->i",rawp[rawf[:,0]].astype(float),np.cross(rawp[rawf[:,1]].astype(float),rawp[rawf[:,2]].astype(float))).sum()/6)
oldn=np.cross(p[f[:,1]].astype(float)-p[f[:,0]],p[f[:,2]].astype(float)-p[f[:,0]]);oldlen=np.linalg.norm(oldn,axis=1)
constraints={};x=p.copy()
for iteration in range(4):
 moved=[h.forward(x,local,w,row,t,pm) for pm in poses]
 added=[]
 for i,world in enumerate(moved):
  rec=exact(world,f)
  for key,tar,before in targets[i]:
   for aa,bb in ci._audit_pair(rec,tar,same_surface=False)["triangle_pairs"]:
    if (int(orig[aa]),int(bb)) not in before:added.append((i,key,int(aa),int(bb)))
 rr={"iteration":iteration,"added_candidate_pairs":len(added)}
 report["iterations"].append(rr);save()
 print(json.dumps(rr),flush=True)
 if not added:
  checks=[selfaudit(x)["count"]]+[selfaudit(m)["count"] for m in moved]
  rr["source_and_pose_self_counts"]=checks
  if not any(checks):
   dest=O/"stable-126-position-unadmitted.npz"
   np.savez(dest,vertices6=np.column_stack([x,h.vertex_normals(x,f)]).astype("<f4"),faces=f,face_origins=orig,binding_indices=local,weights=w)
   report["selected"]={"path":str(dest),"sha256":sha(dest),"full_target_audit_required":True}
  save();break
 for i,key,aa,bb in added:
  triangle=target_triangles[i][key][bb]
  gap,n=sat(native_baselines[i][rawf[int(orig[aa])]],triangle)
  if gap<=1e-10:raise RuntimeError("No positive native-parent separating witness: "+str((i,key,aa,bb,gap)))
  threshold=float((triangle.astype(float)@n).max()+min(gap*.5,5e-7))
  for vi in f[aa]:
   ck=(i,key,aa,bb,int(vi))
   constraints[ck]=(n@jacs[i][vi],threshold-float(worlds[i][vi].astype(float)@n),gap)
 movable=np.asarray(sorted({k[-1] for k in constraints}),int);lookup={v:j for j,v in enumerate(movable)};nv=len(movable)*3
 assert not np.any((w[movable]>0).sum(axis=1)==1),"Do not move original onehot attachment proxies"
 mat=[];rhs=[]
 for key,(coef,need,gap) in constraints.items():
  line=np.zeros(nv*2);j=lookup[key[-1]]*3;line[j:j+3]=-coef;mat.append(line);rhs.append(-need)
 # L1 minimization in metres; numerical scaling makes optimization operate in mm.
 aa=np.asarray(mat);bb=np.asarray(rhs)*1000
 for j in range(nv):
  v=np.zeros(nv*2);v[j]=1;v[nv+j]=-1;aa=np.vstack([aa,v]);bb=np.append(bb,0)
  v=v.copy();v[j]=-1;aa=np.vstack([aa,v]);bb=np.append(bb,0)
 np.savez(O/f"linear-system-{iteration}.npz",A_ub=aa,b_ub_mm=bb,movable=movable,source_positions=p,faces=f)
 result=linprog(np.r_[np.zeros(nv),np.ones(nv)],A_ub=aa,b_ub=bb,bounds=[(-bound*1000,bound*1000)]*nv+[(0,None)]*nv,method="highs",options={"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9})
 rr.update(movable_vertices=movable.tolist(),constraints=len(constraints),solver_success=bool(result.success),solver_message=result.message)
 if not result.success:save();break
 delta=result.x[:nv].reshape(-1,3)/1000;x=p.copy();x[movable]=(p[movable].astype(float)+delta).astype(np.float32)
 nn=np.cross(x[f[:,1]].astype(float)-x[f[:,0]],x[f[:,2]].astype(float)-x[f[:,0]]);nl=np.linalg.norm(nn,axis=1)
 metrics={"maximum_extra_displacement_m":float(np.linalg.norm(x.astype(float)-p,axis=1).max()),"minimum_face_area_ratio":float((nl/oldlen).min()),"minimum_normal_cosine":float((np.einsum("ij,ij->i",nn,oldn)/np.maximum(nl*oldlen,1e-30)).min()),"area_relative_change":area(x)/area(p)-1,"volume_difference_from_original_m3":volume(x)-rawvolume}
 rr["metrics"]=metrics
 valid=metrics["maximum_extra_displacement_m"]<=bound+1e-8 and metrics["minimum_face_area_ratio"]>=.2 and metrics["minimum_normal_cosine"]>0 and abs(metrics["area_relative_change"])<=.001 and abs(metrics["volume_difference_from_original_m3"])<=min(1e-9,.001*abs(rawvolume))
 rr["shape_bounds_pass"]=valid;save()
 if not valid:break
 selfcounts=[selfaudit(x)["count"]]+[selfaudit(m)["count"] for m in [h.forward(x,local,w,row,t,pm) for pm in poses]]
 rr["source_and_pose_self_counts"]=selfcounts;save()
 if any(selfcounts):break
report.update(complete=True,inputs_unchanged=all(sha(k)==v for k,v in pins.items()));save()
print(json.dumps({"selected":report["selected"],"last":report["iterations"][-1]}),flush=True)
