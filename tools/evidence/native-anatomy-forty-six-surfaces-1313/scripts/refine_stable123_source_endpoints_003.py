from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"stable-123-source-endpoint-refinement-003";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-anatomy-completion-1276/src")
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_intersections as ci,cardiac_cavity_geometry as cg
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
C=A/"parent-positive-union-lift-001/stable-123-reference-union-row-patch.npz";T=A/"forty-five-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(p):sha(p) for p in [Path(__file__),H,C,T,Path(ca.__file__),Path(ci.__file__),Path(cg.__file__),A/"parent-positive-union-lift-001/report.json"]}
assert sha(C)=="ef6dd2182e1c526315345b47a5de833c9175f09c8d7c07d607952e2e1e12186c"
z=np.load(C);p=z["vertices6"][:,:3];f=z["faces"];w=z["weights"];local=z["binding_indices"];orig=z["face_origins"];t=h.load_tissue(T);row=h.row_data(t,123)
assert orig.min()>=0 and orig.max()<len(row["faces"])
def exact(p,f):
 ids=np.unique(f);rec,deg=h.exact_rows(p[ids],np.searchsorted(ids,f),ci);assert not deg;return rec
def selfaudit(p):
 rec=exact(p,f);return ci._audit_pair(rec,rec,same_surface=True)
assert cg.analyze_topology(p.astype(float).tolist(),f.tolist())["closed_oriented_manifold_candidate"]
assert selfaudit(p)["count"]==0
poses=[];worlds=[];targets={};labels=[]
for arm,steps,base in [("early",[0,4767,5023,5599,6207,6815,7423,8000],A/"forty-five-surface-native-composition-001/baseline/native-run"),("baseline",h.STEPS,h.PAIR/"baseline/native-run"),("intervention",h.STEPS,h.PAIR/"intervention/native-run"),("K1",[8000],A/"mtp-passive-native-sensitivity-one-001/baseline/native-run")]:
 for step in steps:
  rp=base/"accepted-geometry"/f"step-{step}.receipt.json";pins[str(rp)]=sha(rp);rc=json.loads(rp.read_text());pm={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc.get("accepted_body_poses",rc.get("accepted_registered_body_poses"))}
  i=len(poses);poses.append(pm);labels.append({"arm":arm,"step":step});world=h.forward(p,local,w,row,t,pm);worlds.append(world);audit=selfaudit(world);assert audit["count"]==0,(arm,step,audit)
  if True:
   pack=rp.with_name(f"step-{step}.mrvpack");pins[str(pack)]=sha(pack);assert rc["pack_file_sha256"]==pins[str(pack)]
   pos,surfs,_=ca._pack_surfaces(pack,{(51005,123),(51005,131)})
   sf=np.asarray(surfs[(51005,123)]["faces"]);off=sf-row["faces"];assert np.all(off==off.flat[0]);actual=pos[int(off.flat[0]):int(off.flat[0])+row["vc"]]
   tf=np.asarray(surfs[(51005,131)]["faces"]);target=exact(pos,tf)
   bp=set(tuple(x) for x in ci._audit_pair(exact(actual,row["faces"]),target,same_surface=False)["triangle_pairs"])
   ap={(int(orig[a]),int(b)) for a,b in ci._audit_pair(exact(world,f),target,same_surface=False)["triangle_pairs"]}
   targets[i]=(target,bp)
   print(json.dumps({"step":step,"source_self":0,"added_131_pairs":sorted(ap-bp)}),flush=True)
report={"scope":"Bounded passive source-position correction of the existing positive-winding right palmaris longus. Source reference ancestry and all maps unchanged. Cumulative0.5mm source displacement bound includes prior maximum ancestry displacement; all25 exact self and eight native45 target131 checks, final all859/shared-insertion classification still required.","pins":pins,"pose_labels":labels,"trials":[],"selected":None,"complete":False}
parent_report=A/"bounded-halfmm-source-resolution-001/report.json";pins[str(parent_report)]=sha(parent_report)
pr=next(x for x in json.loads(parent_report.read_text())["rows"] if x["stable_id"]==123)
prior_bound=max(x["cumulative_ancestry_displacement_m"] for x in pr["attempts"] if x.get("accepted_for_next_offline_iteration"));assert prior_bound<.0005
report["prior_cumulative_endpoint_bound_m"]=prior_bound
empty={"count":0,"triangle_pairs":[],"degenerate_face_rows":[]}
def selfcheck(base,cand):
 return ca.audit_incremental_skin_self_intersections(baseline_world_positions=base,candidate_world_positions=cand,baseline_faces=f,candidate_faces=f,baseline_self_audit=empty,expected_baseline_world_f32_sha256=hashlib.sha256(base.tobytes()).hexdigest(),expected_face_index_sha256=ca._face_index_sha256(f),expected_baseline_self_pair_table_sha256=ca._baseline_self_pair_table_sha256(empty,len(f)))["count"]
def save():(O/"report.json").write_text(json.dumps(report,indent=2)+"\n")
area=lambda x,ff:float(np.linalg.norm(np.cross(x[ff[:,1]].astype(float)-x[ff[:,0]],x[ff[:,2]].astype(float)-x[ff[:,0]]),axis=1).sum()/2)
volume=lambda x,ff:float(np.einsum("ij,ij->i",x[ff[:,0]].astype(float),np.cross(x[ff[:,1]].astype(float),x[ff[:,2]].astype(float))).sum()/6)
oldn=np.cross(p[f[:,1]].astype(float)-p[f[:,0]],p[f[:,2]].astype(float)-p[f[:,0]]);oldlen=np.linalg.norm(oldn,axis=1);basearea=area(p,f);rawvol=volume(row["positions"],row["faces"]);volbound=min(1e-9,.001*abs(rawvol));norm=h.vertex_normals(p,f)
report["bounds"]={"cumulative_source_displacement_m":.0005,"maximum_absolute_volume_difference_from_original_m3":volbound,"maximum_incremental_area_relative_change":.001,"minimum_face_area_ratio":.2,"minimum_normal_cosine_exclusive":0}
start=time.monotonic()
for source_id in (4039,1670):
 if report["selected"]:break
 for fraction in (.5,.25,.1):
  if report["selected"]:break
  for vertex in (1679,):
   j=vertex;js=np.asarray([1641,vertex],dtype=int);di=source_id;um=fraction
   x=p.copy();x[1641]=row["positions"][3723];x[j]=(p[j].astype(float)+(row["positions"][source_id].astype(float)-p[j])*fraction).astype(np.float32);delta=float(np.linalg.norm(x[js].astype(float)-p[js].astype(float),axis=1).max())
   if prior_bound+delta>.0005:continue
   nn=np.cross(x[f[:,1]].astype(float)-x[f[:,0]],x[f[:,2]].astype(float)-x[f[:,0]]);nl=np.linalg.norm(nn,axis=1);ratio=float((nl/oldlen).min());cos=float((np.einsum("ij,ij->i",nn,oldn)/np.maximum(nl*oldlen,1e-30)).min())
   ar=area(x,f)/basearea-1;vd=volume(x,f)-rawvol
   if ratio<.2 or cos<=0 or abs(ar)>.001 or abs(vd)>volbound:continue
   target,bp=targets[14];probe=worlds[14].copy();probe[js]=h.forward(x[js],local[js],w[js],row,t,poses[14])
   ap={(int(orig[a]),int(b)) for a,b in ci._audit_pair(exact(probe,f),target,same_surface=False)["triangle_pairs"]}
   rr={"vertex":j,"also_restore_vertex1641_from_original3723":True,"direction_index":di,"source_endpoint_id":source_id,"source_endpoint_fraction":fraction,"actual_source_displacement_m":delta,"cumulative_endpoint_displacement_bound_m":prior_bound+delta,"baseline154431_added_pairs":sorted(ap-bp),"minimum_area_ratio":ratio,"minimum_normal_cosine":cos,"incremental_area_relative_change":ar,"volume_difference_from_original_m3":vd};report["trials"].append(rr)
   if ap-bp:continue
   rr["source_self_count"]=selfcheck(p,x)
   if rr["source_self_count"]:save();continue
   moved=[]
   for i,base in enumerate(worlds):
    world=base.copy();world[js]=h.forward(x[js],local[js],w[js],row,t,poses[i]);moved.append(world)
   rr["self_counts"]=[]
   for i,base in enumerate(worlds):
    rr["self_counts"].append(selfcheck(base,moved[i]))
    if rr["self_counts"][-1]:break
   if any(rr["self_counts"]):save();continue
   rr["target131_added_pairs"]={}
   for i,(tar,before) in targets.items():
    after={(int(orig[a]),int(b)) for a,b in ci._audit_pair(exact(moved[i],f),tar,same_surface=False)["triangle_pairs"]}
    rr["target131_added_pairs"][i]=sorted(after-before)
   save();print(json.dumps(rr),flush=True)
   if any(rr["target131_added_pairs"].values()):continue
   dest=O/"stable-123-position-unadmitted.npz";np.savez(dest,vertices6=np.column_stack([x,h.vertex_normals(x,f)]).astype("<f4"),faces=f,face_origins=orig,binding_indices=local,weights=w)
   report["selected"]={"candidate_path":str(dest),"candidate_sha256":sha(dest),"proposal":rr};break
 save()
report.update(complete=True,inputs_unchanged=all(sha(k)==v for k,v in pins.items()),elapsed_seconds=time.monotonic()-start);save();print(json.dumps({"selected":report["selected"],"trials":len(report["trials"])}),flush=True)
