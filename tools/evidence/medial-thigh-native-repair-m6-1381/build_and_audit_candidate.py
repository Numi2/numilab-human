import gc,hashlib,json,struct,sys,time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
ROOT=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
N=ROOT/"back-forearm-native-014-018-back412-recovery-16s"
RUN=N/"baseline/native-run"
SRC=ROOT/"skin-candidate-30-pose-preparation-001/result-005/bodyparts3d-myosim-skinned-shell.nhskin"
RIG=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid")
PUB=Path("/Users/n/numi-human-left-pq-publication-1344/src")
INV=Path("/Users/n/numi-human-resting-evidence-20261005/native-complete-skin-containment-audit-890/pair-summary-v3.csv")
OUT=ROOT/"skin-native-self-repair-1379"
STEPS=[0,3807,4767,5023,5599,6207,7423,8000]
RIGHT=np.array([24825,24842,24843,24845,24884,25678])
LEFT=np.array([29325,30182,30206,30207,30208,30224,30225])
AMP=.0012; RADIUS=.024; SKIN_KEY=(51007,1)
SKIN_SHA="24f0d0bf288bb501ee7f0464f53e089126c4d4850ad638044b0015a286497d6a"
NATIVE_SHA="147c413bdfa84687f4afa3ce4e0d37576a853a3b4d5f421866e4faefeab4f174"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
 return h.hexdigest()
def rotmat(q):
 x,y,z,w=map(float,q)
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def rot32(q,v):
 f=np.float32; q=np.asarray(q,dtype=f); v=np.asarray(v,dtype=f); a=q[:3]
 cr=np.cross(a,v); tw=np.asarray([f(2*x) for x in cr],dtype=f)
 co=np.asarray([f(f(f(q[3]*tw[0])+f(a[1]*tw[2]))-f(a[2]*tw[1])),f(f(f(q[3]*tw[1])+f(a[2]*tw[0]))-f(a[0]*tw[2])),f(f(f(q[3]*tw[2])+f(a[0]*tw[1]))-f(a[1]*tw[0]))],dtype=f)
 return np.asarray([f(v[i]+co[i]) for i in range(3)],dtype=f)
def rot32(q,v):
 f=np.float32; q=np.asarray(q,dtype=f); v=np.asarray(v,dtype=f); a=q[:3]
 cr=np.cross(a,v); tw=np.asarray(f(2)*cr,dtype=f)
 co=np.asarray(q[3]*tw+np.cross(a,tw),dtype=f)
 return np.asarray(v+co,dtype=f)
def skin_forward(vertices,skin,poses,rig):
 out=np.zeros((len(vertices),3),dtype="<f4")
 for j,(body,tr,qb,scale) in enumerate(skin["bindings"]):
  ids=np.flatnonzero(skin["weights"][:,j]>0)
  if not len(ids): continue
  v=np.asarray(vertices[ids],dtype="<f4"); tr=np.asarray(tr,dtype="<f4"); qb=np.asarray(qb,dtype="<f4")
  if int(body) in poses: bp,bq=poses[int(body)]
  else: q=rig["core_poses"][int(body)]; bp=np.asarray(q[:3],dtype="<f4"); bq=np.asarray(q[3:7],dtype="<f4")
  local=np.asarray(tr+rot32(qb,v)*np.float32(scale),dtype="<f4")
  world=np.asarray(bp+rot32(bq,local),dtype="<f4")
  w=np.asarray(skin["weights"][ids,j],dtype="<f4")
  out[ids]=np.asarray(out[ids]+w[:,None]*world,dtype="<f4")
 return out
def main():
 OUT.mkdir(exist_ok=True)
 outdir=OUT/"attempt-004"; outdir.mkdir(exist_ok=False)
 sys.path.insert(0,str(PUB))
 from numilab_human import common_atlas_skin_clearance as ca, cardiac_cavity_intersections as ci
 from numilab_human.resting_scene import load_rigid,load_skin,source_shell_world
 assert sha(SRC)==SKIN_SHA
 native=Path("/Users/n/numi-lab-parent-bindings-build-1353-003/bin/numi-human-native"); assert sha(native)==NATIVE_SHA
 rig=load_rigid(RIG); sk=load_skin(SRC,rig); raw=sk["raw"]; nv=sk["vertex_count"]; nb=sk["binding_count"]; ni=sk["index_count"]
 vo=60+36*nb; io=vo+56*nv; faces=np.frombuffer(raw,"<u4",ni,io).reshape(-1,3).astype(np.int64)
 assert faces.shape==(109211,3) and nv==54949 and nb==86
 rest,_,_=source_shell_world(rig,sk)
 edges=np.unique(np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
 el=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1)
 graph=coo_matrix((np.r_[el,el],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(nv,nv)).tocsr()
 dr=dijkstra(graph,directed=False,indices=RIGHT,min_only=True); dl=dijkstra(graph,directed=False,indices=LEFT,min_only=True)
 def fall(d):
  u=np.clip(d/RADIUS,0,1); x=1-10*u**3+15*u**4-6*u**5; x[d>=RADIUS]=0; return x
 fr,fl=fall(dr),fall(dl); assert not np.any((fr>0)&(fl>0))
 sr=np.flatnonzero(fr>0); sl=np.flatnonzero(fl>0); support=np.unique(np.r_[sr,sl])
 assert len(support)<nv//5
 records=[]; pmap=[]; rots=[]
 for step in STEPS:
  rp=RUN/"accepted-geometry"/f"step-{step}.receipt.json"; rec=json.loads(rp.read_text()); pack=RUN/"accepted-geometry"/f"step-{step}.mrvpack"
  assert rec["accepted_step"]==step and rec["physical_endpoint"]=="accepted" and rec["surface_audit_endpoint"]=="passed"
  assert sha(pack)==rec["pack_file_sha256"] and rec["common_field_source_anatomy_payload_sha256"]=="1c0c37af76ab3f8e86870fd6cd3abab00b7bcdae51fe934e3461722ca306c241"
  pm={int(x["body_index"]):(np.asarray(x["position_m"],dtype="<f4"),np.asarray(x["quaternion_xyzw"],dtype="<f4")) for x in rec["accepted_body_poses"]}
  assert len(pm)==157
  pmap.append(pm); rots.append({i:rotmat(q) for i,(_,q) in pm.items()}); records.append((step,rec,pack))
 R0=rotmat(rig["core_poses"][128][3:7])
 axes=[]
 for r in rots:
  a=r[128]@R0.T@np.array([1.,0.,0.]); axes.append(a/np.linalg.norm(a))
 bindR=[rotmat(q) for _,_,q,_ in sk["bindings"]]
 delta=np.zeros((nv,3),dtype=np.float64)
 for v in support:
  side=-1 if fr[v]>0 else 1; f=fr[v] if fr[v]>0 else fl[v]; aa=[]; bb=[]
  for pi,pm in enumerate(pmap):
   J=np.zeros((3,3))
   for j,(body,tr,q,scale) in enumerate(sk["bindings"]):
    w=float(sk["weights"][v,j])
    if w==0: continue
    R=rots[pi].get(int(body),rotmat(rig["core_poses"][int(body)][3:7]))
    J+=w*float(scale)*(R@bindR[j])
   aa.append(J); bb.append(side*AMP*f*axes[pi])
  sol,_,rank,_=np.linalg.lstsq(np.vstack(aa),np.concatenate(bb),rcond=None)
  assert rank==3 and np.isfinite(sol).all(); delta[v]=sol
 candv=np.asarray(sk["vertices"]+np.asarray(delta,dtype="<f4"),dtype="<f4")
 changed=np.flatnonzero(np.any(candv.view("<u4")!=np.asarray(sk["vertices"],dtype="<f4").view("<u4"),axis=1))
 maxsrc=float(np.linalg.norm(candv.astype(float)-sk["vertices"],axis=1).max()); assert maxsrc<=.0025
 # The two anatomical ocular loops remain bit-identical.
 receipt=json.loads((N/"resting-anatomy-receipt.json").read_text()); ocular=set()
 for row in receipt["mass_geometry_accounting"]["skin_boundary_repair"]["retained_anatomical_interfaces"]:
  if "ocular" in row.get("interface",""): ocular.update(map(int,row["source_vertex_ids"]))
 assert len(ocular)==115 and not ocular.intersection(map(int,changed))
 edited=bytearray(raw)
 for v in changed: struct.pack_into("<3f",edited,vo+int(v)*56,*map(float,candv[v]))
 cand=outdir/"candidate.nhskin"; cand.write_bytes(edited)
 csk=load_skin(cand,rig); assert np.array_equal(csk["weights"],sk["weights"]) and np.array_equal(np.asarray(csk["vertices"],dtype="<f4").view("<u4"),candv.view("<u4"))
 orig_rows=np.frombuffer(raw,dtype="<f4",count=14*nv,offset=vo).reshape(nv,14); cand_rows=np.frombuffer(edited,dtype="<f4",count=14*nv,offset=vo).reshape(nv,14); assert np.array_equal(orig_rows[:,3:].view("<u4"),cand_rows[:,3:].view("<u4")) and edited[vo+56*nv:]==raw[vo+56*nv:]
 assert sha(SRC)==SKIN_SHA
 inventory,_,_,_=ca._load_target_inventory(INV); allkeys=set(inventory)|{(51005,s) for s in range(151,165)}
 selected={k for k in allkeys if (k[0]==51005 and 151<=k[1]<=164) or (k[0]==51010 and 381<=k[1]<=397) or (k[0]==51011 and 464<=k[1]<=524)}
 assert len(selected)==92
 ref=np.unique(faces); compact=np.searchsorted(ref,faces); poses_result=[]; maxres=0.; maxworld=0.
 for pi,(step,rec,pack) in enumerate(records):
  xyz,surfs,_=ca._pack_surfaces(pack,selected|{SKIN_KEY}); sf=np.asarray(surfs[SKIN_KEY]["faces"],int); base=int(sf.min())
  assert base==int(rec["skin_source_mapping"]["pack_first_vertex"]) and np.array_equal(sf-base,faces)
  actual=np.asarray(xyz[base:base+nv],dtype="<f4"); basef=skin_forward(sk["vertices"],sk,pmap[pi],rig); candf=skin_forward(candv,csk,pmap[pi],rig)
  residual=np.linalg.norm(basef[changed].astype(float)-actual[changed].astype(float),axis=1); maxres=max(maxres,float(residual.max(initial=0)))
  moved=np.asarray(actual.astype(float)+(candf.astype(float)-basef.astype(float)),dtype="<f4")
  disp=np.linalg.norm(moved.astype(float)-actual.astype(float),axis=1); maxworld=max(maxworld,float(disp.max(initial=0)))
  sr=ca._exact_surface_records(moved[ref],compact); sa=ci._audit_pair(sr,sr,same_surface=True)
  tf={k:np.asarray(surfs[k]["faces"],dtype=np.int64) for k in sorted(selected)}
  ta=ca._target_intersection_audit(sr,tf,xyz); tcount=int(sum(v["count"] for v in ta.values()))
  nonzero={f"{k[0]}:{k[1]}":int(v["count"]) for k,v in ta.items() if v["count"]}
  poses_result.append({"step":step,"time_s":rec["accepted_time_s"],"native_pack_sha256":rec["pack_file_sha256"],
    "skin_exact_self_pairs":int(sa["count"]),"degenerate_faces":len(sa.get("degenerate_face_rows",[])),
    "skin_vs_92_targets_exact_pairs":tcount,"nonzero_targets":nonzero,
    "changed_vertices":len(changed),"max_forward_residual_on_changed_m":float(residual.max(initial=0)),
    "max_predicted_world_move_m":float(disp.max(initial=0))})
  print(json.dumps({"step":step,"self":sa["count"],"target":tcount,"max_delta_mm":float(disp.max(initial=0))*1000,"forward_residual_um":float(residual.max(initial=0))*1e6}),flush=True)
  del xyz,surfs,sr,ta,basef,candf,moved; gc.collect()
 payload={"schema":"numi.human.medial-thigh-skin-self-repair-candidate.v1","status":"review_only_candidate_pending_root_assessment_and_later_stress_gate",
  "qualification":"CPU exact Float32 predicates on predicted source-corrected skin geometry at eight accepted N18 native poses; no native replay, no install, no anatomical admission.",
  "changes":{"candidate_path":str(cand),"sha256":sha(cand),"changed_source_vertices":int(len(changed)),"changed_referenced_vertices":int(np.count_nonzero(np.isin(ref,changed))),
   "XYZ_only":True,"normals_bindings_weights_topology_byte_identical":True,"ocular_115_vertices_bit_identical":True,
   "target_seed_movement_mm_each_side":AMP*1000,"geodesic_support_radius_mm":RADIUS*1000,"falloff":"quintic C2 smoothstep","max_source_displacement_mm":maxsrc*1000,"max_world_displacement_mm":maxworld*1000},
  "witnesses":{"native_sha256":NATIVE_SHA,"skin_sha256":SKIN_SHA,"step_3807_faces":[49688,60436,60437],"step_4767_faces":[49760,49761,60478,60479,60515,60516],"right_seed_vertices":RIGHT.tolist(),"left_seed_vertices":LEFT.tolist()},
  "pose_gates":{"capture_count":len(poses_result),"all8_skin_self_exact_zero":all(x["skin_exact_self_pairs"]==0 and x["degenerate_faces"]==0 for x in poses_result),"all8_skin_vs_92_targets_exact_zero":all(x["skin_vs_92_targets_exact_pairs"]==0 for x in poses_result),"max_baseline_forward_residual_changed_vertices_um":maxres*1e6,"captures":poses_result},
  "later_stress_gate":{"status":"pending","reason":"No later accepted-pose cohort using this exact N18 payload; older 300/310-second runs use different anatomy/runtime inputs."},
  "guardrails":{"native_payload_modified":False,"native_launch":False,"gpu":False,"back412_bytes_changed":False,"root_pose_or_forces_changed":False,"source_skin_sha256_unchanged":sha(SRC)==SKIN_SHA},
  "pins":{"source_skin":sha(SRC),"rigid":sha(RIG),"native":sha(native),"receipt":sha(N/"resting-anatomy-receipt.json"),"clearance_module":sha(Path(ca.__file__)),"predicate_module":sha(Path(ci.__file__))}}
 report=outdir/"candidate-and-pose-prediction-report.json"; report.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({"candidate":str(cand),"sha256":sha(cand),"changed_vertices":len(changed),"source_mm":maxsrc*1000,"self_zero":payload["pose_gates"]["all8_skin_self_exact_zero"],"targets_zero":payload["pose_gates"]["all8_skin_vs_92_targets_exact_zero"],"max_forward_residual_um":maxres*1e6,"report":str(report)},sort_keys=True),flush=True)
 assert payload["pose_gates"]["all8_skin_self_exact_zero"] and payload["pose_gates"]["all8_skin_vs_92_targets_exact_zero"]
 assert sha(SRC)==SKIN_SHA and sha(native)==NATIVE_SHA
if __name__=="__main__": main()
