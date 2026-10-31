"""Batch connected O/I/D plus volume quotients using existing exact predicates. Construction only."""
from pathlib import Path
import argparse,collections,hashlib,importlib.util,json,sys,time
from itertools import combinations
import numpy as np
H=Path(__file__).resolve().parent
def imp(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
T=imp("quality624T",H/"transfer_surface_contractions_cross_tissue_470_root.py");M=T.m
A=imp("quality624A",H/"fixed-boundary-quality-topology-196/prepare_current_earclip_adapter_024.py")
B=imp("quality624B",H/"fixed-boundary-quality-topology-196/run_fixed_boundary_dp_batch_197.py")
J=imp("quality624J",H/"audit_joint_native_apex_272_root.py")
U=imp("quality624U",H/"paired-I-local-cavity-527-root/run055-current-I50-D264-source-I-baseline-context/canonicalize_run055_current_parent_unions_v3.py")
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def pin(p):return {"path":str(p),"sha256":sha(p)}
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def csr(t,n):
 order=np.argsort(t.ravel(),kind="stable");off=np.r_[0,np.cumsum(np.bincount(t.ravel(),minlength=n))];return order,off
def incident(c,n,width):return np.unique(c[0][c[1][n]:c[1][n+1]]//width)
def link(t,vertices):
 z=set()
 for tet in t:
  rest=sorted(set(map(int,tet))-set(vertices))
  for k in range(len(rest)+1):z.update(combinations(rest,k))
 return z
def volume_link(v,u,w):
 ru=incident(v["csr"],u,4);rw=incident(v["csr"],w,4);edge=np.intersect1d(ru,rw)
 return len(edge)>0 and (link(v["base_t"][ru],(u,))&link(v["base_t"][rw],(w,)))==link(v["base_t"][edge],(u,w))
def typed(parent,sp,sf,ops,role,dest,parent_pin):
 mp=np.arange(len(sp));index={}
 for oi,op in enumerate(ops):
  mp[op["collapsed_input_point_row"]]=op["retained_input_point_row"]
  rows=op["retired_parent_face_rows"];un=B.union_tuples(parent,op["full_endpoint_face_rows"]);an=B.ancestry_union(parent,op["full_endpoint_face_rows"])
  assert un and an
  for row in rows:assert row not in index;index[row]=(oi,un,an)
 ff=mp[sf];keep=~np.any(np.diff(np.sort(ff,axis=1),axis=1)==0,axis=1);fm=np.flatnonzero(keep);ff=ff[keep]
 unions=[];ancestry=[];oi=np.full(len(ff),-1,np.int64)
 for i,k in enumerate(fm):
  if np.array_equal(ff[i],sf[k]):unions.append(A.rows_union(parent,int(k)));ancestry.append(A.rows_anc(parent,int(k)))
  else:
   oi[i],un,an=index[int(k)];unions.append(un);ancestry.append(an)
 uo,ud=A.csr(unions,7);ao,ad=A.csr(ancestry,3);typ=np.asarray([u[0] for u in unions],np.int64)
 data=dict(parent)
 for key in list(data):
  if key.startswith("source_") and key.endswith("_operation_rows"):data.pop(key)
 data["typed_parent_candidate_path"]=np.asarray([parent_pin["path"]]);data["typed_parent_candidate_sha256"]=np.asarray([parent_pin["sha256"]])
 faceid=parent["quality_star_face_output_to_input_row"][fm].copy();faceid[oi>=0]=-1
 data.update(faces=ff,quality_star_face_output_to_input_row=faceid,quality_star_patch_union_offsets=uo,face_ancestry_offsets=ao,source_input_face_rows=fm,source_input_vertex_quotient=mp,source_collapse_operation_rows=oi)
 data.update({k:typ[:,j] for j,k in enumerate(A.T)});data.update({k:ud[:,j] for j,k in enumerate(A.U)});data.update({k:ad[:,j] for j,k in enumerate(A.A)})
 b0,t0=B.topology(sf);b1,t1=B.topology(ff);assert np.array_equal(b0,b1)
 assert all(t0[k]==t1[k] for k in ("boundary_edges","nonmanifold_edges","same_direction_edges"))
 assert np.all(J.score(sp,ff)>=J.GATE)
 np.savez_compressed(dest,**data);return data
def cache_key(role,p,f):
 v=np.ascontiguousarray(p[np.asarray(f,np.int64)],dtype=np.float32).view(np.uint32)
 order=np.lexsort((v[:,2],v[:,1],v[:,0]))
 return role+":"+v[order].tobytes().hex()
def run(args):
 start=time.monotonic();state=json.loads(args.state.read_text());state=state.get("current_verified_state_20261031",state)
 out=args.output;out.mkdir(parents=True,exist_ok=False)
 V={};S={};skeys={}
 for role,e in state["source_surfaces"].items():
  assert sha(e["path"])==e["sha256"];a=load(e["path"]);S[role]={"a":a,"p":a["points_world_m_f32"],"f":a["faces"],"ops":[]}
  skeys[role]={T.key(a["points_world_m_f32"][x]) for x in np.unique(a["faces"])}
 for role in ("Skin","Fat"):
  e=state[role+"_current"];assert sha(e["path"])==e["sha256"];a=load(e["path"]);p=a["points_world_m"];t=a["tetrahedra"]
  d,q=M.measurements(p,t);assert np.all(d>0);bd=T.boundary(t);assert np.array_equal(T.boundary_identity(bd),T.boundary_identity(a["desired_boundary_faces"]))
  c=csr(t,len(p));look={T.key(x):i for i,x in enumerate(p)};assert len(look)==len(p)
  V[role]={"arrays":a,"path":Path(e["path"]),"p":p,"t":t.copy(),"base_t":t,"mat":a["material_indices"],"base_boundary":bd,"boundary_keys":{tuple(sorted(map(int,f))) for f in bd},"look":look,"csr":c,"node_incidence_order":c[0],"node_incidence_offsets":c[1],"node_incidence_added":{},"active":np.ones(len(t),bool),"map":np.arange(len(p)),"hits":np.bincount(t[~M.passes(d,q)].ravel(),minlength=len(p)),"used":set(),"ops":[],"bad_before":int((~M.passes(d,q)).sum()),"volume":float(d.sum()/6),"absolute_delta":0.}
 scope=H/"current-back412-scope-597-root/run003/output-001/protected-back412-aabb-current-O558-I50-D264-rows.npz"
 assert sha(scope)=="1abbf3c5887cfa507c303e845dafbb78d247e6b7a4893722a5d81ed202fd2676"
 old=load(scope);routes=json.loads(A.PROT.read_text())["surface_routes"]
 collision_cache=set(json.loads(args.collision_cache.read_text())) if args.collision_cache and args.collision_cache.exists() else set()
 allowed=None if not args.allowed else {tuple(x) for x in json.loads(args.allowed.read_text())}
 exclusions=set()
 if args.exclude:
  exclusions={tuple(x) for x in json.loads(args.exclude.read_text())}
 reject=collections.Counter();frontier=[]
 for role in ("D","I","O"):
  ss=S[role];p=ss["p"];f=ss["f"];sc=csr(f,len(p));protected=set(map(int,old[role+"_vertex_rows"]));back_protected=set(protected);anchors=set()
  es=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);e,count=np.unique(es,axis=0,return_counts=True)
  anchors.update(map(int,e[count==1].ravel()))
  for route in routes:
   if route["role"]==role:
    hit=np.flatnonzero(np.all(p==np.asarray(route["target_world_m_f32"],np.float32),axis=1));assert len(hit)==1;anchors.add(int(hit[0]))
  targets=("Skin","Fat") if role=="I" else ("Skin",) if role=="O" else ("Fat",)
  maps={r:np.asarray([V[r]["look"].get(T.key(x),-1) for x in p],np.int64) for r in targets}
  score=np.zeros(len(p),np.int64)
  for r in targets:
   hit=maps[r]>=0;score[hit]+=V[r]["hits"][maps[r][hit]]
  length=np.linalg.norm(p[e[:,0]].astype(float)-p[e[:,1]],axis=1)
  original_cap={"O":.002,"I":.006,"D":.006}[role];cap=.3
  take=(count==2)&(length<=cap)&((score[e[:,0]]+score[e[:,1]])>0)
  e=e[take];length=length[take];order=np.lexsort((e[:,1],e[:,0],length,-(score[e[:,0]]+score[e[:,1]])))
  used=set();tested=0
  for edge,ell in zip(e[order],length[order]):
   if time.monotonic()-start>args.seconds:break
   a,b=map(int,edge);tested+=1
   if (role,min(a,b),max(a,b)) in exclusions:reject[role+":retained_exact_witness_exclusion"]+=1;continue
   fr_a=incident(sc,a,3);fr_b=incident(sc,b,3);er=np.intersect1d(fr_a,fr_b)
   na=set(map(int,f[fr_a].ravel()))-{a};nb=set(map(int,f[fr_b].ravel()))-{b}
   if len(er)!=2 or na&nb!=set(map(int,f[er].ravel()))-{a,b}:reject[role+":surface_link"]+=1;continue
   for keep,drop in ((a,b),(b,a)):
    if allowed is not None and (role,keep,drop) not in allowed:continue
    fr=fr_a if drop==a else fr_b;full=np.union1d(fr_a,fr_b);vertices=set(map(int,f[full].ravel()))
    if vertices&back_protected or drop in anchors:reject[role+":back_or_retired_anchor"]+=1;continue
    if vertices&used:reject[role+":dependent_surface"]+=1;continue
    if any(T.key(p[drop]) in skeys[other] for other in S if other!=role):reject[role+":shared_role_vertex"]+=1;continue
    nf=np.where(f[fr]==drop,keep,f[fr]);deg=np.any(np.diff(np.sort(nf,axis=1),axis=1)==0,axis=1);nf=nf[~deg]
    if not len(nf):reject[role+":empty_surface"]+=1;continue
    oldxyz=p[f[fr]].astype(float);newxyz=p[nf].astype(float)
    no=np.cross(oldxyz[:,1]-oldxyz[:,0],oldxyz[:,2]-oldxyz[:,0])[~deg];nn=np.cross(newxyz[:,1]-newxyz[:,0],newxyz[:,2]-newxyz[:,0])
    den=np.linalg.norm(no,axis=1)*np.linalg.norm(nn,axis=1)
    normal_ok=bool(np.all(den>0) and np.all(np.einsum("ij,ij->i",no,nn)>=.9*den))
    if np.any(J.score(p,nf)<J.GATE):reject[role+":apex"]+=1;continue
    planar_bound=None
    if ell>original_cap or not normal_ok:
     old_full=p[f[fr]].astype(float)
     old_norm=np.cross(old_full[:,1]-old_full[:,0],old_full[:,2]-old_full[:,0])
     normal=old_norm.sum(axis=0);normal_length=np.linalg.norm(normal)
     if normal_length<=0:reject[role+":no_planar_normal"]+=1;continue
     normal/=normal_length
     points_local=p[np.unique(f[fr])].astype(float);height=(points_local-p[drop].astype(float))@normal
     span=float(np.ptp(height))
     if span>.0002 or np.any(old_norm@normal<=1e-16) or np.any(nn@normal<=1e-16):reject[role+":not_bounded_planar_graph"]+=1;continue
     planar_bound={"common_slab_width_m":span,"maximum_surface_separation_bound_m":span,"positive_projected_faces":True,"fixed_projected_boundary":"endpoint fan quotient","per_face_normal_construction_check_replaced":not normal_ok}
    if collision_cache and any(cache_key(role,p,ff) in collision_cache for ff in nf):reject[role+":cached_exact_static_bone_witness"]+=1;continue
    op={"retained_input_point_row":keep,"collapsed_input_point_row":drop,"retired_parent_face_rows":fr.tolist(),"full_endpoint_face_rows":full.tolist(),"edge_length_m":float(ell),"extended_edge_planar_bound":planar_bound}
    trials={};why=None;delta=0;role_delta={}
    for vr in targets:
     v=V[vr]
     trial,reason=T.try_op(v,p,f,op,allow_quality_exchange=True)
     if trial is None:why=vr+":"+reason;break
     u,w=trial["kept_volume_node"],trial["retired_volume_node"];both=np.union1d(incident(v["csr"],u,4),incident(v["csr"],w,4))
     if set(map(int,v["base_t"][both].ravel()))&v["used"]:why=vr+":dependent_fullstar";break
     if not volume_link(v,u,w):why=vr+":volume_link";break
     # Same exact Float32 dyadic determinant owner, using only this local cavity.
     nodes=np.unique(np.r_[trial["old"].ravel(),trial["new"][~trial["deleted"]].ravel()]);idx={int(x):i for i,x in enumerate(nodes)};ip=M.integer_points(v["p"][nodes])
     ds0=sum(M.exact_det(ip,[idx[int(x)] for x in tet]) for tet in trial["old"])
     ds1=sum(M.exact_det(ip,[idx[int(x)] for x in tet]) for tet in trial["new"][~trial["deleted"]])
     dv=ds1-ds0;delta+=dv;dm=float(dv)*2.**(-447)/6.
     if role!="I" and (abs(dv)*100>ds0 or v["absolute_delta"]+abs(dm)>v["volume"]*1e-4):why=vr+":geometric_volume_budget";break
     trial["fullstars"]=both;trial["exact_delta"]=str(dv);trial["delta_m3"]=dm;role_delta[vr]=dv;trials[vr]=trial
    if why:reject[role+":"+why]+=1;continue
    if role=="I" and delta!=0:reject[role+":combined_exact_volume"]+=1;continue
    gain=sum(t["bad_before"]-t["bad_after"] for t in trials.values())
    if gain<=0:reject[role+":no_strict_quality_gain"]+=1;continue
    op["native_quality_gain"]=gain;op["volume_trials"]={}
    for vr,tr in trials.items():
     v=V[vr];v["used"].update(map(int,v["base_t"][tr["fullstars"]].ravel()))
     v["t"][tr["rows"]]=tr["new"];v["active"][tr["rows"][tr["deleted"]]]=False;v["map"][tr["retired_volume_node"]]=tr["kept_volume_node"]
     v["node_incidence_added"].setdefault(tr["kept_volume_node"],set()).update(map(int,tr["rows"][~tr["deleted"]]))
     v["absolute_delta"]+=abs(tr["delta_m3"])
     item={"role":role,"operation":len(ss["ops"]),**{k:tr[k] for k in ("kept_volume_node","retired_volume_node","bad_before","bad_after","new_retained_quality_failures","exact_delta","delta_m3")}, "source_cell_rows":tr["rows"].tolist(),"full_endpoint_star_rows":tr["fullstars"].tolist()}
     v["ops"].append(item);op["volume_trials"][vr]=item
    used.update(vertices);ss["ops"].append(op);break
  frontier.append({"role":role,"candidate_edges":len(e),"tested":tested,"complete":tested==len(e),"operations":len(ss["ops"])})
  print(json.dumps({"role":role,"operations":len(ss["ops"]),"frontier":frontier[-1],"elapsed_s":time.monotonic()-start}),flush=True)
 surfaces={}
 for role,ss in S.items():
  if not ss["ops"]:surfaces[role]=state["source_surfaces"][role];continue
  dest=out/("candidate-"+role+"-typed-current.npz");data=typed(ss["a"],ss["p"],ss["f"],ss["ops"],role,dest,state["source_surfaces"][role])
  surfaces[role]=pin(dest)
 report={"schema":"h706.combined-current-exact-surface-diff-config.v1","source_surfaces":surfaces,"source_state":pin(args.state),"physical_admission":False,"native_steps":0,"frontier":frontier,"rejections":dict(reject),"operations":{r:ss["ops"] for r,ss in S.items()},"producer":pin(Path(__file__)),"elapsed_s":time.monotonic()-start}
 for vr,v in V.items():
  oldt=v["base_t"];origin=np.flatnonzero(v["active"]);t=v["t"][origin];q=v["map"];bd=T.boundary(t);target=q[v["arrays"]["desired_boundary_faces"]]
  target=target[~np.any(np.diff(np.sort(target,axis=1),axis=1)==0,axis=1)]
  assert np.array_equal(T.boundary_identity(target),T.boundary_identity(bd));assert len(np.unique(np.sort(t,axis=1),axis=0))==len(t)
  d,qual=M.measurements(v["p"],t);assert np.all(d>0);bad=int((~M.passes(d,qual)).sum())
  assert bad==v["bad_before"]-sum(o["bad_before"]-o["bad_after"] for o in v["ops"])
  changed=np.any(t!=oldt[origin],axis=1);changed_out=np.flatnonzero(changed)
  stars=[np.asarray(o["full_endpoint_star_rows"],np.int64) for o in v["ops"]]
  _,po,parents,ng,ncg,total=U.make_parent_unions(len(oldt),stars,origin[changed])
  oi=origin.copy();oi[changed]=-1;io=np.full(len(oldt),-1,np.int64);io[origin[~changed]]=np.flatnonzero(~changed)
  payload=dict(v["arrays"])
  for key in list(payload):
   if key.startswith(("desired_boundary_source_","mapped_current_boundary_","proposed_current_desired_boundary_")):payload.pop(key)
  payload.update(tetrahedra=t,material_indices=v["mat"][origin],boundary_faces=bd,desired_boundary_faces=target,current_geometry_parent_path=np.asarray([str(v["path"])]),current_geometry_parent_sha256=np.asarray([sha(v["path"])]),output_cell_to_input_cell=oi,input_cell_to_output_cell=io,retained_input_cell_rows=origin[~changed],replacement_parent_offsets=po,replacement_parent_cell_rows=parents,replacement_tet_offsets=np.asarray([0,len(changed_out)],np.int64),replacement_tetrahedra=t[changed],input_node_to_output_node=q,output_node_to_input_node=np.arange(len(v["p"]),dtype=np.int64),output_point_to_input_point_row=np.arange(len(v["p"]),dtype=np.int64),output_point_parent_offsets=np.zeros(len(v["p"])+1,np.int64),output_point_parent_input_cell_rows=np.empty(0,np.int64),canonical_parent_union_schema=np.asarray("numi.run055.current-immediate-parent-unions.v1"),canonical_parent_union_operation_order=np.asarray("all-quality624-connected-OID"))
  for role,e in surfaces.items():
   payload["typed_"+role+"_candidate_path"]=np.asarray([e["path"]]);payload["typed_"+role+"_candidate_sha256"]=np.asarray([e["sha256"]])
  payload.pop("source_cell_operation_rows",None)
  for key in ("source_I_vertex_quotient","source_O_vertex_quotient","source_interface_collapse_rows"):payload.pop(key,None)
  dest=out/(vr.lower()+"-connected-quality-volume-proposal.npz");np.savez_compressed(dest,**payload)
  report[vr+"_current"]={**pin(dest),"nodes":len(v["p"]),"cells":len(t),"native_quality_failures":bad,"nonpositive":0,"missing_target_faces":0,"extra_actual_faces":0,"absolute_geometric_volume_change_m3":v["absolute_delta"],"operations":len(v["ops"])}
  print(json.dumps({vr:report[vr+"_current"]}),flush=True)
 report["boundary_source_binding_status"]="PENDING_METADATA_REBIND_NO_GEOMETRY_CHANGE";report["bounded_graph_edge_cap_m"]=.3;report["source_metadata_current_step_only"]=True
 report["surface_crossings"]="NOT_YET_AUDITED";report["elapsed_s"]=time.monotonic()-start
 (out/"state.json").write_text(json.dumps(report,indent=2)+"\n")
 print("STATE",str(out/"state.json"),flush=True)
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--state",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--seconds",type=float,default=180);ap.add_argument("--exclude",type=Path);ap.add_argument("--allowed",type=Path);ap.add_argument("--collision-cache",type=Path);run(ap.parse_args())
