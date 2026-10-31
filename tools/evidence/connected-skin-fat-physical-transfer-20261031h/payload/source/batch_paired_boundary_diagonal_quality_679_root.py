"""Batch paired material edge-star diagonal flips for source-boundary triangle conditioning."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np
H=Path(__file__).resolve().parent
def imp(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
C=imp("boundary677C",H/"batch_connected_precise_anchor_quality_671_root.py");T=C.T;M=C.M;A=C.A;B=C.B;J=C.J
L=imp("boundary677L",H/"boundary_edge_flip_676_local.py")
E=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def pin(p):return {"path":str(p),"sha256":sha(p)}
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def cs(t,n):
 order=np.argsort(t.ravel(),kind="stable");off=np.r_[0,np.cumsum(np.bincount(t.ravel(),minlength=n))];return order,off
def inc(c,v,width):return np.unique(c[0][c[1][v]:c[1][v+1]]//width)
def overlap(c,vs,width=4):
 vv=sorted(map(int,vs),key=lambda x:int(c[1][x+1]-c[1][x]));rows=inc(c,vv[0],width)
 for v in vv[1:]:rows=np.intersect1d(rows,inc(c,v,width),assume_unique=True)
 return rows
def cyclic(f):
 f=tuple(map(int,f));return min(f,f[1:]+f[:1],f[2:]+f[:2])
def shape(p,f):
 q=p[f].astype(float);cross=np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1);ell2=np.maximum.reduce([np.sum((q[:,i]-q[:,j])**2,axis=1) for i,j in ((0,1),(0,2),(1,2))]);return cross/ell2
def score(p,t):
 d,q=M.measurements(p,t);r=np.minimum(np.minimum(q/1e-5,d/6e-18),1.);return int((~M.passes(d,q)).sum()),float(-np.log(np.maximum(r,1e-300)).sum())
def addface(counter,ff,sgn):
 key=tuple(sorted(map(int,ff)));sign=1 if cyclic(ff)==cyclic(key) else -1;counter[key]+=sgn*sign
 if counter[key]==0:del counter[key]
def typed(parent,faces,ops,dest,parent_pin):
 data=dict(parent);u=[];an=[];opidx=np.full(len(faces),-1,np.int64);indexes={}
 for oi,op in enumerate(ops):
  rows=op["source_face_rows"];un=B.union_tuples(parent,rows);aa=B.ancestry_union(parent,rows);assert un and aa
  for r in rows:assert r not in indexes;indexes[r]=(oi,un,aa)
 for row in range(len(faces)):
  if row in indexes:oi,un,aa=indexes[row];opidx[row]=oi;u.append(un);an.append(aa)
  else:u.append(A.rows_union(parent,row));an.append(A.rows_anc(parent,row))
 uo,ud=A.csr(u,7);ao,ad=A.csr(an,3);typ=np.asarray([x[0] for x in u],np.int64)
 faceid=parent["quality_star_face_output_to_input_row"].copy();faceid[opidx>=0]=-1
 data["source_input_vertex_quotient"]=np.arange(len(parent["points_world_m_f32"]),dtype=np.int64);data["source_collapse_operation_rows"]=np.full(len(faces),-1,np.int64)
 data.update(faces=faces,quality_star_face_output_to_input_row=faceid,quality_star_patch_union_offsets=uo,face_ancestry_offsets=ao,source_input_face_rows=np.arange(len(faces),dtype=np.int64),source_diagonal_flip_operation_rows=opidx)
 data.update({k:typ[:,j] for j,k in enumerate(A.T)});data.update({k:ud[:,j] for j,k in enumerate(A.U)});data.update({k:ad[:,j] for j,k in enumerate(A.A)})
 b0,t0=B.topology(parent["faces"]);b1,t1=B.topology(faces);assert np.array_equal(b0,b1)
 assert all(t0[k]==t1[k] for k in ("boundary_edges","nonmanifold_edges","same_direction_edges"));assert np.all(J.score(parent["points_world_m_f32"],faces)>=J.GATE)
 data["typed_parent_candidate_path"]=np.asarray([parent_pin["path"]]);data["typed_parent_candidate_sha256"]=np.asarray([parent_pin["sha256"]])
 np.savez_compressed(dest,**data)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--state",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--seconds",type=float,default=90);ap.add_argument("--allowed",type=Path);a=ap.parse_args();start=time.monotonic()
 st=json.loads(a.state.read_text());st=st.get("current_verified_state_20261031",st);out=a.output;out.mkdir(parents=True,exist_ok=False);allowed=None if not a.allowed else {tuple(x) for x in json.loads(a.allowed.read_text())}
 S={};V={};reject=Counter()
 for role,e in st["source_surfaces"].items():
  assert sha(e["path"])==e["sha256"];ar=load(e["path"]);S[role]={"a":ar,"p":ar["points_world_m_f32"],"f":ar["faces"].copy(),"ops":[],"parent":e}
 for vr in ("Skin","Fat"):
  e=st[vr+"_current"];assert sha(e["path"])==e["sha256"];ar=load(e["path"]);p=ar["points_world_m"];t=ar["tetrahedra"];d,q=M.measurements(p,t);assert np.all(d>0)
  bd=T.boundary(t);assert np.array_equal(T.boundary_identity(bd),T.boundary_identity(ar["desired_boundary_faces"]))
  V[vr]={"a":ar,"p":p,"t":t,"mat":ar["material_indices"],"csr":cs(t,len(p)),"lookup":{T.key(x):i for i,x in enumerate(p)},"used":set(),"ops":[],"bad":~M.passes(d,q),"boundary":{tuple(sorted(map(int,f))):f for f in bd},"volume":float(d.sum()/6),"absolute_delta":0.}
 masks=load(H/"current-back412-scope-597-root/run003/output-001/protected-back412-aabb-current-O558-I50-D264-rows.npz")
 for role in ("D","I","O"):
  ss=S[role];p=ss["p"];f=ss["f"];basef=f.copy();targets=("Skin","Fat") if role=="I" else ("Skin",) if role=="O" else ("Fat",);used=set();protected=set(map(int,masks[role+"_vertex_rows"]))
  maps={vr:np.asarray([V[vr]["lookup"].get(T.key(x),-1) for x in p],np.int64) for vr in targets};hits=np.zeros(len(p),np.int64)
  for vr in targets:
   h=np.bincount(V[vr]["t"][V[vr]["bad"]].ravel(),minlength=len(V[vr]["p"]));valid=maps[vr]>=0;hits[valid]+=h[maps[vr][valid]]
  ed=np.sort(basef[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1);order=np.lexsort((ed[:,1],ed[:,0]));se=ed[order];starts=np.r_[0,1+np.flatnonzero(np.any(se[1:]!=se[:-1],axis=1))];count=np.diff(np.r_[starts,len(se)]);starts=starts[count==2]
  rows=order[np.c_[starts,starts+1]]//3;edges=se[starts];rank=np.lexsort((edges[:,1],edges[:,0],-(hits[edges[:,0]]+hits[edges[:,1]])))
  sourceedges={tuple(map(int,x)) for x in se}
  for ei in rank:
   if time.monotonic()-start>a.seconds:break
   edge=tuple(map(int,edges[ei]));fr=rows[ei]
   if not (hits[edge[0]]+hits[edge[1]]):continue
   if allowed is not None and (role,*edge) not in allowed:continue
   oldf=basef[fr];quad=set(map(int,oldf.ravel()))
   if len(quad)!=4 or quad&used or quad&protected:reject[role+":dependent_or_back_quad"]+=1;continue
   aa,bb=edge
   for i in range(3):
    if {int(oldf[0,i]),int(oldf[0,(i+1)%3])}==set(edge):aa=int(oldf[0,i]);bb=int(oldf[0,(i+1)%3]);break
   cc=next(iter(set(map(int,oldf[0]))-set(edge)));dd=next(iter(set(map(int,oldf[1]))-set(edge)))
   if (min(cc,dd),max(cc,dd)) in sourceedges:reject[role+":source_new_diagonal_exists"]+=1;continue
   nf=np.asarray([[cc,dd,bb],[dd,cc,aa]],np.int64)
   oldshape=shape(p,oldf);newshape=shape(p,nf)
   if newshape.min()<=oldshape.min()*(1+1e-4):reject[role+":source_shape_no_improvement"]+=1;continue
   if np.any(J.score(p,nf)<J.GATE):reject[role+":native_source_apex"]+=1;continue
   xyz=p[oldf].astype(float);on=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]);normal=on.sum(axis=0);nnorm=np.linalg.norm(normal)
   if nnorm<=0:continue
   normal/=nnorm;newxyz=p[nf].astype(float);newnormal=np.cross(newxyz[:,1]-newxyz[:,0],newxyz[:,2]-newxyz[:,0]);span=float(np.ptp((p[sorted(quad)].astype(float)-p[aa].astype(float))@normal))
   if span>.0002 or np.any(on@normal<=1e-16) or np.any(newnormal@normal<=1e-16):reject[role+":nonplanar_or_projected_fold"]+=1;continue
   trials={};why=None;totalgain=0;totalenergy=0.;delta=0
   for vr in targets:
    v=V[vr];mp=maps[vr];qmap=mp[np.asarray([aa,bb,cc,dd])]
    if np.any(qmap<0):why="source_point_missing";break
    ar,br,cr,dr=map(int,qmap);rr=overlap(v["csr"],(ar,br));old=v["t"][rr];verts=np.unique(old)
    if len(rr)<2 or len(rr)>32 or set(map(int,verts))&v["used"]:why="empty_large_or_dependent_volume_star";break
    if len(np.unique(v["mat"][rr]))!=1:why="mixed_material";break
    lookup={int(x):i for i,x in enumerate(verts)};localold=np.asarray([[lookup[int(x)] for x in tet] for tet in old],np.int64);lp=v["p"][verts];vold=np.asarray([[mp[int(x)] for x in face] for face in oldf]);vn=np.asarray([[mp[int(x)] for x in face] for face in nf])
    actual=[]
    for ff in vold:
     got=v["boundary"].get(tuple(sorted(map(int,ff))))
     if got is None:why="source_not_material_boundary";break
     actual.append(got)
    if why:break
    localfaces=np.asarray([[lookup[int(x)] for x in ff] for ff in actual],np.int64)
    try:result=L.propose_boundary_flip(M,lp,localold,edge=(lookup[ar],lookup[br]),old_source_faces=localfaces)
    except (ValueError,KeyError) as ex:why="local:"+str(ex);break
    if not result.get("accepted"):why="local:"+result.get("reason","NO");break
    nt=verts[result["new_tets"]];oldchain=Counter(M.chain(old));expected=Counter(oldchain)
    actual=np.asarray(actual,np.int64)
    reverse=cyclic(actual[0])!=cyclic(vold[0]);reverse_second=cyclic(actual[1])!=cyclic(vold[1])
    if reverse_second!=reverse:why="inconsistent_source_material_orientation";break
    expectednew=vn[:,[0,2,1]] if reverse else vn
    for ff in actual:addface(expected,ff,-1)
    for ff in expectednew:addface(expected,ff,1)
    if dict(M.chain(nt))!=dict(expected):why="target_oriented_chain";break
    oldedges={tuple(sorted(map(int,x))) for x in old[:,E].reshape(-1,2)};newedges={tuple(sorted(map(int,x))) for x in nt[:,E].reshape(-1,2)};cavity=set(map(int,rr))
    if any(set(map(int,overlap(v["csr"],ed)))-cavity for ed in newedges-oldedges):why="external_new_edge";break
    oldfaces={tuple(sorted(map(int,x))) for x in old[:,M.PAT].reshape(-1,3)};newfaces={tuple(sorted(map(int,x))) for x in nt[:,M.PAT].reshape(-1,3)}
    if any(set(map(int,overlap(v["csr"],ff)))-cavity for ff in newfaces-oldfaces):why="external_new_face";break
    if any(set(map(int,overlap(v["csr"],tt)))-cavity for tt in nt):why="external_duplicate";break
    before,es0=score(v["p"],old);after,es1=score(v["p"],nt)
    if after>before:why="native_bad_count_increase";break
    nodes=np.unique(np.r_[old.ravel(),nt.ravel()]);li={int(x):i for i,x in enumerate(nodes)};ip=M.integer_points(v["p"][nodes]);od=sum(M.exact_det(ip,[li[int(x)] for x in tt]) for tt in old);nds=[M.exact_det(ip,[li[int(x)] for x in tt]) for tt in nt]
    if min(nds)<=0:why="exact_new_nonpositive";break
    dv=sum(nds)-od;delta+=dv;dm=float(dv)*2.**(-447)/6.
    if role!="I" and (abs(dv)*100>od or v["absolute_delta"]+abs(dm)>v["volume"]*1e-4):why="external_volume_budget";break
    totalgain+=before-after;totalenergy+=es0-es1
    trials[vr]={"old_rows":rr.tolist(),"new_tetrahedra":nt.tolist(),"old_source_faces":actual.tolist(),"new_source_faces":expectednew.tolist(),"bad_before":before,"bad_after":after,"deficit_before":es0,"deficit_after":es1,"delta_exact":str(dv),"delta_m3":dm,"vertices":verts.tolist()}
   if why:reject[role+":"+why]+=1;continue
   if role=="I" and delta!=0:reject[role+":combined_I_volume"]+=1;continue
   if totalgain<0 or (totalgain==0 and totalenergy<=1e-6):reject[role+":no_lexicographic_gain"]+=1;continue
   op={"source_edge":[aa,bb],"source_face_rows":fr.tolist(),"old_source_faces":oldf.tolist(),"new_source_faces":nf.tolist(),"minimum_source_shape_before":float(oldshape.min()),"minimum_source_shape_after":float(newshape.min()),"geometric_slab_width_m":span,"native_bad_gain":totalgain,"native_log_deficit_gain":totalenergy,"volume_trials":trials}
   ss["ops"].append(op);f[fr]=nf;used.update(quad)
   for vr,tr in trials.items():
    v=V[vr];v["used"].update(tr["vertices"]);v["ops"].append(tr);v["absolute_delta"]+=abs(tr["delta_m3"])
  print("SOURCE_DONE",role,len(ss["ops"]),time.monotonic()-start,flush=True)
 result=dict(st);sources={};operations={}
 for role,ss in S.items():
  dest=out/("candidate-"+role+"-typed-current.npz");typed(ss["a"],ss["f"],ss["ops"],dest,ss["parent"]);sources[role]=pin(dest);operations[role]=ss["ops"]
 for vr,v in V.items():
  old=v["a"];remove=[];new=[];nm=[];po=[0];parents=[];target=Counter(L._face_chain(old["desired_boundary_faces"]))
  for tr in v["ops"]:
   rr=np.asarray(tr["old_rows"],np.int64);nt=np.asarray(tr["new_tetrahedra"],np.int64);remove.extend(rr);new.append(nt);nm.extend([int(v["mat"][rr[0]])]*len(nt))
   for tt in nt:parents.extend(rr.tolist());po.append(len(parents))
   for ff in tr["old_source_faces"]:addface(target,ff,-1)
   for ff in tr["new_source_faces"]:addface(target,ff,1)
  keep=np.ones(len(v["t"]),bool);keep[remove]=False;retained=np.flatnonzero(keep);nt=np.vstack(new) if new else np.empty((0,4),np.int64);tt=np.vstack((v["t"][keep],nt));mm=np.r_[v["mat"][keep],np.asarray(nm,v["mat"].dtype)]
  bd=T.boundary(tt);assert dict(L._face_chain(bd))==dict(target);assert len(np.unique(np.sort(tt,axis=1),axis=0))==len(tt);dd,qq=M.measurements(v["p"],tt);bad=int((~M.passes(dd,qq)).sum());assert np.all(dd>0)
  origin=np.r_[retained,np.full(len(nt),-1,np.int64)];inverse=np.full(len(v["t"]),-1,np.int64);inverse[retained]=np.arange(len(retained))
  payload=dict(old)
  for key in list(payload):
   if key.startswith(("desired_boundary_source_","mapped_current_boundary_","proposed_current_desired_boundary_")):payload.pop(key)
  payload.update(input_node_to_output_node=np.arange(len(v["p"]),dtype=np.int64),output_node_to_input_node=np.arange(len(v["p"]),dtype=np.int64),output_point_to_input_point_row=np.arange(len(v["p"]),dtype=np.int64),output_point_parent_offsets=np.zeros(len(v["p"])+1,np.int64),output_point_parent_input_cell_rows=np.empty(0,np.int64),tetrahedra=tt,material_indices=mm,boundary_faces=bd,desired_boundary_faces=bd.copy(),current_geometry_parent_path=np.asarray([st[vr+"_current"]["path"]]),current_geometry_parent_sha256=np.asarray([st[vr+"_current"]["sha256"]]),output_cell_to_input_cell=origin,input_cell_to_output_cell=inverse,retained_input_cell_rows=retained,replacement_parent_offsets=np.asarray(po,np.int64),replacement_parent_cell_rows=np.asarray(parents,np.int64),replacement_tet_offsets=np.asarray([0,len(nt)],np.int64),replacement_tetrahedra=nt,canonical_parent_union_operation_order=np.asarray("paired-boundary-diagonal677"))
  for role,e in sources.items():payload["typed_"+role+"_candidate_path"]=np.asarray([e["path"]]);payload["typed_"+role+"_candidate_sha256"]=np.asarray([e["sha256"]])
  dest=out/(vr.lower()+"-boundary-diagonal-quality-volume.npz");np.savez_compressed(dest,**payload);result[vr+"_current"]={**st[vr+"_current"],**pin(dest),"cells":len(tt),"native_quality_failures":bad,"nonpositive":0,"absolute_geometric_volume_change_m3":v["absolute_delta"]}
  print("QUALITY",vr,bad,flush=True)
 result.update(source_surfaces=sources,operations=operations,source_state=pin(a.state),producer=pin(Path(__file__)),local_owner=pin(Path(L.__file__)),rejections=dict(reject),physical_admission=False,native_steps=0,boundary_source_binding_status="PENDING_METADATA_REBIND_NO_GEOMETRY_CHANGE",surface_crossings="NOT_YET_AUDITED",elapsed_s=time.monotonic()-start)
 for k in ("independent_replay","exact_context"):result.pop(k,None)
 (out/"state.json").write_text(json.dumps(result,indent=2)+"\n");print("STATE",str(out/"state.json"),flush=True)
if __name__=="__main__":main()
