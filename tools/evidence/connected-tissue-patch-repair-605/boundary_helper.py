#!/usr/bin/env python3
from pathlib import Path
import collections, hashlib, importlib.util, json, math, time
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001")
INPUT=ROOT/"run-009-actual-f32-witness-batch/candidate-f64.npz"
INPUT_REPORT=ROOT/"run-013-simultaneous-f32-class-repair/report.json"
OUT=ROOT/"run-022-guide-plane-star-remesh"
TOPO=ROOT/"contract_d_connected_f32_classes_rerun013.py"
EMITTER=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/emit_f32.py")

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def ear_clip(loop,P,guide):
 xyz=P[np.asarray(loop,np.int64)]
 centered=xyz-xyz.mean(0);gn=np.asarray(guide,np.float64)/np.linalg.norm(guide)
 u=centered[1].copy();u-=float(u@gn)*gn
 if np.linalg.norm(u)<=1e-15:
  axis=np.eye(3)[int(np.argmin(np.abs(gn)))]
  u=np.cross(axis,gn)
 u/=np.linalg.norm(u);v=np.cross(gn,u)
 uv=centered@np.stack((u,v),axis=1)
 scale=max(float(np.linalg.norm(uv.max(0)-uv.min(0))),1e-30)
 eps=np.finfo(float).eps*scale*scale*32
 signed=float(sum(uv[i,0]*uv[(i+1)%len(loop),1]-uv[(i+1)%len(loop),0]*uv[i,1] for i in range(len(loop)))*0.5)
 if abs(signed)<=eps:raise ValueError("projected boundary loop has zero area")
 orient=1.0 if signed>0 else -1.0
 def cr(a,b,c):return float((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
 def inside(p,a,b,c):
  x=orient*cr(a,b,p);y=orient*cr(b,c,p);z=orient*cr(c,a,p)
  return x>=-eps and y>=-eps and z>=-eps
 def quality(a,b,c):
  area2=orient*cr(uv[a],uv[b],uv[c])
  lens=[np.linalg.norm(uv[a]-uv[b]),np.linalg.norm(uv[b]-uv[c]),np.linalg.norm(uv[c]-uv[a])]
  return float((2*math.sqrt(3.0)*area2)/(sum(x*x for x in lens)+1e-300))
 def search(ids):
  if len(ids)==3:
   a,b,c=ids
   if orient*cr(uv[a],uv[b],uv[c])<=eps:return None
   n=np.cross(P[loop[b]]-P[loop[a]],P[loop[c]]-P[loop[a]])
   if float(n@guide)<=0:return None
   return [(loop[a],loop[b],loop[c])],[quality(a,b,c)]
  ears=[]
  for j,cur in enumerate(ids):
   prev=ids[(j-1)%len(ids)];nxt=ids[(j+1)%len(ids)]
   a,b,c=uv[prev],uv[cur],uv[nxt]
   area2=orient*cr(a,b,c)
   if area2<=eps:continue
   n=np.cross(P[loop[cur]]-P[loop[prev]],P[loop[nxt]]-P[loop[prev]])
   if float(n@guide)<=0:continue
   if any(inside(uv[k],a,b,c) for k in ids if k not in (prev,cur,nxt)):continue
   q=quality(prev,cur,nxt)
   ears.append((q,cur,j,prev,nxt))
  for q,cur,j,prev,nxt in sorted(ears,key=lambda x:(-x[0],x[1])):
   rest=ids[:j]+ids[j+1:]
   result=search(rest)
   if result is not None:
    faces,scores=result
    return [(loop[prev],loop[cur],loop[nxt])]+faces,[q]+scores
  return None
 result=search(tuple(range(len(loop))))
 if result is None:raise ValueError("no outward-wound boundary-preserving triangulation exists")
 out,scores=result
 return out,{"projected_signed_area_m2":signed,"projection_max_residual_m":float(np.max(np.abs(centered@gn))),"min_projected_triangle_quality":float(min(scores)),"outward_projection_normal":gn.tolist()}
def boundary_cycles(F,rows):
 ef=collections.defaultdict(list)
 for r in rows:
  a,b,c=map(int,F[r])
  for u,v in ((a,b),(b,c),(c,a)):ef[tuple(sorted((u,v)))].append((int(r),u,v))
 bd=[vals[0][1:] for vals in ef.values() if len(vals)==1]
 out=collections.defaultdict(list);inc=collections.defaultdict(list)
 for a,b in bd:out[a].append(b);inc[b].append(a)
 if not all(len(x)==1 for x in out.values()) or not all(len(x)==1 for x in inc.values()):
  raise ValueError("patch boundary directed degrees are not one-in/one-out")
 nxt={a:x[0] for a,x in out.items()};todo=set(nxt);cycles=[]
 while todo:
  s=min(todo);c=s;loop=[]
  for _ in range(len(nxt)+1):
   if c in loop:raise ValueError("patch boundary repeated vertex")
   loop.append(c);todo.discard(c);c=nxt[c]
   if c==s:break
  if c!=s:raise ValueError("patch boundary cycle did not close")
  cycles.append(loop)
 return cycles,bd
def edges(F):
 em=collections.defaultdict(list)
 for r,(a,b,c) in enumerate(F):
  for u,v in ((int(a),int(b)),(int(b),int(c)),(int(c),int(a))):em[tuple(sorted((u,v)))].append(r)
 return em
def main():
 start=time.monotonic();OUT.mkdir(parents=True,exist_ok=False)
 z=np.load(INPUT,allow_pickle=False);d={k:z[k] for k in z.files}
 P=np.asarray(d["points_world_m_f64"],np.float64);F=np.asarray(d["faces_f64"],np.int64)
 Q=P.astype(np.float32);b=Q.view(np.uint32)
 t=Q[F].astype(np.float64);cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);zero=np.all(cross==0,axis=1)
 collapsed=((b[F[:,0]]==b[F[:,1]]).all(1)|(b[F[:,1]]==b[F[:,2]]).all(1)|(b[F[:,2]]==b[F[:,0]]).all(1))
 witness=np.flatnonzero(zero|collapsed)
 reportclass=json.loads(INPUT_REPORT.read_text())
 classes=[set(map(int,x["members"])) for x in reportclass["class_decisions"] if x["status"]=="ACCEPTED"]
 incidence=[set() for _ in range(len(P))]
 for r,tr in enumerate(F):
  for v in tr:incidence[int(v)].add(r)
 seeds=classes+[set(map(int,F[r])) for r in np.flatnonzero(zero&~collapsed)]
 patchsets=[]
 for seed in seeds:
  rows=set().union(*(incidence[v] for v in seed))
  mergedseed=set(seed);remain=[]
  for q in patchsets:
   if rows&q["rows"]:rows|=q["rows"];mergedseed|=q["seeds"]
   else:remain.append(q)
  remain.append({"seeds":mergedseed,"rows":rows});patchsets=remain
 patchsets.sort(key=lambda x:min(x["seeds"]))
 registry={704434:(23,32),706662:(46,74),709339:(59,95),715855:(92,15)}
 attrs={k:np.asarray(d[k]) for k in ("face_cache_row","face_source_layer","face_source_body","face_source_bone","face_run_original_id","face_id","face_source_identity0","face_source_identity1")}
 off=np.asarray(d["face_ancestry_offsets"],np.int64);ar=np.asarray(d["face_ancestry_input_cache_rows"],np.int64)
 anc=[set(map(int,ar[off[i]:off[i+1]])) for i in range(len(F))]
 patch_reports=[];remove=set();new_faces=[];new_anc=[];new_attrs={k:[] for k in attrs};patch_volume_delta=0.0;boundary_before=set()
 for pi,patch in enumerate(patchsets):
  rows=sorted(patch["rows"])
  if remove.intersection(rows):raise ValueError("patch face sets overlap after connected merge")
  remove.update(rows)
  loops,bd=boundary_cycles(F,rows)
  if len(loops)!=1:raise ValueError(f"patch {pi} has {len(loops)} boundary cycles")
  loop=loops[0]
  guide=np.sum(np.cross(P[F[rows]][:,1]-P[F[rows]][:,0],P[F[rows]][:,2]-P[F[rows]][:,0]),axis=0)
  if np.linalg.norm(guide)==0:raise ValueError(f"patch {pi} has no stable outward guide")
  tri,planar=ear_clip(loop,P,guide)
  newf=np.asarray(tri,np.int64)
  # Exact boundary preservation and orientation on the new local patch.
  old_directed=collections.Counter((a,b0) for a,b0 in bd)
  ef=collections.defaultdict(list)
  for r,(a,b0,c) in enumerate(newf):
   for u,v in ((int(a),int(b0)),(int(b0),int(c)),(int(c),int(a))):ef[tuple(sorted((u,v)))].append((u,v))
  new_bd=collections.Counter(vals[0] for vals in ef.values() if len(vals)==1)
  if old_directed!=new_bd:raise ValueError(f"patch {pi} boundary orientation/edges changed")
  newn=np.cross(P[newf][:,1]-P[newf][:,0],P[newf][:,2]-P[newf][:,0])
  if np.any(np.einsum("ij,j->i",newn,guide)<=0):raise ValueError(f"patch {pi} triangulation reverses local outward orientation")
  norm=np.linalg.norm(newn,axis=1);maxedge=np.max(np.stack([np.linalg.norm(P[newf[:,1]]-P[newf[:,0]],axis=1),np.linalg.norm(P[newf[:,2]]-P[newf[:,1]],axis=1),np.linalg.norm(P[newf[:,0]]-P[newf[:,2]],axis=1)]),axis=0)
  altitude=norm/maxedge
  origins=sorted(set(int(attrs["face_run_original_id"][r]) for r in rows))
  cats=sorted(set((int(attrs["face_source_layer"][r]),int(attrs["face_source_body"][r]),int(attrs["face_source_bone"][r])) for r in rows))
  source_union=set().union(*(anc[r] for r in rows))
  for nf in newf:
   new_faces.append(nf);new_anc.append(set(source_union))
   layer=next(iter({c[0] for c in cats})) if len({c[0] for c in cats})==1 else -1
   body=next(iter({c[1] for c in cats})) if len({c[1] for c in cats})==1 else -1
   bonecat=next(iter({c[2] for c in cats})) if len({c[2] for c in cats})==1 else -1
   runid=origins[0] if len(origins)==1 else -1
   for k,v in (("face_cache_row",-1),("face_source_layer",layer),("face_source_body",body),("face_source_bone",bonecat),("face_run_original_id",runid),("face_id",-1),("face_source_identity0",-1),("face_source_identity1",-1)):new_attrs[k].append(v)
  triold=P[F[np.asarray(rows,np.int64)]];trinew=P[newf]
  volold=float(np.einsum("ij,ij->i",triold[:,0],np.cross(triold[:,1],triold[:,2])).sum()/6.0)
  volnew=float(np.einsum("ij,ij->i",trinew[:,0],np.cross(trinew[:,1],trinew[:,2])).sum()/6.0)
  patch_volume_delta+=volnew-volold
  boundary_before.update(tuple(sorted(e)) for e in (tuple(x) for x in old_directed))
  patch_reports.append({"patch":pi,"seed_vertices":sorted(map(int,patch["seeds"])),"removed_current_face_rows":rows,"removed_face_count":len(rows),"boundary_loop_vertex_rows":loop,"boundary_vertex_count":len(loop),"new_face_count":int(len(newf)),"source_run_ids":origins,"source_categories":[{"layer":c[0],"body":c[1],"bone":c[2]} for c in cats],"ancestry_union_source_face_count":len(source_union),"planarity":planar,"min_new_face_altitude_m":float(altitude.min()),"max_new_face_altitude_m":float(altitude.max()),"min_projected_quality":min(planar["min_projected_triangle_quality"],1.0),"signed_volume_delta_m3":volnew-volold})
 keep=np.ones(len(F),bool);keep[list(remove)]=False
 Fout=np.vstack((F[keep],np.asarray(new_faces,np.int64)))
 outrow=np.r_[np.flatnonzero(keep).astype(np.int64),np.full(len(new_faces),-1,np.int64)]
 ances=[anc[r] for r in np.flatnonzero(keep)]+new_anc
 outattrs={k:np.concatenate((np.asarray(v)[keep],np.asarray(new_attrs[k],dtype=np.asarray(v).dtype))) for k,v in attrs.items()}
 offsets=[0];arows=[]
 for a0 in ances:arows.extend(sorted(a0));offsets.append(len(arows))
 offsets=np.asarray(offsets,np.int64);arows=np.asarray(arows,np.int64)
 meta={}
 for row,oid,fid,i0,i1 in zip(np.asarray(d["face_ancestry_input_cache_rows"],np.int64),np.asarray(d["face_ancestry_original_ids"],np.int64),np.asarray(d["face_ancestry_face_ids"],np.int64),np.asarray(d["face_ancestry_identity0"],np.int64),np.asarray(d["face_ancestry_identity1"],np.int64)):
  meta[int(row)]=(int(oid),int(fid),int(i0),int(i1))
 if any(int(r) not in meta for r in arows):raise ValueError("ancestry metadata missing")
 aoid=np.asarray([meta[int(r)][0] for r in arows],np.int64);afid=np.asarray([meta[int(r)][1] for r in arows],np.int64);ai0=np.asarray([meta[int(r)][2] for r in arows],np.int64);ai1=np.asarray([meta[int(r)][3] for r in arows],np.int64)
 topo=loadmod("topology_tools",TOPO)
 topo64=topo.topology(P,Fout);loops64=topo.loops(Fout)
 postT=Q[Fout].astype(np.float64);postcr=np.cross(postT[:,1]-postT[:,0],postT[:,2]-postT[:,0]);zero32=np.flatnonzero(np.all(postcr==0.,axis=1))
 # Ensure source ancestry of all untouched faces is bitwise unchanged and each remesh face has full CSR.
 f64out=OUT/"candidate-f64.npz"
 patchmeta={"operation":"connected_vertex_star_boundary_preserving_retriangulation","patches":patch_reports,"removed_input_candidate_rows":sorted(remove),"input_witness_rows":list(map(int,witness)),"new_face_count":len(new_faces),"signed_volume_delta_total_m3":patch_volume_delta}
 packed=dict(points_world_m_f64=P,faces_f64=Fout,vertex_input_to_representative=d["vertex_input_to_representative"],face_cache_row=outattrs["face_cache_row"],face_source_layer=outattrs["face_source_layer"],face_source_body=outattrs["face_source_body"],face_source_bone=outattrs["face_source_bone"],face_run_original_id=outattrs["face_run_original_id"],face_id=outattrs["face_id"],face_source_identity0=outattrs["face_source_identity0"],face_source_identity1=outattrs["face_source_identity1"],face_ancestry_offsets=offsets,face_ancestry_input_cache_rows=arows,face_ancestry_original_ids=aoid,face_ancestry_face_ids=afid,face_ancestry_identity0=ai0,face_ancestry_identity1=ai1,source_points_f32=d["source_points_f32"],removed_exact_cap_cache_rows=d["removed_exact_cap_cache_rows"],source_cache_sha256=d["source_cache_sha256"],source_geometry_sha256=np.asarray([hashlib.sha256(np.ascontiguousarray(P).view(np.uint8).tobytes()+np.ascontiguousarray(Fout).view(np.uint8).tobytes()).hexdigest()]),inherited_prior_edge_collapse_json=d.get("inherited_prior_edge_collapse_json",np.asarray(["[]"])),prior_edge_collapse_json=d.get("prior_edge_collapse_json",d.get("edge_collapse_json",np.asarray(["[]"]))),edge_patch_remesh_json=np.asarray([json.dumps(patchmeta,separators=(",",":"))]))
 np.savez_compressed(f64out,**packed)
 structure=(topo64["component_count"]==1 and topo64["boundary_edge_count"]==45 and not any(int(k)>2 for k in topo64["edge_incidence"]) and topo64["components"][0]["euler"]==0 and loops64==[21,24] and not len(zero32))
 emission={"status":"NOT_RUN","reason":"F64 structural or Float32 area gate failed"};selfgate={"status":"NOT_RUN"};f32out=None;topo32=None;loops32=None
 if structure:
  emod=loadmod("numi_emit_f32",EMITTER)
  y,f32,ereport=emod.emit(P,Fout,np.asarray(d["source_points_f32"],np.float32))
  c32=np.cross(y[f32][:,1].astype(float)-y[f32][:,0].astype(float),y[f32][:,2].astype(float)-y[f32][:,0].astype(float))
  zeros=np.flatnonzero(np.all(c32==0.,axis=1))
  if not len(zeros):
   topo32=topo.topology(y.astype(float),f32);loops32=topo.loops(f32)
   f32out=OUT/"candidate-f32.npz"
   np.savez_compressed(f32out,points_world_m=y,faces=f32,face_cache_row=packed["face_cache_row"],face_source_layer=packed["face_source_layer"],face_source_body=packed["face_source_body"],face_source_bone=packed["face_source_bone"],face_run_original_id=packed["face_run_original_id"],face_id=packed["face_id"],face_source_identity0=packed["face_source_identity0"],face_source_identity1=packed["face_source_identity1"],face_ancestry_offsets=offsets,face_ancestry_input_cache_rows=arows,face_ancestry_original_ids=aoid,face_ancestry_face_ids=afid,face_ancestry_identity0=ai0,face_ancestry_identity1=ai1,vertex_input_old_to_emitted=np.asarray(ereport["input_vertex_old_to_emitted"],np.int64),vertex_emitted_input_parent_offsets=np.asarray(ereport["emitted_vertex_input_parent_offsets"],np.int64),vertex_emitted_input_parent_rows=np.asarray(ereport["emitted_vertex_input_parent_rows"],np.int64),vertex_emitted_source_point_offsets=np.asarray(ereport["emitted_vertex_source_point_offsets"],np.int64),vertex_emitted_source_point_rows=np.asarray(ereport["emitted_vertex_source_point_rows"],np.int64),float32_emission_json=np.asarray([json.dumps(ereport,separators=(",",":"))]))
   emission={"status":"PASS","path":str(f32out),"sha256":sha(f32out),"zero_area_faces":0}
 else:
  emission={"status":"FAIL_OR_NOT_RUN","zero_area_F32_faces_before_emission":int(len(zero32))}
 report={"input":str(INPUT),"input_sha256":sha(INPUT),"output_f64":str(f64out),"output_f64_sha256":sha(f64out),"output_f32":str(f32out) if f32out else None,"output_f32_sha256":sha(f32out) if f32out else None,"input_faces":int(len(F)),"output_faces":int(len(Fout)),"patches":patch_reports,"removed_current_face_rows":sorted(remove),"input_F32_witness_rows":list(map(int,witness)),"input_zero_area_count":int(zero.sum()),"input_collapsed_count":int(collapsed.sum()),"output_pre_emission_zero_area_F32_count":int(len(zero32)),"F64_topology":topo64,"F64_boundary_loops":loops64,"structural_F64_and_pre_emission_F32_gate":bool(structure),"F32_topology":topo32,"F32_boundary_loops":loops32,"F32_emission":emission,"full_D_exact_self_gate":{"status":"NOT_RUN_queued_after_I_topology_audit"},"signed_volume_delta_m3":patch_volume_delta,"elapsed_s":time.monotonic()-start,"admission":False}
 (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({k:v for k,v in report.items() if k!="patches"},indent=2))
if __name__=="__main__":main()

