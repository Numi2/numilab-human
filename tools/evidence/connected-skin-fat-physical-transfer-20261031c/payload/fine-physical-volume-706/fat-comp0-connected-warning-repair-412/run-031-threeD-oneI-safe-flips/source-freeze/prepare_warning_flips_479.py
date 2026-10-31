#!/usr/bin/env python3
"""Build fixed-coordinate typed 2-to-2 D/I flips for three verified D warning segments and one I warning segment.
No admission; exact full-context audit and 697 owner load are separate receipts.
"""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import argparse, hashlib, importlib.util, itertools, json, sys, time
import numpy as np

H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
A_PATH=H/'fixed-boundary-quality-topology-196/prepare_current_earclip_adapter_024.py'
B_PATH=H/'fixed-boundary-quality-topology-196/run_fixed_boundary_dp_batch_197.py'
J_PATH=H/'audit_joint_native_apex_272_root.py'
U_KEYS=('quality_star_patch_union_face_source_layer','quality_star_patch_union_face_source_body','quality_star_patch_union_face_source_bone','quality_star_patch_union_face_source_face_id','quality_star_patch_union_face_source_original_id','quality_star_patch_union_face_source_identity0','quality_star_patch_union_face_source_identity1')
T_KEYS=('face_source_layer','face_source_body','face_source_bone','face_source_face_id','face_source_original_id','face_source_identity0','face_source_identity1')
A_KEYS=('face_ancestry_source_face_rows','face_ancestry_source_original_ids','face_ancestry_parent_face_rows')
OPS=(
 {'role':'D','id':'D-warning-1','edge_volume_rows':(135551,135485),'expected_opposite_volume_rows':(135550,135518)},
 {'role':'D','id':'D-warning-2','edge_volume_rows':(135832,136517),'expected_opposite_volume_rows':(136256,137500)},
 {'role':'D','id':'D-warning-4','edge_volume_rows':(139234,140575),'expected_opposite_volume_rows':(139233,140078)},
 {'role':'I','id':'I-warning-3','edge_volume_rows':(33206,32869),'expected_opposite_volume_rows':(32895,33270)},
)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path)
 if s is None or s.loader is None:raise RuntimeError(f'cannot import {path}')
 m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def load(path):
 with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def f32key(x):return np.ascontiguousarray(np.asarray(x,np.float32)).view(np.uint32).tobytes()
def directed_chain(faces):
 c=Counter()
 for tri in np.asarray(faces,np.int64):
  a,b,d=map(int,tri)
  for u,v in ((a,b),(b,d),(d,a)):c[(u,v)]+=1
 return {e:n-c.get((e[1],e[0]),0) for e,n in c.items() if n-c.get((e[1],e[0]),0)}
def oriented_flip(old):
 old=np.asarray(old,np.int64).reshape((2,3))
 common=sorted(set(map(int,old[0]))&set(map(int,old[1])))
 opp=sorted((set(map(int,old[0]))|set(map(int,old[1])))-set(common))
 if len(common)!=2 or len(opp)!=2:raise ValueError('old faces do not form one 2-2 quadrilateral')
 want=directed_chain(old);solutions={}
 def cyclic(t):
  q=tuple(map(int,t));return min(q,q[1:]+q[:1],q[2:]+q[:2])
 a,b=opp;u,v=common
 for x in itertools.permutations((a,b,u)):
  for y in itertools.permutations((a,b,v)):
   q=np.asarray((x,y),np.int64)
   if directed_chain(q)==want:solutions[tuple(sorted((cyclic(x),cyclic(y))))]=q
 if len(solutions)!=1:raise ValueError(f'expected one oriented replacement, found {len(solutions)}')
 return next(iter(solutions.values())),common,opp,want
def topology(faces):
 edges=Counter();same=0;nonmanifold=0
 for tri in np.asarray(faces,np.int64):
  a,b,c=map(int,tri)
  for u,v in ((a,b),(b,c),(c,a)):
   key=tuple(sorted((u,v)));edges[key]+=1
 if any(n>2 for n in edges.values()):nonmanifold=sum(n>2 for n in edges.values())
 directed=Counter()
 for tri in np.asarray(faces,np.int64):
  a,b,c=map(int,tri)
  for u,v in ((a,b),(b,c),(c,a)):directed[(u,v)]+=1
 for (u,v),n in directed.items():
  if u<v and edges[(u,v)]==2 and not (n==1 and directed[(v,u)]==1):same+=1
 return {'faces':int(len(faces)),'used_vertices':int(len(np.unique(faces))),'unique_edges':len(edges),'boundary_edges':sum(n==1 for n in edges.values()),'nonmanifold_edges':nonmanifold,'same_direction_edges':same}
def build_role(role,parent_path,parent,vol_points,vol_point_lookup,ops,out,A,J):
 p32=np.asarray(parent['points_world_m_f32'],np.float32)
 p64=np.asarray(parent['points_world_m'],np.float64)
 if p32.ndim!=2 or p32.shape[1]!=3 or not np.isfinite(p32).all():raise ValueError(f'{role}: malformed F32 points')
 if p64.shape!=p32.shape or not np.array_equal(p64.astype(np.float32).view(np.uint32),p32.view(np.uint32)):raise ValueError(f'{role}: F64/F32 point views disagree')
 if len({f32key(x) for x in p32})!=len(p32):raise ValueError(f'{role}: duplicate F32 coordinates')
 point_to_volume=[]
 for x in p32:
  row=vol_point_lookup.get(f32key(x))
  if row is None:raise ValueError(f'{role}: source point missing from current volume coordinate domain')
  point_to_volume.append(row)
 point_to_volume=np.asarray(point_to_volume,np.int64)
 f0=np.asarray(parent['faces'],np.int64);f=f0.copy()
 if f.ndim!=2 or f.shape[1]!=3 or np.any(f<0) or np.any(f>=len(p32)):raise ValueError(f'{role}: invalid faces')
 before=topology(f0);chain0=directed_chain(f0)
 ip=J.M.integer_points(p32)
 changed_rows=set();used_vertices=set();operation_docs=[];operation_parent_rows={}
 for op in ops:
  if op['role']!=role:continue
  edge_global=tuple(map(int,op['edge_volume_rows']))
  edge=[]
  for v in edge_global:
   if v<0 or v>=len(vol_points):raise ValueError(f'{op["id"]}: volume point row outside source')
   li={k:i for i,k in enumerate([f32key(x) for x in p32])}.get(f32key(vol_points[v]))
   if li is None:raise ValueError(f'{op["id"]}: edge point absent from {role} candidate')
   edge.append(li)
  u,v=edge
  adj=np.flatnonzero(np.sum(np.isin(f,np.asarray([u,v],np.int64)),axis=1)==2)
  if len(adj)!=2:raise ValueError(f'{op["id"]}: expected exactly two faces on edge {edge_global}, found {len(adj)}')
  rows=list(map(int,adj))
  if any(r in changed_rows for r in rows):raise ValueError(f'{op["id"]}: operation overlaps an earlier changed face')
  old=f[rows].copy();new,common,opposite,_=oriented_flip(old)
  if set(map(int,common))!=set(edge):raise ValueError(f'{op["id"]}: selected shared edge mismatch')
  opposite_global=tuple(sorted(int(point_to_volume[int(x)]) for x in opposite))
  if opposite_global!=tuple(sorted(map(int,op['expected_opposite_volume_rows']))):
   raise ValueError(f'{op["id"]}: opposite source vertices differ: {opposite_global}')
  touched=set(map(int,np.r_[old.ravel(),new.ravel()]))
  if used_vertices & touched:raise ValueError(f'{op["id"]}: flip patches overlap at a point')
  used_vertices|=touched
  old_chain=directed_chain(old);new_chain=directed_chain(new)
  if old_chain!=new_chain:raise ValueError(f'{op["id"]}: local oriented boundary changed')
  old_score=np.asarray(J.score(p32,old),float);new_score=np.asarray(J.score(p32,new),float)
  exact_reasons=[J.exact_terms(ip,x)[0] for x in new]
  if any(x is not None for x in exact_reasons):raise ValueError(f'{op["id"]}: exact joint apex rejection {exact_reasons}')
  if np.any(new_score<float(np.float32(1e-5))):raise ValueError(f'{op["id"]}: new surface face below current joint apex gate {new_score.tolist()}')
  f[rows]=new;changed_rows.update(rows);operation_parent_rows[op['id']]=rows
  union=sorted(set(tuple(map(int,q)) for r in rows for q in A.rows_union(parent,r)))
  ancestry=sorted(set(tuple(map(int,q)) for r in rows for q in A.rows_anc(parent,r)))
  if not union or not ancestry:raise ValueError(f'{op["id"]}: empty complete typed union or source ancestry')
  operation_docs.append({'operation':'oriented_surface_2_to_2_flip','operation_id':op['id'],'role':role,
   'input_candidate_path':str(parent_path),'input_candidate_sha256':sha(parent_path),
   'retired_parent_face_rows':rows,'new_output_face_rows':rows,
   'shared_edge_parent_point_rows':list(edge),'shared_edge_current_volume_point_rows':list(edge_global),
   'opposite_parent_point_rows':list(opposite),'opposite_current_volume_point_rows':list(opposite_global),
   'old_faces_parent_point_rows':old.tolist(),'new_faces_output_point_rows':new.tolist(),
   'old_faces_current_volume_point_rows':point_to_volume[old].tolist(),
   'new_faces_current_volume_point_rows':point_to_volume[new].tolist(),
   'complete_typed_parent_union':[list(q) for q in union],
   'source_face_ancestry_union':[list(q) for q in ancestry],
   'lineage_semantics':'complete patch-level union across both retired faces; not per-triangle overlap',
   'old_joint_apex_scores':old_score.tolist(),'new_joint_apex_scores':new_score.tolist(),
   'exact_new_joint_apex_reasons':exact_reasons,'all_source_point_coordinates_unchanged':True,
   'physical_admission':False,'native_steps':0})
 if not changed_rows:raise ValueError(f'{role}: no operations applied')
 if len({tuple(sorted(map(int,x))) for x in f})!=len(f):raise ValueError(f'{role}: duplicate face after flips')
 after=topology(f)
 if after!=before:raise ValueError(f'{role}: full topology counters changed: {before} -> {after}')
 if directed_chain(f)!=chain0:raise ValueError(f'{role}: global directed boundary changed')
 # Rebuild complete typed patch unions and source ancestry for every row.
 rows_union=[];rows_ancestry=[];op_for_row={r:opid for opid,rs in operation_parent_rows.items() for r in rs}
 for r in range(len(f)):
  if r not in op_for_row:
   rows_union.append(A.rows_union(parent,r));rows_ancestry.append(A.rows_anc(parent,r))
  else:
   opid=op_for_row[r];rs=operation_parent_rows[opid]
   rows_union.append(sorted(set(tuple(map(int,q)) for pr in rs for q in A.rows_union(parent,pr))))
   rows_ancestry.append(sorted(set(tuple(map(int,q)) for pr in rs for q in A.rows_anc(parent,pr))))
 if any(not x for x in rows_union) or any(not x for x in rows_ancestry):raise ValueError(f'{role}: output has empty typed/ancestry CSR row')
 uoff,udata=A.csr(rows_union,7);aoff,adata=A.csr(rows_ancestry,3)
 scalar=np.asarray([x[0] for x in rows_union],np.int64)
 fmap=np.arange(len(f),dtype=np.int64);fmap[np.asarray(sorted(changed_rows),np.int64)]=-1
 payload=dict(parent)
 payload['faces']=f
 payload['quality_star_point_output_to_input_row']=np.asarray(parent.get('quality_star_point_output_to_input_row',np.arange(len(p32))),np.int64).copy()
 qface=np.asarray(parent.get('quality_star_face_output_to_input_row',np.arange(len(f))),np.int64).copy();qface[np.asarray(sorted(changed_rows),np.int64)]=-1
 payload['quality_star_face_output_to_input_row']=qface
 payload['quality_star_patch_union_offsets']=uoff
 payload['face_ancestry_offsets']=aoff
 payload.update({k:scalar[:,j] for j,k in enumerate(T_KEYS)})
 payload.update({k:udata[:,j] for j,k in enumerate(U_KEYS)})
 payload.update({k:adata[:,j] for j,k in enumerate(A_KEYS)})
 payload['point_output_to_parent_rows']=np.arange(len(p32),dtype=np.int64)
 payload['face_output_to_parent_rows']=fmap
 payload['retired_parent_face_rows']=np.asarray(sorted(changed_rows),np.int64)
 payload['changed_parent_face_rows']=np.asarray(sorted(changed_rows),np.int64)
 payload['changed_face_rows']=np.asarray(sorted(changed_rows),np.int64)
 payload['source_current_candidate_path']=np.asarray([str(parent_path)])
 payload['source_current_candidate_sha256']=np.asarray([sha(parent_path)])
 payload['typed_parent_candidate_path']=np.asarray([str(parent_path)])
 payload['typed_parent_candidate_sha256']=np.asarray([sha(parent_path)])
 payload['parent_candidate_sha256']=np.asarray([sha(parent_path)])
 cand=out/f'candidate-{role}-warning-flips-typed.npz';np.savez_compressed(cand,**payload)
 npzmap=out/f'{role.lower()}-parent-row-maps.npz'
 np.savez_compressed(npzmap,point_output_to_parent_rows=np.arange(len(p32),dtype=np.int64),face_output_to_parent_rows=fmap,retired_parent_face_rows=np.asarray(sorted(changed_rows),np.int64),changed_parent_face_rows=np.asarray(sorted(changed_rows),np.int64))
 docs=[]
 for doc in operation_docs:
  path=out/f'{doc["operation_id"]}.json';path.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');docs.append({'path':str(path),'sha256':sha(path)})
 return cand,npzmap,operation_docs,docs,{'before':before,'after':after,'full_oriented_boundary_unchanged':True,'changed_face_rows':sorted(changed_rows),'operation_count':len(operation_docs),'point_coordinates_bitwise_unchanged':True}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--parent-d',required=True,type=Path);ap.add_argument('--parent-d-sha256',required=True);ap.add_argument('--parent-i',required=True,type=Path);ap.add_argument('--parent-i-sha256',required=True);ap.add_argument('--current-volume',required=True,type=Path);ap.add_argument('--current-volume-sha256',required=True);ap.add_argument('--current-o',required=True,type=Path);ap.add_argument('--current-o-sha256',required=True);ap.add_argument('--output',required=True,type=Path)
 a=ap.parse_args();t0=time.monotonic();out=a.output.resolve()
 if out.exists():raise FileExistsError(out)
 for path,digest in ((a.parent_d,a.parent_d_sha256),(a.parent_i,a.parent_i_sha256),(a.current_volume,a.current_volume_sha256),(a.current_o,a.current_o_sha256)):
  if sha(path)!=digest:raise ValueError(f'input SHA mismatch: {path}')
 out.mkdir(parents=True)
 D=load(a.parent_d);I=load(a.parent_i);V=load(a.current_volume);O=load(a.current_o)
 vp=np.asarray(V['points_world_m'],np.float32)
 if vp.ndim!=2 or vp.shape[1]!=3 or not np.isfinite(vp).all():raise ValueError('current volume F32 point table invalid')
 vl={}
 for i,x in enumerate(vp):
  k=f32key(x)
  if k in vl:raise ValueError('current volume contains duplicate F32 coordinate')
  vl[k]=i
 A=module('earclip024_warning476',A_PATH);J=module('joint272_warning476',J_PATH)
 dpath,dmap,dops,ddocs,dsum=build_role('D',a.parent_d.resolve(),D,vp,vl,OPS,out,A,J)
 ipath,imap,iops,idocs,isum=build_role('I',a.parent_i.resolve(),I,vp,vl,OPS,out,A,J)
 config={'schema':'h706.combined-current-exact-surface-diff-config.v1','baseline_candidates':{'D':{'path':str(a.parent_d.resolve()),'sha256':a.parent_d_sha256},'I':{'path':str(a.parent_i.resolve()),'sha256':a.parent_i_sha256},'O':{'path':str(a.current_o.resolve()),'sha256':a.current_o_sha256}},'candidate_surfaces':{'D':{'path':str(dpath),'sha256':sha(dpath)},'I':{'path':str(ipath),'sha256':sha(ipath)},'O':{'path':str(a.current_o.resolve()),'sha256':a.current_o_sha256}},'changed_roles':['D','I']}
 cp=out/'combined-audit-config.json';cp.write_text(json.dumps(config,indent=2,sort_keys=True)+'\n')
 report={'schema':'h706.current-target-tetgen-warning-surface-flips.v1','status':'PASS_TYPED_D_I_WARNING_FLIP_DERIVATIVE_LOCAL_ONLY','physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN','source_script':str(Path(__file__).resolve()),'source_script_sha256':sha(Path(__file__).resolve()),'parent_candidates':{'D':{'path':str(a.parent_d.resolve()),'sha256':a.parent_d_sha256},'I':{'path':str(a.parent_i.resolve()),'sha256':a.parent_i_sha256},'O':{'path':str(a.current_o.resolve()),'sha256':a.current_o_sha256}},'current_volume':{'path':str(a.current_volume.resolve()),'sha256':a.current_volume_sha256},'candidate_D':{'path':str(dpath),'sha256':sha(dpath),'maps_path':str(dmap),'maps_sha256':sha(dmap),'topology':dsum},'candidate_I':{'path':str(ipath),'sha256':sha(ipath),'maps_path':str(imap),'maps_sha256':sha(imap),'topology':isum},'operation_count':len(dops)+len(iops),'operations':ddocs+idocs,'expected_warning_surface_segments_removed':[{'role':'D','edge_volume_rows':[135551,135485]},{'role':'D','edge_volume_rows':[135832,136517]},{'role':'I','edge_volume_rows':[33206,32869]},{'role':'D','edge_volume_rows':[139234,140575]}],'boundary_steiner_edge_flip':'EXCLUDED_AFTER_EXACT_CONTEXT_COLLISION','full_context_config':{'path':str(cp),'sha256':sha(cp)},'full_OID_walls_all185_audit':'PENDING','actual_697_D_I_load':'PENDING','fat_and_skin_target_rebind':'PENDING','volume_boundary_rebuild':'PENDING','strict_native_thresholds_unchanged':True,'all_surface_points_fixed':True,'limitations':['The 2-to-2 flips change the inferred piecewise-linear D/I triangulation while preserving point coordinates and directed open/closed boundaries.','No full-context exact collision audit, actual 697 load, current Skin/Fat target rebind, volume transfer, mass mapping, or native qualification is claimed.'],'elapsed_s':time.monotonic()-t0}
 rp=out/'report.json';rp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':report['status'],'D':report['candidate_D'],'I':report['candidate_I'],'config':report['full_context_config'],'report':str(rp),'report_sha256':sha(rp),'elapsed_s':report['elapsed_s']},indent=2))
if __name__=='__main__':main()
