#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,time
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
P=H/'fixed-boundary-quality-topology-196'; ROOT=P/'earclip-current-024-run001'
PARENT=P/'current-typed-adapter-023-run001'
OUT=P/'current-typed-adapter-024-run001'
REPORT=ROOT/'report.json'
PROT=H/'verified-volume-preparation-010-merged038005/rim/protected149-current-routes.json'
ROLES=('O','I','D')
T=('face_source_layer','face_source_body','face_source_bone','face_source_face_id','face_source_original_id','face_source_identity0','face_source_identity1')
U=('quality_star_patch_union_face_source_layer','quality_star_patch_union_face_source_body','quality_star_patch_union_face_source_bone','quality_star_patch_union_face_source_face_id','quality_star_patch_union_face_source_original_id','quality_star_patch_union_face_source_identity0','quality_star_patch_union_face_source_identity1')
RU=('new_face_patch_union_layer','new_face_patch_union_body','new_face_patch_union_bone','new_face_patch_union_face_id','new_face_patch_union_original_id','new_face_patch_union_identity0','new_face_patch_union_identity1')
A=('face_ancestry_source_face_rows','face_ancestry_source_original_ids','face_ancestry_parent_face_rows')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def rows_union(x,i):
 o=x['quality_star_patch_union_offsets'];lo,hi=map(int,o[i:i+2])
 if hi>lo:return sorted(set(tuple(int(x[k][j]) for k in U) for j in range(lo,hi)))
 return [tuple(int(x[k][i]) for k in T)]
def rows_anc(x,i):
 o=x['face_ancestry_offsets'];lo,hi=map(int,o[i:i+2])
 return sorted(set(tuple(int(x[k][j]) for k in A) for j in range(lo,hi)))
def csr(rows,width):
 off=np.zeros(len(rows)+1,np.int64);flat=[]
 for i,r in enumerate(rows):flat.extend(r);off[i+1]=len(flat)
 return off,np.asarray(flat,np.int64).reshape((-1,width)) if flat else np.empty((0,width),np.int64)
def qbound(p,f):
 t=np.asarray(p,np.float32)[np.asarray(f,np.int64)].astype(np.float64);a=t[:,1]-t[:,0];b=t[:,2]-t[:,0];c=t[:,2]-t[:,1]
 ar=np.linalg.norm(np.cross(a,b),axis=1);long=np.maximum.reduce((np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1),np.linalg.norm(c,axis=1)))
 return np.divide(ar,long*long,out=np.zeros_like(ar),where=long>0)
def main():
 OUT.mkdir(parents=True,exist_ok=False);t0=time.monotonic()
 prc=PARENT/'receipt.json'; rr=json.loads(prc.read_text()); rep=json.loads(REPORT.read_text())
 if rr.get('status')!='PASS_CURRENT_ROW_TYPED_ANCESTRY_ADAPTER_ONLY':raise ValueError('parent typed adapter not PASS')
 if sha(prc)!='aa589940b583fb54cfa1bb9c5fb7e25bbcc79e12392af1505b265c0e3a7e68e7':raise ValueError('parent adapter receipt hash mismatch')
 if sha(REPORT)!='d4fec402b5dbbd600d2f5bf4b01fb678b1ecbd904fcf8939ee89dcc67e7f0f7d':raise ValueError('DP report hash mismatch')
 gate=rep.get('joint_changed_faces_vs_complete_OID_walls_all185',rep.get('joint_changed_vs_complete_OID_walls_all185',{}))
 if gate.get('pair_count')!=0:raise ValueError('current changed-vs-complete exact gate not zero')
 if gate.get('joint_new_face_self_pairs',gate.get('new_face_self_pairs'))!=0:raise ValueError('current patch self gate not zero')
 output={}
 for role in ROLES:
  parent_rec=rr['candidate_roles'][role]; pp=Path(parent_rec['path'])
  if sha(pp)!=parent_rec['sha256']:raise ValueError('parent candidate adapter hash mismatch '+role)
  parent=load(pp); rawrec=rep['outputs'][role]; cp=Path(rawrec['path'])
  if sha(cp)!=rawrec['sha256']:raise ValueError('DP output hash mismatch '+role)
  cur=load(cp); nf=len(cur['faces']); PF=np.asarray(parent['faces'],np.int64); F=np.asarray(cur['faces'],np.int64)
  if not np.array_equal(np.asarray(parent['points_world_m'],np.float64),np.asarray(cur['points_world_m'],np.float64)):raise ValueError('DP point coordinates changed '+role)
  if not np.array_equal(np.asarray(parent['points_world_m_f32'],np.float32),np.asarray(cur['points_world_m_f32'],np.float32)):raise ValueError('DP F32 point coordinates changed '+role)
  if str(cur['immediate_parent_candidate_sha256'].reshape(-1)[0])!=parent_rec['raw_current_sha256']:raise ValueError('DP immediate raw parent pin mismatch '+role)
  fm=np.asarray(cur['face_output_to_parent_rows'],np.int64);pm=np.asarray(cur['point_output_to_parent_rows'],np.int64)
  if len(fm)!=nf or np.any((fm>=len(PF))):raise ValueError('face parent map out of range '+role)
  if np.any((pm>=len(parent['points_world_m']))):raise ValueError('point parent map out of range '+role)
  ops={}
  for s in rep.get('accepted_operations',[]):
   if s.get('role')!=role:continue
   opath=Path(s['operation_json_path'])
   if sha(opath)!=s['operation_json_sha256']:raise ValueError('operation json hash mismatch')
   op=json.loads(opath.read_text());un=sorted(set(tuple(map(int,t)) for t in op['complete_typed_parent_union']));anc=sorted(set(tuple(map(int,t)) for t in op.get('source_face_ancestry_union',[])))
   if not un:raise ValueError('empty operation union')
   for r in map(int,op['new_output_face_rows']):
    if r in ops or fm[r]>=0:raise ValueError('duplicate or mapped operation row')
    ops[r]={'union':un,'anc':anc,'op_path':str(opath),'op_sha':s['operation_json_sha256']}
  actual_new=set(map(int,np.flatnonzero(fm<0)))
  if actual_new!=set(ops):raise ValueError('new faces are not covered exactly by accepted operation docs')
  nt=np.asarray(cur['new_faces_parent_point_rows'],np.int64).reshape((-1,3)); uoff=np.asarray(cur['new_face_patch_union_offsets'],np.int64)
  if len(uoff)!=len(nt)+1 or int(uoff[-1])!=len(cur[RU[0]]):raise ValueError('bad output raw union offsets')
  for k in RU:
   if k not in cur:raise ValueError('missing raw union member '+k)
  new_by_tri={}
  for j,t in enumerate(nt):
   key=tuple(map(int,t)); lo,hi=map(int,uoff[j:j+2])
   vals=sorted(set(tuple(int(cur[k][q]) for k in RU) for q in range(lo,hi)))
   if key in new_by_tri:raise ValueError('duplicate generated parent triangle')
   new_by_tri[key]=vals
  union_rows=[];anc_rows=[]
  for i,f in enumerate(F):
   parentrow=int(fm[i])
   if parentrow>=0:
    mapped=tuple(map(int,pm[f]))
    if min(mapped)<0 or mapped!=tuple(map(int,PF[parentrow])):raise ValueError('retained output face not exact oriented parent '+role)
    union_rows.append(rows_union(parent,parentrow));anc_rows.append(rows_anc(parent,parentrow))
   else:
    if i not in ops:raise ValueError('unregistered generated output face')
    tri=tuple(map(int,pm[f]))
    if min(tri)<0:raise ValueError('generated face has point without parent')
    vals=new_by_tri.get(tri)
    if vals is None or vals!=ops[i]['union']:raise ValueError('generated raw union differs from operation union')
    if 'new_faces_input_point_rows' in cur:
     hit=np.flatnonzero(np.all(nt==np.asarray(tri,np.int64)[None,:],axis=1))
     if len(hit)!=1:raise ValueError('new point triangle not unique')
     j=int(hit[0]);src=np.asarray(cur['point_output_to_input_rows'],np.int64)[f]
     if not np.array_equal(src,np.asarray(cur['new_faces_input_point_rows'],np.int64)[j]):raise ValueError('source015 point triple mismatch')
    union_rows.append(ops[i]['union']);anc_rows.append(ops[i]['anc'])
  if any(not u for u in union_rows):raise ValueError('empty current face typed ancestry')
  uoff2,ud=csr(union_rows,7);aoff,ad=csr(anc_rows,3);rep_t=np.asarray([u[0] for u in union_rows],np.int64)
  out=OUT/('candidate-'+role+'-typed-current-024.npz')
  pmap=np.asarray(cur['point_output_to_input_rows'],np.int64);fmap=np.asarray(cur['face_output_to_input_rows'],np.int64)
  if not np.array_equal(pmap,np.asarray(parent['quality_star_point_output_to_input_row'],np.int64)):raise ValueError('source015 point map changed unexpectedly')
  expected=np.full(nf,-1,np.int64); keep=fm>=0;expected[keep]=np.asarray(parent['quality_star_face_output_to_input_row'],np.int64)[fm[keep]]
  if not np.array_equal(fmap,expected):raise ValueError('source015 face map does not compose through immediate parent')
  np.savez_compressed(out,points_world_m=np.asarray(cur['points_world_m'],np.float64),points_world_m_f32=np.asarray(cur['points_world_m_f32'],np.float32),faces=F,
   quality_star_point_output_to_input_row=pmap,quality_star_face_output_to_input_row=fmap,
   face_source_layer=rep_t[:,0],face_source_body=rep_t[:,1],face_source_bone=rep_t[:,2],face_source_face_id=rep_t[:,3],face_source_original_id=rep_t[:,4],face_source_identity0=rep_t[:,5],face_source_identity1=rep_t[:,6],
   quality_star_patch_union_offsets=uoff2,quality_star_patch_union_face_source_layer=ud[:,0],quality_star_patch_union_face_source_body=ud[:,1],quality_star_patch_union_face_source_bone=ud[:,2],quality_star_patch_union_face_source_face_id=ud[:,3],quality_star_patch_union_face_source_original_id=ud[:,4],quality_star_patch_union_face_source_identity0=ud[:,5],quality_star_patch_union_face_source_identity1=ud[:,6],
   face_ancestry_offsets=aoff,face_ancestry_source_face_rows=ad[:,0],face_ancestry_source_original_ids=ad[:,1],face_ancestry_parent_face_rows=ad[:,2],
   source_current_candidate_path=np.asarray([str(cp)]),source_current_candidate_sha256=np.asarray([rawrec['sha256']]),typed_parent_candidate_path=np.asarray([parent_rec['path']]),typed_parent_candidate_sha256=np.asarray([parent_rec['sha256']]))
  output[role]={'path':str(out),'sha256':sha(out),'raw_current_path':str(cp),'raw_current_sha256':rawrec['sha256'],'parent_adapter_path':str(pp),'parent_adapter_sha256':parent_rec['sha256'],'points':int(len(cur['points_world_m'])),'faces':nf,'generated_faces':len(ops),'typed_union_tuples':int(len(ud)),'ancestry_tuples':int(len(ad))}
 # Current qbound witnesses on the new exact output surfaces.
 wr=[];rrr=[];rat=[]
 for role in ROLES:
  z=load(output[role]['path']);q=qbound(z['points_world_m_f32'],z['faces']);bad=np.flatnonzero(q<1e-5);wr.extend([role]*len(bad));rrr.extend(map(int,bad));rat.extend(map(float,q[bad]))
 protected={r:[] for r in ROLES}; pr=json.loads(PROT.read_text())
 for x in pr['surface_routes']:
  role=x['role'];z=load(output[role]['path']);target=np.asarray(x['target_world_m_f32'],np.float32);hit=np.flatnonzero(np.all(z['points_world_m_f32']==target[None,:],axis=1))
  if len(hit)!=1:raise ValueError('protected route mapping mismatch '+role)
  protected[role].append(int(hit[0]))
 offs=[0];allprot=[]
 for r in ROLES:allprot.extend(sorted(protected[r]));offs.append(len(allprot))
 wp=OUT/'current-qbound-witnesses.npz'
 np.savez_compressed(wp,surface_role=np.asarray(wr,dtype='U1'),candidate_face_rows=np.asarray(rrr,np.int64),bound_ratio=np.asarray(rat,np.float64),protected_rows_by_surface_role=np.asarray(allprot,np.int64),protected_rows_by_surface_role_offsets=np.asarray(offs,np.int64))
 rec={'schema':'h706.current-typed-earclip-adapter-024.v1','status':'PASS_CURRENT_ROW_TYPED_ANCESTRY_ADAPTER_ONLY','physical_admission':False,'native_steps':0,
  'parent_adapter_receipt_path':str(prc),'parent_adapter_receipt_sha256':sha(prc),'dp_report_path':str(REPORT),'dp_report_sha256':sha(REPORT),'protected149_path':str(PROT),'protected149_sha256':sha(PROT),
  'witness_path':str(wp),'witness_sha256':sha(wp),'candidate_roles':output,'witness_counts':{r:wr.count(r) for r in ROLES},
  'scope':'One-step typed-union and source-face-ancestry propagation through exact earclip output; geometry/faces/points copied bitwise. Metadata preparation only, no geometry re-audit.'}
 (OUT/'receipt.json').write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':rec['status'],'output':output,'receipt_path':str(OUT/'receipt.json'),'receipt_sha256':sha(OUT/'receipt.json'),'witness_counts':rec['witness_counts'],'elapsed_s':time.monotonic()-t0},indent=2),flush=True)
if __name__=='__main__':main()
