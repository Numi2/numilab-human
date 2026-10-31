#!/usr/bin/env python3
"""Bind fixed-coordinate raw topology proposals through the existing typed face-union owner.
No geometric repair or physical admission is performed here.
"""
from pathlib import Path
import argparse,json,hashlib,importlib.util,sys,time
import numpy as np
H=Path(__file__).resolve().parent;S=H/'fixed-boundary-quality-topology-196'
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
A=module('typed024for260',S/'prepare_current_earclip_adapter_024.py')
B=module('base197for260',S/'run_fixed_boundary_dp_batch_197.py')
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def apply(report_path,out):
 tic=time.monotonic();out.mkdir(parents=True,exist_ok=False);rep=json.loads(report_path.read_text())
 assert rep['physical_admission'] is False and rep['native_steps']==0
 assert rep['joint_complete_surface_walls_all185_pairs']==rep['joint_changed_self_pairs']==0
 result={};opdocs=[]
 for role,rr in rep['outputs'].items():
  pp=Path(rr['parent']);cp=Path(rr['path']);assert sha(pp)==rr['parent_sha256'] and sha(cp)==rr['sha256']
  par=A.load(pp);cur=A.load(cp);P=np.asarray(cur['points_world_m'],np.float64);PF=np.asarray(par['faces'],np.int64);F=np.asarray(cur['faces'],np.int64)
  assert np.array_equal(P,par['points_world_m']) and np.array_equal(cur['points_world_m_f32'],par['points_world_m_f32'])
  pm=np.asarray(cur['point_output_to_parent_rows'],np.int64);fm=np.asarray(cur['face_output_to_parent_rows'],np.int64)
  assert np.array_equal(pm,np.arange(len(P))) and len(fm)==len(F) and np.all(fm<len(PF))
  assert str(cur['immediate_parent_candidate_sha256'][0])==sha(pp)
  ops={}
  for j,op in enumerate(x for x in rep['accepted_operations'] if x['role']==role):
   rows=list(map(int,op.get('retired_parent_face_rows',op.get('parent_face_rows',[]))))
   newrows=list(map(int,op.get('new_output_face_rows',rows)))
   un=B.union_tuples(par,rows);anc=B.ancestry_union(par,rows)
   assert un==op['typed_parent_union'] and B.union_digest(un)==op['typed_parent_union_sha256'] and anc==op['source_face_ancestry_union']
   assert len(newrows)==len(op['new_faces']) and np.array_equal(F[newrows],np.asarray(op['new_faces'],np.int64))
   doc={**op,'schema':'h706.fixed-boundary-quality-earclip-operation.v1','complete_typed_parent_union':un,
        'new_output_face_rows':newrows,'retired_parent_face_rows':rows,'typed_current_adapter_path':str(pp),
        'typed_current_adapter_sha256':sha(pp),'lineage_semantics':'complete patch-level parent union, not per-triangle overlap',
        'physical_admission':False,'native_steps':0}
   dp=out/f'operation-{role}-{j:04d}.json';dp.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
   opdocs.append({'path':str(dp),'sha256':sha(dp)})
   for i in newrows:
    assert i not in ops and fm[i]<0;ops[i]=(un,anc)
  assert set(ops)==set(map(int,np.flatnonzero(fm<0)))
  nt=np.asarray(cur['new_faces_parent_point_rows'],np.int64);uoff=cur['new_face_patch_union_offsets'];newmap={}
  assert len(uoff)==len(nt)+1
  for j,t in enumerate(nt):
   lo,hi=map(int,uoff[j:j+2]);un=sorted(set(tuple(int(cur[k][n]) for k in A.RU) for n in range(lo,hi)))
   key=tuple(map(int,t));assert key not in newmap;newmap[key]=un
  unions=[];ancestries=[]
  for i,f in enumerate(F):
   row=int(fm[i])
   if row>=0:
    assert np.array_equal(pm[f],PF[row]);unions.append(A.rows_union(par,row));ancestries.append(A.rows_anc(par,row))
   else:
    un,anc=ops[i];assert newmap[tuple(map(int,pm[f]))]==un
    unions.append(un);ancestries.append(anc)
  assert all(unions) and all(ancestries)
  uoff,ud=A.csr(unions,7);aoff,ad=A.csr(ancestries,3);typed=np.asarray([u[0] for u in unions],np.int64)
  pmap=np.asarray(par['quality_star_point_output_to_input_row'],np.int64)[pm]
  fmap=np.full(len(F),-1,np.int64);keep=fm>=0;fmap[keep]=par['quality_star_face_output_to_input_row'][fm[keep]]
  payload={'points_world_m':P,'points_world_m_f32':np.asarray(cur['points_world_m_f32'],np.float32),'faces':F,
   'quality_star_point_output_to_input_row':pmap,'quality_star_face_output_to_input_row':fmap,
   'quality_star_patch_union_offsets':uoff,'face_ancestry_offsets':aoff,
   'source_current_candidate_path':np.asarray([str(cp)]),'source_current_candidate_sha256':np.asarray([sha(cp)]),
   'typed_parent_candidate_path':np.asarray([str(pp)]),'typed_parent_candidate_sha256':np.asarray([sha(pp)])}
  payload.update({k:typed[:,j] for j,k in enumerate(A.T)});payload.update({k:ud[:,j] for j,k in enumerate(A.U)});payload.update({k:ad[:,j] for j,k in enumerate(A.A)})
  # Coordinate-only protected routes remain exact; no relabeling of open boundaries.
  prot=json.loads(A.PROT.read_text());routes=0
  for r in prot['surface_routes']:
   if r['role']==role:
    assert np.count_nonzero(np.all(payload['points_world_m_f32']==np.asarray(r['target_world_m_f32'],np.float32),axis=1))==1;routes+=1
  b0,t0=B.topology(PF);b1,t1=B.topology(F);assert np.array_equal(b0,b1) and t0==t1
  path=out/f'candidate-{role}-typed-current.npz';np.savez_compressed(path,**payload)
  reread=A.load(path);assert all(np.array_equal(reread[k],v) for k,v in payload.items())
  qq=B.qbound(payload['points_world_m_f32'],F)
  result[role]={'path':str(path),'sha256':sha(path),'parent':str(pp),'parent_sha256':sha(pp),'raw_path':str(cp),'raw_sha256':sha(cp),'faces':len(F),'points':len(P),'generated_faces':len(ops),'typed_tuples':len(ud),'ancestry_tuples':len(ad),'protected_routes_unchanged':routes,'no_native_apex_faces':int(np.count_nonzero(np.sqrt(3)/2*qq<1e-5)),'construction_floor_bad_faces':int(np.count_nonzero(qq<rep['construction_surface_floor']))}
 receipt={'status':'PASS_CURRENT_ROW_TYPED_ANCESTRY_ADAPTER_ONLY','physical_admission':False,'native_steps':0,'source_sha256':sha(Path(__file__)),'typed_owner_sha256':sha(Path(A.__file__)),'input_report':str(report_path),'input_report_sha256':sha(report_path),'candidate_roles':result,'operation_docs':opdocs,'elapsed_s':time.monotonic()-tic}
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='operation_docs'},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();apply(a.report,a.output)
