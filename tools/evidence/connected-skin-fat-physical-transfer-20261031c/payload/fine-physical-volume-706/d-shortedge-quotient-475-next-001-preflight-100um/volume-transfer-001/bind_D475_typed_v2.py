#!/usr/bin/env python3
"""Bind the frozen D475 edge quotient through the existing typed-union owners."""
from pathlib import Path
import hashlib, importlib.util, json, sys, time
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
RUN=H/'d-shortedge-quotient-475-next-001-preflight-100um/volume-transfer-001'
DSEED=H/'d-shortedge-quotient-475-next-001-preflight-100um/preflight-output/identity-seed-D.npz'
DSEED_SHA='02c6364b0a6100306496beb6fdb48b96b2bf0f56fb9d65526dae7df46c48564d'
SURF=H/'d-shortedge-quotient-475-next-001-preflight-100um/surface-batch-001/candidate-D-short-edge-collapse-raw.npz'
SURF_SHA='b055cbcb6f14637fcef3ac7dc5c2461cae33307731705f3a8b0337741900d8bd'
REPORT=H/'d-shortedge-quotient-475-next-001-preflight-100um/surface-batch-001/report.json'
REPORT_SHA='28be53d5bc5d9505907d1f55951a2ec17b43ac67409e78cb5504a4f85a768632'
A_PATH=H/'fixed-boundary-quality-topology-196/prepare_current_earclip_adapter_024.py'
B_PATH=H/'fixed-boundary-quality-topology-196/run_fixed_boundary_dp_batch_197.py'
PROT=H/'verified-volume-preparation-010-merged038005/rim/protected149-current-routes.json'
PROT_SHA='ed6e15c6dc9324b09cd4514e1fc3a149fc92a07215128a9820d72fca3f350a71'
OUT=RUN/'typed-bind-002'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p)
 if s is None or s.loader is None:raise RuntimeError('cannot load '+str(p))
 m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}

def euler(fs):
 es=np.concatenate((fs[:,[0,1]],fs[:,[1,2]],fs[:,[2,0]]))
 return len(np.unique(fs))-len(np.unique(np.sort(es,axis=1),axis=0))+len(fs)

for p,w in [(DSEED,DSEED_SHA),(SURF,SURF_SHA),(REPORT,REPORT_SHA),(PROT,PROT_SHA)]:
 if not p.exists() or sha(p)!=w:raise ValueError('input pin mismatch: '+str(p))
rep=json.loads(REPORT.read_text())
if rep.get('status')!='PASS_BOUNDED_EXACT_SHORT_EDGE_COLLAPSE_PROPOSAL' or rep.get('physical_admission') is not False or rep.get('native_steps')!=0:raise ValueError('raw source report is not a source-only pass')
ctx=rep.get('joint_changed_vs_complete_OID_walls_all185',{})
if ctx.get('pair_count')!=0:raise ValueError('raw final joint context count is not zero')
if rep.get('outputs',{}).get('D',{}).get('path')!=str(SURF) or rep['outputs']['D'].get('sha256')!=SURF_SHA:raise ValueError('report does not pin raw D candidate')
ops=[x for x in rep.get('accepted_operations',[]) if x.get('role')=='D']
if len(ops)!=1846:raise ValueError('expected exactly 1,846 D operations')
A=module('typed024_D475',A_PATH);B=module('base197_D475',B_PATH)
par=load(DSEED);cur=load(SURF);P=np.asarray(cur['points_world_m'],np.float64);PF=np.asarray(par['faces'],np.int64);F=np.asarray(cur['faces'],np.int64)
pm=np.asarray(cur['point_output_to_parent_rows'],np.int64);fm=np.asarray(cur['face_output_to_parent_rows'],np.int64)
if not np.array_equal(P,par['points_world_m']) or not np.array_equal(cur['points_world_m_f32'],par['points_world_m_f32']):raise ValueError('D475 changed registered point bytes')
if not np.array_equal(pm,np.arange(len(P),dtype=np.int64)) or len(fm)!=len(F) or np.any(fm>=len(PF)):raise ValueError('point/face output maps invalid')
if str(cur['immediate_parent_candidate_sha256'][0])!=DSEED_SHA or str(cur['typed_parent_candidate_sha256'][0])!='c285291a36182ad3a025b439ec10b1c1bb15595a1554d4fbb5d943d2a00da7a6':raise ValueError('raw D immediate/typed parent pin mismatch')
t0=time.monotonic();OUT.mkdir(parents=True,exist_ok=False);new_op_faces={};opdocs=[]
for oi,op in enumerate(ops):
 retired=list(map(int,op['retired_parent_face_rows']));union_rows=list(map(int,op['affected_parent_face_rows']));newrows=list(map(int,op['new_output_face_rows']))
 un=B.union_tuples(par,union_rows);anc=B.ancestry_union(par,union_rows)
 if un!=[tuple(map(int,v)) for v in op['complete_typed_parent_union']] or B.union_digest(un)!=op['typed_parent_union_sha256'] or anc!=op['source_face_ancestry_union']:
  raise ValueError(f'complete typed/source-face ancestry mismatch operation {oi}')
 expected_new=np.asarray(op.get('new_faces',op.get('new_faces_current_point_rows')),np.int64)
 if len(newrows)!=len(expected_new) or not np.array_equal(F[newrows],expected_new):raise ValueError(f'new face output map mismatch operation {oi}')
 if any(int(fm[r])>=0 for r in newrows):raise ValueError(f'generated output face has a parent row operation {oi}')
 for r in newrows:
  if int(r) in new_op_faces:raise ValueError('generated face assigned to multiple operations')
  new_op_faces[int(r)]=(un,anc)
 doc={**op,'schema':'h706.fixed-boundary-short-edge-contraction-operation.v1','complete_typed_parent_union':un,'new_output_face_rows':newrows,'retired_parent_face_rows':retired,'affected_parent_face_rows':union_rows,'typed_current_adapter_path':str(DSEED),'typed_current_adapter_sha256':DSEED_SHA,'lineage_semantics':'complete deduplicated typed and face ancestry union over the entire affected edge star, including retained faces; not per-triangle overlap','physical_admission':False,'native_steps':0}
 dp=OUT/f'operation-D-{oi:04d}.json';dp.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');opdocs.append({'path':str(dp),'sha256':sha(dp)})
if set(new_op_faces)!=set(map(int,np.flatnonzero(fm<0))):raise ValueError('operation ledger does not exactly cover generated output faces')
RU=A.RU;AOFF=np.asarray(cur['new_face_patch_union_offsets'],np.int64)
if AOFF.shape!=(len(cur['new_faces_parent_point_rows'])+1,) or AOFF[0]!=0 or np.any(np.diff(AOFF)<=0):raise ValueError('raw new-face typed union CSR malformed')
packed={}
for j,tri in enumerate(np.asarray(cur['new_faces_parent_point_rows'],np.int64)):
 lo,hi=map(int,AOFF[j:j+2]);packed[tuple(map(int,tri))]=sorted(set(tuple(int(cur[k][n]) for k in RU) for n in range(lo,hi)))
if len(packed)!=len(cur['new_faces_parent_point_rows']):raise ValueError('duplicate raw generated face union key')
unions=[];ancestries=[]
for i,f in enumerate(F):
 row=int(fm[i])
 if row>=0:
  if not np.array_equal(pm[f],PF[row]):raise ValueError('retained face point rows differ from source parent')
  unions.append(A.rows_union(par,row));ancestries.append(A.rows_anc(par,row))
 else:
  un,anc=new_op_faces[i]
  if packed[tuple(map(int,pm[f]))]!=un:raise ValueError('raw patch-union CSR disagrees with operation full union')
  unions.append(un);ancestries.append(anc)
if not all(unions) or not all(ancestries):raise ValueError('empty typed or ancestry row')
uoff,ud=A.csr(unions,7);aoff,ad=A.csr(ancestries,3);typed=np.asarray([u[0] for u in unions],np.int64)
pmap=np.asarray(par['quality_star_point_output_to_input_row'],np.int64)[pm]
fmap=np.full(len(F),-1,np.int64);keep=fm>=0;fmap[keep]=par['quality_star_face_output_to_input_row'][fm[keep]]
payload={'points_world_m':P,'points_world_m_f32':np.asarray(cur['points_world_m_f32'],np.float32),'faces':F,'quality_star_point_output_to_input_row':pmap,'quality_star_face_output_to_input_row':fmap,'quality_star_patch_union_offsets':uoff,'face_ancestry_offsets':aoff,'source_current_candidate_path':np.asarray([str(SURF)]),'source_current_candidate_sha256':np.asarray([SURF_SHA]),'typed_parent_candidate_path':np.asarray([str(DSEED)]),'typed_parent_candidate_sha256':np.asarray([DSEED_SHA])}
payload.update({k:typed[:,j] for j,k in enumerate(A.T)});payload.update({k:ud[:,j] for j,k in enumerate(A.U)});payload.update({k:ad[:,j] for j,k in enumerate(A.A)})
if 'face_source_identity_schema_by_original_id_json' in par:payload['face_source_identity_schema_by_original_id_json']=np.asarray(par['face_source_identity_schema_by_original_id_json'])
prot=json.loads(PROT.read_text());routes=0
for r in prot['surface_routes']:
 if r['role']=='D':
  if np.count_nonzero(np.all(payload['points_world_m_f32']==np.asarray(r['target_world_m_f32'],np.float32),axis=1))!=1:raise ValueError('protected D route point not unique')
  routes+=1
if routes!=59:raise ValueError(f'expected 59 protected D routes, got {routes}')
b0,t0=B.topology(PF);b1,t1=B.topology(F)
if not np.array_equal(b0,b1) or any(t0[k]!=t1[k] for k in ('boundary_edges','nonmanifold_edges','same_direction_edges')):raise ValueError('D quotient changed closed/rim topology invariants')
if euler(PF)!=euler(F):raise ValueError('D quotient changed Euler characteristic')
out=OUT/'candidate-D-typed-current.npz';np.savez_compressed(out,**payload);rr=load(out)
for k,v in payload.items():
 if not np.array_equal(rr[k],v):raise ValueError('typed candidate readback mismatch '+k)
receipt={'schema':'h706.d-shortedge-475-typed-lineage-binding.v1','status':'PASS_CURRENT_ROW_TYPED_ANCESTRY_ADAPTER_ONLY','typed_source_path':str(out),'typed_source_sha256':sha(out),'raw_surface_path':str(SURF),'raw_surface_sha256':SURF_SHA,'surface_report_path':str(REPORT),'surface_report_sha256':REPORT_SHA,'typed_parent_path':str(DSEED),'typed_parent_sha256':DSEED_SHA,'typed_union_owner_path':str(A_PATH),'typed_union_owner_sha256':sha(A_PATH),'ancestry_owner_path':str(B_PATH),'ancestry_owner_sha256':sha(B_PATH),'face_count':len(F),'point_count':len(P),'generated_face_count':len(new_op_faces),'operation_count':len(ops),'typed_union_tuple_count':len(ud),'flattened_ancestry_tuple_count':len(ad),'protected_D_route_count':routes,'topology_euler_before':euler(PF),'topology_euler_after':euler(F),'full_surface_context_pairs':int(ctx['pair_count']),'new_face_self_pairs':int(sum(int(x.get('exact_replacement_self_pairs',0)) for x in ops)),'physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN','elapsed_s':time.monotonic()-t0,'limitations':['Typed source-face and 7-lane group ancestry are bound through the existing owners.','This is source-only; latest O484 combined context, 697 loader, volume, mass mapping, and native acceptance remain separate gates.']}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:receipt[k] for k in ['status','typed_source_path','typed_source_sha256','operation_count','generated_face_count','protected_D_route_count','typed_union_tuple_count','flattened_ancestry_tuple_count','elapsed_s']},sort_keys=True))
