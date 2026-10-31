#!/usr/bin/env python3
"""Readback verification and receipt for the frozen D475 typed bind output."""
from pathlib import Path
import hashlib,json
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
RUN=H/'d-shortedge-quotient-475-next-001-preflight-100um/volume-transfer-001'
DSEED=H/'d-shortedge-quotient-475-next-001-preflight-100um/preflight-output/identity-seed-D.npz'
SURF=H/'d-shortedge-quotient-475-next-001-preflight-100um/surface-batch-001/candidate-D-short-edge-collapse-raw.npz'
REPORT=H/'d-shortedge-quotient-475-next-001-preflight-100um/surface-batch-001/report.json'
BIND=RUN/'bind_D475_typed_v2.py'; OUT=RUN/'typed-bind-002'; CAND=OUT/'candidate-D-typed-current.npz'
DSEED_SHA='02c6364b0a6100306496beb6fdb48b96b2bf0f56fb9d65526dae7df46c48564d'
SURF_SHA='b055cbcb6f14637fcef3ac7dc5c2461cae33307731705f3a8b0337741900d8bd'
REPORT_SHA='28be53d5bc5d9505907d1f55951a2ec17b43ac67409e78cb5504a4f85a768632'
CAND_SHA='97ebce376ec202cb63c3339f4bc433c80a724fc6d63362db56d47d3bf0313bbb'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
for p,w in [(DSEED,DSEED_SHA),(SURF,SURF_SHA),(REPORT,REPORT_SHA),(CAND,CAND_SHA)]:
 if sha(p)!=w:raise ValueError('frozen artifact hash mismatch '+str(p))
rep=json.loads(REPORT.read_text());raw=load(SURF);par=load(DSEED);can=load(CAND)
if sha(BIND)!='f5ee1ed4e0790bb8909ca4882ccfa925dd314f16871055950690077e433682f9':raise ValueError('typed binder source changed')
if not np.array_equal(can['faces'],raw['faces']) or not np.array_equal(can['points_world_m_f32'],raw['points_world_m_f32']) or not np.array_equal(can['points_world_m'],raw['points_world_m']):raise ValueError('typed candidate geometry differs from raw D proposal')
if not np.array_equal(can['points_world_m_f32'],par['points_world_m_f32']):raise ValueError('typed candidate moved parent point bytes')
fm=np.asarray(raw['face_output_to_parent_rows'],np.int64);F=np.asarray(raw['faces'],np.int64);P=np.asarray(raw['points_world_m_f32'],np.float32)
if not np.array_equal(can['quality_star_face_output_to_input_row'][fm>=0],par['quality_star_face_output_to_input_row'][fm[fm>=0]]):raise ValueError('retained face input ancestry map differs')
if str(can['typed_parent_candidate_sha256'][0])!=DSEED_SHA or str(can['source_current_candidate_sha256'][0])!=SURF_SHA:raise ValueError('typed candidate parent/source pins differ')
N=len(F);uoff=np.asarray(can['quality_star_patch_union_offsets'],np.int64);aoff=np.asarray(can['face_ancestry_offsets'],np.int64)
if uoff.shape!=(N+1,) or aoff.shape!=(N+1,) or uoff[0]!=0 or aoff[0]!=0 or uoff[-1]!=len(can['quality_star_patch_union_face_source_layer']) or aoff[-1]!=len(can['face_ancestry_source_face_rows']) or np.any(np.diff(uoff)<=0) or np.any(np.diff(aoff)<=0):raise ValueError('typed CSR offsets invalid/nonempty')
lanes=('layer','body','bone','face_id','original_id','identity0','identity1')
anc_lanes=('face_ancestry_source_face_rows','face_ancestry_source_original_ids','face_ancestry_parent_face_rows')
if len({len(can['quality_star_patch_union_face_source_'+k]) for k in lanes})!=1 or len({len(can[k]) for k in anc_lanes})!=1:raise ValueError('typed CSR lane length mismatch')
ops=[x for x in rep['accepted_operations'] if x.get('role')=='D'];new_rows=set()
for oi,op in enumerate(ops):
 docp=OUT/f'operation-D-{oi:04d}.json';doc=json.loads(docp.read_text())
 if doc.get('patch_id')!=op.get('patch_id') or doc.get('typed_parent_union_sha256')!=op.get('typed_parent_union_sha256'):raise ValueError(f'op doc/report mismatch {oi}')
 union=sorted(set(tuple(map(int,r)) for r in doc['complete_typed_parent_union']))
 ancestry=sorted(set(tuple(map(int,r)) for r in doc['source_face_ancestry_union']))
 for fr in map(int,doc['new_output_face_rows']):
  if fr<0 or fr>=N or fr in new_rows:raise ValueError('generated face row invalid/duplicate')
  new_rows.add(fr);lo,hi=map(int,uoff[fr:fr+2]);got=sorted(set(tuple(int(can['quality_star_patch_union_face_source_'+k][n]) for k in lanes) for n in range(lo,hi)))
  if got!=union:raise ValueError(f'generated face typed union mismatch op {oi} face {fr}')
  lo,hi=map(int,aoff[fr:fr+2]);got=sorted(set(tuple(int(can[k][n]) for k in anc_lanes) for n in range(lo,hi)))
  if got!=ancestry:raise ValueError(f'generated face source ancestry mismatch op {oi} face {fr}')
new_raw=set(map(int,np.flatnonzero(fm<0)))
if new_rows!=new_raw:raise ValueError('operation docs do not cover exactly the generated output faces')
if len(ops)!=1846 or len(new_rows)!=6514:raise ValueError('unexpected operation/generated face count')
# Protected 149 routes remain unique in the typed point array.
prot=H/'verified-volume-preparation-010-merged038005/rim/protected149-current-routes.json'
if sha(prot)!='ed6e15c6dc9324b09cd4514e1fc3a149fc92a07215128a9820d72fca3f350a71':raise ValueError('protected route pin mismatch')
pr=json.loads(prot.read_text());nr=0
for r in pr['surface_routes']:
 if r.get('role')=='D':
  if np.count_nonzero(np.all(P==np.asarray(r['target_world_m_f32'],np.float32),axis=1))!=1:raise ValueError('D protected route point missing/ambiguous')
  nr+=1
if nr!=59:raise ValueError('protected D route count mismatch')
if not np.array_equal(can['faces'],F):raise ValueError('final typed face readback differs')
receipt={'schema':'h706.d-shortedge-475-typed-lineage-binding.v1','status':'PASS_CURRENT_ROW_TYPED_ANCESTRY_ADAPTER_ONLY','typed_source_path':str(CAND),'typed_source_sha256':sha(CAND),'raw_surface_path':str(SURF),'raw_surface_sha256':SURF_SHA,'surface_report_path':str(REPORT),'surface_report_sha256':REPORT_SHA,'typed_parent_path':str(DSEED),'typed_parent_sha256':DSEED_SHA,'typed_binder_source_path':str(BIND),'typed_binder_source_sha256':sha(BIND),'binding_readback_verifier_path':str(Path(__file__)),'binding_readback_verifier_sha256':sha(Path(__file__)),'face_count':N,'point_count':len(P),'generated_face_count':len(new_rows),'operation_count':len(ops),'typed_union_tuple_count':int(uoff[-1]),'flattened_ancestry_tuple_count':int(aoff[-1]),'protected_D_route_count':nr,'generated_faces_union_ancestry_checked':len(new_rows),'full_surface_context_pairs':int(rep['joint_changed_vs_complete_OID_walls_all185']['pair_count']),'new_face_self_pairs':int(sum(int(x.get('exact_replacement_self_pairs',0)) for x in ops)),'failed_initial_binder_attempt':{'source_sha256':'b24fb1a184763e9818de8e9a751d5efbc213766402a421d29682bf489607e4e9','failure':'incorrect extra prefix was applied while reading raw union lane names; candidate generation completed in v2 before receipt-time variable-shadow error'},'physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN','limitations':['Typed source-face and 7-lane group ancestry are bound through the existing owners and read back against all generated faces.','Current O484 combined-context audit, actual 697 loader, volume/mass transfer, and native acceptance are separate gates.']}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:receipt[k] for k in ['status','typed_source_sha256','operation_count','generated_face_count','protected_D_route_count','typed_union_tuple_count','flattened_ancestry_tuple_count','full_surface_context_pairs']},sort_keys=True))
