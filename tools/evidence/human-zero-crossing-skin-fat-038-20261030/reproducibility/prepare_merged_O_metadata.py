from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

BASE=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
TASK=BASE/'quality-warning-connected-union-038-typed-metadata-005-001'
RUN=TASK/'O-merged-run004-run005-001'
MERGE=BASE/'quality-warning-connected-union-038/merge-run004-run005-O'
MERGE_REPORT=MERGE/'merge-report.json'
PROPOSAL=MERGE/'O-merged-run004-run005-proposal.npz'
PARENT_PINS=BASE/'quality-warning-connected-union-038-typed-metadata-001/parent-037-pins.json'
WRITER=BASE/'quality-warning-connected-union-038-typed-metadata-001/prepare_typed_connected_union.py'
PROTECTED=BASE/'microedge-collar-014-saved-op-replay/receipt.json'
GEO=MERGE/'full-check/independent-verification-001/report.json'
CORR004=BASE/'quality-operation-hash-correction-038-001/operation-hash-correction-038.json'
CORR005=BASE/'quality-operation-hash-correction-005-001/operation-hash-correction-005.json'
OUT_NPZ=RUN/'merged-O-proposal-operation-list-normalized.npz'
OPS=RUN/'normalized-operations'
WRITER_OUT=RUN/'typed-output'
PYTHON=Path('/Users/numi/numi-human-prep-venv-20261005/bin/python')
EXPECTED={
 'merge_report':'7bdfd7b8b7d088926a0afb3ea126d4eb7d082a2fbfc8df39667d1d2bf624661e',
 'proposal':'9e0a7abfd1e192c37cdbdabe17335e46d79024e19386c9e635a46ff7564516c3',
 'geometry':'af18d3e4d3f02284acc4a27740c5c9256464c17fb99ca443a7186716a836cdf7',
 'corr004':'b9770e626752a835d32225434f12eecd399429cf34402d5304d1fca78e255c89',
 'corr005':'da58a44d5135aa8b4370ff5db060ba4c10ed7e4c7de8f2e183568b416d128991',
 'parent_pins':'65925c87b122b9ede7716e7d7aa81783afd91a149f53c44995e3b4d1c5377034',
 'protected':'a7eaac55c3cacf88aa5d31c2f2844b6da957a3348c98e76951bfc96c32968a01',
}
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load_npz(p:Path)->dict[str,np.ndarray]:
 with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]).copy() for k in z.files}
def require(p:Path,h:str,label:str):
 got=sha(p)
 if got!=h:raise ValueError(f'{label} SHA mismatch: {p} {got} != {h}')
def role_record(doc:dict,role:str)->dict:
 found=[x for x in doc['roles'] if x.get('role')==role]
 if len(found)!=1:raise ValueError(f'expected one correction role {role}')
 return found[0]

if RUN.exists():raise SystemExit(f'refusing preexisting run directory: {RUN}')
for key,path in [('merge_report',MERGE_REPORT),('proposal',PROPOSAL),('geometry',GEO),('corr004',CORR004),('corr005',CORR005),('parent_pins',PARENT_PINS),('protected',PROTECTED)]:
 require(path,EXPECTED[key],key)
merge=json.loads(MERGE_REPORT.read_text())
if merge.get('status')!='PASS_MAP_COMPOSITION_GEOMETRY_PROOF_PENDING' or merge.get('physical_admission') is not False or merge.get('coordinates_changed') is not False:
 raise ValueError('merged proposal/report status or geometry scope changed')
geo=json.loads(GEO.read_text())
if geo.get('status') not in ('PASS_FIXED_COORDINATE_GEOMETRY_DELTA','PASS_FIXED_COORDINATE_GEOMETRY') or geo.get('physical_admission') is not False:
 raise ValueError('independent geometry result is not the expected fixed-coordinate pass')
c004=json.loads(CORR004.read_text()); c005=json.loads(CORR005.read_text())
if c004.get('status')!='PASS_198_OF_198_PRE_FINALIZATION_HASHES_AND_MAPS_RECONSTRUCTED' or c004.get('operation_count')!=198:raise ValueError('run004 operation correction did not pass 198 ops')
if c005.get('status')!='PASS_RUN005_FINALIZED_OPERATION_HASHES_AND_ROW_MAPS' or c005.get('operation_count')!=515:raise ValueError('run005 operation correction did not pass 515 ops')
# The already completed source-only verification pins the full 005 correction evidence.
ver=TASK/'independent-005-source-correction-verification.json'
require(ver,'5af7f8ce8b38f219af31ad4fb9dfd21a1df1da4c432f3bdcd19f426bf1d27bb3','005 independent correction verifier')
if json.loads(ver.read_text()).get('status')!='PASS_515_OF_515_FINALIZATION_AND_IMMEDIATE_037_PLUS_ULTIMATE_015_ROW_MAPS':raise ValueError('005 independent correction verifier did not pass')
pins=json.loads(PARENT_PINS.read_text())
parent_pin=pins['roles']['O']
parent_typed=BASE/parent_pin['typed_parent']['path']; parent_maps=BASE/parent_pin['typed_parent_maps']['path']; parent_raw=BASE/parent_pin['raw_parent']['path']
for p,k in ((parent_typed,'typed_parent'),(parent_maps,'typed_parent_maps'),(parent_raw,'raw_parent')):require(p,parent_pin[k]['sha256'],'O '+k)
ops=merge['operation_selection']['merged_operations']
if len(ops)!=202 or len({x['patch_id'] for x in ops})!=202:raise ValueError('merged operation table is not 202 unique operations')
raw=load_npz(PROPOSAL)
for key in ('points_world_m','points_world_m_f32','faces','candidate_patch_operation_ids','face_output_to_parent_rows','face_output_to_input_rows','point_output_to_parent_rows','point_output_to_input_rows','new_faces_parent_point_rows','new_faces_input_point_rows'):
 if key not in raw:raise ValueError('merged proposal missing '+key)
old_ids=np.asarray(raw['candidate_patch_operation_ids']).astype(str).tolist()
op_ids=[str(x['patch_id']) for x in ops]
if len(old_ids)!=203 or old_ids[0]!='037-O-warning-000' or old_ids[1:]!=op_ids:
 raise ValueError('merged patch-id registry does not equal parent 037 entry plus ordered 202-operation table')
faces=np.asarray(raw['faces'],np.int64); fparent=np.asarray(raw['face_output_to_parent_rows'],np.int64); pparent=np.asarray(raw['point_output_to_parent_rows'],np.int64)
if faces.ndim!=2 or faces.shape[1]!=3 or fparent.shape!=(len(faces),) or pparent.shape!=(len(raw['points_world_m']),):raise ValueError('merged surface map shapes malformed')
if int(np.count_nonzero(fparent<0))!=1647:raise ValueError('new output face count is not 1647')
point_inverse=np.full(int(pparent.max())+1,-1,np.int64)
if np.any(pparent<0) or len(np.unique(pparent))!=len(pparent):raise ValueError('merged point→037 map is not injective')
# Domain is established by the pinned 037 parent raw point array.
parent_raw_arrays=load_npz(parent_raw)
if int(pparent.max(initial=-1))>=len(parent_raw_arrays['points_world_m']):raise ValueError('merged point parent row outside 037 point domain')
point_inverse=np.full(len(parent_raw_arrays['points_world_m']),-1,np.int64);point_inverse[pparent]=np.arange(len(pparent),dtype=np.int64)
retired=set();out_rows=set(); correction_by_run={
 '004':{x['path']:x for x in role_record(c004,'O')['operations']},
 '005':{x['path']:x for x in role_record(c005,'O')['operations']},
}
for op in ops:
 opid=op['patch_id']; src=Path(op['source_operation_path'])
 if src.is_symlink(): raise ValueError(f'operation source unexpectedly symlinked: {src}')
 op_sha=sha(src)
 if op_sha!=op['source_operation_sha256']:raise ValueError(f'operation source SHA mismatch: {opid}')
 source=json.loads(src.read_text())
 if source.get('role')!='O' or source.get('patch_id')!=opid:raise ValueError(f'operation source identity mismatch: {opid}')
 if source.get('source_lineage')!=op.get('source_lineage'):raise ValueError(f'merge report lineage differs from source operation: {opid}')
 if source.get('retired_input_face_rows')!=op['retired_037_face_rows']:raise ValueError(f'immediate 037 retired rows differ from op source: {opid}')
 if source.get('new_faces_input_point_rows')!=op['new_faces_037_point_rows']:raise ValueError(f'immediate 037 point triangles differ from op source: {opid}')
 if source.get('new_faces_source_point_rows')!=op['new_faces_015_point_rows']:raise ValueError(f'ultimate 015 point triangles differ from op source: {opid}')
 corr=correction_by_run[op['source_run']].get(str(src))
 if corr is None:raise ValueError(f'operation absent from correction receipt: {opid}')
 corr_sha=corr.get('finalized_sha256',corr.get('current_finalized_sha256'))
 if corr_sha!=op_sha:raise ValueError(f'operation correction SHA differs: {opid}')
 if str(op.get('source_report_sha256'))!=str(merge[f"run{op['source_run']}_report"]['sha256']):raise ValueError(f'operation report pin mismatch: {opid}')
 rr=np.asarray(op['retired_037_face_rows'],np.int64); oo=np.asarray(op['merged_output_face_rows'],np.int64); tri=np.asarray(op['new_faces_037_point_rows'],np.int64)
 if rr.ndim!=1 or not len(rr) or len(np.unique(rr))!=len(rr) or np.any(rr<0):raise ValueError(f'invalid 037 retired map: {opid}')
 if oo.ndim!=1 or tri.shape!=(len(oo),3) or not len(oo) or np.any(oo<0) or np.any(oo>=len(faces)):raise ValueError(f'invalid merged face map: {opid}')
 if np.any(fparent[oo]!=-1):raise ValueError(f'merged operation output is not a new immediate-parent face: {opid}')
 if any(int(r) in retired for r in rr):raise ValueError(f'operations overlap retired 037 parent faces: {opid}')
 if any(int(r) in out_rows for r in oo):raise ValueError(f'operations overlap merged output rows: {opid}')
 if np.any(tri<0) or np.any(tri>=len(point_inverse)) or np.any(point_inverse[tri]<0):raise ValueError(f'operation triangle outside mapped 037 point rows: {opid}')
 if not np.array_equal(faces[oo],point_inverse[tri]):raise ValueError(f'merged triangle connectivity differs: {opid}')
 retired.update(map(int,rr));out_rows.update(map(int,oo))
if out_rows!=set(map(int,np.flatnonzero(fparent<0))):raise ValueError('202 ops do not exactly cover new merged output faces')
# Build canonical operation docs in a new task-owned directory; originals remain untouched.
RUN.mkdir(parents=True); OPS.mkdir()
for idx,op in enumerate(ops):
 src=Path(op['source_operation_path']); data=json.loads(src.read_text()); nested=dict(data['patches'][0])
 rows=list(map(int,op['merged_output_face_rows'])); retired_rows=list(map(int,op['retired_037_face_rows'])); tris=np.asarray(op['new_faces_037_point_rows'],np.int64).tolist()
 data['new_output_face_rows']=rows;data['retired_input_face_rows']=retired_rows;data['new_faces_input_point_rows']=tris
 nested['new_output_face_rows']=rows;nested['retired_input_face_rows']=retired_rows;nested['new_faces_input_point_rows']=tris
 data['patches']=[nested]
 data['merge_binding']={'merge_report_path':str(MERGE_REPORT),'merge_report_sha256':EXPECTED['merge_report'],'source_operation_path':str(src),'source_operation_sha256':op['source_operation_sha256'],'source_run':op['source_run'],'source_report_sha256':op['source_report_sha256'],'merged_output_face_rows':rows,'retired_037_face_rows':retired_rows,'new_faces_037_point_rows':tris,'lineage_semantics':op['lineage_semantics']}
 filename=f'{idx:03d}-{opid}.json'
 (OPS/filename).write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n')
# Metadata-only proposal normalization: candidate_patch_operation_ids is an operation list,
# whereas the merged producer's registry also includes retained 037 parent patch 0.
normalized={k:v.copy() for k,v in raw.items()}
normalized['candidate_patch_operation_ids']=np.asarray(op_ids,dtype=np.asarray(raw['candidate_patch_operation_ids']).dtype)
np.savez_compressed(OUT_NPZ,**normalized)
with np.load(OUT_NPZ,allow_pickle=False) as z:
 if set(z.files)!=set(raw):raise ValueError('normalized NPZ member set changed')
 for k in raw:
  a=np.asarray(z[k]);b=np.asarray(normalized[k])
  if k=='candidate_patch_operation_ids':
   if not np.array_equal(a,np.asarray(op_ids,dtype=a.dtype)):raise ValueError('normalized operation ID list mismatch')
  elif a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():raise ValueError(f'metadata normalization changed member {k}')
# Verify exact geometry and all source/parent maps are byte-identical after the one metadata field change.
for k in ('points_world_m','points_world_m_f32','faces','face_output_to_parent_rows','face_output_to_input_rows','point_output_to_parent_rows','point_output_to_input_rows','new_faces_parent_point_rows','new_faces_input_point_rows'):
 if normalized[k].dtype!=raw[k].dtype or normalized[k].shape!=raw[k].shape or normalized[k].tobytes()!=raw[k].tobytes():raise ValueError(f'geometry/map member changed during normalization: {k}')
normalization={'schema':'h706.quality-warning-038-merged-O-operation-list-normalization.v1','status':'PASS_METADATA_ONLY_OPERATION_LIST_NORMALIZATION','source_proposal_path':str(PROPOSAL),'source_proposal_sha256':EXPECTED['proposal'],'normalized_proposal_path':str(OUT_NPZ),'normalized_proposal_sha256':sha(OUT_NPZ),'only_changed_member':'candidate_patch_operation_ids','reason':'The merged proposal registry includes a retained 037 patch identity; this stage adds only the 202 newly-created output patches relative to the authenticated immediate 037 typed parent.','retained_037_registry_patch_id':old_ids[0],'ordered_operation_ids':op_ids,'operation_count':len(ops),'normalized_operation_directory':str(OPS),'operation_sources':[{'patch_id':x['patch_id'],'source_run':x['source_run'],'source_operation_path':x['source_operation_path'],'source_operation_sha256':x['source_operation_sha256'],'source_report_sha256':x['source_report_sha256'],'output_face_count':len(x['merged_output_face_rows']),'retired_parent_face_count':len(x['retired_037_face_rows'])} for x in ops],'geometry_and_parent_maps_bitwise_identical':True,'geometry_proof_path':str(GEO),'geometry_proof_sha256':EXPECTED['geometry'],'geometry_status':geo['status'],'physical_admission':False}
normal_path=RUN/'operation-list-normalization-receipt.json';normal_path.write_text(json.dumps(normalization,sort_keys=True,indent=2)+'\n')
cmd=[str(PYTHON),str(WRITER),'--role','O','--parent-typed',str(parent_typed),'--parent-maps',str(parent_maps),'--parent-raw',str(parent_raw),'--proposal',str(OUT_NPZ),'--operation-dir',str(OPS),'--composed-map',str(OUT_NPZ),'--protected-receipt',str(PROTECTED),'--outdir',str(WRITER_OUT),'--generation-label','038','--operation-glob','*.json','--operation-retired-face-key','retired_input_face_rows','--operation-new-point-key','new_faces_input_point_rows','--proposal-parent-face-map-key','face_output_to_parent_rows','--proposal-parent-point-map-key','point_output_to_parent_rows']
proc=subprocess.run(cmd,text=True,capture_output=True)
(RUN/'writer.stdout.txt').write_text(proc.stdout)
(RUN/'writer.stderr.txt').write_text(proc.stderr)
if proc.returncode:raise SystemExit(f'generic typed writer failed ({proc.returncode}); see {RUN}/writer.stderr.txt')
result=json.loads(proc.stdout.strip().splitlines()[-1])
inv={'schema':'h706.quality-warning-038-merged-O-typed-build-invocation.v1','status':'PASS_TYPED_METADATA_BUILD','argv':cmd,'writer_sha256':sha(WRITER),'parent_typed_path':str(parent_typed),'parent_typed_sha256':sha(parent_typed),'parent_maps_path':str(parent_maps),'parent_maps_sha256':sha(parent_maps),'parent_raw_path':str(parent_raw),'parent_raw_sha256':sha(parent_raw),'source_merge_report_path':str(MERGE_REPORT),'source_merge_report_sha256':sha(MERGE_REPORT),'source_merged_proposal_path':str(PROPOSAL),'source_merged_proposal_sha256':sha(PROPOSAL),'normalized_proposal_sha256':sha(OUT_NPZ),'operation_normalization_receipt':str(normal_path),'operation_normalization_receipt_sha256':sha(normal_path),'geometry_proof_path':str(GEO),'geometry_proof_sha256':sha(GEO),'run004_correction_path':str(CORR004),'run004_correction_sha256':sha(CORR004),'run005_correction_path':str(CORR005),'run005_correction_sha256':sha(CORR005),'operation_correction_005_verification_path':str(ver),'operation_correction_005_verification_sha256':sha(ver),'protected_receipt_sha256':sha(PROTECTED),'generic_writer_result':result,'physical_admission':False}
invpath=RUN/'build-invocation.json';invpath.write_text(json.dumps(inv,sort_keys=True,indent=2)+'\n')
print(json.dumps({'status':'PASS_TYPED_METADATA_BUILD','role':'O','writer_result':result,'invocation_path':str(invpath),'invocation_sha256':sha(invpath),'normalization_receipt_sha256':sha(normal_path)},sort_keys=True))
