from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

BASE=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
RUN=BASE/'quality-warning-connected-union-038/run-005'
COR=BASE/'quality-operation-hash-correction-005-001/operation-hash-correction-005.json'
COR_EXPECTED='da58a44d5135aa8b4370ff5db060ba4c10ed7e4c7de8f2e183568b416d128991'
REPORT=RUN/'report.json'
REPORT_EXPECTED='e81f5753da8d4ae6c05de0f24bbe70f107ecdc34a0c42f571813e8d23a7bcc56'
PARENT_PINS=BASE/'quality-warning-connected-union-038-typed-metadata-001/parent-037-pins.json'
PROTECTED=BASE/'microedge-collar-014-saved-op-replay/receipt.json'
PROTECTED_EXPECTED='a7eaac55c3cacf88aa5d31c2f2844b6da957a3348c98e76951bfc96c32968a01'
OUT=BASE/'quality-warning-connected-union-038-typed-metadata-005-001/independent-005-source-correction-verification.json'

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def ck(ok,msg):
    if not ok: raise ValueError(msg)

def load_npz(path):
    with np.load(path,allow_pickle=False) as z: return {k:np.asarray(z[k]).copy() for k in z.files}

ck(sha(COR)==COR_EXPECTED,'005 correction receipt hash mismatch')
cor=json.loads(COR.read_text())
ck(cor['status']=='PASS_RUN005_FINALIZED_OPERATION_HASHES_AND_ROW_MAPS','005 correction status mismatch')
ck(sha(REPORT)==REPORT_EXPECTED==cor['report_sha256'],'005 report hash mismatch')
report=json.loads(REPORT.read_text())
ck(report['status']=='EXACT_WARNING_PRIORITY_QUALITY_UNION_PROPOSALS_READY','005 report is not proposal-ready')
accepted={x['patch_id']:x for x in report['accepted_patches']}
ck(len(accepted)==515,'accepted patch count mismatch')
pins=json.loads(PARENT_PINS.read_text())
ck(sha(PROTECTED)==PROTECTED_EXPECTED,'protected149 receipt hash mismatch')
roles={x['role']:x for x in cor['roles']}
ck(set(roles)=={'O','I','D'},'correction roles mismatch')
counts={}; checks=0; role_records=[]
for role, rec in roles.items():
    op_records=rec['operations']; ids=[x['patch_id'] for x in op_records]
    ck(len(ids)==len(set(ids)),'duplicate operation IDs in correction '+role)
    counts[role]=len(ids)
    proposal_path=Path(rec['proposal_path'])
    ck(sha(proposal_path)==rec['proposal_sha256'],'proposal hash mismatch '+role)
    prop=load_npz(proposal_path)
    parent_pin=pins['roles'][role]
    parent_raw_path=BASE/parent_pin['raw_parent']['path']
    parent_typed_path=BASE/parent_pin['typed_parent']['path']
    parent_maps_path=BASE/parent_pin['typed_parent_maps']['path']
    ck(sha(parent_raw_path)==parent_pin['raw_parent']['sha256'],'parent raw hash mismatch '+role)
    ck(sha(parent_typed_path)==parent_pin['typed_parent']['sha256'],'parent typed hash mismatch '+role)
    ck(sha(parent_maps_path)==parent_pin['typed_parent_maps']['sha256'],'parent maps hash mismatch '+role)
    praw=load_npz(parent_raw_path); ptyped=load_npz(parent_typed_path); pmaps=load_npz(parent_maps_path)
    ck(str(np.asarray(prop['source_candidate_sha256']).reshape(-1)[0])==str(np.asarray(praw['source_candidate_sha256']).reshape(-1)[0]),'ultimate source015 SHA mismatch '+role)
    ck(str(np.asarray(prop['immediate_parent_candidate_sha256']).reshape(-1)[0])==parent_pin['raw_parent']['sha256'],'immediate parent SHA mismatch '+role)
    ck(proposal_path==Path(rec['proposal_path']),'proposal path inconsistency')
    pids=np.asarray(prop['candidate_patch_operation_ids']).astype(str).tolist()
    ck(len(pids)==len(set(pids)) and set(pids)==set(ids),'proposal and correction operation IDs differ '+role)
    point_to_parent=np.asarray(prop['point_output_to_parent_rows'],np.int64)
    point_to_source=np.asarray(prop['point_output_to_input_rows'],np.int64)
    face_to_parent=np.asarray(prop['face_output_to_parent_rows'],np.int64)
    face_to_source=np.asarray(prop['face_output_to_input_rows'],np.int64)
    faces=np.asarray(prop['faces'],np.int64)
    ck(point_to_parent.shape==(len(prop['points_world_m']),) and point_to_source.shape==point_to_parent.shape,'point maps malformed '+role)
    ck(face_to_parent.shape==(len(faces),) and face_to_source.shape==face_to_parent.shape,'face maps malformed '+role)
    ck(np.all(point_to_parent>=0) and np.all(point_to_parent<len(praw['points_world_m'])) and len(np.unique(point_to_parent))==len(point_to_parent),'immediate point map not injective '+role)
    ck(np.array_equal(point_to_source,np.asarray(praw['point_output_to_input_rows'],np.int64)[point_to_parent]),'ultimate point015 map does not compose through immediate 037 '+role)
    parent_face_map=np.asarray(pmaps['output_face_to_raw_proposal_row'] if 'output_face_to_raw_proposal_row' in pmaps else pmaps['output_face_to_raw_row'],np.int64)
    raw037_face_to_typed=np.full(len(praw['faces']),-1,np.int64); raw037_face_to_typed[parent_face_map]=np.arange(len(parent_face_map),dtype=np.int64)
    ck(np.all(face_to_parent>=-1) and np.all(face_to_parent<len(praw['faces'])),'immediate face map out of range '+role)
    retained=face_to_parent>=0
    ck(len(np.unique(face_to_parent[retained]))==int(retained.sum()),'duplicate retained parent faces '+role)
    ck(np.array_equal(face_to_source[retained],np.asarray(praw['face_output_to_input_rows'],np.int64)[face_to_parent[retained]]),'ultimate face015 map does not compose through immediate 037 '+role)
    new_rows=np.flatnonzero(~retained)
    inverse_point=np.full(len(praw['points_world_m']),-1,np.int64); inverse_point[point_to_parent]=np.arange(len(point_to_parent),dtype=np.int64)
    covered_new=[]; covered_retired=[]; op_checks=0
    for o in op_records:
        path=Path(o['path']); raw_bytes=path.read_bytes(); op=json.loads(raw_bytes)
        ck(hashlib.sha256(raw_bytes).hexdigest()==o['current_finalized_sha256'],'final operation hash mismatch '+o['patch_id'])
        ck(op['patch_id']==o['patch_id'] and op['role']==role,'operation identity mismatch '+o['patch_id'])
        pre=dict(op); pre.pop('new_output_face_rows',None); pre.pop('patches',None)
        pre_sha=hashlib.sha256((json.dumps(pre,sort_keys=True,indent=2)+'\n').encode()).hexdigest()
        ck(pre_sha==o['reconstructed_pre_finalization_sha256']==o['report_pinned_pre_finalization_sha256'],'pre-finalization hash mismatch '+o['patch_id'])
        a=accepted.get(o['patch_id']); ck(a is not None,'report patch missing '+o['patch_id'])
        ck(a['role']==role and a['source_operation_path']==str(path) and a['source_operation_sha256']==pre_sha,'report operation pin mismatch '+o['patch_id'])
        ck(a['new_face_count']==o['new_output_face_count'] and a['retired_parent_face_count']==o['retired_parent_face_count'],'report operation counts mismatch '+o['patch_id'])
        patch=op['patches']; ck(len(patch)==1 and patch[0]['patch_id']==op['patch_id'],'operation must contain one matching patch '+o['patch_id'])
        for key in ('new_output_face_rows','retired_input_face_rows','new_faces_input_point_rows','new_faces_source_point_rows'):
            ck(op[key]==patch[0].get(key,op[key]),'top/nested row mismatch '+o['patch_id']+' '+key)
        rows=np.asarray(op['new_output_face_rows'],np.int64); tris=np.asarray(op['new_faces_input_point_rows'],np.int64); src_tris=np.asarray(op['new_faces_source_point_rows'],np.int64); retired=np.asarray(op['retired_input_face_rows'],np.int64)
        ck(tris.shape==(len(rows),3) and src_tris.shape==(len(rows),3),'operation triangle maps malformed '+o['patch_id'])
        ck(np.all(rows>=0) and np.all(rows<len(faces)) and np.all(face_to_parent[rows]==-1),'new output rows not derived faces '+o['patch_id'])
        ck(np.all(tris>=0) and np.all(tris<len(inverse_point)) and np.all(inverse_point[tris]>=0),'new immediate-parent point refs unmapped '+o['patch_id'])
        ck(np.array_equal(faces[rows],inverse_point[tris]),'output triangles differ from immediate 037 point rows '+o['patch_id'])
        ck(np.array_equal(src_tris,np.asarray(praw['point_output_to_input_rows'],np.int64)[tris]),'source015 triangles differ from immediate 037 point map '+o['patch_id'])
        ck(np.all(retired>=0) and np.all(retired<len(praw['faces'])),'retired rows outside immediate 037 face domain '+o['patch_id'])
        covered_new.extend(rows.tolist()); covered_retired.extend(retired.tolist()); op_checks+=1; checks+=1
    ck(len(covered_new)==len(set(covered_new)) and set(covered_new)==set(map(int,new_rows)),'operation output rows do not partition proposal derived faces '+role)
    retired_parent=np.asarray(prop['retired_parent_face_rows'],np.int64)
    ck(len(covered_retired)==len(set(covered_retired)) and set(covered_retired)==set(map(int,retired_parent)),'operation retired rows do not partition proposal retired-parent rows '+role)
    expected_retired=np.setdiff1d(np.arange(len(praw['faces']),dtype=np.int64),face_to_parent[retained],assume_unique=True)
    ck(set(map(int,expected_retired))==set(covered_retired),'retained/retired immediate-parent face partition mismatch '+role)
    role_records.append({'role':role,'proposal_path':str(proposal_path),'proposal_sha256':sha(proposal_path),'typed_parent_sha256':parent_pin['typed_parent']['sha256'],'parent_maps_sha256':parent_pin['typed_parent_maps']['sha256'],'parent_raw_sha256':parent_pin['raw_parent']['sha256'],'operation_count':len(op_records),'derived_face_count':len(new_rows),'retired_parent_face_count':len(covered_retired),'operation_rows_independently_checked':op_checks})
ck(sum(counts.values())==515,'total operation count mismatch')
OUT.write_text(json.dumps({'schema':'quality-warning-connected-union-005-independent-operation-correction-verification.v1','status':'PASS_515_OF_515_FINALIZATION_AND_IMMEDIATE_037_PLUS_ULTIMATE_015_ROW_MAPS','correction_receipt_path':str(COR),'correction_receipt_sha256':sha(COR),'run_report_path':str(REPORT),'run_report_sha256':sha(REPORT),'protected149_receipt_path':str(PROTECTED),'protected149_receipt_sha256':sha(PROTECTED),'operation_counts_by_role':counts,'independently_checked_operation_count':checks,'roles':role_records,'scope':'Read-only operation JSON, source hashes, proposal triangle maps, and row partition verification. No geometry, native runtime, volume, mass, or admission claim.'},sort_keys=True,indent=2)+'\n')
print(json.dumps({'status':'PASS_INDEPENDENT_005_CORRECTION_VERIFY','receipt':str(OUT),'receipt_sha256':sha(OUT),'operations_checked':checks,'operation_counts_by_role':counts}))
