#!/usr/bin/env python3
"""Replay the actual O 37-face DP plan through candidate emission and typed binder.
The face patch is reduced to a local fixture, but keeps the complete pinned point
row domain so the existing protected-route checks run unchanged.
"""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np

H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
ROOT=H/'joint-native-OI-ears-279-root'
PARENT=ROOT/'O-batch002-run001/typed-280-002/candidate-O-typed-current.npz'
RAW=ROOT/'O-batch006-dp001/candidate-O-joint-native-apex-connected-raw.npz'
REPORT=ROOT/'O-batch006-dp001/report.json'
TYPED=ROOT/'O-batch006-dp001/typed-280-003/candidate-O-typed-current.npz'
TYPED_RECEIPT=ROOT/'O-batch006-dp001/typed-280-003/receipt.json'
BINDER=H/'bind_joint_boundary_typed_274_root.py'
OUT=ROOT/'controls-006/typed-fixture-output'
FIX=ROOT/'controls-006/fixture-input'
RESULT=ROOT/'controls-006/result.json'

def sha(p):
    return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()

def mod(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f'cannot load {path}')
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def save_npz(path, arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)

def subset_csr(offsets, lanes, rows, width):
    offsets=np.asarray(offsets,np.int64); out_offsets=np.zeros(len(rows)+1,np.int64); flat=[]
    for i,row in enumerate(map(int,rows)):
        lo,hi=map(int,offsets[row:row+2]);flat.extend(tuple(int(lane[j]) for lane in lanes) for j in range(lo,hi));out_offsets[i+1]=len(flat)
    values=np.asarray(flat,np.int64).reshape((-1,width)) if flat else np.empty((0,width),np.int64)
    return out_offsets,values

def main():
    if OUT.exists() or FIX.exists() or RESULT.exists(): raise FileExistsError('fixture output already exists')
    for p,expected in (
        (PARENT,'a1fd7fe287584b6fbe2e1c97ecafcd376dfc7a97f0165cb0c5a56c557d575dfc'),
        (RAW,'3cfd683d97c211cb07a976ef92469689276f618e183c9130401d7259320906c0'),
        (TYPED,'4e0c46662721fed33d4bd4efb390c6668ff67581d2484fcd9ba9058834fa7a20'),
        (REPORT,'35812a03c529505797a89857dc2b09eb438e45bb00e9d4b42fa4635cc5985792'),
        (TYPED_RECEIPT,'256ad35954cf5f99638d785b6138ad20a9c62e38855d6306b08193adb959642c')):
        actual=sha(p)
        if actual!=expected: raise ValueError(f'frozen input SHA mismatch {p}: {actual}')
    binder=mod('binder_fixture_actual_o',BINDER)
    B=binder.B; A=binder.A
    rep=json.loads(REPORT.read_text())
    receipt=json.loads(TYPED_RECEIPT.read_text())
    assert rep['status']=='PASS_BOUNDED_CONNECTED_CLUSTER_INTERVAL_DP_FIXED_COORDINATE_PROPOSAL'
    assert rep['physical_admission'] is False and rep['native_steps']==0
    assert rep['joint_complete_surface_walls_all185_pairs']==0 and rep['joint_changed_self_pairs']==0
    assert receipt['status']=='PASS_CURRENT_ROW_TYPED_ANCESTRY_ADAPTER_ONLY'
    opmeta=receipt['operation_docs']
    assert len(opmeta)==1 and opmeta[0]['sha256']=='801fe6e046511278de763219ff1f5d55718416e7613270bbbac731332d6933cf'
    op=json.loads(Path(opmeta[0]['path']).read_text())
    assert op['role']=='O' and op['seed_face']==44399
    with np.load(PARENT,allow_pickle=False) as z: parent={k:np.array(z[k],copy=True) for k in z.files}
    with np.load(RAW,allow_pickle=False) as z: raw={k:np.array(z[k],copy=True) for k in z.files}
    raw_rec=rep['outputs']['O'];assert raw_rec['path']==str(RAW) and raw_rec['sha256']==sha(RAW)
    assert raw_rec['parent']==str(PARENT) and raw_rec['parent_sha256']==sha(PARENT)
    rows=np.asarray(op['retired_parent_face_rows'],np.int64)
    assert len(rows)==37 and len(np.unique(rows))==37
    out_rows=np.asarray(op['new_output_face_rows'],np.int64)
    new_faces=np.asarray(op['new_faces'],np.int64)
    assert len(out_rows)==len(new_faces)==13
    assert np.array_equal(raw['faces'][out_rows],new_faces)
    assert np.array_equal(raw['faces'][out_rows],np.asarray(TYPED and np.load(TYPED,allow_pickle=False)['faces'][out_rows],np.int64))
    # Recompute complete patch support from the actual 37 parent faces.
    union=B.union_tuples(parent,rows); ancestry=B.ancestry_union(parent,rows)
    assert union==[tuple(map(int,x)) for x in op['complete_typed_parent_union']]
    assert B.union_digest(union)==op['typed_parent_union_sha256']
    assert {tuple(map(int,x)) for x in ancestry}=={tuple(map(int,x)) for x in op['source_face_ancestry_union']}
    # Build a fixture parent with the full protected point domain but only the
    # actual retired local 37-face patch; all ancestry rows are sliced exactly.
    face_keys=('face_source_layer','face_source_body','face_source_bone','face_source_face_id','face_source_original_id','face_source_identity0','face_source_identity1')
    ancestry_keys=('face_ancestry_source_face_rows','face_ancestry_source_original_ids','face_ancestry_parent_face_rows')
    union_keys=('quality_star_patch_union_face_source_layer','quality_star_patch_union_face_source_body','quality_star_patch_union_face_source_bone','quality_star_patch_union_face_source_face_id','quality_star_patch_union_face_source_original_id','quality_star_patch_union_face_source_identity0','quality_star_patch_union_face_source_identity1')
    fixture_parent={k:np.array(v,copy=True) for k,v in parent.items() if k not in face_keys+ancestry_keys+union_keys+('face_ancestry_offsets','quality_star_patch_union_offsets','faces','quality_star_face_output_to_input_row')}
    fixture_parent['faces']=np.asarray(parent['faces'][rows],np.int64)
    fixture_parent['quality_star_face_output_to_input_row']=np.asarray(parent['quality_star_face_output_to_input_row'],np.int64)[rows]
    for k in face_keys: fixture_parent[k]=np.asarray(parent[k])[rows]
    fixture_parent['face_ancestry_offsets'],a_flat=subset_csr(parent['face_ancestry_offsets'],[parent[k] for k in ancestry_keys],rows,3)
    for j,k in enumerate(ancestry_keys):fixture_parent[k]=a_flat[:,j]
    fixture_parent['quality_star_patch_union_offsets'],u_flat=subset_csr(parent['quality_star_patch_union_offsets'],[parent[k] for k in union_keys],rows,7)
    for j,k in enumerate(union_keys):fixture_parent[k]=u_flat[:,j]
    FIX.mkdir(parents=True)
    parent_path=FIX/'parent-37-face-patch.npz';save_npz(parent_path,fixture_parent)
    parent_sha=sha(parent_path)
    # Emitted candidate contains the exact found plan only. This exercises the
    # binder's local-face/global-point namespace and its output CSR construction.
    fixture_raw={
        'points_world_m':np.asarray(raw['points_world_m']).copy(),
        'points_world_m_f32':np.asarray(raw['points_world_m_f32'],np.float32).copy(),
        'faces':new_faces.copy(),
        'point_output_to_parent_rows':np.arange(len(raw['points_world_m']),dtype=np.int64),
        'face_output_to_parent_rows':np.full(len(new_faces),-1,dtype=np.int64),
        'immediate_parent_candidate_sha256':np.asarray([parent_sha]),
        'new_faces_parent_point_rows':new_faces.copy(),
    }
    u_offsets=np.zeros(len(new_faces)+1,np.int64);u_rows=[]
    for i in range(len(new_faces)):
        u_rows.extend(union);u_offsets[i+1]=len(u_rows)
    u_array=np.asarray(u_rows,np.int64).reshape((-1,7))
    fixture_raw['new_face_patch_union_offsets']=u_offsets
    for j,k in enumerate(A.RU):fixture_raw[k]=u_array[:,j]
    raw_path=FIX/'raw-found-plan-37-to-13.npz';save_npz(raw_path,fixture_raw);raw_sha=sha(raw_path)
    local_op={
      'role':'O','seed_face':int(op['seed_face']),
      'retired_parent_face_rows':list(range(len(rows))),
      'parent_face_rows':list(range(len(rows))),
      'new_output_face_rows':list(range(len(new_faces))),
      'new_faces':new_faces.tolist(),
      'typed_parent_union':[list(map(int,x)) for x in union],
      'typed_parent_union_sha256':B.union_digest(union),
      'source_face_ancestry_union':[list(map(int,x)) for x in ancestry],
    }
    local_report={
      'status':'PASS_LOCAL_CAPTURED_FOUND_PLAN_FIXTURE','physical_admission':False,'native_steps':0,
      'joint_complete_surface_walls_all185_pairs':0,'joint_changed_self_pairs':0,
      'outputs':{'O':{'path':str(raw_path),'sha256':raw_sha,'parent':str(parent_path),'parent_sha256':parent_sha}},
      'accepted_operations':[local_op],
      'quality_metric':'joint_volume_shape_apex_upper_bound',
      'joint_bound_owner_sha256':rep['joint_bound_owner_sha256'],
      'construction_surface_floor':rep['construction_surface_floor'],
    }
    local_report_path=FIX/'captured-plan-report.json';local_report_path.write_text(json.dumps(local_report,indent=2,sort_keys=True)+'\n')
    binder_out=OUT/'typed';OUT.mkdir(parents=True)
    typed_receipt=binder.apply(local_report_path,binder_out)
    typed_path=binder_out/'candidate-O-typed-current.npz'
    assert typed_path.is_file()
    with np.load(typed_path,allow_pickle=False) as z: typed={k:np.array(z[k],copy=True) for k in z.files}
    assert np.array_equal(typed['faces'],new_faces)
    assert np.array_equal(typed['points_world_m_f32'],fixture_raw['points_world_m_f32'])
    assert len(typed['quality_star_patch_union_offsets'])==len(new_faces)+1
    assert len(typed['face_ancestry_offsets'])==len(new_faces)+1
    assert np.array_equal(typed['quality_star_face_output_to_input_row'],np.full(len(new_faces),-1,np.int64))
    for i in range(len(new_faces)):
        lo,hi=map(int,typed['quality_star_patch_union_offsets'][i:i+2])
        assert [tuple(map(int,row)) for row in np.stack([typed[k][lo:hi] for k in union_keys],axis=1)]==union
        lo,hi=map(int,typed['face_ancestry_offsets'][i:i+2])
        assert [tuple(map(int,row)) for row in np.stack([typed[k][lo:hi] for k in ancestry_keys],axis=1)]==ancestry
    result={
      'status':'PASS_ACTUAL_O_FOUND_PLAN_FULL_TYPED_BINDER_LOCAL_FIXTURE',
      'physical_admission':False,'native_steps':0,
      'original_exact_plan_report':{'path':str(REPORT),'sha256':sha(REPORT)},
      'original_typed_receipt':{'path':str(TYPED_RECEIPT),'sha256':sha(TYPED_RECEIPT)},
      'original_parent':{'path':str(PARENT),'sha256':sha(PARENT)},
      'original_raw_candidate':{'path':str(RAW),'sha256':sha(RAW)},
      'original_typed_candidate':{'path':str(TYPED),'sha256':sha(TYPED)},
      'operation_document':{'path':opmeta[0]['path'],'sha256':opmeta[0]['sha256'],
                            'parent_face_count':len(rows),'output_face_count':len(new_faces),
                            'full_typed_union_tuples':len(union),'full_face_ancestry_tuples':len(ancestry)},
      'fixture':{'parent_path':str(parent_path),'parent_sha256':parent_sha,'parent_faces':len(rows),
                 'raw_path':str(raw_path),'raw_sha256':raw_sha,'raw_faces':len(new_faces),
                 'same_full_point_domain':True,'actual_found_plan_replayed':True},
      'typed_binder':{'source_sha256':sha(BINDER),'typed_owner_sha256':sha(Path(binder.A.__file__)),
                      'typed_output_path':str(typed_path),'typed_output_sha256':sha(typed_path),
                      'typed_receipt':typed_receipt,'union_csr_matches_complete_patch_union_for_every_face':True,
                      'source_face_ancestry_csr_matches_for_every_face':True,
                      'generated_face_source015_rows_all_minus_one':True,
                      'protected_routes_unchanged':True},
    }
    RESULT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))
    print('result_sha256',sha(RESULT))

if __name__=='__main__':main()
