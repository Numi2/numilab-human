#!/usr/bin/env python3
"""Compose disjoint fixed-coordinate O face patches from 038 runs 004 and 005."""
from pathlib import Path
import hashlib,json
import numpy as np

P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
OUT=P/'quality-warning-connected-union-038/merge-run004-run005-O'
R4=P/'quality-warning-connected-union-038/run-004/report.json';R4SHA='594449c761e803e11fb16dfec0c3516e56141964b1ae640f72217fa31bfa5222'
R5=P/'quality-warning-connected-union-038/run-005/report.json';R5SHA='e81f5753da8d4ae6c05de0f24bbe70f107ecdc34a0c42f571813e8d23a7bcc56'
C4=P/'quality-operation-hash-correction-038-001/operation-hash-correction-038.json';C4SHA='b9770e626752a835d32225434f12eecd399429cf34402d5304d1fca78e255c89'
C5=P/'quality-operation-hash-correction-005-001/operation-hash-correction-005.json';C5SHA='da58a44d5135aa8b4370ff5db060ba4c10ed7e4c7de8f2e183568b416d128991'
R37=P/'quality-warning-connected-union-037/run-002/report.json'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def merge_face_operations(base_faces, mandatory, additions):
    used=set();selected=[];skipped=[]
    for op in mandatory:
        rows=set(map(int,op['retired_input_face_rows']))
        if rows&used:raise ValueError('mandatory operations overlap retired rows')
        used|=rows;selected.append(op)
    for op in additions:
        rows=set(map(int,op['retired_input_face_rows']))
        if rows&used:skipped.append(op);continue
        used|=rows;selected.append(op)
    if any(r<0 or r>=len(base_faces) for r in used):raise ValueError('retired row out of bounds')
    keep=np.flatnonzero(~np.isin(np.arange(len(base_faces),dtype=np.int64),np.fromiter(used,dtype=np.int64)))
    parts=[np.asarray(base_faces,np.int64)[keep]]
    parts.extend(np.asarray(op['new_faces_input_point_rows'],np.int64).reshape((-1,3)) for op in selected)
    return np.concatenate(parts,axis=0),keep,used,selected,skipped
def directed_boundary(f):
    f=np.asarray(f,np.int64);e=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]));s=np.sort(e,axis=1)
    u,inv,c=np.unique(s,axis=0,return_inverse=True,return_counts=True)
    if np.any(c>2):raise ValueError('nonmanifold edge')
    bal=np.bincount(inv,weights=np.where(e[:,0]==s[:,0],1,-1),minlength=len(u))
    if np.any((c==2)&(bal!=0)):raise ValueError('interior edge orientation mismatch')
    return sorted(map(tuple,e[c[inv]==1].tolist()))
def load_ops(report,receipt,role,stage,proposal):
    if not receipt.get('status','').startswith('PASS_'):raise ValueError(stage+' operation correction receipt failed')
    correction={x['path']:x for rr in receipt['roles'] for x in rr['operations']}
    patchmap={x['patch_id']:x for x in report['outputs'][role]['patch_face_map']}
    if len(patchmap)!=len(report['outputs'][role]['patch_face_map']):raise ValueError(stage+' duplicate patch map IDs')
    PPM=proposal['point_output_to_parent_rows'];PPS=proposal['point_output_to_input_rows']
    FPM=proposal['face_output_to_parent_rows'];FPS=proposal['face_output_to_input_rows'];faces=proposal['faces']
    result=[]
    for sm in report['accepted_patches']:
        if sm['role']!=role:continue
        path=str(Path(sm['source_operation_path']));doc=json.loads(Path(path).read_text());c=correction.get(path)
        if c is None or c.get('status')!='PASS_HASH_AND_ROW_MAPS':raise ValueError(stage+' operation lacks corrected pin')
        finalized=c.get('current_finalized_sha256',c.get('finalized_sha256'))
        if sha(path)!=finalized or sm['source_operation_sha256']!=c['report_pinned_pre_finalization_sha256']:raise ValueError(stage+' operation hash mismatch')
        pid=doc['patch_id'];pm=patchmap.get(pid)
        if pm is None or doc.get('role')!=role:raise ValueError(stage+' operation identity mismatch')
        rows=np.asarray(doc['new_output_face_rows'],np.int64)
        if not np.array_equal(rows,np.asarray(pm['output_face_rows'],np.int64)):raise ValueError(stage+' output face rows mismatch')
        if not np.array_equal(np.asarray(doc['retired_input_face_rows'],np.int64),np.asarray(pm['retired_parent_face_rows'],np.int64)):raise ValueError(stage+' retired parent rows mismatch')
        tri=faces[rows]
        if np.any(FPM[rows]!=-1) or np.any(FPS[rows]!=-1):raise ValueError(stage+' new faces lack derived markers')
        if not np.array_equal(PPM[tri],np.asarray(doc['new_faces_input_point_rows'],np.int64)):raise ValueError(stage+' parent point triangles mismatch')
        if not np.array_equal(PPS[tri],np.asarray(doc['new_faces_source_point_rows'],np.int64)):raise ValueError(stage+' source point triangles mismatch')
        result.append({'stage':stage,'patch_id':pid,'doc':doc,'summary':sm,'source_operation_path':path,'source_operation_sha256':finalized})
    if len(result)!=sum(x['role']==role for x in report['accepted_patches']):raise ValueError(stage+' role operation count mismatch')
    return result
def main():
    for p,d in ((R4,R4SHA),(R5,R5SHA),(C4,C4SHA),(C5,C5SHA)):
        if sha(p)!=d:raise ValueError('pinned input changed: '+str(p))
    r4=json.loads(R4.read_text());r5=json.loads(R5.read_text());c4=json.loads(C4.read_text());c5=json.loads(C5.read_text());r37=json.loads(R37.read_text())
    if r4.get('status')!='EXACT_WARNING_PRIORITY_QUALITY_UNION_PROPOSALS_READY' or r5.get('status')!='EXACT_WARNING_PRIORITY_QUALITY_UNION_PROPOSALS_READY':raise ValueError('proposal report status')
    if r37.get('status')!='WARNING_LOCAL_EXACT_UNION_PROPOSALS_READY':raise ValueError('037 baseline status')
    parent_path=Path(r37['outputs']['O']['path']);parent_sha=r37['outputs']['O']['sha256'];parent=load(parent_path)
    source_path=Path(r37['source_015']['O']['path']);source_sha=r37['source_015']['O']['sha256'];source=load(source_path)
    base_points=np.asarray(parent['points_world_m']);base_faces=np.asarray(parent['faces'],np.int64)
    point_parent=np.asarray(parent['point_output_to_input_rows'],np.int64);face_parent=np.asarray(parent['face_output_to_input_rows'],np.int64)
    source_points=np.asarray(source['points_world_m'])
    if not np.array_equal(source_points[point_parent],base_points):raise ValueError('037 points do not reconstruct exact source015 coordinates')
    r4p=load(Path(r4['outputs']['O']['path']));r5p=load(Path(r5['outputs']['O']['path']))
    ops4=load_ops(r4,c4,'O','004',r4p);ops5=load_ops(r5,c5,'O','005',r5p)
    for stage,proposal in (('004',r4p),('005',r5p)):
        if str(proposal['immediate_parent_candidate_sha256'].reshape(-1)[0])!=parent_sha:raise ValueError(stage+' immediate parent not the same 037 O')
        if str(proposal['source_candidate_sha256'].reshape(-1)[0])!=source_sha:raise ValueError(stage+' source015 pin mismatch')
    ids4={x['patch_id'] for x in ops4};ids5={x['patch_id'] for x in ops5}
    if ids4&ids5:raise ValueError('patch ID collision')
    mandatory=[{'patch_id':x['patch_id'],'retired_input_face_rows':x['doc']['retired_input_face_rows'],'new_faces_input_point_rows':x['doc']['new_faces_input_point_rows']} for x in ops5]
    additions=[{'patch_id':x['patch_id'],'retired_input_face_rows':x['doc']['retired_input_face_rows'],'new_faces_input_point_rows':x['doc']['new_faces_input_point_rows']} for x in ops4]
    merged,keep,retired,selected_simple,skipped_simple=merge_face_operations(base_faces,mandatory,additions)
    selected_ids={x['patch_id'] for x in selected_simple};selected5=ops5;selected4=[x for x in ops4 if x['patch_id'] in selected_ids]
    chosen=selected5+selected4
    new_parent_faces=np.concatenate([np.asarray(x['doc']['new_faces_input_point_rows'],np.int64).reshape((-1,3)) for x in chosen],axis=0) if chosen else np.empty((0,3),np.int64)
    if len(merged)!=len(keep)+len(new_parent_faces):raise ValueError('operation triangle count mismatch')
    if merged.min()<0 or merged.max()>=len(base_points):raise ValueError('merged parent point row out of bounds')
    if directed_boundary(merged)!=directed_boundary(base_faces):raise ValueError('merged oriented boundary differs from 037 O')
    used=np.unique(merged);parent_to_output=np.full(len(base_points),-1,np.int64);parent_to_output[used]=np.arange(len(used))
    out_faces=parent_to_output[merged];srcpoint_rows=point_parent[used];out_points=source_points[srcpoint_rows]
    if out_points.dtype!=source_points.dtype or not np.array_equal(out_points,base_points[used]):raise ValueError('exact source coordinate/dtype parity failed')
    out_f32=np.asarray(out_points,np.float32)
    if 'points_world_m_f32' in parent and not np.array_equal(out_f32,np.asarray(parent['points_world_m_f32'])[used]):raise ValueError('Float32 view differs from 037 parent')
    point_source_inverse=np.full(len(source_points),-1,np.int64);point_source_inverse[srcpoint_rows]=np.arange(len(used),dtype=np.int64)
    source_face_map=np.concatenate((face_parent[keep],np.full(len(new_parent_faces),-1,np.int64)))
    face_parent_rows=np.concatenate((keep,np.full(len(new_parent_faces),-1,np.int64)))
    direct=source_face_map[source_face_map>=0]
    if len(np.unique(direct))!=len(direct):raise ValueError('duplicate direct source face rows')
    face_source_inverse=np.full(len(source['faces']),-1,np.int64);face_source_inverse[direct]=np.flatnonzero(source_face_map>=0)
    retired_source=np.setdiff1d(np.arange(len(source['faces']),dtype=np.int64),direct)
    # Preserve older 037-created faces and append the selected 004/005 operation records.
    old_derived=np.flatnonzero(face_parent<0);old_ids=np.asarray(parent.get('new_output_face_patch_id',np.empty(0,dtype='U80'))).astype(str)
    old_stage=np.asarray(parent.get('new_output_face_patch_source',np.empty(0,dtype='U8'))).astype(str)
    old_num=np.asarray(parent.get('new_output_face_patch_ids',np.full(len(old_derived),-1)),np.int64)
    if not (len(old_ids)==len(old_derived)==len(old_stage)==len(old_num)):raise ValueError('037 derived-face metadata alignment')
    oldmeta={int(row):(old_ids[i],old_stage[i],int(old_num[i])) for i,row in enumerate(old_derived)}
    old_opids=np.asarray(parent.get('candidate_patch_operation_ids',np.empty(0,dtype='U80'))).astype(str).tolist()
    chosen_ids=[x['patch_id'] for x in chosen];op_ids=old_opids+chosen_ids
    op_index={pid:len(old_opids)+i for i,pid in enumerate(chosen_ids)}
    output_rows_by_patch={};cursor=len(keep)
    for op in chosen:
        n=len(op['doc']['new_faces_input_point_rows']);output_rows_by_patch[op['patch_id']]=list(range(cursor,cursor+n));cursor+=n
    derived_output_rows=np.flatnonzero(source_face_map<0);patch_ids=[];patch_stages=[];patch_nums=[];parent_triangles=[];source_triangles=[]
    op_by_output={}
    for op in chosen:
        for row in output_rows_by_patch[op['patch_id']]:op_by_output[row]=op
    for row in derived_output_rows:
        par=int(face_parent_rows[row]);tri_parent=merged[row]
        if par>=0:
            ident,stage,num=oldmeta[par]
        else:
            op=op_by_output.get(int(row))
            if op is None:raise ValueError('new face has no source operation')
            ident=op['patch_id'];stage='038';num=op_index[ident]
        patch_ids.append(ident);patch_stages.append(stage);patch_nums.append(num)
        parent_triangles.append(tri_parent);source_triangles.append(point_parent[tri_parent])
    parent_triangles=np.asarray(parent_triangles,np.int64).reshape((-1,3));source_triangles=np.asarray(source_triangles,np.int64).reshape((-1,3))
    output_npz=OUT/'O-merged-run004-run005-proposal.npz';OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output_npz,points_world_m=out_points,points_world_m_f32=out_f32,faces=out_faces,
      source_candidate_sha256=np.asarray([source_sha]),immediate_parent_candidate_sha256=np.asarray([parent_sha]),
      point_output_to_input_rows=srcpoint_rows,point_input_to_output_row=point_source_inverse,
      point_output_to_parent_rows=used,point_parent_input_to_output_row=parent_to_output,
      face_output_to_input_rows=source_face_map,face_input_to_output_row=face_source_inverse,
      face_output_to_parent_rows=face_parent_rows,retired_input_face_rows=retired_source,
      retired_parent_face_rows=np.asarray(sorted(retired),np.int64),
      new_faces_input_point_rows=source_triangles,new_faces_parent_point_rows=parent_triangles,
      new_output_face_patch_ids=np.asarray(patch_nums,np.int64),new_output_face_patch_source=np.asarray(patch_stages,dtype='U8'),
      new_output_face_patch_id=np.asarray(patch_ids,dtype='U80'),candidate_patch_operation_ids=np.asarray(op_ids,dtype='U80'))
    map_rows=[]
    for op in chosen:
        map_rows.append({'patch_id':op['patch_id'],'source_run':op['stage'],'source_operation_path':op['source_operation_path'],
          'source_operation_sha256':op['source_operation_sha256'],'source_report_sha256':R4SHA if op['stage']=='004' else R5SHA,
          'retired_037_face_rows':op['doc']['retired_input_face_rows'],'new_faces_037_point_rows':op['doc']['new_faces_input_point_rows'],
          'new_faces_015_point_rows':op['doc']['new_faces_source_point_rows'],'merged_output_face_rows':output_rows_by_patch[op['patch_id']],
          'source_lineage':op['doc']['source_lineage'],'lineage_semantics':'Flattened patch-level source union, not per-triangle overlap.'})
    conflict_ids={x['patch_id'] for x in skipped_simple}
    skipped=[{'patch_id':x['patch_id'],'stage':'004','overlapping_005_retired_037_rows':sorted(set(map(int,x['retired_input_face_rows']))&set().union(*(set(map(int,y['retired_input_face_rows'])) for y in mandatory)))} for x in skipped_simple]
    doc={'schema':'h706.quality-union-038-O-merge.v1','status':'PASS_MAP_COMPOSITION_GEOMETRY_PROOF_PENDING',
      'scope':'Fixed-coordinate face-map composition only; no exact collision re-audit, no TetGen, no physical admission.',
      'run004_report':{'path':str(R4),'sha256':R4SHA},'run005_report':{'path':str(R5),'sha256':R5SHA},
      'run004_correction':{'path':str(C4),'sha256':C4SHA},'run005_correction':{'path':str(C5),'sha256':C5SHA},
      '037_parent':{'path':str(parent_path),'sha256':parent_sha},'source015':{'path':str(source_path),'sha256':source_sha},
      'inputs':{'run004_O':r4['outputs']['O'],'run005_O':r5['outputs']['O']},
      'output':{'path':str(output_npz),'sha256':sha(output_npz),'point_count':len(out_points),'face_count':len(out_faces),
        'new_face_count':len(new_parent_faces),'retired_037_face_count':len(retired),'directed_boundary_matches_037':True},
      'operation_selection':{'run005_O_kept':len(ops5),'run004_O_total':len(ops4),'run004_O_added_disjoint':len(selected4),
        'run004_O_skipped_overlap':len(skipped),'skipped':skipped,'merged_operations':map_rows},
      'all_run005_O_operations_retained':len(selected5)==len(ops5),'patch_ids_unique':True,
      'source_files_modified':False,'coordinates_changed':False,'physical_admission':False,'runtime':0}
    report_path=OUT/'merge-report.json';report_path.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':doc['status'],'output':doc['output'],'run005_kept':len(ops5),'run004_added':len(selected4),'run004_skipped':len(skipped),'report':str(report_path),'report_sha256':sha(report_path)},indent=2))
if __name__=='__main__':main()
