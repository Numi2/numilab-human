#!/usr/bin/env python3
"""Read-only operation-hash and row-map correction for quality-union run-005."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
RUN=ROOT/'quality-warning-connected-union-038/run-005'
REPORT_SHA='e81f5753da8d4ae6c05de0f24bbe70f107ecdc34a0c42f571813e8d23a7bcc56'
REMOVED=['new_output_face_rows','patches']
def sha(b): return hashlib.sha256(b).hexdigest()
def file_sha(p): return sha(Path(p).read_bytes())
def prefinal(doc):
 d=dict(doc); gone=[]
 for k in REMOVED:
  if k in d: gone.append(k); del d[k]
 if gone!=REMOVED: raise ValueError('expected exactly both finalization fields')
 return (json.dumps(d,indent=2,sort_keys=True)+'\n').encode()
def check_op(doc, summary, report_map, a):
 if (doc.get('patch_id'),doc.get('role'))!=(summary.get('patch_id'),summary.get('role')): raise ValueError('operation identity mismatch')
 if len(doc.get('patches',[]))!=1: raise ValueError('expected one finalized nested patch')
 nested=doc['patches'][0]
 for k in ('patch_id','new_output_face_rows','retired_input_face_rows','source_initial_face_rows','new_faces_input_point_rows','source_lineage'):
  if nested.get(k)!=doc.get(k): raise ValueError('nested operation mismatch: '+k)
 outrows=np.asarray(doc['new_output_face_rows'],np.int64)
 if not np.array_equal(outrows,np.asarray(report_map['output_face_rows'],np.int64)): raise ValueError('operation/report output rows differ')
 if not np.array_equal(np.asarray(doc['retired_input_face_rows'],np.int64),np.asarray(report_map['retired_parent_face_rows'],np.int64)): raise ValueError('operation/report retired parent rows differ')
 if len(outrows)!=int(summary['new_face_count']): raise ValueError('output face count mismatch')
 pos=np.asarray([a['row_to_derived'][int(x)] for x in outrows],np.int64)
 if np.any(pos<0): raise ValueError('operation output row absent from derived domain')
 tri=a['faces'][outrows]
 if np.any(a['face_parent'][outrows]!=-1) or np.any(a['face_source'][outrows]!=-1): raise ValueError('derived output face has source face')
 if not np.array_equal(a['point_parent'][tri],np.asarray(doc['new_faces_input_point_rows'],np.int64)): raise ValueError('immediate-parent triangle points differ')
 if not np.array_equal(a['point_source'][tri],np.asarray(doc['new_faces_source_point_rows'],np.int64)): raise ValueError('ultimate-source triangle points differ')
 if not np.all(a['patch_id'][pos]==doc['patch_id']): raise ValueError('patch ID array differs')
 return {'patch_id':doc['patch_id'],'new_output_face_count':len(outrows),'retired_parent_face_count':len(doc['retired_input_face_rows'])}
def inspect_role(role, report):
 out=report['outputs'][role]; path=Path(out['path'])
 if file_sha(path)!=out['sha256']: raise ValueError(role+' proposal SHA mismatch')
 with np.load(path,allow_pickle=False) as z: z={k:np.array(z[k],copy=True) for k in z.files}
 a={'faces':z['faces'],'face_parent':z['face_output_to_parent_rows'],'face_source':z['face_output_to_input_rows'],
    'point_parent':z['point_output_to_parent_rows'],'point_source':z['point_output_to_input_rows'],'patch_id':z['new_output_face_patch_id'].astype(str)}
 derived=np.flatnonzero(a['face_parent']==-1)
 source_derived=np.flatnonzero(a['face_source']==-1)
 if not set(map(int,derived)).issubset(set(map(int,source_derived))): raise ValueError(role+' immediate-derived faces absent from source-derived domain')
 a['row_to_derived']=np.full(len(a['faces']),-1,np.int64); a['row_to_derived'][source_derived]=np.arange(len(source_derived))
 ops=[x for x in report['accepted_patches'] if x['role']==role]; maps={x['patch_id']:x for x in out['patch_face_map']}
 if len(maps)!=len(out['patch_face_map']) or set(maps)!={x['patch_id'] for x in ops}: raise ValueError(role+' patch inventory mismatch')
 covered=[]; retired=[]; rec=[]
 for sm in ops:
  op_path=Path(sm['source_operation_path']); doc=json.loads(op_path.read_text()); final_sha=file_sha(op_path)
  expected=sm['source_operation_sha256']; reconstructed=sha(prefinal(doc))
  if reconstructed!=expected or final_sha==expected: raise ValueError(role+'/'+sm['patch_id']+' report hash reconstruction failed')
  detail=check_op(doc,sm,maps[sm['patch_id']],a)
  covered.extend(map(int,doc['new_output_face_rows'])); retired.extend(map(int,doc['retired_input_face_rows']))
  rec.append({**detail,'path':str(op_path),'report_pinned_pre_finalization_sha256':expected,
    'reconstructed_pre_finalization_sha256':reconstructed,'current_finalized_sha256':final_sha,
    'removed_finalization_fields':REMOVED,'status':'PASS_HASH_AND_ROW_MAPS'})
 if len(set(covered))!=len(covered) or set(covered)!=set(map(int,derived)): raise ValueError(role+' operation output rows do not cover derived faces exactly')
 if len(set(retired))!=len(retired) or not np.array_equal(np.sort(np.asarray(retired,np.int64)),np.sort(z['retired_parent_face_rows'])): raise ValueError(role+' retired parent-face union mismatch')
 return {'role':role,'proposal_path':str(path),'proposal_sha256':out['sha256'],'operation_count':len(rec),'derived_face_count':len(derived),'retired_parent_face_count':len(retired),'operations':rec}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,default=RUN/'report.json');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if file_sha(a.report)!=REPORT_SHA: raise ValueError('run-005 report pin mismatch')
 r=json.loads(a.report.read_text())
 if r.get('status')!='EXACT_WARNING_PRIORITY_QUALITY_UNION_PROPOSALS_READY': raise ValueError('run-005 status')
 launch=json.loads((RUN/'launch.json').read_text()); roles=[inspect_role(x,r) for x in ('O','I','D')]; n=sum(x['operation_count'] for x in roles)
 if n!=r['accepted_patch_count']: raise ValueError('operation count mismatch')
 doc={'schema':'h706-quality-operation-finalization-correction.v1','status':'PASS_RUN005_FINALIZED_OPERATION_HASHES_AND_ROW_MAPS',
  'scope':'Read-only hash reconstruction and output-row validation; no source operation, report, geometry, physics, volume, or admission changes.',
  'report_path':str(a.report),'report_sha256':REPORT_SHA,'producer_script_path':launch['script'],'producer_script_sha256':launch['script_sha256'],
  'operation_count':n,'roles':roles,'canonical_reconstruction':'UTF-8 JSON sort_keys=true indent=2 plus final newline after removing exactly new_output_face_rows and patches.',
  'source_files_modified':False,'physical_admission':False,'runtime':0}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':doc['status'],'operation_count':n,'out':str(a.out),'sha256':file_sha(a.out)},indent=2))
if __name__=='__main__': main()
