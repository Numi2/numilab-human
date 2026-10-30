#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np

TYPED_FIELDS=("face_source_layer","face_source_body","face_source_bone","face_source_face_id","face_source_original_id","face_source_identity0","face_source_identity1")
FACE_CSR={
 "face_ancestry_offsets":("face_ancestry_source_face_rows","face_ancestry_source_original_ids","face_ancestry_parent_face_rows"),
 "face_parent_offsets":("face_parent_rows",),
 "edge_flip_parent_offsets":("edge_flip_parent_face_rows",),
 "microedge_parent_input_face_offsets":("microedge_parent_input_face_rows",),
 "candidate_face_ancestry_offsets":("candidate_face_ancestry_rows",),
 "union_face_ancestry_offsets":("union_face_ancestry_rows","union_face_ancestry_run_original_ids","union_face_ancestry_face_ids"),
 "operational_output_face_parent_offsets":("operational_output_face_parent_rows",),
}
POINT_CSR={
 "microedge_parent_input_point_offsets":("microedge_parent_input_point_rows",),
 "reference_vertex_parent_face_offsets":("reference_vertex_parent_face_rows","reference_vertex_parent_face_barycentric"),
 "reference_original_input_vertex_offsets":("reference_original_input_vertex_rows",),
 "output_vertex_input_parent_offsets":("output_vertex_input_parent_rows",),
}
SEMANTICS=("Each connected-union replacement triangle carries the complete deduplicated seven-field typed identity union of every retired parent face, recursively flattening earlier quality-star patch unions. This is patch-level source support, not a per-triangle overlap or single-face ancestry claim.")

def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def load_npz(path:Path)->dict[str,np.ndarray]:
 with np.load(path,allow_pickle=False) as z:return {k:np.asarray(z[k]).copy() for k in z.files}

def canonical_json(x:Any)->str:return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True)
def hash_json(x:Any)->str:return hashlib.sha256(canonical_json(x).encode('utf-8')).hexdigest()
def aligned_tuples(arrays:dict[str,np.ndarray],rows:np.ndarray)->list[tuple[int,...]]:
 rows=np.asarray(rows,np.int64)
 if rows.ndim!=1 or not len(rows) or np.any(rows<0) or np.any(rows>=len(arrays[TYPED_FIELDS[0]])) or len(np.unique(rows))!=len(rows):
  raise ValueError('retired parent face rows are empty, repeated, or out of range')
 return sorted({tuple(int(arrays[k][r]) for k in TYPED_FIELDS) for r in rows})

def face_union_for_parent_rows(parent:dict[str,np.ndarray],rows:np.ndarray)->list[tuple[int,...]]:
 rows=np.asarray(rows,np.int64)
 if np.any(rows<0) or np.any(rows>=len(parent['faces'])) or len(np.unique(rows))!=len(rows) or not len(rows):
  raise ValueError('invalid retired typed-parent face rows')
 offsets=parent.get('quality_star_patch_union_offsets')
 lane_names=[f'quality_star_patch_union_{k}' for k in TYPED_FIELDS]
 if offsets is None or any(k not in parent for k in lane_names):return aligned_tuples(parent,rows)
 offsets=np.asarray(offsets,np.int64)
 if offsets.shape!=(len(parent['faces'])+1,) or offsets[0]!=0 or np.any(offsets[1:]<offsets[:-1]):raise ValueError('parent patch-union offsets malformed')
 lanes=[np.asarray(parent[k]) for k in lane_names]
 if any(len(a)!=int(offsets[-1]) for a in lanes):raise ValueError('parent patch-union lanes misaligned')
 out=set()
 for r in rows:
  a,b=map(int,offsets[r:r+2])
  if b>a:
   out.update(tuple(int(v) for v in row) for row in np.column_stack([x[a:b] for x in lanes]))
  else:
   out.add(tuple(int(parent[k][r]) for k in TYPED_FIELDS))
 return sorted(out)

def csr_tuples(offsets:np.ndarray,lanes:dict[str,np.ndarray],rows:np.ndarray)->list[tuple[Any,...]]:
 offsets=np.asarray(offsets,np.int64); rows=np.asarray(rows,np.int64)
 records=set()
 for r in rows:
  if r<0 or r+1>=len(offsets):raise ValueError('CSR parent face row out of domain')
  a,b=map(int,offsets[r:r+2])
  for j in range(a,b):
   rec=[]
   for v in lanes.values():
    x=np.asarray(v)[j]
    rec.append(tuple(np.asarray(x).reshape(-1).tolist()) if np.ndim(x) else x.item() if hasattr(x,'item') else x)
   records.add(tuple(rec))
 return sorted(records)

def rebuild_face_csr(parent:dict[str,np.ndarray],out_to_parent:np.ndarray,patch_by_out:dict[int,dict[str,Any]],offset_key:str,lane_keys:tuple[str,...])->tuple[np.ndarray,dict[str,np.ndarray]]:
 offsets=np.asarray(parent[offset_key],np.int64); n=len(parent['faces'])
 if offsets.shape!=(n+1,) or offsets[0]!=0 or np.any(offsets[1:]<offsets[:-1]):raise ValueError(f'{offset_key} invalid')
 lanes={k:np.asarray(parent[k]) for k in lane_keys}
 if any(len(v)!=int(offsets[-1]) for v in lanes.values()):raise ValueError(f'{offset_key} lane length mismatch')
 chunks={k:[] for k in lane_keys}; out_offsets=[0]
 for outrow,srcrow in enumerate(np.asarray(out_to_parent,np.int64)):
  if srcrow>=0:
   a,b=map(int,offsets[srcrow:srcrow+2]); records=None
   for k in lane_keys:chunks[k].append(lanes[k][a:b])
   out_offsets.append(out_offsets[-1]+b-a)
  else:
   patch=patch_by_out.get(outrow)
   if patch is None:raise ValueError(f'{offset_key} derived face {outrow} lacks patch parents')
   records=csr_tuples(offsets,lanes,np.asarray(patch['retired_parent_rows'],np.int64))
   if not records:raise ValueError(f'{offset_key} derived face {outrow} has empty flattened ancestry')
   for index,k in enumerate(lane_keys):
    src=lanes[k]
    vals=[r[index] for r in records]
    if src.ndim==1:chunks[k].append(np.asarray(vals,dtype=src.dtype))
    else:chunks[k].append(np.asarray(vals,dtype=src.dtype).reshape((-1,)+src.shape[1:]))
   out_offsets.append(out_offsets[-1]+len(records))
 result={}
 for k,parts in chunks.items():
  a=lanes[k]
  result[k]=np.concatenate(parts,axis=0).astype(a.dtype,copy=False) if out_offsets[-1] else np.empty((0,)+a.shape[1:],dtype=a.dtype)
 return np.asarray(out_offsets,np.int64),result

def rebuild_point_csr(parent:dict[str,np.ndarray],out_to_parent:np.ndarray,offset_key:str,lane_keys:tuple[str,...])->tuple[np.ndarray,dict[str,np.ndarray]]:
 offsets=np.asarray(parent[offset_key],np.int64); n=len(parent['points_world_m'])
 if offsets.shape!=(n+1,) or offsets[0]!=0 or np.any(offsets[1:]<offsets[:-1]):raise ValueError(f'{offset_key} invalid')
 lanes={k:np.asarray(parent[k]) for k in lane_keys}
 if any(len(v)!=int(offsets[-1]) for v in lanes.values()):raise ValueError(f'{offset_key} lane length mismatch')
 chunks={k:[] for k in lane_keys}; out_offsets=[0]
 for row in np.asarray(out_to_parent,np.int64):
  if row<0 or row>=n:raise ValueError('new connected-union point has no parent point map')
  a,b=map(int,offsets[row:row+2])
  for k in lane_keys:chunks[k].append(lanes[k][a:b])
  out_offsets.append(out_offsets[-1]+b-a)
 result={k:(np.concatenate(v,axis=0).astype(lanes[k].dtype,copy=False) if out_offsets[-1] else np.empty((0,)+lanes[k].shape[1:],dtype=lanes[k].dtype)) for k,v in chunks.items()}
 return np.asarray(out_offsets,np.int64),result

def fill_new_faces(src:np.ndarray,out_to_parent:np.ndarray)->np.ndarray:
 src=np.asarray(src); shape=(len(out_to_parent),)+src.shape[1:]
 if np.issubdtype(src.dtype,np.integer): out=np.full(shape,-1,dtype=np.result_type(src.dtype,np.int64))
 elif np.issubdtype(src.dtype,np.floating):out=np.full(shape,np.nan,dtype=src.dtype)
 elif src.dtype.kind in ('U','S'):out=np.full(shape,'',dtype=src.dtype)
 elif src.dtype.kind=='b':out=np.zeros(shape,dtype=src.dtype)
 else:raise ValueError(f'unsupported per-face field {src.dtype}')
 keep=np.flatnonzero(np.asarray(out_to_parent)>=0); out[keep]=src[np.asarray(out_to_parent)[keep]]; return out

def operation_files(op_dir:Path,role:str,stage:str)->dict[str,tuple[dict[str,Any],Path,str]]:
 result={}
 pattern=f'{stage}-{role}-warning-*.json' if stage=='037' else f'036-{role}-union-*.json'
 for path in sorted(op_dir.glob(pattern)):
  source=json.loads(path.read_text()); data=dict(source); patches=source.get('patches')
  if not isinstance(patches,list) or len(patches)!=1:raise ValueError(f'{path} does not have exactly one connected patch')
  patch=patches[0]
  if source.get('role')!=role or source.get('patch_id')!=patch.get('patch_id'):raise ValueError(f'{path} operation identity mismatch')
  if source.get('new_output_face_rows')!=patch.get('new_output_face_rows'):raise ValueError(f'{path} top/nested output row maps differ')
  lineage=source.get('source_lineage',{})
  if lineage!=patch.get('source_lineage',lineage):raise ValueError(f'{path} top/nested lineage differs')
  if stage=='037':
   # The operation has both ultimate-015 and immediate-036 row domains.
   if 'source_036_face_rows' not in source or 'new_faces_036_point_rows' not in source:
    raise ValueError(f'{path} lacks immediate 036-domain operation maps')
   data['retired_input_face_rows']=source['source_036_face_rows']
   data['new_faces_input_point_rows']=source['new_faces_036_point_rows']
  else:
   if source.get('retired_input_face_rows')!=patch.get('retired_input_face_rows'):
    raise ValueError(f'{path} top/nested retired row maps differ')
  pid=str(source['patch_id'])
  if pid in result:raise ValueError(f'duplicate operation id {pid}')
  result[pid]=(data,path,sha256(path))
 return result

def verify_lineage_hint(role:str,op:dict[str,Any],expected:list[tuple[int,...]],parent:dict[str,np.ndarray],retired_parent:np.ndarray)->None:
 lineage=op.get('source_lineage',{})
 if not isinstance(lineage,dict):raise ValueError('operation source_lineage is not a mapping')
 for col,key in enumerate(TYPED_FIELDS):
  hint=lineage.get(key)
  if hint is not None and set(map(int,np.asarray(hint).reshape(-1)))!=set(int(row[col]) for row in expected):
   raise ValueError(f'{role} {op["patch_id"]} {key} hint differs from recursively flattened typed union')
 if role=='O' and 'complete_face_ancestry_union' in lineage and all(k in parent for k in ('face_ancestry_offsets','face_ancestry_source_face_rows','face_ancestry_source_original_ids','face_ancestry_parent_face_rows')):
  got=csr_tuples(parent['face_ancestry_offsets'],{k:parent[k] for k in ('face_ancestry_source_face_rows','face_ancestry_source_original_ids','face_ancestry_parent_face_rows')},retired_parent)
  hinted=sorted({tuple(map(int,x)) for x in lineage['complete_face_ancestry_union']})
  if got!=hinted:raise ValueError(f'O {op["patch_id"]} complete face ancestry hint differs from parent CSR union')
 if role=='D' and 'union_face_ancestry_tuples' in lineage and all(k in parent for k in ('union_face_ancestry_offsets','union_face_ancestry_rows','union_face_ancestry_run_original_ids','union_face_ancestry_face_ids')):
  got_rows=csr_tuples(parent['union_face_ancestry_offsets'],{k:parent[k] for k in ('union_face_ancestry_rows','union_face_ancestry_run_original_ids','union_face_ancestry_face_ids')},retired_parent)
  # The operation JSON serializes the semantic triple as (run_original_id, face_id, row),
  # while the NPZ lanes are stored (row, run_original_id, face_id).
  got=sorted((int(run),int(face),int(row)) for row,run,face in got_rows)
  hinted=sorted({tuple(map(int,x)) for x in lineage['union_face_ancestry_tuples']})
  if got!=hinted:raise ValueError(f'D {op["patch_id"]} union ancestry hint differs from parent CSR union')
 if role=='D' and 'candidate_face_ancestry_union' in lineage and 'candidate_face_ancestry_offsets' in parent:
  cand=csr_tuples(parent['candidate_face_ancestry_offsets'],{'candidate_face_ancestry_rows':parent['candidate_face_ancestry_rows']},retired_parent)
  got=sorted(int(x[0]) for x in cand)
  hinted=sorted({int(x) for x in lineage['candidate_face_ancestry_union']})
  if got!=hinted:raise ValueError(f'D {op["patch_id"]} candidate ancestry hint differs from parent CSR union')


def build_role(role:str,parent_typed_path:Path,parent_maps_path:Path,parent_raw_path:Path,proposal_path:Path,operation_dir:Path,composed_path:Path,protected_receipt_path:Path,outdir:Path)->dict[str,Any]:
 parent=load_npz(parent_typed_path); maps=load_npz(parent_maps_path); praw=load_npz(parent_raw_path); prop=load_npz(proposal_path); composed=load_npz(composed_path)
 hashes={"parent_typed":sha256(parent_typed_path),"parent_maps":sha256(parent_maps_path),"parent_raw":sha256(parent_raw_path),"proposal":sha256(proposal_path),"composed":sha256(composed_path),"protected_receipt":sha256(protected_receipt_path)}
 stage='037' if 'immediate_parent_candidate_sha256' in prop else '036'
 nparent=len(parent['points_world_m']); fparent=len(parent['faces']); nraw=len(praw['points_world_m']); fraw=len(praw['faces'])
 parent_chain_raw=np.asarray(parent.get('quality_star_map_chain_json',np.asarray(['']))).reshape(-1)[0]
 try: parent_chain=json.loads(str(parent_chain_raw))
 except Exception as exc: raise ValueError('typed parent has no parseable immutable source map chain') from exc
 if not parent_chain.get('chain') or parent_chain['chain'][-1].get('sha256')!=hashes['parent_raw']:
  raise ValueError('typed-parent map chain does not terminate at the pinned immediate raw parent')
 if stage=='036':
  if str(np.asarray(prop['source_candidate_sha256']).reshape(-1)[0])!=hashes['parent_raw']:
   raise ValueError('036 source SHA is not the immediate parent raw proposal')
  if str(np.asarray(praw['source_candidate_sha256']).reshape(-1)[0])!=str(np.asarray(composed['source_candidate_sha256']).reshape(-1)[0]):
   raise ValueError('036 raw and composed map do not declare the same immutable source')
 else:
  if str(np.asarray(prop['immediate_parent_candidate_sha256']).reshape(-1)[0])!=hashes['parent_raw']:
   raise ValueError('037 immediate-parent SHA differs from pinned 036 raw proposal')
  if str(np.asarray(prop['source_candidate_sha256']).reshape(-1)[0])!=parent_chain.get('original',{}).get('sha256'):
   raise ValueError('037 ultimate source SHA differs from the authenticated original 015 input')
  if Path(composed_path).resolve()!=Path(proposal_path).resolve():
   raise ValueError('037 composed map must be its proposal, which carries the authenticated ultimate 015 maps')
 raw_point_to_parent=np.asarray(maps['raw_output_point_to_compact_row'],np.int64)
 if raw_point_to_parent.shape!=(nraw,) or np.any(raw_point_to_parent < -1) or np.any(raw_point_to_parent>=nparent):raise ValueError('parent raw→typed point map malformed')
 parent_face_map_key='output_face_to_raw_row' if 'output_face_to_raw_row' in maps else 'output_face_to_raw_proposal_row'
 parent_typed_to_raw=np.asarray(maps[parent_face_map_key],np.int64)
 if parent_typed_to_raw.shape!=(fparent,) or np.any(parent_typed_to_raw<0) or np.any(parent_typed_to_raw>=fraw) or len(np.unique(parent_typed_to_raw))!=fparent:
  raise ValueError('parent typed→raw face map malformed')
 raw_face_to_parent=np.full(fraw,-1,np.int64); raw_face_to_parent[parent_typed_to_raw]=np.arange(fparent,dtype=np.int64)
 if set(map(int,raw_face_to_parent[raw_face_to_parent>=0]))!=set(range(fparent)):raise ValueError('parent raw face map is not reversible')
 raw_face_source=np.asarray(prop['face_output_to_parent_rows'] if stage=='037' else prop['face_output_to_input_rows'],np.int64)
 raw_point_source=np.asarray(prop['point_output_to_parent_rows'] if stage=='037' else prop['point_output_to_input_rows'],np.int64)
 raw_faces=np.asarray(prop['faces'],np.int64)
 raw_points=np.asarray(prop['points_world_m'])
 if raw_point_source.shape!=(len(raw_points),) or np.any(raw_point_source<0) or np.any(raw_point_source>=nraw) or len(np.unique(raw_point_source))!=len(raw_point_source):raise ValueError('connected-union point map is not injective into immediate raw parent')
 p_to_parent=raw_point_to_parent[raw_point_source]
 used=np.unique(raw_faces.reshape(-1))
 if np.any(p_to_parent[used]<0) or len(np.unique(p_to_parent[used]))!=len(used):raise ValueError('used connected-union points lack a unique typed-parent row')
 if raw_points.dtype!=parent['points_world_m'].dtype or raw_points.tobytes()!=parent['points_world_m'][p_to_parent].tobytes():raise ValueError('connected-union point coordinates differ from mapped typed-parent bytes')
 if not np.array_equal(np.asarray(prop['points_world_m_f32']),raw_points.astype(np.float32)):raise ValueError('connected-union F32 view is not the exact cast of source coordinates')
 if raw_faces.ndim!=2 or raw_faces.shape[1]!=3 or np.any(raw_faces<0) or np.any(raw_faces>=len(raw_points)):raise ValueError('connected-union faces malformed')
 if np.any((raw_faces[:,0]==raw_faces[:,1])|(raw_faces[:,1]==raw_faces[:,2])|(raw_faces[:,2]==raw_faces[:,0])):raise ValueError('connected-union face repeats point rows')
 if raw_face_source.shape!=(len(raw_faces),) or np.any(raw_face_source < -1) or np.any(raw_face_source>=fraw):raise ValueError('connected-union face map outside immediate raw parent domain')
 retained=raw_face_source>=0
 if len(np.unique(raw_face_source[retained]))!=int(retained.sum()):raise ValueError('connected-union duplicates retained parent face rows')
 if stage=='037':
  retired_raw=np.setdiff1d(np.arange(fraw,dtype=np.int64),raw_face_source[raw_face_source>=0],assume_unique=False)
 else:
  retired_raw=np.asarray(prop['retired_input_face_rows'],np.int64)
 if len(np.unique(retired_raw))!=len(retired_raw) or np.any(retired_raw<0) or np.any(retired_raw>=fraw):raise ValueError('connected-union retired raw face rows invalid')
 if set(map(int,raw_face_source[retained]))!=set(range(fraw))-set(map(int,retired_raw)):raise ValueError('retained and retired raw parent faces do not partition input')
 out_to_parent=np.full(len(raw_faces),-1,np.int64)
 out_to_parent[retained]=raw_face_to_parent[raw_face_source[retained]]
 if np.any(out_to_parent[retained]<0):raise ValueError('retained output face does not map to a typed parent face')
 # Exact retained oriented connectivity check after the point-map composition.
 for outrow in np.flatnonzero(retained):
  expected=parent['faces'][out_to_parent[outrow]]
  got=p_to_parent[raw_faces[outrow]]
  if not np.array_equal(got,expected):raise ValueError(f'retained face {outrow} connectivity/winding differs from typed parent row {out_to_parent[outrow]}')
 newrows=np.flatnonzero(~retained)
 op_ids=np.asarray(prop['candidate_patch_operation_ids']).astype(str)
 if len(op_ids)!=len(np.unique(op_ids)):raise ValueError('operation IDs are repeated')
 operations=operation_files(operation_dir,role,stage)
 if set(op_ids.tolist())!=set(operations):raise ValueError('raw operation ID list differs from operation JSON inventory')
 patch_ids=np.full(len(newrows),-1,np.int64)
 new_row_lookup={int(row):i for i,row in enumerate(newrows)}
 new_faces_parent_raw=np.full((len(newrows),3),-1,np.int64)
 for pid,op_id in enumerate(op_ids.tolist()):
  data,_,_=operations[op_id]
  op_out=np.asarray(data['new_output_face_rows'],np.int64)
  op_new=np.asarray(data['new_faces_input_point_rows'],np.int64)
  if op_new.shape!=(len(op_out),3):raise ValueError(f'{op_id} immediate-parent triangle map malformed')
  for row,tri in zip(op_out,op_new):
   idx=new_row_lookup.get(int(row))
   if idx is None or patch_ids[idx]>=0:raise ValueError(f'{op_id} output face does not uniquely partition newly-derived rows')
   patch_ids[idx]=pid; new_faces_parent_raw[idx]=tri
 if np.any(patch_ids<0):raise ValueError('operation rows do not cover all newly-derived faces')
 if np.any(new_faces_parent_raw<0) or np.any(new_faces_parent_raw>=nraw):raise ValueError('new face point references outside immediate raw parent')
 if np.any(raw_point_to_parent[new_faces_parent_raw]<0):raise ValueError('new face references a point absent from typed parent')
 inverse_rawpoint=np.full(nraw,-1,np.int64); inverse_rawpoint[raw_point_source]=np.arange(len(raw_point_source),dtype=np.int64)
 if np.any(inverse_rawpoint[new_faces_parent_raw]<0) or not np.array_equal(raw_faces[newrows],inverse_rawpoint[new_faces_parent_raw]):raise ValueError('new face triangles differ from immediate-parent point rows')
 if set(map(int,np.unique(patch_ids)))!=set(range(len(op_ids))):raise ValueError('operation IDs do not define dense patch ordinal domain')
 retired_parent=np.asarray(raw_face_to_parent[retired_raw],np.int64)
 if np.any(retired_parent<0) or len(np.unique(retired_parent))!=len(retired_parent):raise ValueError('retired raw parent rows do not map uniquely to typed-parent face rows')
 op_to_rows={}; patch_by_out={}; retired_seen=set(); operation_records=[]
 for pid,op_id in enumerate(op_ids.tolist()):
  data,path,opsha=operations[op_id]
  idx=np.flatnonzero(patch_ids==pid); outs=newrows[idx]
  if not len(outs):raise ValueError(f'{op_id} has no new output faces')
  op_out=np.asarray(data['new_output_face_rows'],np.int64)
  op_new=np.asarray(data['new_faces_input_point_rows'],np.int64)
  op_ret=np.asarray(data['retired_input_face_rows'],np.int64)
  if not np.array_equal(op_out,outs) or not np.array_equal(op_new,new_faces_parent_raw[idx]):raise ValueError(f'{op_id} output rows/new triangles mismatch NPZ maps')
  if np.any(op_ret<0) or np.any(op_ret>=fraw):raise ValueError(f'{op_id} retired rows outside raw parent face domain')
  op_parent=raw_face_to_parent[op_ret]
  if np.any(op_parent<0) or len(np.unique(op_parent))!=len(op_parent):raise ValueError(f'{op_id} retired rows do not map uniquely to typed parent faces')
  if not set(map(int,op_ret)).issubset(set(map(int,retired_raw))):raise ValueError(f'{op_id} retired faces missing from proposal retirement map')
  overlap=retired_seen & set(map(int,op_parent))
  if overlap:raise ValueError(f'{op_id} overlaps retired parent-face set from another operation')
  retired_seen.update(map(int,op_parent))
  expected=face_union_for_parent_rows(parent,op_parent)
  verify_lineage_hint(role,data,expected,parent,op_parent)
  for row in outs:
   patch_by_out[int(row)]={"pid":pid,"op_id":op_id,"retired_parent_rows":op_parent.copy(),"typed_union":expected}
  operation_records.append({"role":role,"patch_id":op_id,"local_patch_ordinal":pid,"output_face_rows":outs.tolist(),"retired_parent_face_rows":op_parent.tolist(),"retired_raw_face_rows":op_ret.tolist(),"typed_union_count":len(expected),"typed_union_sha256":hash_json([list(x) for x in expected]),"operation_json":str(path),"operation_json_sha256":opsha,"source_lineage":data.get('source_lineage',{}),"source_lineage_sha256":hash_json(data.get('source_lineage',{}))})
 if retired_seen!=set(map(int,retired_parent)):raise ValueError('operation retired-face sets do not cover proposal retired set exactly')
 # Compact output point rows; never coordinate-weld distinct point identities.
 raw_to_compact=np.full(len(raw_points),-1,np.int64); raw_to_compact[used]=np.arange(len(used),dtype=np.int64)
 faces=raw_to_compact[raw_faces]
 points=raw_points[used].copy(); points_f32=np.asarray(prop['points_world_m_f32'])[used].copy()
 final_to_parent=p_to_parent[used]
 if len(np.unique(final_to_parent))!=len(final_to_parent):raise ValueError('output point→typed parent map is not injective')
 final_direct_points=np.asarray(parent.get('quality_star_point_output_to_input_row'),np.int64)[final_to_parent]
 if np.any(final_direct_points<0) or len(np.unique(final_direct_points))!=len(final_direct_points):raise ValueError('composed immutable015 point map is missing or noninjective')
 parent_direct_faces=np.asarray(parent.get('quality_star_face_output_to_input_row'),np.int64)
 if parent_direct_faces.shape!=(fparent,):raise ValueError('typed parent has no valid immutable015 face map')
 final_direct_faces=np.full(len(faces),-1,np.int64)
 keep=np.flatnonzero(out_to_parent>=0); final_direct_faces[keep]=parent_direct_faces[out_to_parent[keep]]
 # The root 045 composer is independent evidence; require exact agreement.
 if stage=='037':
  cpoints=np.asarray(composed['point_output_to_input_rows'],np.int64)[used]
  cfaces=np.asarray(composed['face_output_to_input_rows'],np.int64)
  cpoints_world=np.asarray(composed['points_world_m'])[used]
  cfaces_world=raw_to_compact[np.asarray(composed['faces'],np.int64)]
 else:
  cpoints=np.asarray(composed['point_output_to_input_rows'],np.int64)
  cfaces=np.asarray(composed['face_output_to_input_rows'],np.int64)
  cpoints_world=np.asarray(composed['points_world_m'])
  cfaces_world=np.asarray(composed['faces'])
 if not np.array_equal(cpoints,final_direct_points):raise ValueError('composed immutable-source point map differs from metadata composition')
 if not np.array_equal(cfaces,final_direct_faces):raise ValueError('composed immutable-source face map differs from metadata composition')
 if cpoints_world.dtype!=points.dtype or cpoints_world.tobytes()!=points.tobytes() or not np.array_equal(cfaces_world,faces):raise ValueError('composed geometry differs from compact output')
 # Preserve all face-/point-owned fields from parent. New face scalar descriptors use -1/blank/NaN until explicitly set.
 face_csr_keys={offset for offset in FACE_CSR if offset in parent}
 face_csr_lanes={k for offset in face_csr_keys for k in FACE_CSR[offset]}
 point_csr_keys={offset for offset in POINT_CSR if offset in parent}
 point_csr_lanes={k for offset in point_csr_keys for k in POINT_CSR[offset]}
 if 'quality_star_patch_union_offsets' in parent:
  face_csr_keys.add('quality_star_patch_union_offsets')
  face_csr_lanes.update(f'quality_star_patch_union_{k}' for k in TYPED_FIELDS)
 for k,v in parent.items():
  if k in ('points_world_m','points_world_m_f32','faces','face_source_identity_schema_by_original_id_json','quality_star_patch_union_semantics','quality_star_patch_union_offsets') or k in face_csr_keys or k in face_csr_lanes or k in point_csr_keys or k in point_csr_lanes or k in ('quality_star_patch_id','quality_star_patch_operation_id','quality_star_face_output_to_input_row','quality_star_point_output_to_input_row','quality_star_immediate_face_output_to_input_row','quality_star_immediate_point_output_to_input_row','quality_star_raw_proposal_sha256','quality_star_map_chain_json'):
   continue
  if v.ndim>0 and v.shape[0]==fparent:out_value=fill_new_faces(v,out_to_parent)
  elif v.ndim>0 and v.shape[0]==nparent:out_value=v[final_to_parent].copy()
  else:out_value=v.copy()
  # Keep named variables local, then store in output dict below.
  if 'out' not in locals():out={}
  out[k]=out_value
 if 'out' not in locals():out={}
 out['points_world_m']=points; out['points_world_m_f32']=points_f32; out['faces']=faces
 # Rebuild all recognized face CSR; fail closed if an unknown face CSR offset domain exists.
 known_offsets=set(FACE_CSR)|{'quality_star_patch_union_offsets'}
 for k,v in parent.items():
  if k.endswith('_offsets') and np.asarray(v).shape==(fparent+1,) and k not in known_offsets:
   raise ValueError(f'unrecognized face-owned CSR schema {k}; refusing to drop ancestry')
 for key in sorted(face_csr_keys):
  if key=='quality_star_patch_union_offsets':continue
  offsets,lanes= rebuild_face_csr(parent,out_to_parent,patch_by_out,key,FACE_CSR[key])
  out[key]=offsets; out.update(lanes)
 # Rebuild point CSR rows in exact output point order.
 for key in sorted(point_csr_keys):
  offsets,lanes=rebuild_point_csr(parent,final_to_parent,key,POINT_CSR[key]); out[key]=offsets; out.update(lanes)
 # Rebuild recursively flattened full 7-field typed union. Retained parent rows keep their prior union exactly.
 qoff=[]; qlanes={k:[] for k in TYPED_FIELDS}; qoff=[0]
 oldq=parent.get('quality_star_patch_union_offsets')
 if oldq is not None:
  oldq=np.asarray(oldq,np.int64)
  oldqlanes={k:np.asarray(parent[f'quality_star_patch_union_{k}']) for k in TYPED_FIELDS}
  if oldq.shape!=(fparent+1,) or oldq[0]!=0 or np.any(oldq[1:]<oldq[:-1]) or any(len(x)!=int(oldq[-1]) for x in oldqlanes.values()):raise ValueError('parent quality patch-union CSR malformed')
 for outrow,parentrow in enumerate(out_to_parent):
  if parentrow>=0 and oldq is not None:
   a,b=map(int,oldq[parentrow:parentrow+2]); tuples=[tuple(int(oldqlanes[k][j]) for k in TYPED_FIELDS) for j in range(a,b)]
   if not tuples:tuples=[tuple(int(parent[k][parentrow]) for k in TYPED_FIELDS)]
  elif parentrow>=0:
   tuples=[tuple(int(parent[k][parentrow]) for k in TYPED_FIELDS)]
  else:
   tuples=patch_by_out[outrow]['typed_union']
  tuples=sorted(set(tuples))
  for tup in tuples:
   for k,x in zip(TYPED_FIELDS,tup):qlanes[k].append(x)
  qoff.append(qoff[-1]+len(tuples))
 # Add one derived typed identity per connected patch, preserving mixed values as unknown sentinel -1.
 schema_raw=parent.get('face_source_identity_schema_by_original_id_json')
 if schema_raw is None:raise ValueError('typed parent has no source identity schema')
 schema=json.loads(str(np.asarray(schema_raw).reshape(-1)[0]))
 existing_ids=set(map(int,np.unique(parent['face_source_original_id'])))
 schema_ids={int(k) for k in schema if str(k).lstrip('-').isdigit()}
 next_source_id=max(existing_ids|schema_ids|{699_999_999})+1
 existing_patch=np.asarray(parent.get('quality_star_patch_id',np.full(fparent,-1)),np.int64)
 next_patch=int(existing_patch.max(initial=-1))+1
 source_out={k:fill_new_faces(parent[k],out_to_parent).astype(np.int64,copy=False) for k in TYPED_FIELDS}
 patch_id_out=fill_new_faces(existing_patch,out_to_parent).astype(np.int64,copy=False)
 patch_operation_out=fill_new_faces(np.asarray(parent.get('quality_star_patch_operation_id',np.full(fparent,'',dtype='<U1'))),out_to_parent)
 patch_records=[]; schemas_added=[]
 for pid,op_id in enumerate(op_ids.tolist()):
  rows=np.flatnonzero(patch_ids==pid); out_rows=newrows[np.flatnonzero(patch_ids==pid)]
  parent_rows=patch_by_out[int(out_rows[0])]['retired_parent_rows']
  typed_union=face_union_for_parent_rows(parent,parent_rows)
  expected=np.asarray(typed_union,np.int64).reshape((-1,7))
  fields=[]
  for col in (0,1,2):
   vals=np.unique(expected[:,col]); fields.append(int(vals[0]) if len(vals)==1 else -1)
  layer_value,body_value,bone_value=fields
  global_pid=next_patch+pid
  synthetic=next_source_id+pid
  op,path,opsha=operations[op_id]
  lineage=op.get('source_lineage',{})
  union_hash=hash_json([list(x) for x in typed_union])
  for row in out_rows:
   if out_to_parent[row]>=0 or row not in patch_by_out:raise ValueError('derived output face ownership map inconsistent')
   source_out['face_source_layer'][row]=layer_value
   source_out['face_source_body'][row]=body_value
   source_out['face_source_bone'][row]=bone_value
   source_out['face_source_face_id'][row]=-1
   source_out['face_source_original_id'][row]=synthetic
   source_out['face_source_identity0'][row]=int(row)
   source_out['face_source_identity1'][row]=global_pid
   patch_id_out[row]=global_pid; patch_operation_out[row]=op_id
  schema[str(synthetic)]={"role":"DERIVED_PATCH","derived_operation":("quality_star_warning_connected_union_fixed_boundary_dp_patch_retriangulation" if stage=='037' else "quality_star_connected_union_fixed_boundary_dp_patch_retriangulation"),"derived_patch_id":op_id,"derived_patch_ordinal":global_pid,"face_count":int(len(out_rows)),"identity_keys":["face_source_identity0","face_source_identity1"],"identity_face_id_key":"face_source_face_id","parent_support_semantics":SEMANTICS,"parent_face_row_domain":"immediate_parent_typed_candidate_faces","parent_union_tuple_count":int(len(typed_union)),"parent_union_tuple_sha256":union_hash,"complete_typed_parent_union":[list(t) for t in typed_union],"source_lineage":lineage,"source_lineage_sha256":hash_json(lineage),"operation_json_path":str(path),"operation_json_sha256":opsha,"source_scalar_summary":{"face_source_layer":layer_value,"face_source_body":body_value,"face_source_bone":bone_value}}
  schemas_added.append(str(synthetic))
  patch_records.append({"role":role,"patch_id":op_id,"local_patch_ordinal":pid,"global_patch_ordinal":global_pid,"synthetic_face_source_original_id":synthetic,"output_face_rows":out_rows.tolist(),"retired_parent_typed_face_rows":parent_rows.tolist(),"typed_parent_union":[list(t) for t in typed_union],"typed_union_sha256":union_hash,"union_hash_encoding":"UTF-8 compact JSON of sorted 7-integer tuples, sorted keys, separators=(',',':'), ensure_ascii=True","source_lineage":lineage,"source_lineage_sha256":hash_json(lineage),"operation_json_path":str(path),"operation_json_sha256":opsha})
 for k in TYPED_FIELDS:out[k]=source_out[k]
 out['face_source_identity_schema_by_original_id_json']=np.asarray([canonical_json(schema)])
 out['quality_star_patch_union_offsets']=np.asarray(qoff,np.int64)
 for k,values in qlanes.items():out[f'quality_star_patch_union_{k}']=np.asarray(values,np.int64)
 out['quality_star_patch_union_semantics']=np.asarray([SEMANTICS])
 out['quality_star_patch_id']=patch_id_out
 out['quality_star_patch_operation_id']=patch_operation_out
 out['quality_star_face_output_to_input_row']=final_direct_faces
 out['quality_star_point_output_to_input_row']=final_direct_points
 out['quality_star_immediate_face_output_to_input_row']=out_to_parent
 out['quality_star_immediate_point_output_to_input_row']=final_to_parent
 out['quality_star_raw_proposal_sha256']=np.asarray([hashes['proposal']])
 if stage=='037':
  chain=json.loads(str(parent_chain_raw)); chain.setdefault('chain',[]).append({'path':str(proposal_path),'sha256':hashes['proposal']}); out['quality_star_map_chain_json']=np.asarray([canonical_json(chain)])
 else:
  out['quality_star_map_chain_json']=np.asarray([str(composed.get('fixed_coordinate_map_chain_json',np.asarray(['']).reshape(-1)[0]))])
 # Ensure every output face is mapped once and all new faces are represented by exactly one operation.
 if len(out['face_source_layer'])!=len(faces) or set(patch_by_out)!=set(map(int,newrows)):raise ValueError('derived face tuple coverage mismatch')
 if np.any(np.diff(out['face_ancestry_offsets'])<=0):raise ValueError('final face ancestry CSR has empty rows')
 for k in ('candidate_face_ancestry_offsets','union_face_ancestry_offsets','face_parent_offsets','edge_flip_parent_offsets','microedge_parent_input_face_offsets','operational_output_face_parent_offsets'):
  if k in out and np.any(np.diff(out[k])[newrows]<=0):raise ValueError(f'{k} has an empty derived-face row')
 # D direct reader requires scalar source-field tuple arrays and full union/candidate CSR. Unknown mixed source is -1; full union remains authoritative.
 if role=='D':
  for k in ('union_input_run_original_id','union_input_face_id','union_input_face_rows'):
   if k not in out or out[k].shape!=(len(faces),):raise ValueError(f'D 217 scalar union field {k} missing/misaligned')
  for row in newrows:
   out['union_input_run_original_id'][row]=-1; out['union_input_face_id'][row]=-1; out['union_input_face_rows'][row]=-1
  for k in ('union_face_ancestry_offsets','union_face_ancestry_rows','union_face_ancestry_run_original_ids','union_face_ancestry_face_ids','candidate_face_ancestry_offsets','candidate_face_ancestry_rows'):
   if k not in out:raise ValueError(f'D 217 requires {k}')
 # Reconcile map-chain exact rows with canonical protected routes and produce row crosswalk.
 receipt=json.loads(protected_receipt_path.read_text()); routes=receipt['protected149']['surface_routes']
 if len(routes)!=149 or int(receipt['protected149']['count'])!=149:raise ValueError('protected route receipt is not canonical 149')
 inv=np.full(int(final_direct_points.max(initial=-1))+1,-1,np.int64); inv[final_direct_points]=np.arange(len(final_direct_points))
 route_rows=[]
 for route in routes:
  if str(route['role'])!=role:continue
  srcrow=int(route['surface_vertex_row']); target=np.asarray(route['target_world_m_f32'],np.float32)
  hits=np.flatnonzero(final_direct_points==srcrow)
  if len(hits)!=1 or target.shape!=(3,) or points[hits[0]].astype(np.float32).tobytes()!=target.tobytes():raise ValueError(f'protected route {route["canonical_protected_row"]} missing or moved')
  route_rows.append({"canonical_protected_row":int(route['canonical_protected_row']),"role":role,"input_surface_vertex_row":srcrow,"output_surface_vertex_row":int(hits[0]),"target_world_m_f32":target.tolist()})
 if len(route_rows)!=(45 if role in ('O','I') else 59):raise ValueError(f'{role} protected route cardinality mismatch')
 outpath=outdir/f'candidate-{role}-connected-union-typed.npz'; mappath=outdir/f'row-maps-{role}.npz'; unionspath=outdir/f'patch-unions-{role}.json'
 np.savez_compressed(outpath,**out)
 input_point_to_output=np.full(nparent,-1,np.int64); input_point_to_output[final_to_parent]=np.arange(len(final_to_parent))
 input_face_to_output=np.full(fparent,-1,np.int64);
 for row,parentrow in enumerate(out_to_parent):
  if parentrow>=0:input_face_to_output[parentrow]=row
 output_raw_face=np.arange(len(faces),dtype=np.int64)
 raw_face_to_typed=np.arange(len(faces),dtype=np.int64)
 np.savez_compressed(mappath,output_point_to_input_row=final_to_parent,input_point_to_output_row=input_point_to_output,output_point_to_raw_proposal_row=used,raw_output_point_to_compact_row=raw_to_compact,raw_output_point_to_input_row=raw_point_source,output_face_to_input_row=out_to_parent,input_face_to_output_row=input_face_to_output,output_face_to_raw_proposal_row=output_raw_face,raw_output_face_to_compact_row=raw_face_to_typed,raw_output_face_to_input_row=raw_face_source,output_face_to_immutable015_row=final_direct_faces,output_point_to_immutable015_row=final_direct_points,retired_input_face_rows=retired_parent,new_output_face_patch_ids=patch_id_out)
 unions={"schema":"quality-star-connected-union-typed-patch-union.v1","role":role,"immediate_parent_typed_sha256":hashes['parent_typed'],"immediate_parent_raw_sha256":hashes['parent_raw'],"connected_proposal_sha256":hashes['proposal'],"composed_map_sha256":hashes['composed'],"typed_fields":list(TYPED_FIELDS),"semantics":SEMANTICS,"patches":patch_records}
 unionspath.write_text(json.dumps(unions,sort_keys=True,indent=2,allow_nan=False)+'\n')
 return {"role":role,"input_hashes":hashes,"output_path":str(outpath),"output_sha256":sha256(outpath),"row_maps_path":str(mappath),"row_maps_sha256":sha256(mappath),"patch_union_path":str(unionspath),"patch_union_sha256":sha256(unionspath),"input_points":nparent,"output_points":len(points),"retired_points":int(np.count_nonzero(input_point_to_output<0)),"input_faces":fparent,"output_faces":len(faces),"retired_faces":int(len(retired_parent)),"new_faces":int(len(newrows)),"patch_count":len(op_ids),"mixed_scalar_patch_count":sum(1 for p in patch_records if any(p['typed_parent_union'][0][i] != p['typed_parent_union'][-1][i] for i in (0,1,2))),"protected_route_rows":route_rows,"immutable015_point_rows":int(len(final_direct_points)),"immutable015_face_map_nonnegative":int(np.count_nonzero(final_direct_faces>=0)),"typed_patch_union_tuple_count":int(qoff[-1]),"operation_json_count":len(operations)}

def main():
 ap=argparse.ArgumentParser(description='Compose typed ancestry and point/face row maps through one fixed-coordinate connected-union surface stage.')
 for name in ('role','parent_typed','parent_maps','parent_raw','proposal','operation_dir','composed_map','protected_receipt','outdir'):
  ap.add_argument('--'+name.replace('_','-'),required=True,choices=('O','I','D') if name=='role' else None)
 a=ap.parse_args(); outdir=Path(a.outdir)
 if outdir.exists():raise FileExistsError(f'refusing to reuse output directory {outdir}')
 outdir.mkdir(parents=True)
 report=build_role(a.role,*[Path(getattr(a,k)) for k in ('parent_typed','parent_maps','parent_raw','proposal','operation_dir','composed_map','protected_receipt')],outdir)
 report['metadata_only']=True; report['geometry_or_physics_admission']=False; report['tetgen_run']=False; report['native_runtime_run']=False
 report['limitations']=['Coordinates and triangulation are copied from the immutable connected-union proposal; this writer adds/reindexes typed provenance only.','Patch ancestry is a complete recursively flattened parent union, not a per-triangle overlap claim.','No exact surface, volume, mass, or native qualification is performed here.']
 receipt=outdir/f'{a.role}-metadata-receipt.json';receipt.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n');print(json.dumps({"status":"PASS_TYPED_METADATA_ONLY","receipt":str(receipt),"receipt_sha256":sha256(receipt),"role":a.role,"output_sha256":report['output_sha256'],"new_faces":report['new_faces'],"patches":report['patch_count']}))
if __name__=='__main__':main()
