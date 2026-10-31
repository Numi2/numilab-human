from pathlib import Path
import hashlib, json
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
OUT=H/'fat614-minimal-f32-target-split-002/run-005'
PARENT=H/'fat614-minimal-f32-target-split-001/candidate-D-minimally-rounded-yy-edge-split.npz'
PARENT_SHA='602036c9b164bd1eec83b344675b8cabf453665920e45b558b50fe8115c67493'
TARGET=H/'fat614-typed-target-001/fat-current-typed-target-input.npz'
TARGET_SHA='5fa7a5149a2380151023d332e68e40dfca69340ab8dcb6abcd444d64ae1c3464'
VOL=H/'fat614-residual-pass-001/remeshed/connected-fixed-surface-volume-proposal.npz'
VOL_SHA='c21b708fa591476b1c5ab5f33329362ac8086c6429c73a117e63100555984e8b'
CENSUS=H/'fat614-minimal-f32-target-split-001/remaining-component-census-001/identified-target-edges.json'
CENSUS_SHA='f396afb6a1e9f6316fe08561951598f6737b6d3ef016bda17307cd4b6d26a56e'
PREV_ADD=H/'fat614-minimal-f32-target-split-001/face-source-identity-schema-additions.json'
PREV_ADD_SHA='4c2f78a4cd48ed80906446de9e0bd13f55eca3b2041d83d8b0a10d60809081b2'
SELECTED={(3,3383),(4,455),(5,173),(5,174),(7,135),(8,44),(8,45),(10,54),(11,20)}
LANES=['layer','body','bone','face_id','original_id','identity0','identity1']
ANC_FIELDS=['face_ancestry_source_face_rows','face_ancestry_source_original_ids','face_ancestry_parent_face_rows']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hjson(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
for p,want in ((PARENT,PARENT_SHA),(TARGET,TARGET_SHA),(VOL,VOL_SHA),(CENSUS,CENSUS_SHA),(PREV_ADD,PREV_ADD_SHA)):
 if sha(p)!=want:raise ValueError(f'input hash mismatch {p}: {sha(p)}')
if OUT.exists() and any(OUT.iterdir()):raise FileExistsError(f'refusing to overwrite nonempty {OUT}')
OUT.mkdir(parents=True,exist_ok=True)
x=load(PARENT); target=load(TARGET); vol=load(VOL); cdata=json.loads(CENSUS.read_text()); psha=sha(PARENT)
P64=np.asarray(x['points_world_m']); P32=np.asarray(x['points_world_m_f32'],np.float32); F=np.asarray(x['faces'],np.int64)
if P64.dtype!=np.float64 or P32.shape!=P64.shape or not np.array_equal(P64.astype(np.float32).view(np.uint32),P32.view(np.uint32)):raise ValueError('parent point views differ')
V=np.asarray(vol['points_world_m'],np.float32); D2G=np.full(len(P32),-1,np.int64)
lookup={}
for i,q in enumerate(V):
 k=np.ascontiguousarray(q,dtype='<f4').tobytes()
 if k in lookup:raise ValueError('volume coordinate aliases prevent global map')
 lookup[k]=i
for i,q in enumerate(P32):D2G[i]=lookup.get(np.ascontiguousarray(q,dtype='<f4').tobytes(),-1)
if np.any(D2G[:120633]<0) or len(np.unique(D2G[:120633]))!=120633:raise ValueError('original D point prefix lacks unique exact volume mapping')
if np.any(D2G[120633:]!=-1):raise ValueError('previous split points were unexpectedly found in old volume')
roles=np.asarray(target['desired_surface_face_role'],np.uint8); src=np.asarray(target['desired_surface_face_source_row'],np.int64); TF=np.asarray(target['desired_boundary_faces'],np.int64)
if json.loads(str(target['desired_surface_face_role_json'].item())).get('2')!='typed-current-D-reversed-for-Fat':raise ValueError('D target role changed')
selected=[]
for item in cdata['nodes']:
 key=(int(item['component']),int(item['output_node_id']))
 if key not in SELECTED:continue
 edgehit=item['shared_target_edge_hits']
 if len(edgehit)!=1:raise ValueError(f'{key}: expected one unambiguous D edge, got {len(edgehit)}')
 eh=edgehit[0]; edge=list(map(int,eh['global_volume_edge_rows'])); p64=np.asarray(item['output_coordinate_f64_m'],np.float64); p32=np.asarray(item['f32_nearest_m'],np.float32)
 A=V[edge[0]].astype(np.float64);B=V[edge[1]].astype(np.float64);d=B-A
 t64=float((p64-A)@d/(d@d)); t32=float((p32.astype(np.float64)-A)@d/(d@d)); residual=float(np.linalg.norm(np.cross(p32.astype(np.float64)-A,d))/np.linalg.norm(d)); displacement=float(np.linalg.norm(p32.astype(np.float64)-p64))
 if not 0<t64<1 or not 0<t32<1:raise ValueError(f'{key}: projected point left open edge segment')
 if residual>1e-7:raise ValueError(f'{key}: rounded point edge residual unexpectedly large: {residual}')
 parent_src=list(map(int,eh['source_D_face_rows'])); target_rows=list(map(int,eh['target_face_rows']))
 if len(parent_src)!=2 or len(set(parent_src))!=2 or len(target_rows)!=2:raise ValueError(f'{key}: target edge not covered by exactly two D faces')
 for tr,pr in zip(target_rows,parent_src):
  if int(roles[tr])!=2 or int(src[tr])!=pr:raise ValueError(f'{key}: role/source face crosswalk mismatch')
  if not set(edge).issubset(set(map(int,TF[tr]))):raise ValueError(f'{key}: target parent face lacks both split-edge endpoints')
  if pr>=len(F) or np.any(np.asarray(x['face_output_to_parent_rows'])[pr:pr+1]<0):raise ValueError(f'{key}: target source parent was already retired')
  mapped=np.sort(D2G[F[pr]])
  if not np.array_equal(mapped,np.sort(TF[tr])):raise ValueError(f'{key}: D face/target face point crosswalk mismatch')
 if any(np.array_equal(p32.view(np.uint32),q.view(np.uint32)) for q in P32):raise ValueError(f'{key}: rounded point duplicates current D point')
 selected.append({'component':key[0],'node_id':key[1],'edge_global_volume_rows':edge,'parent_D_face_rows':parent_src,'parent_target_face_rows':target_rows,'tetgen_f64_m':p64.tolist(),'new_point_f32_m':p32.tolist(),'new_point_f32_bits':[f'0x{int(v):08x}' for v in p32.view(np.uint32)],'parameter_f64':t64,'parameter_f32_projection':t32,'support_residual_m':residual,'witness_round_displacement_m':displacement,'output_candidate_faces':item['containing_target_faces']})
if len(selected)!=len(SELECTED):raise ValueError(f'selected boundary-node inventory mismatch: {len(selected)}')
selected.sort(key=lambda o:(o['component'],o['node_id']))
byedge={}
for op in selected:byedge.setdefault(tuple(op['edge_global_volume_rows']),[]).append(op)
for edge,ops in byedge.items():
 ops.sort(key=lambda o:o['parameter_f32_projection'])
 if any(ops[i]['parameter_f32_projection']>=ops[i+1]['parameter_f32_projection'] for i in range(len(ops)-1)):raise ValueError(f'edge point order/collision: {edge}')
 if len({tuple(int(v,16) for v in o['new_point_f32_bits']) for o in ops})!=len(ops):raise ValueError(f'rounded point collision on {edge}')
G2D={int(g):i for i,g in enumerate(D2G) if g>=0}
if len(G2D)!=120633:raise ValueError('D to volume map not unique over original D prefix')
for op in selected:op['endpoint_D_rows']=[int(G2D[g]) for g in op['edge_global_volume_rows']]
old_nface=len(F);old_npoint=len(P32);old_union_off=np.asarray(x['quality_star_patch_union_offsets'],np.int64);old_anc_off=np.asarray(x['face_ancestry_offsets'],np.int64)
if len(old_union_off)!=old_nface+1 or len(old_anc_off)!=old_nface+1:raise ValueError('parent CSR dimensions malformed')
all_orig=np.concatenate([np.asarray(x['face_source_original_id'],np.int64),np.asarray(x['quality_star_patch_union_face_source_original_id'],np.int64)])
next_id=int(all_orig.max())+1
old_changed=list(map(int,np.asarray(x['changed_parent_face_rows'],np.int64)));old_retired=list(map(int,np.asarray(x['retired_parent_face_rows'],np.int64)))
patch_specs=[];newF=F.copy();appended=[]
for edge,ops in sorted(byedge.items()):
 parents=sorted(set(p for op in ops for p in op['parent_D_face_rows']))
 if len(parents)!=2:raise ValueError(f'edge {edge} has non-two parent faces')
 for pr in parents:
  tri=list(map(int,F[pr]));endpoint_set=set(op['endpoint_D_rows'][0] for op in ops[:1])|set(op['endpoint_D_rows'][1] for op in ops[:1])
  k=next((k for k in range(3) if {tri[k],tri[(k+1)%3]}==endpoint_set),None)
  if k is None:raise ValueError(f'parent face {pr} is not adjacent to edge {edge}')
  u,v,c=tri[k],tri[(k+1)%3],tri[(k+2)%3];g_u=int(D2G[u]);g_v=int(D2G[v]);ordered=[]
  for op in ops:
   t=op['parameter_f32_projection']
   if (g_u,g_v)==tuple(edge):to=t
   elif (g_v,g_u)==tuple(edge):to=1.0-t
   else:raise ValueError('D edge endpoint map changed')
   ordered.append((to,op))
  ordered.sort(key=lambda q:q[0]);children=[]
  verts=[u]+[old_npoint+selected.index(op) for _,op in ordered]+[v]
  for aa,bb in zip(verts[:-1],verts[1:]):children.append([int(aa),int(bb),int(c)])
  patch={'parent_face':pr,'edge_global_volume_rows':list(edge),'ordered_component_node_pairs':[[op['component'],op['node_id']] for _,op in ordered],'children':children}
  tuples=[];lo,hi=map(int,old_union_off[pr:pr+2])
  if hi>lo:tuples=[tuple(int(x[f'quality_star_patch_union_face_source_{lane}'][j]) for lane in LANES) for j in range(lo,hi)]
  else:tuples=[tuple(int(x[f'face_source_{lane}'][pr]) for lane in LANES)]
  union=sorted(set(tuples));alo,ahi=map(int,old_anc_off[pr:pr+2]);anc=sorted(set(tuple(int(x[k][j]) for k in ANC_FIELDS) for j in range(alo,ahi)))
  if not union or not anc:raise ValueError(f'empty typed parent union/ancestry at face {pr}')
  # Keep the current parent's declared scalar descriptor exactly; the full union remains authoritative and may span multiple historical layer/body/bone tuples.
  parent_scalar={lane:int(x[f'face_source_{lane}'][pr]) for lane in ('layer','body','bone')}
  if not any(tuple(t[:3])==tuple(parent_scalar[lane] for lane in ('layer','body','bone')) for t in union):raise ValueError(f'patch {pr}: current scalar descriptor is not present in complete parent union')
  patch.update({'ordinal':4+len(patch_specs),'parent_face_row':pr,'complete_typed_parent_union':[list(t) for t in union],'parent_union_tuple_count':len(union),'parent_union_tuple_sha256':hjson([list(t) for t in union]),'complete_face_ancestry_union':[list(t) for t in anc],'face_ancestry_union_sha256':hjson([list(t) for t in anc]),'synthetic_original_id':next_id,'source_layer':parent_scalar['layer'],'source_body':parent_scalar['body'],'source_bone':parent_scalar['bone'],'parent_support_semantics':'complete typed ancestry union of the retired parent face; scalar layer/body/bone descriptors preserve the current parent candidate values; generated edge-split children make no exact per-child overlap or measured single-face claim'})
  next_id+=1
  childrows=[pr]
  for child in children[1:]:appended.append(child);childrows.append(old_nface+len(appended)-1)
  newF[pr]=np.asarray(children[0],np.int64);patch['output_face_rows']=childrows;patch['union_tuples']=union;patch['ancestry_tuples']=anc
  patch_specs.append(patch)
if len({p['parent_face'] for p in patch_specs})!=len(patch_specs):raise ValueError('overlapping target edge parent faces in batch')
new_points=np.asarray([op['new_point_f32_m'] for op in selected],np.float32);newP32=np.vstack((P32,new_points));newP64=np.vstack((P64,new_points.astype(np.float64)))
if len({tuple(map(int,row)) for row in newP32.view(np.uint32)})!=len(newP32):raise ValueError('duplicate F32 points after append')
newF=np.vstack((newF,np.asarray(appended,np.int64)))
changed_new=sorted({r for patch in patch_specs for r in patch['output_face_rows']});retired_new=sorted(p['parent_face_row'] for p in patch_specs)
if len(newF)!=old_nface+18 or len(appended)!=18 or len(retired_new)!=14 or len(changed_new)!=32:raise ValueError(f'unexpected batch dimensions faces={len(newF)} appended={len(appended)} retired={len(retired_new)} changed={len(changed_new)}')
for lane in LANES:x[f'face_source_{lane}']=np.r_[np.asarray(x[f'face_source_{lane}']).copy(),np.zeros(len(appended),dtype=np.asarray(x[f'face_source_{lane}']).dtype)]
row_to_patch={r:p for p in patch_specs for r in p['output_face_rows']}
for row,patch in row_to_patch.items():
 for lane,value in (('layer',patch['source_layer']),('body',patch['source_body']),('bone',patch['source_bone']),('face_id',-1),('original_id',patch['synthetic_original_id']),('identity0',row),('identity1',patch['ordinal'])):x[f'face_source_{lane}'][row]=value
uvals={lane:[] for lane in LANES};newuo=np.zeros(len(newF)+1,np.int64);avals={k:[] for k in ANC_FIELDS};newao=np.zeros(len(newF)+1,np.int64)
for row in range(len(newF)):
 patch=row_to_patch.get(row)
 if patch is None:
  lo,hi=map(int,old_union_off[row:row+2]);tuples=[tuple(int(x[f'quality_star_patch_union_face_source_{lane}'][j]) for lane in LANES) for j in range(lo,hi)]
  lo,hi=map(int,old_anc_off[row:row+2]);anc=[tuple(int(x[k][j]) for k in ANC_FIELDS) for j in range(lo,hi)]
 else:tuples=patch['union_tuples'];anc=patch['ancestry_tuples']
 for tup in tuples:
  for j,lane in enumerate(LANES):uvals[lane].append(int(tup[j]))
 newuo[row+1]=newuo[row]+len(tuples)
 for tup in anc:
  for j,k in enumerate(ANC_FIELDS):avals[k].append(int(tup[j]))
 newao[row+1]=newao[row]+len(anc)
x['quality_star_patch_union_offsets']=newuo;x['face_ancestry_offsets']=newao
for lane in LANES:x[f'quality_star_patch_union_face_source_{lane}']=np.asarray(uvals[lane],np.int64)
for k in ANC_FIELDS:x[k]=np.asarray(avals[k],np.int64)
pmap=np.r_[np.arange(old_npoint,dtype=np.int64),np.full(len(selected),-1,np.int64)];fmap=np.r_[np.arange(old_nface,dtype=np.int64),np.full(len(appended),-1,np.int64)];fmap[np.asarray(retired_new,np.int64)]=-1
x['point_output_to_parent_rows']=pmap;x['face_output_to_parent_rows']=fmap
qpoint=np.asarray(x['quality_star_point_output_to_input_row'],np.int64).copy();x['quality_star_point_output_to_input_row']=np.r_[qpoint,np.full(len(selected),-1,np.int64)]
qface=np.asarray(x['quality_star_face_output_to_input_row'],np.int64).copy();qface[np.asarray(retired_new,np.int64)]=-1;x['quality_star_face_output_to_input_row']=np.r_[qface,np.full(len(appended),-1,np.int64)]
x['points_world_m']=newP64;x['points_world_m_f32']=newP32;x['faces']=newF
x['immediate_parent_candidate_path']=np.asarray([str(PARENT)],dtype='U512');x['immediate_parent_candidate_sha256']=np.asarray([psha],dtype='U64')
x['retired_immediate_parent_face_rows']=np.asarray(retired_new,np.int64);x['retired_parent_face_rows']=np.asarray(sorted(set(old_retired+retired_new)),np.int64);x['changed_parent_face_rows']=np.asarray(sorted(set(old_changed+changed_new)),np.int64)
prev=json.loads(PREV_ADD.read_text());additions=dict(prev['additions']);new_ids=[]
for patch in patch_specs:
 entry={'synthetic_original_id':int(patch['synthetic_original_id']),'parent_face_row':int(patch['parent_face_row']),'derived_operation':'nearest_float32_rounding_of_tetgen_boundary_steiner_on_current_D_edge','parent_support_semantics':patch['parent_support_semantics'],'complete_typed_parent_union':patch['complete_typed_parent_union'],'parent_union_tuple_count':patch['parent_union_tuple_count'],'parent_union_tuple_sha256':patch['parent_union_tuple_sha256'],'complete_face_ancestry_union':patch['complete_face_ancestry_union'],'face_ancestry_union_sha256':patch['face_ancestry_union_sha256'],'source_component':next(op['component'] for op in selected if tuple(op['edge_global_volume_rows'])==tuple(patch['edge_global_volume_rows'])),'source_tetgen_node_ids':[n for c,n in patch['ordered_component_node_pairs']]}
 additions[str(patch['synthetic_original_id'])]=entry;new_ids.append(int(patch['synthetic_original_id']))
reg={'schema':'numi.face-source-identity-schema-additions.v1','status':'ADDITIONS_ONLY_MERGE_WITH_PARENT_REGISTRY','parent_candidate_path':str(PARENT),'parent_candidate_sha256':psha,'inherited_additions_path':str(PREV_ADD),'inherited_additions_sha256':sha(PREV_ADD),'additions':additions,'new_addition_ids':new_ids}
regp=OUT/'face-source-identity-schema-additions.json';regp.write_text(json.dumps(reg,sort_keys=True,indent=2)+'\n')
prev_op_paths=list(map(str,np.asarray(x.get('derived_boundary_split_operation_json_paths',[])).reshape(-1)));prev_op_shas=list(map(str,np.asarray(x.get('derived_boundary_split_operation_json_sha256s',[])).reshape(-1)));ops=[]
for ordinal,op in enumerate(selected,start=2):
 edge=tuple(op['edge_global_volume_rows']);op_patches=[p for p in patch_specs if tuple(p['edge_global_volume_rows'])==edge]
 obj={'schema':'numi.fat614.yy-derived-target-edge-split-operation.v1','component_id':op['component'],'tetgen_output_node_id':op['node_id'],'parent_candidate_path':str(PARENT),'parent_candidate_sha256':psha,'target_input_path':str(TARGET),'target_input_sha256':sha(TARGET),'volume_parent_path':str(VOL),'volume_parent_sha256':sha(VOL),'edge_global_volume_point_rows':list(edge),'endpoint_D_point_rows':op['endpoint_D_rows'],'edge_parameter_f64':op['parameter_f64'],'edge_parameter_after_f32_rounding':op['parameter_f32_projection'],'tetgen_witness_f64_m':op['tetgen_f64_m'],'rounded_f32_point_m':op['new_point_f32_m'],'rounded_f32_bits':op['new_point_f32_bits'],'support_residual_m':op['support_residual_m'],'witness_round_displacement_m':op['witness_round_displacement_m'],'parent_D_face_rows':op['parent_D_face_rows'],'parent_target_face_rows':op['parent_target_face_rows'],'new_output_D_point_row':old_npoint+ordinal-2,'face_patches':[{'parent_face_row':p['parent_face_row'],'target_edge_global_volume_rows':p['edge_global_volume_rows'],'output_face_rows':p['output_face_rows'],'child_faces_D_rows':p['children'],'synthetic_original_id':p['synthetic_original_id'],'complete_typed_parent_union':p['complete_typed_parent_union'],'parent_union_tuple_sha256':p['parent_union_tuple_sha256'],'complete_face_ancestry_union':p['complete_face_ancestry_union'],'face_ancestry_union_sha256':p['face_ancestry_union_sha256']} for p in op_patches],'status':'DERIVED_TARGET_SUPPORT_REPAIR_ROUNDING_OFF_EXACT_EDGE; FULL_CONTEXT_AND_VOLUME_REPLAY_REQUIRED','physical_admission':False,'native_steps':0}
 opath=OUT/f'edge-split-operation-{ordinal:02d}.json';opath.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n');ops.append({'path':str(opath),'sha256':sha(opath),'data':obj})
x['derived_boundary_split_operation_json_paths']=np.asarray(prev_op_paths+[o['path'] for o in ops],dtype='U512');x['derived_boundary_split_operation_json_sha256s']=np.asarray(prev_op_shas+[o['sha256'] for o in ops],dtype='U64')
x['derived_boundary_split_operation_json_sha256s_source_sha256']=np.asarray([sha(PARENT)],dtype='U64')
prev_pointrows=np.asarray(x.get('derived_boundary_split_point_rows',np.array([],np.int64)),np.int64).reshape(-1);x['derived_boundary_split_point_rows']=np.r_[prev_pointrows,np.arange(old_npoint,old_npoint+len(selected),dtype=np.int64)]
prev_edges=np.asarray(x.get('derived_boundary_split_point_source_global_volume_edge_rows',np.empty((0,2),np.int64)),np.int64).reshape(-1,2);x['derived_boundary_split_point_source_global_volume_edge_rows']=np.vstack((prev_edges,np.asarray([o['edge_global_volume_rows'] for o in selected],np.int64)))
prev_dedges=np.asarray(x.get('derived_boundary_split_point_edge_D_rows',np.empty((0,2),np.int64)),np.int64).reshape(-1,2);x['derived_boundary_split_point_edge_D_rows']=np.vstack((prev_dedges,np.asarray([o['endpoint_D_rows'] for o in selected],np.int64)))
for name,newvals,dtype in [('derived_boundary_split_point_parameters',[o['parameter_f64'] for o in selected],np.float64),('derived_boundary_split_point_witness_projection_parameters',[o['parameter_f64'] for o in selected],np.float64),('derived_boundary_split_point_support_residual_m',[o['support_residual_m'] for o in selected],np.float64),('derived_boundary_split_point_witness_perpendicular_residual_m',[o['support_residual_m'] for o in selected],np.float64),('derived_boundary_split_point_witness_displacement_m',[o['witness_round_displacement_m'] for o in selected],np.float64)]:
 old=np.asarray(x.get(name,np.array([],dtype=dtype)),dtype=dtype).reshape(-1);x[name]=np.r_[old,np.asarray(newvals,dtype=dtype)]
prev_ord=np.asarray(x.get('derived_boundary_split_operation_ordinals',np.array([],np.int32)),np.int32).reshape(-1);x['derived_boundary_split_operation_ordinals']=np.r_[prev_ord,np.arange(2,2+len(selected),dtype=np.int32)]
prev_pidx=np.asarray(x.get('derived_boundary_split_face_patch_ordinals',np.array([],np.int32)),np.int32).reshape(-1);x['derived_boundary_split_face_patch_ordinals']=np.r_[prev_pidx,np.asarray([row_to_patch[r]['ordinal'] for r in changed_new],np.int32)]
x['face_source_identity_schema_additions_json']=np.asarray([str(regp)],dtype='U512');x['face_source_identity_schema_additions_sha256']=np.asarray([sha(regp)],dtype='U64')
def bad_edges(Faces):
 d={}
 for tri in Faces:
  for u,v in ((int(tri[0]),int(tri[1])),(int(tri[1]),int(tri[2])),(int(tri[2]),int(tri[0]))):d.setdefault(tuple(sorted((u,v))),[]).append((u,v))
 return {k:v for k,v in d.items() if len(v)!=2 or v[0]!=v[1][::-1]}
old_boundary_edges=bad_edges(F); new_boundary_edges=bad_edges(newF)
if new_boundary_edges!=old_boundary_edges:
 old_keys=set(old_boundary_edges); new_keys=set(new_boundary_edges)
 print('NEW_BAD_EDGES',list(new_keys-old_keys)[:60])
 print('REPAIRED_OR_CHANGED_OLD_BOUNDARY',list(old_keys-new_keys)[:60])
 print('PATCHES',[(p['edge_global_volume_rows'],p['parent_face_row'],p['children']) for p in patch_specs])
 raise ValueError(f'D boundary-edge delta changed: old={len(old_boundary_edges)} new={len(new_boundary_edges)}')
if len({tuple(sorted(map(int,f))) for f in newF})!=len(newF):raise ValueError('duplicate D faces')
for patch in patch_specs:
 old=F[patch['parent_face']];opn=np.cross(P32[old[1]].astype(float)-P32[old[0]],P32[old[2]].astype(float)-P32[old[0]])
 for tri in patch['children']:
  q=newP32[np.asarray(tri,np.int64)].astype(float);nn=np.cross(q[1]-q[0],q[2]-q[0])
  if float(opn@nn)<=0:raise ValueError(f'orientation failure at parent {patch["parent_face"]}')
x['derived_boundary_split_new_point_f32_bits']=np.asarray([[int(v,16) for v in op['new_point_f32_bits']] for op in selected],np.uint32)
x['derived_boundary_split_new_face_parent_rows']=np.asarray([p['parent_face_row'] for p in patch_specs],np.int64)
x['derived_boundary_split_new_face_patch_output_rows_json']=np.asarray([json.dumps(p['output_face_rows'],separators=(',',':')) for p in patch_specs],dtype='U256')
x['derived_boundary_split_complete_parent_union_sha256']=np.asarray([p['parent_union_tuple_sha256'] for p in patch_specs],dtype='U64')
x['derived_boundary_split_complete_ancestry_union_sha256']=np.asarray([p['face_ancestry_union_sha256'] for p in patch_specs],dtype='U64')
x['face_source_identity_schema_additions_parent_path']=np.asarray([str(PREV_ADD)],dtype='U512');x['face_source_identity_schema_additions_parent_sha256']=np.asarray([sha(PREV_ADD)],dtype='U64')
candidate=OUT/'candidate-D-remaining-yy-target-edge-splits.npz';np.savez_compressed(candidate,**x)
report={'schema':'numi.fat614.remaining-yy-target-edge-splits.v1','status':'PASS_TYPED_D_LOCAL_BATCH_AWAITING_FULL_CONTEXT_AUDIT','physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN','inputs':{str(p):{'sha256':sha(p),'size_bytes':p.stat().st_size} for p in (PARENT,TARGET,VOL,CENSUS,PREV_ADD)},'outputs':{'candidate':{'path':str(candidate),'sha256':sha(candidate),'size_bytes':candidate.stat().st_size},'identity_additions':{'path':str(regp),'sha256':sha(regp),'size_bytes':regp.stat().st_size},'operations':[{'path':o['path'],'sha256':o['sha256']} for o in ops]},'candidate_geometry':{'points_before_after':[old_npoint,len(newP32)],'faces_before_after':[old_nface,len(newF)],'point_prefix_f32_exact':True,'point_prefix_f64_exact':True,'new_point_rows':list(range(old_npoint,len(newP32))),'new_point_count':len(selected),'new_face_count':len(appended),'retired_parent_face_rows':retired_new,'changed_output_face_rows':changed_new,'face_level_patch_count':len(patch_specs),'bad_directed_edges':len(new_boundary_edges),'inherited_open_boundary_edges':len(old_boundary_edges),'boundary_edge_delta_equal':new_boundary_edges==old_boundary_edges,'duplicate_faces':0,'parent_face_orientation_preserved':True},'operations':[o['data'] for o in ops],'untouched_components':{'return_code_200_with_no_exact_current_D_target_edge_support':[9,12],'return_code_not_200_not_used':[0,1,13]},'limitations':'Only target-edge Steiner points exactly lying on an authenticated current D edge (before componentwise F32 rounding) are included. Components 9 and 12 are cut-boundary additions, not current target-edge inserts. Components 0/1 have return code 3 and are not accepted TetGen results. This candidate still requires full O/I/D + both walls + all185 exact context audit, actual 697 loader, and paired volume replay; it is non-admitting.'}
rp=OUT/'report.json';rp.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({'candidate':str(candidate),'candidate_sha256':sha(candidate),'report':str(rp),'report_sha256':sha(rp),'new_points':len(selected),'new_faces':len(appended),'patches':len(patch_specs),'changed_rows':len(changed_new),'points':[(o['component'],o['node_id'],o['edge_global_volume_rows'],o['parameter_f32_projection'],o['support_residual_m'],o['new_point_f32_bits']) for o in selected]},indent=2))
