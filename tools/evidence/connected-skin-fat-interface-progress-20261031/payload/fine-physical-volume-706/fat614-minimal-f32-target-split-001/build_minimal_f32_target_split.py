from pathlib import Path
from fractions import Fraction
from math import gcd
from functools import reduce
import hashlib, json, importlib.util, sys
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
OUT=H/'fat614-minimal-f32-target-split-001'
D=H/'native-apex-D-singleton-after-paired-293-root/run-001/typed-001/candidate-D-typed-current.npz'
TARGET=H/'fat614-typed-target-001/fat-current-typed-target-input.npz'
VOL=H/'fat614-residual-pass-001/remeshed/connected-fixed-surface-volume-proposal.npz'
WIT=H/'fat614-candidate-structural-audit-001/yy-witness-map.json'
PINS={str(D):'214f888227a38da66bbbbeff70932927f393f9d6b34b1a2690f328a01b6a4dca',str(TARGET):'5fa7a5149a2380151023d332e68e40dfca69340ab8dcb6abcd444d64ae1c3464',str(VOL):'c21b708fa591476b1c5ab5f33329362ac8086c6429c73a117e63100555984e8b',str(WIT):'a1bab8f1f0d58673d2c65947767582c11772d04c59a1b46b343a8faf8aa00ee2'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def hjson(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
for p,want in PINS.items():
 got=sha(p)
 if got!=want:raise ValueError(f'input hash mismatch {p}: {got}')
x=load(D); target=load(TARGET); vol=load(VOL); witness=json.loads(WIT.read_text())
P64=np.asarray(x['points_world_m']); P32=np.asarray(x['points_world_m_f32'],np.float32); F=np.asarray(x['faces'],np.int64)
if P64.dtype!=np.float64 or P32.shape!=P64.shape or not np.array_equal(P64.astype(np.float32).view(np.uint32),P32.view(np.uint32)):raise ValueError('D point views differ')
if target['points_world_m'].dtype!=np.float32 or vol['points_world_m'].dtype!=np.float32:raise ValueError('target/base point domains are not F32')
if not np.array_equal(target['points_world_m'],vol['points_world_m'][:len(target['points_world_m'])]) and len(target['points_world_m'])==len(vol['points_world_m']):raise ValueError('target/base points differ unexpectedly')
# Coordinate-index the existing D point rows into the frozen volume namespace.
vol_xyz=np.asarray(vol['points_world_m'],np.float32)
coord_to_global={}
for i,q in enumerate(vol_xyz):
 k=np.ascontiguousarray(q).tobytes()
 if k in coord_to_global:raise ValueError('volume F32 point coordinates are not unique')
 coord_to_global[k]=i
D_to_global=np.full(len(P32),-1,np.int64)
for i,q in enumerate(P32):
 g=coord_to_global.get(np.ascontiguousarray(q).tobytes())
 if g is not None:D_to_global[i]=g
if np.any(D_to_global<0):raise ValueError('some D points have no exact current volume point mapping')
# Current target face source rows are the authenticated D local-face rows.
roles=np.asarray(target['desired_surface_face_role'],np.uint8); source_rows=np.asarray(target['desired_surface_face_source_row'],np.int64); target_faces=np.asarray(target['desired_boundary_faces'],np.int64)
if json.loads(str(target['desired_surface_face_role_json'].item())).get('2')!='typed-current-D-reversed-for-Fat':raise ValueError('target role 2 identity changed')
ops=[]
for comp,edge,wanted_src_faces in [(6,[188532,198523],[238319,238320]),(2,[217282,221017],[238357,238358])]:
 c=next(c for c in witness['components'] if c['id']==comp)
 nodes=[n for n in c['new_boundary_points'] if n.get('mapped_edge_global_rows')==edge]
 # Pick TetGen's main interior witness nearest the component's primary edge parameter.
 preferred=21 if comp==6 else 1393
 n=next(q for q in nodes if q['output_node_id']==preferred)
 g0,g1=map(int,edge)
 if not np.array_equal(vol_xyz[[g0,g1]].view(np.uint32),np.asarray([P32[np.flatnonzero(D_to_global==g0)[0]],P32[np.flatnonzero(D_to_global==g1)[0]]]).view(np.uint32)):raise ValueError('D/global endpoint coordinate mapping failed')
 t=float(n['edge_parameter']); witness64=np.asarray(n['output_coordinate_f64'],np.float64)
 rounded=np.asarray(witness64,dtype=np.float32)
 # The rounded F32 is componentwise nearest to TetGen witness coordinate; record exact line residual.
 A=vol_xyz[g0].astype(np.float64);B=vol_xyz[g1].astype(np.float64);v=B-A
 tc=float(np.dot(witness64-A,v)/np.dot(v,v)); exact_line=A+tc*v
 cross=np.cross(rounded.astype(np.float64)-A,v); residual=float(np.linalg.norm(cross)/np.linalg.norm(v))
 displacement=float(np.linalg.norm(rounded.astype(np.float64)-witness64))
 if not 0<t<1 or not 0<tc<1:raise ValueError('witness projection is not interior')
 # Exact arithmetic proof: all possible F32 coordinates over each monotone interval lie on these finest power-of-two grids.
 Afrac=[Fraction(float(q)) for q in vol_xyz[g0]]; Bfrac=[Fraction(float(q)) for q in vol_xyz[g1]]
 emin=[]
 for aa,bb in zip(Afrac,Bfrac):
  mn=min(abs(aa),abs(bb));e=mn.numerator.bit_length()-mn.denominator.bit_length()
  if Fraction(2)**e>mn:e-=1
  emin.append(e)
 def lattice(row):
  out=[]
  for val,e in zip(row,emin):
   r=val/(Fraction(2)**(e-23))
   if r.denominator!=1:raise ValueError('endpoint is not on computed binary32 lattice')
   out.append(r.numerator)
  return out
 q0=lattice(Afrac);q1=lattice(Bfrac);delta=[y-z for z,y in zip(q0,q1)];gg=reduce(gcd,(abs(q) for q in delta if q),0)
 exact_interior_t=[]
 if gg>1:
  for m in range(1,gg):
   if any((d*m)%gg for d in delta):continue
   row=[q0[i]+delta[i]*m//gg for i in range(3)]
   isf32=True
   for q,e0 in zip(row,emin):
    val=Fraction(q)*(Fraction(2)**(e0-23)); f=np.float32(float(val))
    if Fraction(float(f))!=val:isf32=False;break
   if isf32:exact_interior_t.append(Fraction(m,gg))
 if exact_interior_t:raise ValueError(f'unexpected exact F32 interior points: {exact_interior_t}')
 # Find the exactly two D source faces from role/source-row metadata, then validate their global triangles.
 target_rows=[];parents=[]
 for sr in wanted_src_faces:
  hit=np.flatnonzero((roles==2)&(source_rows==sr))
  if len(hit)!=1:raise ValueError(f'D source face {sr} lacks unique target face row')
  tr=int(hit[0]);target_rows.append(tr);parents.append(int(sr))
 parents=np.asarray(parents,np.int64)
 if any(not np.array_equal(np.sort(D_to_global[F[r]]),np.sort(target_faces[tr])) for r,tr in zip(parents,target_rows)):raise ValueError('D source face/global target crosswalk mismatch')
 if not set(edge).issubset(set(map(int,D_to_global[F[parents]].ravel()))):raise ValueError('source faces do not contain selected edge')
 if np.any(np.all(P32.view(np.uint32)==rounded.view(np.uint32),axis=1)):raise ValueError('rounded witness point already exists in D surface')
 ops.append({'component_id':comp,'edge_global_volume_point_rows':edge,'endpoint_D_point_rows':[int(np.flatnonzero(D_to_global==g)[0]) for g in edge],'parent_D_face_rows':parents.tolist(),'parent_target_face_rows':target_rows,'parent_global_faces':[target_faces[r].tolist() for r in target_rows],'tetgen_node_id':preferred,'tetgen_witness_f64_m':witness64.tolist(),'tetgen_edge_parameter':t,'witness_projection_parameter':tc,'new_point_f32_m':rounded.tolist(),'new_point_f32_bits':[f'0x{int(y):08x}' for y in rounded.view(np.uint32)],'perpendicular_support_residual_m':residual,'nearest_round_displacement_from_witness_m':displacement,'finest_grid_exponents':emin,'endpoint_lattice_integer_rows':[q0,q1],'lattice_delta':delta,'lattice_delta_gcd':gg,'exact_f32_interior_parameters':[]})
# Build D face/point derivative. Each operation retires two D faces, subdivides them, and appends one F32 point.
lanes=['layer','body','bone','face_id','original_id','identity0','identity1']; anc_fields=['face_ancestry_source_face_rows','face_ancestry_source_original_ids','face_ancestry_parent_face_rows']
if not all(k in x for k in [f'face_source_{k}' for k in lanes]+['quality_star_patch_union_offsets','face_ancestry_offsets']+anc_fields):raise ValueError('typed D ancestry fields incomplete')
old_nface=len(F);old_npoint=len(P32)
new_P32=np.vstack([P32,np.asarray([o['new_point_f32_m'] for o in ops],np.float32)])
new_P64=np.vstack([P64,np.asarray([o['new_point_f32_m'] for o in ops],np.float64)])
if len({tuple(map(int,row)) for row in new_P32.view(np.uint32)})!=len(new_P32):raise ValueError('D points duplicate after append')
newF=F.copy();patches=[];retired=[];appended=[]
for oi,o in enumerate(ops):
 a,b=o['endpoint_D_point_rows'];parents=np.asarray(o['parent_D_face_rows'],np.int64);pnew=old_npoint+oi
 edge_patch_rows=[]
 for pr in parents:
  tri=list(map(int,F[int(pr)]));k=next((k for k in range(3) if {tri[k],tri[(k+1)%3]}=={a,b}),None)
  if k is None:raise ValueError('parent D face not adjacent to split edge')
  u,w,c=tri[k],tri[(k+1)%3],tri[(k+2)%3]
  children=[[u,pnew,c],[pnew,w,c]]
  newF[int(pr)]=np.asarray(children[0],np.int64);outrows=[int(pr),old_nface+len(appended)]
  appended.append(children[1]);retired.append(int(pr))
  patch={'ordinal':len(patches),'edge_operation_ordinal':oi,'parent_face':int(pr),'rows':outrows,'faces':children,'new_point_D_row':pnew}
  patches.append(patch);edge_patch_rows.extend(outrows)
 o['new_output_D_face_rows']=edge_patch_rows
 o['new_output_D_faces']=[newF[r].tolist() for r in edge_patch_rows if r<old_nface]+[[int(q) for q in row] for row in appended[-2:]]
 o['source_face_patch_ordinals']=[p['ordinal'] for p in patches if p['edge_operation_ordinal']==oi]
newF=np.vstack([newF,np.asarray(appended,np.int64)])
changed_rows=sorted(r for patch in patches for r in patch['rows']);retired=sorted(set(retired))
if len(newF)!=old_nface+4 or len(changed_rows)!=8 or len(retired)!=4 or len(patches)!=4:raise ValueError('unexpected split row counts')
# Each generated child keeps the complete typed ancestry union of its own retired parent face.
union_for_patch=[];ancestry_for_patch=[];union_sha=[];ancestry_sha=[];synthetic=[]
all_orig=np.concatenate([x['face_source_original_id'],x['quality_star_patch_union_face_source_original_id']])
next_id=int(np.max(all_orig))+1
for pi,patch in enumerate(patches):
 pr=patch['parent_face'];lo,hi=map(int,x['quality_star_patch_union_offsets'][pr:pr+2])
 if hi>lo:tuples=[tuple(int(x[f'quality_star_patch_union_face_source_{lane}'][j]) for lane in lanes) for j in range(lo,hi)]
 else:tuples=[tuple(int(x[f'face_source_{lane}'][pr]) for lane in lanes)]
 alo,ahi=map(int,x['face_ancestry_offsets'][pr:pr+2]);anc=[tuple(int(x[k][j]) for k in anc_fields) for j in range(alo,ahi)]
 union=sorted(set(tuples));anc=sorted(set(anc))
 if not union or not anc:raise ValueError('parent typed union/ancestry is empty')
 union_for_patch.append(union);ancestry_for_patch.append(anc);union_sha.append(hjson([list(t) for t in union]));ancestry_sha.append(hjson([list(t) for t in anc]));synthetic.append(next_id+pi)
 patch.update({'parent_typed_union':union,'parent_union_tuple_count':len(union),'parent_union_tuple_sha256':union_sha[-1],'complete_face_ancestry_union':anc,'face_ancestry_union_sha256':ancestry_sha[-1],'synthetic_original_id':synthetic[-1],'parent_support_semantics':'complete typed ancestry union of the one retired parent face; generated split faces do not claim exact per-child overlap or measured single-face identity'})
# Extend typed scalar columns, mark each output child face as derived under its own one-parent patch ID.
for lane in lanes:x[f'face_source_{lane}']=np.r_[np.asarray(x[f'face_source_{lane}']).copy(),np.zeros(4,dtype=x[f'face_source_{lane}'].dtype)]
for pi,patch in enumerate(patches):
 parent=patch['parent_face']; scalar={lane:int(x[f'face_source_{lane}'][parent]) for lane in ['layer','body','bone']}
 for r in patch['rows']:
  values={'layer':scalar['layer'],'body':scalar['body'],'bone':scalar['bone'],'face_id':-1,'original_id':synthetic[pi],'identity0':int(r),'identity1':int(pi)}
  for lane,v in values.items():x[f'face_source_{lane}'][r]=v
# Rebuild standard tuple and ancestry CSRs; preserve all unchanged rows byte-for-byte semantically.
row_to_patch={r:i for i,patch in enumerate(patches) for r in patch['rows']}
old_union_off=np.asarray(x['quality_star_patch_union_offsets'],np.int64);old_anc_off=np.asarray(x['face_ancestry_offsets'],np.int64)
new_union_off=np.zeros(len(newF)+1,np.int64);uvals={lane:[] for lane in lanes};new_anc_off=np.zeros(len(newF)+1,np.int64);avals={k:[] for k in anc_fields}
for r in range(len(newF)):
 if r in row_to_patch:
  pi=row_to_patch[r];tuples=union_for_patch[pi];anc=ancestry_for_patch[pi]
 else:
  lo,hi=map(int,old_union_off[r:r+2]);tuples=[tuple(int(x[f'quality_star_patch_union_face_source_{lane}'][j]) for lane in lanes) for j in range(lo,hi)]
  lo,hi=map(int,old_anc_off[r:r+2]);anc=[tuple(int(x[k][j]) for k in anc_fields) for j in range(lo,hi)]
 for tup in tuples:
  for j,lane in enumerate(lanes):uvals[lane].append(int(tup[j]))
 new_union_off[r+1]=new_union_off[r]+len(tuples)
 for tup in anc:
  for j,k in enumerate(anc_fields):avals[k].append(int(tup[j]))
 new_anc_off[r+1]=new_anc_off[r]+len(anc)
x['quality_star_patch_union_offsets']=new_union_off
x['face_ancestry_offsets']=new_anc_off
for lane in lanes:x[f'quality_star_patch_union_face_source_{lane}']=np.asarray(uvals[lane],np.int64)
for k in anc_fields:x[k]=np.asarray(avals[k],np.int64)
# Exact immediate maps to the typed D parent.
pmap=np.r_[np.arange(old_npoint,dtype=np.int64),np.full(len(ops),-1,np.int64)]
fmap=np.r_[np.arange(old_nface,dtype=np.int64),np.full(4,-1,np.int64)];fmap[np.asarray(retired,np.int64)]=-1
x['point_output_to_parent_rows']=pmap;x['face_output_to_parent_rows']=fmap
x['quality_star_point_output_to_input_row']=np.r_[np.asarray(x['quality_star_point_output_to_input_row'],np.int64),np.full(len(ops),-1,np.int64)]
qf=np.asarray(x['quality_star_face_output_to_input_row'],np.int64).copy();qf[np.asarray(retired,np.int64)]=-1
x['quality_star_face_output_to_input_row']=np.r_[qf,np.full(4,-1,np.int64)]
x['points_world_m']=new_P64;x['points_world_m_f32']=new_P32;x['faces']=newF
x['retired_parent_face_rows']=np.asarray(retired,np.int64);x['changed_parent_face_rows']=np.asarray(changed_rows,np.int64)
x['immediate_parent_candidate_sha256']=np.asarray([sha(D)],dtype='U64')
x['derived_boundary_split_operation_json_sha256s']=np.asarray(['pending'],dtype='U64')
x['derived_boundary_split_point_rows']=np.asarray([old_npoint,old_npoint+1],np.int64)
x['derived_boundary_split_point_source_global_volume_edge_rows']=np.asarray([o['edge_global_volume_point_rows'] for o in ops],np.int64)
x['derived_boundary_split_point_parameters']=np.asarray([o['tetgen_edge_parameter'] for o in ops],np.float64)
x['derived_boundary_split_point_support_residual_m']=np.asarray([o['perpendicular_support_residual_m'] for o in ops],np.float64)
# Local D closedness, boundary signature, unique faces, orientation, native joint apex envelope.
def edgesig(G):
 ed={}
 for tri in G:
  for u,v in ((int(tri[0]),int(tri[1])),(int(tri[1]),int(tri[2])),(int(tri[2]),int(tri[0]))):ed.setdefault(tuple(sorted((u,v))),[]).append((u,v))
 return {k:tuple(sorted(v)) for k,v in ed.items() if len(v)!=2 or v[0][0]==v[1][0]}
before=edgesig(F);after=edgesig(newF)
if before!=after:raise ValueError(f'D closed topology changed: bad edges {len(before)}->{len(after)}')
if len({tuple(sorted(map(int,f))) for f in newF})!=len(newF):raise ValueError('duplicate faces after split')
# Load existing exact joint-apex scorer; final receipt binds full changed-D exact audit separately.
modpath=H/'audit_joint_native_apex_272_root.py';spec=importlib.util.spec_from_file_location('joint_apex_272_f32split',modpath);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
child_scores=[];orient=[]
for r in changed_rows:
 pi=row_to_patch[r];parent=patches[pi]['parent_face'];tri=newF[r];
 old=F[parent];oldn=np.cross(P32[old[1]].astype(float)-P32[old[0]],P32[old[2]].astype(float)-P32[old[0]])
 xyz=new_P32[tri].astype(float);nn=np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0]);dot=float(oldn@nn);orient.append(dot>0)
 child_scores.append({'face_row':int(r),'parent_face_row':int(parent),'orientation_dot':dot,'normalized_face_q':float(np.linalg.norm(nn)/max(np.sum((xyz[0]-xyz[1])**2),np.sum((xyz[1]-xyz[2])**2),np.sum((xyz[2]-xyz[0])**2)))})
if not all(orient):raise ValueError('new face orientation does not preserve parent normal')
# Record operation-specific JSON and SHA.
for i,o in enumerate(ops):
 patch_ids=o['source_face_patch_ordinals']; additions=[]
 for pi in patch_ids:
  patch=patches[pi]; addition={'synthetic_original_id':synthetic[pi],'derived_operation':'nearest_float32_projection_of_TetGen_boundary_witness_to_D_target_edge','role':'DERIVED_PATCH','identity0':'output face row','identity1':'face patch ordinal','complete_typed_parent_union':union_for_patch[pi],'parent_union_tuple_count':len(union_for_patch[pi]),'parent_union_tuple_sha256':union_sha[pi]}
  additions.append(addition)
 o['typed_source_identity_schema_additions']=additions;o['new_point_D_row']=old_npoint+i;o['candidate_D_parent_path']=str(D);o['candidate_D_parent_sha256']=sha(D);o['typed_role']='D';o['current_target_role']='typed-current-D-reversed-for-Fat'
 op_path=OUT/f'edge-split-operation-{i:02d}.json';op_path.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n');o['operation_json_path']=str(op_path);o['operation_json_sha256']=sha(op_path)
# Refresh NPZ operation SHA array after all operation JSONs exist.
x['derived_boundary_split_operation_json_paths']=np.asarray([o['operation_json_path'] for o in ops],dtype='U256')
x['derived_boundary_split_operation_json_sha256s']=np.asarray([o['operation_json_sha256'] for o in ops],dtype='U64')
# Per-row point provenance records the source edge and projection weights; no exact-point identity is claimed.
x['derived_boundary_split_point_edge_D_rows']=np.asarray([o['endpoint_D_point_rows'] for o in ops],np.int64)
x['derived_boundary_split_point_witness_projection_parameters']=np.asarray([o['witness_projection_parameter'] for o in ops],np.float64)
x['derived_boundary_split_point_witness_perpendicular_residual_m']=np.asarray([o['perpendicular_support_residual_m'] for o in ops],np.float64)
x['derived_boundary_split_point_witness_displacement_m']=np.asarray([o['nearest_round_displacement_from_witness_m'] for o in ops],np.float64)
x['derived_boundary_split_operation_ordinals']=np.asarray([0,1],np.int32)
x['derived_boundary_split_face_patch_ordinals']=np.asarray([row_to_patch[r] for r in changed_rows],np.int32)
# Write a separate registry-additions JSON: parent D does not carry a complete registry field.
reg={'schema':'numi.face-source-identity-schema-additions.v1','status':'ADDITIONS_ONLY_MERGE_WITH_PARENT_REGISTRY','parent_candidate_sha256':sha(D),'additions':{str(synthetic[pi]):{'synthetic_original_id':synthetic[pi],'parent_face_row':patches[pi]['parent_face'],'derived_operation':'nearest_float32_projection_of_TetGen_boundary_witness_to_D_target_edge','complete_typed_parent_union':union_for_patch[pi],'parent_union_tuple_count':len(union_for_patch[pi]),'parent_union_tuple_sha256':union_sha[pi]} for pi in range(len(patches))}}
regp=OUT/'face-source-identity-schema-additions.json';regp.write_text(json.dumps(reg,indent=2,sort_keys=True)+'\n')
x['face_source_identity_schema_additions_json']=np.asarray([str(regp)],dtype='U')
x['face_source_identity_schema_additions_sha256']=np.asarray([sha(regp)],dtype='U64')
# The new points are intentionally off their old edges: preserve exact residual and keep this proposal non-admitting until complete rerun/audit.
candidate=OUT/'candidate-D-minimally-rounded-yy-edge-split.npz';np.savez_compressed(candidate,**x)
# Geometry-only self checks already run locally; global exact context and volume replay remain separate.
report={'schema':'numi.fat614.minimally-rounded-d-boundary-edge-split.v1','status':'PASS_TYPED_D_LOCAL_SPLIT_CONSTRUCTION_GLOBAL_AUDIT_PENDING','physical_admission':False,'native_steps':0,'inputs':{p:{'sha256':sha(p)} for p in PINS},'candidate':{'path':str(candidate),'sha256':sha(candidate),'points_before_after':[old_npoint,len(new_P32)],'faces_before_after':[old_nface,len(newF)],'new_point_rows':x['derived_boundary_split_point_rows'].tolist(),'retired_parent_face_rows':retired,'changed_output_face_rows':changed_rows,'source_face_patch_count':len(patches)},'operations':ops,'exact_float32_segment_search':{'method':'exact integer lattice from binary32 endpoint dyadics on each monotone coordinate interval; any exact point parameter must make each coordinate lattice index integral','edge_proofs':[{'edge':o['edge_global_volume_point_rows'],'finest_grid_exponents':o['finest_grid_exponents'],'endpoint_lattice_integer_rows':o['endpoint_lattice_integer_rows'],'lattice_delta':o['lattice_delta'],'gcd':o['lattice_delta_gcd'],'exact_interior_f32_parameters':o['exact_f32_interior_parameters']} for o in ops]},'topology':{'bad_directed_edges_before_after':[len(before),len(after)],'closed_signature_unchanged':before==after,'duplicate_unoriented_faces':0},'changed_faces_local':{'orientation_preserved':all(orient),'joint_native_apex_bound_min':float(min(mod.score(new_P32,newF[np.asarray(changed_rows,np.int64)]))),'joint_native_apex_bound_max':float(max(mod.score(new_P32,newF[np.asarray(changed_rows,np.int64)]))),'rows':child_scores},'scope':'Nearest-Float32 rounded TetGen edge witnesses. Each appended point is nanometers off the prior target edge, so target support changes slightly; complete O/I/D, wall, all185 exact context audit and volume remesh/replay are required before promotion. No physical admission.'}
rp=OUT/'report.json';rp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'report_path':str(rp),'report_sha256':sha(rp),'candidate_path':str(candidate),'candidate_sha256':sha(candidate),'ops':[{'op':o['component_id'],'edge':o['edge_global_volume_point_rows'],'parent_D_faces':o['parent_D_face_rows'],'newpoint':o['new_point_f32_m'],'bits':o['new_point_f32_bits'],'residual_m':o['perpendicular_support_residual_m'],'tproj':o['witness_projection_parameter'],'faces':o['output_D_face_rows'],'operation_sha':o['operation_json_sha256']} for o in ops],'status':report['status'],'joint_native_apex_bound_min':report['changed_faces_local']['joint_native_apex_bound_min']},indent=2))
