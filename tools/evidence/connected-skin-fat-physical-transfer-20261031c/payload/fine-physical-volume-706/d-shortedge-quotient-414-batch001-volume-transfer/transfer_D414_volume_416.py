#!/usr/bin/env python3
"""Apply the pinned D414 exact surface quotient to current Fat 414 with indexed379.
Source-only proposal: no mass mapping, native steps, or physical admission.
"""
from pathlib import Path
from collections import Counter
import hashlib,importlib.util,json,sys,time
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
OUT=Path(__file__).resolve().parent/'run002'
BASE=H/'fat-comp0-connected-warning-repair-412/run-012-final-D-current-target-rebind/fat-current-typed-target-rebound.npz'
BASE_SHA='72c851692580a639f89a4916bc4bf5c38331be9b6def89322eccf02ee66ea6ec'
VOLUME414=H/'d-shortedge-quotient-411-batch-final/volume-transfer-414/run001/fat-D411-379-plus-comp1-run041-composed.npz'
VOLUME414_SHA='14f06beaf20a2f78396585e7868df831cff4b9f27e3401e271d7643175073a77'
DSEED=H/'d-shortedge-quotient-414-next-001-preflight-attempt004/identity-seed-D.npz'
DSEED_SHA='fba94572f52f51e58902063da181d24fcdf47473d0b322d68370277d9ab32c03'
SURF=H/'d-shortedge-quotient-414-batch001-surface-proposal/candidate-D-short-edge-collapse-raw.npz'
SURF_SHA='60f89988727fd00d272aaf2bed798e0f19e9f91f98067c308408a935c183cacc'
SURF_REPORT=H/'d-shortedge-quotient-414-batch001-surface-proposal/report.json'
SURF_REPORT_SHA='4d4be7ee3d713f8cc7a4a4e52e0adff63e1c021d710a74015aa24da223899653'
LINEAGE=H/'d-shortedge-quotient-414-batch001-surface-proposal/lineage-crosswalk.json'
LINEAGE_SHA='74e4a4af2916a04b0152916c4a9b6ec48cb24aa3ae753d08af7bcfe0f1af3889'
PREFLIGHT=H/'d-shortedge-quotient-414-next-001-preflight-attempt004/preflight.json'
PREFLIGHT_SHA='b991ced167685079c02d54f3da395278cc609713914178e81ce07c2428bea62f'
WARN_DISJOINT=H/'d-shortedge-quotient-414-batch001-volume-transfer/run017-warning-disjointness.json'
WARN_DISJOINT_SHA='b862fb88cceb21027b5cc2064e10458341e83c76143762a5761728379b598ebc'
AUTH=H/'d-shortedge-quotient-414-next-001-preflight-attempt004/allowed-D-edge-map.npz'
AUTH_SHA='439a1df81c608ebb153992d884cf8356e507cd228e20346e97a2ef46cce0aa95'
COMP0=H/'fat-comp0-connected-warning-repair-412/run-014-current-comp0-edge-star-cavity/comp0-full-edge-star-exclusion.npz'
COMP0_SHA='bfb88500119d48ff09c3af5358a405302ea29f152252d7b8dccbd6b4275ac893'
COMP1=H/'d-shortedge-quotient-414-next-001-input-attempt004/comp1-current-output-protection.npz'
COMP1_SHA='2d578f66c9c4b62787995dc83de3f214edb0adae3002bf7dc38d5bad298deae4'
OWNER=H/'transfer_surface_contractions_indexed_379_root.py'
OWNER_SHA='ea9135fe6c19efc644bd8429bd01e5f36547f304e86da9214aa1bb594441da37'
REG=H/'interior-quality-conditioning-309-banks/condition_registered_interior_309.py'
REPORT_D=H/'fat-comp0-connected-warning-repair-412/run-009-final-D411-comp1-flips-exact-adapter/exact-audit/report.json'
REPORT_D_SHA='44962483a961203a14c52b03eb954e87b2d4827e066f38567074e3d4b75c1a28'
PYTHON='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p)
 if s is None or s.loader is None:raise RuntimeError(f'cannot load {p}')
 m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
for p,h in ((BASE,BASE_SHA),(VOLUME414,VOLUME414_SHA),(DSEED,DSEED_SHA),(SURF,SURF_SHA),(SURF_REPORT,SURF_REPORT_SHA),(LINEAGE,LINEAGE_SHA),(PREFLIGHT,PREFLIGHT_SHA),(WARN_DISJOINT,WARN_DISJOINT_SHA),(AUTH,AUTH_SHA),(COMP0,COMP0_SHA),(COMP1,COMP1_SHA),(OWNER,OWNER_SHA),(REPORT_D,REPORT_D_SHA)):
 got=sha(p)
 if got!=h:raise ValueError(f'input SHA mismatch {p}: {got} != {h}')
owner=module('indexed379_D414',OWNER);m=owner.m;reg=module('exactvolume_delta_D414',REG)
base=load(BASE);volume414=load(VOLUME414);dseed=load(DSEED);surface=load(SURF);sr=json.loads(SURF_REPORT.read_text());auth=load(AUTH);warning=json.loads(WARN_DISJOINT.read_text())
if warning.get('status')!='PASS_WARNING_PATCH_DISJOINT_FROM_D414_PREFLIGHT':raise ValueError('warning overlap proof is not PASS')
for key0 in ('points_world_m','points_world_m_f32','tetrahedra','material_indices','boundary_faces'):
 if not np.array_equal(base[key0],volume414[key0]):raise ValueError(f'current target rebind differs from qualified 414 geometry in {key0}')
if sr.get('status')!='PASS_BOUNDED_EXACT_SHORT_EDGE_COLLAPSE_PROPOSAL' or sr.get('accepted_collapse_count')!=2363:raise ValueError('surface report status/count mismatch')
if sr.get('outputs',{}).get('D',{}).get('path')!=str(SURF) or sr['outputs']['D'].get('sha256')!=SURF_SHA:raise ValueError('surface report does not pin raw D output')
if sr['input_candidates']['D']['sha256']!=DSEED_SHA:raise ValueError('surface report D parent pin mismatch')
if surface['immediate_parent_candidate_sha256'].reshape(-1).tolist()!=[DSEED_SHA]:raise ValueError('surface raw immediate parent pin mismatch')
if surface['typed_parent_candidate_sha256'].reshape(-1).tolist()!=['707bdf3ac95fd0dc9539c148b9bad8e31207dab515d75cd21ebe039a38cb2703']:raise ValueError('surface raw typed parent pin mismatch')
if sr.get('joint_changed_vs_complete_OID_walls_all185',{}).get('pair_count')!=0:raise ValueError('surface complete-context changed-pair count not zero')
# Authenticate the frozen raw output by replaying its face edits over its exact input domain.
sp=np.asarray(dseed['points_world_m_f32'],np.float32);sf=np.asarray(dseed['faces'],np.int64)
if not np.array_equal(surface['points_world_m_f32'],sp) or not np.array_equal(surface['points_world_m'],dseed['points_world_m']):raise ValueError('surface contraction moved/changed D point coordinates')
ops=sr['accepted_operations'];retired=[];newfaces=[]
for op in ops:
 if op.get('role')!='D' or op.get('input_sha256')!=DSEED_SHA:raise ValueError('accepted operation role/input mismatch')
 retired.extend(map(int,op['retired_parent_face_rows']))
 newfaces.extend(np.asarray(op['new_faces_current_point_rows'],np.int64).tolist())
if len(set(retired))!=len(retired):raise ValueError('duplicate retired parent face across accepted operations')
keepface=np.ones(len(sf),bool);keepface[np.asarray(retired,np.int64)]=False
replayed=np.vstack((sf[keepface],np.asarray(newfaces,np.int64).reshape(-1,3)))
if not np.array_equal(replayed,surface['faces']):raise ValueError('raw D output faces do not exactly replay accepted operation ledger')
# Verify each accepted contraction is an exact preflight-authorized orientation with the same full volume-star rows.
orient=np.asarray(auth['D_allowed_orientation_edge_rows'],np.int64);drops=np.asarray(auth['D_allowed_collapsed_point_rows'],np.int64);keeps=np.asarray(auth['D_allowed_retained_point_rows'],np.int64)
off=np.asarray(auth['D_orientation_affected_cell_offsets'],np.int64);arows=np.asarray(auth['D_orientation_affected_cell_rows'],np.int64)
auth_lookup={(int(d),int(k)):i for i,(d,k) in enumerate(zip(drops,keeps))}
if len(auth_lookup)!=len(drops):raise ValueError('duplicate preflight orientation rows')
p=np.asarray(base['points_world_m_f32'],np.float32);t=np.asarray(base['tetrahedra'],np.int64).copy();mat=np.asarray(base['material_indices'],np.int32)
if len(t)!=len(mat) or np.any(mat!=16):raise ValueError('current Fat input material/cell contract mismatch')
if not np.array_equal(base['points_world_m'],p) or len(p)!=len(np.unique(p,axis=0)):raise ValueError('current Fat input F32 point contract/uniqueness failed')
# Confirm current D point crosswalk against the pinned base bundle.
look={owner.key(x):i for i,x in enumerate(p)}
if len(look)!=len(p):raise ValueError('duplicate current Fat F32 point keys')
d_to_v=np.asarray([look.get(owner.key(x),-1) for x in sp],np.int64)
if np.any(d_to_v<0) or len(np.unique(d_to_v))!=len(d_to_v):raise ValueError('D→Fat exact F32 coordinate map incomplete or ambiguous')
if not np.array_equal(d_to_v,np.asarray(base['fat_current_target_rebind_D_point_to_volume_rows'],np.int64)):raise ValueError('D→Fat map disagrees with pinned target bundle')
# Protected comp0 and comp1 cells plus complete point-star cells are reconstructed from exact current row domains.
c0=load(COMP0);c1=load(COMP1)
protected_cells=set(map(int,c0['comp0_full_protected_cell_rows']))|set(map(int,c1['input_cell_rows']))
protected_points=set(map(int,c0['comp0_full_protected_point_rows']))|set(map(int,c1['global_point_rows']))
flat=t.reshape(-1);order=np.argsort(flat,kind='stable');counts=np.bincount(flat,minlength=len(p));inc_off=np.empty(len(p)+1,np.int64);inc_off[0]=0;np.cumsum(counts,out=inc_off[1:])
inc_rows=order//4
for v in protected_points:
 if v<0 or v>=len(p):raise ValueError('protected point outside current volume')
 protected_cells.update(map(int,inc_rows[inc_off[v]:inc_off[v+1]]))
# Cavity point stars and all 59 D rim routes are protected independently.
protected_routes_path=H/'verified-volume-preparation-010-merged038005/rim/protected149-current-routes.json'
protected_routes_sha='ed6e15c6dc9324b09cd4514e1fc3a149fc92a07215128a9820d72fca3f350a71'
if sha(protected_routes_path)!=protected_routes_sha:raise ValueError('protected route source SHA mismatch')
protected_routes=json.loads(protected_routes_path.read_text())
route_keys=set()
for r in protected_routes.get('surface_routes',[]):
 if r.get('role')=='D':route_keys.add(owner.key(np.asarray(r['target_world_m_f32'],np.float32)))
protected_D=set()
for i,x in enumerate(sp):
 if owner.key(x) in route_keys:protected_D.add(i)
if len(protected_D)!=59:raise ValueError(f'expected 59 exact D protected routes, found {len(protected_D)}')
# Build 379 dynamic incidence and exact source tet identity indices.
base_boundary=owner.boundary(t);boundary_keys={tuple(sorted(map(int,f))) for f in base_boundary}
tet_owners={}
for row,tet in enumerate(t):tet_owners.setdefault(tuple(sorted(map(int,tet))),[]).append(row)
flat_order=np.argsort(flat,kind='stable');v={'path':BASE,'p':p,'t':t,'mat':mat,'base_boundary':base_boundary,'boundary_keys':boundary_keys,'look':look,'active':np.ones(len(t),bool),'map':np.arange(len(p),dtype=np.int64),'node_incidence_order':flat_order,'node_incidence_offsets':inc_off,'node_incidence_added':{},'tet_owners':tet_owners}
# 379's helper consults a module global for collision owners; pin that current-volume index.
owner.tet_owners=tet_owners
base_det,base_q=m.measurements(p,t)
if np.any(base_det<=0):raise ValueError('current 414 Fat input has a nonpositive F32 cell')
base_good=m.passes(base_det,base_q);base_bad=int(np.count_nonzero(~base_good))
seen_rows=set();seen_pairs=set();records=[];full_offsets=[0];full_rows=[];changed_offsets=[0];changed_rows=[];removed_offsets=[0];removed_rows=[];old_changed=[];new_changed=[];deleted_changed=[];surf_q=np.arange(len(sp),dtype=np.int64);exact_delta=0;failures=[]
start=time.monotonic()
for oi,op in enumerate(ops):
 drop=int(op['collapsed_input_point_row']);keep=int(op['retained_input_point_row']);idx=auth_lookup.get((drop,keep))
 if idx is None:failures.append({'operation':oi,'reason':'not_an_authorized_preflight_orientation'});break
 expected_rows=np.asarray(arows[int(off[idx]):int(off[idx+1])],np.int64)
 if drop in protected_D or keep in protected_D:failures.append({'operation':oi,'reason':'protected_D_route_endpoint'});break
 result,why=owner.try_op(v,sp,sf,op)
 if result is None:failures.append({'operation':oi,'reason':why});break
 rows=np.asarray(result['rows'],np.int64);old=np.asarray(result['old'],np.int64);new=np.asarray(result['new'],np.int64);deleted=np.asarray(result['deleted'],bool)
 if not np.array_equal(rows,expected_rows):failures.append({'operation':oi,'reason':'actual_full_volume_star_differs_from_preflight','actual_rows':len(rows),'authorized_rows':len(expected_rows)});break
 if set(map(int,rows)) & protected_cells:failures.append({'operation':oi,'reason':'overlaps_comp0_or_comp1_protected_full_stars'});break
 overlap=seen_rows.intersection(map(int,rows))
 if overlap:failures.append({'operation':oi,'reason':'overlaps_prior_D_operation_full_star','overlap_sample':sorted(overlap)[:8]});break
 if (drop,keep) in seen_pairs:failures.append({'operation':oi,'reason':'duplicate_surface_orientation'});break
 changed=deleted|np.any(new!=old,axis=1);cr=rows[changed];rr=rows[deleted]
 if not np.all(mat[rows]==16):failures.append({'operation':oi,'reason':'material_changed'});break
 delta=int(sum(reg.exact_star_determinants(p,old))-sum(reg.exact_star_determinants(p,new[~deleted])))
 exact_delta-=delta
 v['t'][rows]=new;v['active'][rows[deleted]]=False;v['map'][int(result['retired_volume_node'])]=int(result['kept_volume_node'])
 v['node_incidence_added'].setdefault(int(result['kept_volume_node']),set()).update(map(int,rows[~deleted]))
 surf_q[drop]=keep;seen_pairs.add((drop,keep));seen_rows.update(map(int,rows))
 full_rows.extend(map(int,rows));full_offsets.append(len(full_rows));changed_rows.extend(map(int,cr));changed_offsets.append(len(changed_rows));removed_rows.extend(map(int,rr));removed_offsets.append(len(removed_rows))
 old_changed.extend(old[changed].tolist());new_changed.extend(new[changed].tolist());deleted_changed.extend(deleted[changed].tolist())
 records.append({'operation_index':oi,'patch_id':op['patch_id'],'drop_D_point_row':drop,'keep_D_point_row':keep,'drop_volume_input_row':int(result['retired_volume_node']),'keep_volume_input_row':int(result['kept_volume_node']),'full_star_cell_count':len(rows),'changed_cell_count':len(cr),'removed_cell_count':len(rr),'exact_det6_delta_integer':str(-delta),'native_bad_before':int(result['bad_before']),'native_bad_after':int(result['bad_after'])})
if failures:
 OUT.mkdir(parents=True,exist_ok=False)
 rp=OUT/'failed-transfer.json';rp.write_text(json.dumps({'schema':'h706.d-shortedge-414-volume-transfer.v1','status':'FAIL_CLOSED_PARTIAL_TRANSFER_NO_CANDIDATE','physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN','failure':failures[0],'completed_operations':len(records),'expected_operations':len(ops),'surface_report_sha256':SURF_REPORT_SHA},indent=2,sort_keys=True)+'\n')
 print(rp.read_text());raise SystemExit(2)
if len(records)!=2363:raise ValueError(f'expected all 2363 transfer ops, got {len(records)}')
active=v['active'];active_rows=np.flatnonzero(active);tt=v['t'][active];newbd=owner.boundary(tt)
expected_actual=v['map'][base_boundary];es=np.sort(expected_actual,axis=1);expected_actual=expected_actual[~np.any(es[:,1:]==es[:,:-1],axis=1)]
if not np.array_equal(owner.boundary_identity(newbd),owner.boundary_identity(expected_actual)):raise ValueError('volume quotient boundary differs from exact mapped input volume boundary')
# Map the desired current target through both the D surface quotient and the volume quotient.
target=np.asarray(base['desired_boundary_faces'],np.int64);mapped=v['map'][target];ts=np.sort(mapped,axis=1);tdeg=np.any(ts[:,1:]==ts[:,:-1],axis=1);mapped_target=mapped[~tdeg]
def face_counter(f):
 return Counter(tuple(map(int,x)) for x in owner.boundary_identity(np.asarray(f,np.int64)))
actual_target_c=face_counter(newbd);desired_target_c=face_counter(mapped_target)
target_missing=sum((desired_target_c-actual_target_c).values());target_extra=sum((actual_target_c-desired_target_c).values())
# Float32 metrics over the full output; local exact signs were replayed by 379 for every changed star.
used=np.unique(tt);outp=p;ot=tt.copy();det,q=m.measurements(outp,ot)
if np.any(det<=0):raise ValueError('full output has nonpositive measured Float32 cell')
# Independent material and unchanged-cell replay.
if np.any(mat!=16):raise ValueError('input material field is not Fat material 16')
if len(np.unique(ot,axis=0))!=len(ot):raise ValueError('duplicate output tetrahedron')
unchanged=np.ones(len(t),bool);unchanged[np.unique(np.asarray(full_rows,np.int64))]=False
if not np.array_equal(v['t'][unchanged],t[unchanged]) or not np.all(active[unchanged]):raise ValueError('a cell outside the recorded operation stars changed')
# Build exact output point/cell maps and transform every live target-node array.
input_node_to_output=v['map'].copy();output_node_to_input=np.arange(len(p),dtype=np.int64);input_cell_to_output=np.full(len(t),-1,np.int64);input_cell_to_output[active_rows]=np.arange(len(active_rows),dtype=np.int64)
# The surface quotient is a source-row map; apply it before the volume quotient and compaction.
final_D_point_to_output=v['map'][d_to_v[surf_q]]
if np.any(final_D_point_to_output<0):raise ValueError('current D point map has an unmapped output node')
# Reindex desired/current and historical boundary arrays while keeping aligned source metadata.
def map_faces_in_rows(arr):
 a=np.asarray(arr,np.int64);mv=v['map'][a];s=np.sort(mv,axis=1);deg=np.any(s[:,1:]==s[:,:-1],axis=1);return mv[~deg],deg
payload={k:base[k] for k in base}
payload['points_world_m']=outp.astype(np.float32,copy=False);payload['points_world_m_f32']=outp.astype(np.float32,copy=False);payload['tetrahedra']=ot;payload['material_indices']=mat[active];payload['boundary_faces']=newbd.copy()
# Canonical target face array and source lanes are reduced by precisely the degenerate mapped source faces.
for name in ('desired_boundary_faces','previous_desired_boundary_faces','preflip_run041_desired_boundary_faces'):
 if name in payload:
  old=np.asarray(base[name],np.int64);new,deg=map_faces_in_rows(old);payload[name]=new
  if name=='desired_boundary_faces': target_deg=deg
for name in ('desired_boundary_source_rows','desired_boundary_source_group','desired_boundary_source_face_rows'):
 if name in payload and len(payload[name])==len(target_deg):payload[name]=np.asarray(base[name])[~target_deg]
if 'desired_boundary_degenerate_source_rows' in payload:
 prior=np.asarray(base['desired_boundary_degenerate_source_rows'],np.int64)
 src=np.asarray(base['desired_boundary_source_rows'],np.int64)
 payload['desired_boundary_degenerate_source_rows']=np.r_[prior,src[target_deg]]
for name in ('fat_current_target_rebind_D_point_to_volume_rows','fat_current_target_rebind_I_point_to_volume_rows'):
 if name in payload:
  old=np.asarray(base[name],np.int64);new=v['map'][old]
  if np.any(new<0):raise ValueError(f'{name} has a point absent from output')
  payload[name]=new
for name in ('fat_current_target_rebind_wall_faces_volume_rows',):
 if name in payload:
  old=np.asarray(base[name],np.int64);new=v['map'][old]
  if np.any(new<0):raise ValueError(f'{name} has a point absent from output')
  payload[name]=new
# Preserve original provenance lanes and add this transfer's exact map/certificate.
payload.update({'retained_input_cell_rows':active_rows,'input_cell_to_output_cell':input_cell_to_output,'output_cell_to_input_cell':active_rows.copy(),'input_node_to_output_node':input_node_to_output,'output_node_to_input_node':output_node_to_input,'source_D414_surface_quotient':surf_q,'source_D414_vertex_to_output_node':final_D_point_to_output,'source_D414_operation_indices':np.asarray([r['operation_index'] for r in records],np.int64)})
# Rebase the inherited 406 cell keys and comp1 metadata through this output row map.
if 'source_base406_cell_rows' in base:payload['source_base406_cell_rows']=np.asarray(base['source_base406_cell_rows'])[active_rows]
if 'source_comp1_output_cell_rows' in base:
 old=np.asarray(base['source_comp1_output_cell_rows'],np.int64);payload['source_comp1_output_cell_rows']=input_cell_to_output[old]
 if np.any(payload['source_comp1_output_cell_rows']<0):raise ValueError('D batch removed a protected comp1 cell')
if 'source_comp1_added_point_rows' in base:
 old=np.asarray(base['source_comp1_added_point_rows'],np.int64);payload['source_comp1_added_point_rows']=input_node_to_output[old]
 if np.any(payload['source_comp1_added_point_rows']<0):raise ValueError('D batch removed a protected comp1 point')
if 'source_comp1_replacement_start' in base:
 inherited=np.asarray(base['source_comp1_replacement_start'],np.int64).copy();rows=np.asarray(payload['source_comp1_output_cell_rows'],np.int64)
 if len(rows)==0 or not np.array_equal(rows,np.arange(int(rows[0]),int(rows[0])+len(rows),dtype=np.int64)):raise ValueError('protected comp1 output cell block is not contiguous after D transfer')
 payload['source_comp1_replacement_start_inherited']=inherited
 payload['source_comp1_replacement_start']=np.asarray([int(rows[0])],np.int64)
# Keep exact transfer evidence in a compact sidecar, not duplicated inside a giant JSON receipt.
OUT.mkdir(parents=True,exist_ok=False);side=OUT/'operation-cell-star-map.npz'
np.savez_compressed(side,operation_full_star_offsets=np.asarray(full_offsets,np.int64),operation_full_star_cell_rows=np.asarray(full_rows,np.int64),operation_changed_cell_offsets=np.asarray(changed_offsets,np.int64),operation_changed_cell_rows=np.asarray(changed_rows,np.int64),operation_removed_cell_offsets=np.asarray(removed_offsets,np.int64),operation_removed_cell_rows=np.asarray(removed_rows,np.int64),operation_old_changed_tetrahedra=np.asarray(old_changed,np.int64).reshape(-1,4),operation_new_changed_tetrahedra=np.asarray(new_changed,np.int64).reshape(-1,4),operation_changed_deleted=np.asarray(deleted_changed,bool),authorized_orientation_indices=np.asarray([auth_lookup[(r['drop_D_point_row'],r['keep_D_point_row'])] for r in records],np.int64),comp0_comp1_protected_cell_rows=np.asarray(sorted(protected_cells),np.int64),comp0_comp1_protected_point_rows=np.asarray(sorted(protected_points),np.int64),D_source_point_to_414_input_volume=np.asarray(d_to_v,np.int64),D_surface_quotient=surf_q,D414_D_vertex_to_output_node=final_D_point_to_output)
candidate=OUT/'fat-D414-surface-quotient-volume-proposal.npz'
payload.update({'source_D414_operation_cell_star_map_path':np.asarray([str(side)]),'source_D414_operation_cell_star_map_sha256':np.asarray([sha(side)])})
np.savez_compressed(candidate,**payload)
# Read back decisive arrays and maps.
out=load(candidate)
for k,expected in (('points_world_m_f32',outp),('tetrahedra',ot),('material_indices',mat[active]),('boundary_faces',newbd)):
 if not np.array_equal(out[k],expected):raise ValueError(f'candidate readback mismatch in {k}')
if not np.array_equal(owner.boundary_identity(out['boundary_faces']),owner.boundary_identity(newbd)):raise ValueError('output boundary readback mismatch')
if int(np.count_nonzero(~m.passes(det,q)))>base_bad:raise ValueError('global native-bad cell count regressed')
# Honest float summary plus exact local integer-lattice volume-delta witness.
base_vol=float(np.sum(base_det,dtype=np.float64)/6.0);out_vol=float(np.sum(det,dtype=np.float64)/6.0)
report={'schema':'h706.d-shortedge-414-indexed379-volume-transfer.v1','status':'PASS_D414_VOLUME_QUOTIENT_PROPOSAL','physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN','input_volume_path':str(BASE),'input_volume_sha256':BASE_SHA,'qualified_414_geometry_path':str(VOLUME414),'qualified_414_geometry_sha256':VOLUME414_SHA,'surface_report_path':str(SURF_REPORT),'surface_report_sha256':SURF_REPORT_SHA,'surface_lineage_crosswalk_path':str(LINEAGE),'surface_lineage_crosswalk_sha256':LINEAGE_SHA,'preflight_report_path':str(PREFLIGHT),'preflight_report_sha256':PREFLIGHT_SHA,'run017_warning_disjointness_path':str(WARN_DISJOINT),'run017_warning_disjointness_sha256':WARN_DISJOINT_SHA,'raw_D_surface_candidate_path':str(SURF),'raw_D_surface_candidate_sha256':SURF_SHA,'preflight_authorized_orientation_map_path':str(AUTH),'preflight_authorized_orientation_map_sha256':AUTH_SHA,'indexed379_owner_path':str(OWNER),'indexed379_owner_sha256':OWNER_SHA,'D_context_audit_path':str(REPORT_D),'D_context_audit_sha256':REPORT_D_SHA,'protected_routes_path':str(protected_routes_path),'protected_routes_sha256':protected_routes_sha,'comp0_mask_path':str(COMP0),'comp0_mask_sha256':COMP0_SHA,'comp1_protection_path':str(COMP1),'comp1_protection_sha256':COMP1_SHA,'operation_count':len(records),'transferred_orientation_count':len(records),'input_points':len(p),'output_points':len(outp),'unused_output_point_count':int(len(p)-len(used)),'input_cells':len(t),'output_cells':len(ot),'removed_cells':int(len(t)-len(ot)),'all_point_coordinates_preserved':bool(np.array_equal(outp,p)),'all_materials_16':bool(np.all(mat[active]==16)),'protected_comp0_comp1_star_overlap_operations':0,'operation_full_star_cells_unique':len(seen_rows),'operation_full_star_overlap_count':0,'native_bad_cells_before':base_bad,'native_bad_cells_after':int(np.count_nonzero(~m.passes(det,q))),'native_bad_cell_gain':base_bad-int(np.count_nonzero(~m.passes(det,q))),'nonpositive_cells':int(np.count_nonzero(det<=0)),'native_absolute_volume_failures':int(np.count_nonzero(det/6<=1e-18)),'native_normalized_det_failures':int(np.count_nonzero(q<1e-5)),'minimum_det6':float(det.min()),'minimum_volume_m3':float((det/6).min()),'minimum_normalized_det':float(q.min()),'base_total_volume_float_m3':base_vol,'output_total_volume_float_m3':out_vol,'total_volume_float_delta_m3':out_vol-base_vol,'exact_F32_local_det6_delta_integer':str(exact_delta),'exact_F32_det6_scale_denominator':str(1 << 447),'volume_boundary_equals_mapped_current_boundary':True,'volume_boundary_equals_mapped_target_chain':bool(target_missing==0 and target_extra==0), 'mapped_target_boundary_residual':{'missing_oriented_records':int(target_missing),'extra_oriented_records':int(target_extra)},'target_degenerate_faces_removed':int(np.count_nonzero(target_deg)),'final_target_face_count':int(len(out['desired_boundary_faces'])),'current_boundary_face_count':int(len(out['boundary_faces'])),'operation_cell_star_map_path':str(side),'operation_cell_star_map_sha256':sha(side),'candidate_path':str(candidate),'candidate_sha256':sha(candidate),'elapsed_s':time.monotonic()-start,'limitations':['The D target chain is a zero-coordinate topology quotient and the volume is an exact indexed379 proposal.','Full O/I/D + both-wall/all185 context on the transferred candidate, 697 typed binding for the new D, source mass mapping, and native execution remain separate qualification gates.','Mass and physical admission are not claimed.']}
(OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps(report,indent=2,sort_keys=True))
