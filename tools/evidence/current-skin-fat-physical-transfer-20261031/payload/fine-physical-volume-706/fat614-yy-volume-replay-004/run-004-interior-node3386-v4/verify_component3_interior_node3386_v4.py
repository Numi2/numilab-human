#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
import hashlib,importlib.util,json,time
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
OUT=H/'fat614-yy-volume-replay-004/run-004-interior-node3386-v4'
start=time.monotonic()
CASE=H/'fat614-compact-cavities-004-remesh-002/component-0003'
BASE=H/'fat614-residual-pass-001/remeshed/connected-fixed-surface-volume-proposal.npz'
PART=H/'fat614-compact-cavities-004-budget1-satcap-002/connected-fixed-surface-volume-proposal.npz'
PREP=H/'fat614-compact-cavities-004-anchor-stars/connected-cavity-shells.npz'
PARENT=H/'fat614-yy-volume-replay-002/fat-comp2-comp6-rounded-boundary-points-volume-proposal.npz'
DFINAL=H/'fat614-minimal-f32-target-split-002/run-005/candidate-D-remaining-yy-target-edge-splits.npz'
D602=H/'fat614-minimal-f32-target-split-001/candidate-D-minimally-rounded-yy-edge-split.npz'
SEARCH=H/'fat614-yy-volume-replay-004/run-003-ulps-v3/result.json'
EXPECT={'parent':'d4906b176f78283f4073f9bd1e8d202a7301b57550a7b1b71fb101e76c4d236f','d_final':'4ce01a4bd397e7cf2b3c469ee33a3937d9173ff7d585a2687a1a09cf98377ab2','search':'c108c26435a9caa2b64c26ec961ce6f3f0490646669e4aaaa5fb5db7a171d781','node':'7020e2aff109e2eac24bc5ccbc731861da8d4e827a2b4cf2e659d26c6a0d2a65','ele':'fa1539247b9f4d5aa82c109a61e0e477ef8e79c276870b09a7802b40723f3f24','face':'8e3d472115d8435d56241e8e5439215fc5b5b12df91c9232ec3b26fe987db954'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(sp);import sys;sys.modules[name]=mod;sp.loader.exec_module(mod);return mod
def load_npz(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def table(path):
 lines=[x.strip() for x in Path(path).read_text().splitlines() if x.strip() and not x.lstrip().startswith('#')]
 n=int(lines[0].split()[0]);a=np.asarray([[float(v) for v in s.split()] for s in lines[1:1+n]],np.float64)
 if len(a)!=n:raise ValueError(f'{path}: row count mismatch')
 return a
def f32key(x):return np.asarray(x,np.float32).view(np.uint32).tobytes()
def exact_dets(p,t):
 used=np.unique(np.asarray(t,np.int64).ravel());lookup=np.full(len(p),-1,np.int64);lookup[used]=np.arange(len(used));ip=m.integer_points(np.asarray(p,np.float32)[used])
 return [int(m.exact_det(ip,[int(lookup[int(x)]) for x in tet])) for tet in np.asarray(t,np.int64)]
def exact_flux(p,b):
 if not len(b):return 0
 used=np.unique(np.asarray(b,np.int64).ravel());lookup=np.full(len(p),-1,np.int64);lookup[used]=np.arange(len(used));ip=m.integer_points(np.asarray(p,np.float32)[used]);v=0
 for face in b:
  a,c,d=[ip[int(lookup[int(x)])] for x in face]
  v+=a[0]*(c[1]*d[2]-c[2]*d[1])-a[1]*(c[0]*d[2]-c[2]*d[0])+a[2]*(c[0]*d[1]-c[1]*d[0])
 return int(v)
def face_unoriented(b):return Counter(tuple(sorted(map(int,f))) for f in np.asarray(b,np.int64))
def face_oriented(b):
 out=Counter()
 for f in np.asarray(b,np.int64):
  a,b0,c=map(int,f);s=tuple(sorted((a,b0,c)));inv=(a>b0)+(a>c)+(b0>c);out[s]+=1 if inv%2==0 else -1
 return +out
for label,path,want in [('parent',PARENT,EXPECT['parent']),('d_final',DFINAL,EXPECT['d_final']),('search',SEARCH,EXPECT['search']),('node',CASE/'cavity.1.node',EXPECT['node']),('ele',CASE/'cavity.1.ele',EXPECT['ele']),('face',CASE/'cavity.1.face',EXPECT['face'])]:
 got=sha(path)
 if got!=want:raise ValueError(f'{label} SHA mismatch {got}')
# Load the frozen predicates and SAT owner.
r=load('remesh296',H/'remesh_fat614_compact_cavities_296_007.py');topo=r.m;m=r.m.m
caps=load('caps288',H/'direct_connected_skin_caps_288_root.py')
# Search receipt is an exact local-star candidate; do not trust coordinates without recomputing all gates.
search=json.loads(SEARCH.read_text());cand=search.get('candidate')
if search.get('status')!='PASS_F32_INTERIOR_STAR_REPAIR_CANDIDATE' or search.get('physical_admission') is not False or search.get('native_steps')!=0:raise ValueError('search receipt is not source-only PASS')
node=table(CASE/'cavity.1.node');ele=table(CASE/'cavity.1.ele');faces=table(CASE/'cavity.1.face')
node_ids=node[:,0].astype(np.int64);raw_points=node[:,1:4].astype(np.float64);local_points=raw_points.astype(np.float32);id_to_row={int(x):i for i,x in enumerate(node_ids)}
if 3386 not in id_to_row:raise ValueError('TetGen node 3386 absent')
point_row=id_to_row[3386];t=np.asarray([[id_to_row[int(x)] for x in row] for row in ele[:,1:5]],np.int64)
if int(ele[0,0])==0:raise ValueError('unexpected TetGen element IDs')
incident=np.flatnonzero(np.any(t==point_row,axis=1));iflen=len(incident)
if iflen!=52:raise ValueError(f'node3386 star size changed: {iflen}')
face_node_ids=set(map(int,faces[:,1:4].astype(np.int64).ravel()))
poly_lines=[x.strip() for x in (CASE/'cavity.poly').read_text().splitlines() if x.strip() and not x.lstrip().startswith('#')]
poly_n=int(poly_lines[0].split()[0]);poly_ids=set(int(x.split()[0]) for x in poly_lines[1:1+poly_n])
if 3386 in face_node_ids or 3386 in poly_ids:raise ValueError('node3386 is not an inferred interior point')
coord=np.asarray(cand['point_f32'],np.float32)
if coord.shape!=(3,) or not np.all(np.isfinite(coord)):raise ValueError('invalid candidate coordinate')
old_coord=local_points[point_row].copy();if_not_equal=np.array_equal(old_coord.view(np.uint32),coord.view(np.uint32))
if if_not_equal:raise ValueError('candidate does not move the inferred node')
# Parent point domain and new D target-point prefix; output node rows are appended in frozen component order.
parent=load_npz(PARENT);pp=np.asarray(parent['points_world_m'],np.float32);tdesc=np.asarray(parent['tetrahedra'],np.int64)
dfinal=load_npz(DFINAL);d602=load_npz(D602);dfp=np.asarray(dfinal['points_world_m_f32'],np.float32);d602p=np.asarray(d602['points_world_m_f32'],np.float32)
point_list=[x.copy() for x in pp];point_index={f32key(x):i for i,x in enumerate(pp)}
if len(point_index)!=len(pp):raise ValueError('parent volume has duplicate point coordinates')
if not np.array_equal(dfp[:len(d602p)].view(np.uint32),d602p.view(np.uint32)):raise ValueError('D final point prefix mismatch')
for drow,x in enumerate(dfp):
 k=f32key(x)
 if k in point_index:continue
 point_index[k]=len(point_list);point_list.append(x.copy())
new_d_added=len(point_list)-len(pp)
# Every .poly point must map to the authenticated parent or current D target point set.
poly=np.asarray([[float(v) for v in line.split()] for line in poly_lines[1:1+poly_n]],np.float64)
for rec in poly:
 nid=int(rec[0]);key=f32key(np.asarray(rec[1:4],np.float32))
 if key not in point_index:raise ValueError(f'PLC source node {nid} missing from current volume/D point domain')
# Map output nodes exactly as the frozen full replay does.
used_node_ids=set(map(int,ele[:,1:5].astype(np.int64).ravel()))|face_node_ids
node_global=np.full(len(node_ids),-1,np.int64)
for i,x64 in enumerate(raw_points):
 if int(node_ids[i]) not in used_node_ids:continue
 x=np.asarray(x64,np.float32);k=f32key(x)
 if k in point_index:g=point_index[k]
 else:g=len(point_list);point_index[k]=g;point_list.append(x.copy())
 node_global[i]=g
if node_global[point_row]<len(pp)+new_d_added:raise ValueError('interior node unexpectedly aliases a source/D point')
expected_global_row=int(node_global[point_row]);old_key=f32key(old_coord);new_key=f32key(coord)
if point_index.get(old_key)!=expected_global_row:raise ValueError('old interior point map is inconsistent')
if new_key in point_index and point_index[new_key]!=expected_global_row:raise ValueError('candidate coordinate collides with another point')
point_list[expected_global_row]=coord.copy();del point_index[old_key];point_index[new_key]=expected_global_row
out_points=np.asarray(point_list,np.float32)
if not np.array_equal(out_points[:len(pp)].view(np.uint32),pp.view(np.uint32)):raise ValueError('parent volume point prefix changed')
out_tets=np.asarray([[int(node_global[id_to_row[int(n)]]) for n in row] for row in ele[:,1:5]],np.int64)
if np.any(out_tets<0):raise ValueError('used TetGen point row not mapped')
# Exact candidate checks over all 8,459 cells; source F64 signs are recorded separately.
raw_det,_=m.measurements(raw_points,t);base_det,base_q=m.measurements(local_points,t);new_det,new_q=m.measurements(out_points,out_tets)
base_pass=np.asarray(m.passes(base_det,base_q),bool);new_pass=np.asarray(m.passes(new_det,new_q),bool)
new_exact=exact_dets(out_points,out_tets);base_exact=exact_dets(np.asarray(local_points,np.float32),t)
if any(x<=0 for x in new_exact):raise ValueError('corrected comp3 still has exact F32 nonpositive cells')
if np.any(base_pass&~new_pass):raise ValueError('interior correction degrades a formerly passing native-quality cell')
if int(np.count_nonzero(raw_det<=0))!=0:raise ValueError('raw TetGen F64 determinant baseline is not all-positive')
# Star shell, full component shell, exact volume/flux and point-coordinate-only delta.
old_star_local=topo.boundary(t[incident]);old_star=node_global[old_star_local];new_star=topo.boundary(out_tets[incident]);old_component=node_global[topo.boundary(t)];new_component=topo.boundary(out_tets)
if not np.array_equal(topo.boundary_identity(old_star),topo.boundary_identity(new_star)):raise ValueError('moved interior point changed the original star boundary after row-map composition')
if not np.array_equal(topo.boundary_identity(old_component),topo.boundary_identity(new_component)):raise ValueError('moved interior point changed the full component boundary after row-map composition')
star_det=exact_dets(out_points,out_tets[incident]);star_flux=exact_flux(out_points,new_star)
if sum(star_det)!=star_flux:raise ValueError('corrected exact star volume does not equal unchanged star flux')
full_det=sum(new_exact);full_flux=exact_flux(out_points,new_component)
if full_det!=full_flux:raise ValueError('corrected component exact volume does not equal unchanged component boundary flux')
# Verify the corrected local mesh does not overlap its untouched TetGen cells.
local_tree=caps.TetBVH(out_points,out_tets)
local_sat=caps.check_intersections(out_points,out_tets,out_tets[incident],incident,[],local_tree,max_pairs=2_000_000)
if not local_sat.get('pass'):raise ValueError('component-local exact SAT failed: '+str(local_sat))
# Map all seven pending components and exclude their source rows before exact retained-volume SAT.
part=load_npz(PART);prep=load_npz(PREP);pmap={int(x):i for i,x in enumerate(np.asarray(part['retained_input_cell_rows'],np.int64))};dmap={int(x):i for i,x in enumerate(np.asarray(parent['retained_input_cell_rows'],np.int64))}
co=np.asarray(prep['cell_offsets'],np.int64);selected=[]
for ci in (3,4,5,7,8,10,11):
 src=np.asarray(prep['input_cell_rows'][int(co[ci]):int(co[ci+1])],np.int64)
 partial=np.asarray([pmap[int(x)] for x in src],np.int64)
 selected.extend(int(dmap[int(x)]) for x in partial)
if len(set(selected))!=len(selected):raise ValueError('pending component source output-row overlap')
external_tree=caps.TetBVH(pp,tdesc)
external_sat=caps.check_intersections(out_points,tdesc,out_tets[incident],np.asarray(selected,np.int64),[],external_tree,max_pairs=10_000_000)
if not external_sat.get('pass'):raise ValueError('corrected star exact SAT against retained d4906 volume failed: '+str(external_sat))
# Record a local corrected component proposal only; this is not the full Fat candidate.
local_path=OUT/'component-0003-f32-corrected.npz'
used_output_points=np.unique(out_tets)
np.savez_compressed(local_path,points_world_m=out_points[used_output_points],tetrahedra=np.searchsorted(used_output_points,out_tets),output_point_global_volume_rows=used_output_points,material_indices=np.full(len(out_tets),16,np.int32),tetgen_output_node_ids=node_ids,output_node_global_volume_rows=node_global,corrected_tetgen_node_id=np.asarray([3386],np.int64),corrected_parent_volume_point_row=np.asarray([expected_global_row],np.int64),corrected_point_world_m=np.asarray([coord],np.float32),incident_element_ids=ele[incident,0].astype(np.int64))
report={'schema':'numi.fat614.corrected-inferred-interior-point-volume-check.v1','status':'PASS_LOCAL_EXACT_F32_INTERIOR_STAR_AND_RETAINED_TET_SAT','physical_admission':False,'native_steps':0,'mass_mapping_status':'NOT_RUN','input_parent_path':str(PARENT),'input_parent_sha256':sha(PARENT),'d_target_path':str(DFINAL),'d_target_sha256':sha(DFINAL),'search_receipt_path':str(SEARCH),'search_receipt_sha256':sha(SEARCH),'node_file_sha256':sha(CASE/'cavity.1.node'),'element_file_sha256':sha(CASE/'cavity.1.ele'),'face_file_sha256':sha(CASE/'cavity.1.face'),'tetgen_node_id':3386,'local_point_row':int(point_row),'global_appended_point_row':expected_global_row,'mapped_as_new_inferred_interior_point':True,'old_point_f32':old_coord.tolist(),'new_point_f32':coord.tolist(),'offsets_ulp':cand['offsets'],'displacement_m':float(np.linalg.norm((coord-old_coord).astype(np.float64))),'star_tets':len(incident),'star_exact_positive_after':len(star_det),'component_tets':len(out_tets),'component_exact_positive_after':len(new_exact),'raw_tetgen_f64_nonpositive':int(np.count_nonzero(raw_det<=0)),'baseline_f32_exact_nonpositive':int(sum(x<=0 for x in base_exact)),'new_f32_exact_nonpositive':0,'baseline_component_native_failures':int(np.count_nonzero(~base_pass)),'new_component_native_failures':int(np.count_nonzero(~new_pass)),'formerly_good_cells_degraded':0,'exact_star_boundary_unchanged':True,'exact_component_boundary_unchanged':True,'exact_star_volume_equals_boundary_flux':True,'exact_component_volume_equals_boundary_flux':True,'local_component_retained_tet_sat':local_sat,'external_d4906_retained_tet_sat':external_sat,'pending_source_rows_excluded_from_external_sat':len(selected),'candidate_local_npz':str(local_path),'candidate_local_npz_sha256':sha(local_path),'elapsed_s':time.monotonic()-start,'native_admission':'NOT_RUN','mass_mapping':'NOT_RUN'}
report_path=OUT/'report.json';report_path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
