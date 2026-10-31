#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
import hashlib,importlib.util,json,time,sys
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
OUT=H/'fat614-yy-volume-replay-004/run-006-interior-155-55-sat';PARENT=H/'fat614-yy-volume-replay-002/fat-comp2-comp6-rounded-boundary-points-volume-proposal.npz';PART=H/'fat614-compact-cavities-004-budget1-satcap-002/connected-fixed-surface-volume-proposal.npz';PREP=H/'fat614-compact-cavities-004-anchor-stars/connected-cavity-shells.npz';DFINAL=H/'fat614-minimal-f32-target-split-002/run-005/candidate-D-remaining-yy-target-edge-splits.npz';D602=H/'fat614-minimal-f32-target-split-001/candidate-D-minimally-rounded-yy-edge-split.npz';SEARCH=H/'fat614-yy-volume-replay-004/run-005-interior-ulp-155-55/result.json'
sp=importlib.util.spec_from_file_location('remesh296',H/'remesh_fat614_compact_cavities_296_007.py');r=importlib.util.module_from_spec(sp);sys.modules[sp.name]=r;sp.loader.exec_module(r);topo=r.m;m=topo.m
capspec=importlib.util.spec_from_file_location('caps288',H/'direct_connected_skin_caps_288_root.py');caps=importlib.util.module_from_spec(capspec);sys.modules[capspec.name]=caps;capspec.loader.exec_module(caps)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def tab(path):
 ls=[x.strip() for x in Path(path).read_text().splitlines() if x.strip() and not x.lstrip().startswith('#')];n=int(ls[0].split()[0]);a=np.asarray([[float(v) for v in x.split()] for x in ls[1:1+n]],np.float64)
 if len(a)!=n:raise ValueError(f'{path} row count mismatch')
 return a
def fkey(x):return np.asarray(x,np.float32).view(np.uint32).tobytes()
def exact_dets(p,t):
 used=np.unique(np.asarray(t,np.int64).ravel());lut=np.full(len(p),-1,np.int64);lut[used]=np.arange(len(used));ip=m.integer_points(np.asarray(p,np.float32)[used]);return [int(m.exact_det(ip,[int(lut[int(v)]) for v in tet])) for tet in np.asarray(t,np.int64)]
def exact_flux(p,b):
 used=np.unique(np.asarray(b,np.int64).ravel());lut=np.full(len(p),-1,np.int64);lut[used]=np.arange(len(used));ip=m.integer_points(np.asarray(p,np.float32)[used]);v=0
 for f in b:
  a,c,d=[ip[int(lut[int(x)])] for x in f];v+=a[0]*(c[1]*d[2]-c[2]*d[1])-a[1]*(c[0]*d[2]-c[2]*d[0])+a[2]*(c[0]*d[1]-c[1]*d[0])
 return int(v)
def oriented(b):
 out=Counter()
 for f in np.asarray(b,np.int64):
  a,c,d=map(int,f);s=tuple(sorted((a,c,d)));inv=int(a>c)+int(a>d)+int(c>d);out[s]+=1 if inv%2==0 else -1
 return +out

def expected_map(parent_points, d_points, nodes, used_node_ids, node_ids):
 point_list=[x.copy() for x in parent_points];lut={fkey(x):i for i,x in enumerate(point_list)}
 if len(lut)!=len(point_list):raise ValueError('duplicate parent point coordinates')
 for x in d_points:
  k=fkey(x)
  if k not in lut:lut[k]=len(point_list);point_list.append(np.asarray(x,np.float32).copy())
 d_added=len(point_list)-len(parent_points);ng=np.full(len(node_ids),-1,np.int64)
 for i,x64 in enumerate(nodes):
  if int(node_ids[i]) not in used_node_ids:continue
  x=np.asarray(x64,np.float32);k=fkey(x)
  if k in lut:g=lut[k]
  else:g=len(point_list);lut[k]=g;point_list.append(x.copy())
  ng[i]=g
 return np.asarray(point_list,np.float32),ng,d_added

started=time.monotonic();search=json.loads(SEARCH.read_text());parent=load(PARENT);pp=np.asarray(parent['points_world_m'],np.float32);tdesc=np.asarray(parent['tetrahedra'],np.int64);part=load(PART);prep=load(PREP);dfinal=load(DFINAL);d602=load(D602);dfp=np.asarray(dfinal['points_world_m_f32'],np.float32);d602p=np.asarray(d602['points_world_m_f32'],np.float32)
if sha(PARENT)!='d4906b176f78283f4073f9bd1e8d202a7301b57550a7b1b71fb101e76c4d236f' or sha(DFINAL)!='4ce01a4bd397e7cf2b3c469ee33a3937d9173ff7d585a2687a1a09cf98377ab2':raise ValueError('parent or D target changed')
if sha(SEARCH)!='54472a6fbf94ff6e95c19316d1ed04e3743ab9260b401c8593ee5fecabd58a90':raise ValueError('interior ULP search receipt changed')
if search.get('status')!='CANDIDATES_FOUND' or search.get('physical_admission') is not False:raise ValueError('search is not source-only candidate receipt')
if not np.array_equal(dfp[:len(d602p)].view(np.uint32),d602p.view(np.uint32)):raise ValueError('D point prefix mismatch')
# Exclude all seven source cavities exactly as in the full remesh owner.
pmap={int(x):i for i,x in enumerate(np.asarray(part['retained_input_cell_rows'],np.int64))};dmap={int(x):i for i,x in enumerate(np.asarray(parent['retained_input_cell_rows'],np.int64))};co=np.asarray(prep['cell_offsets'],np.int64);deleted=[]
for ci in (3,4,5,7,8,10,11):
 src=np.asarray(prep['input_cell_rows'][int(co[ci]):int(co[ci+1])],np.int64);partial=np.asarray([pmap[int(x)] for x in src],np.int64);deleted.extend(int(dmap[int(x)]) for x in partial)
if len(set(deleted))!=len(deleted):raise ValueError('pending component source rows overlap')
local_tree_cache={};external_tree=caps.TetBVH(pp,tdesc);results={}
for ci,nodeid in ((7,155),(10,55)):
 C=H/f'fat614-compact-cavities-004-remesh-002/component-{ci:04d}';node_path=C/'cavity.1.node';ele_path=C/'cavity.1.ele';face_path=C/'cavity.1.face';poly_path=C/'cavity.poly'
 nd=tab(node_path);el=tab(ele_path);ft=tab(face_path);pl=tab(poly_path);ids=nd[:,0].astype(np.int64);raw=nd[:,1:4].astype(np.float64);p=raw.astype(np.float32);lookup={int(x):i for i,x in enumerate(ids)};t=np.asarray([[lookup[int(x)] for x in rr] for rr in el[:,1:5]],np.int64);row=lookup[nodeid];incident=np.flatnonzero(np.any(t==row,axis=1));cand=search['components'][str(ci)]['candidate']
 if cand is None:raise ValueError(f'component{ci} has no ULP candidate')
 faceids=set(ft[:,1:4].astype(np.int64).ravel());polyids=set(pl[:,0].astype(int))
 if nodeid in faceids or nodeid in polyids:raise ValueError(f'component{ci} node not interior')
 used=set(el[:,1:5].astype(np.int64).ravel())|faceids;out_points,node_global,dadded=expected_map(pp,dfp,p,used,ids)
 if node_global[row]<len(pp)+dadded:raise ValueError(f'component{ci} corrected node not a newly added interior point')
 global_row=int(node_global[row]);old_coord=p[row].copy();new_coord=np.asarray(cand['point_f32'],np.float32);oldkey=fkey(old_coord);newkey=fkey(new_coord)
 if fkey(out_points[global_row])!=oldkey:raise ValueError(f'component{ci} expected old point mismatch')
 index={fkey(x):i for i,x in enumerate(out_points)}
 if newkey in index and index[newkey]!=global_row:raise ValueError(f'component{ci} new point aliases another point')
 out_points[global_row]=new_coord
 # all local tetra rows in global volume point namespace
 out_t=np.asarray([[int(node_global[lookup[int(n)]]) for n in rr] for rr in el[:,1:5]],np.int64)
 if np.any(out_t<0):raise ValueError(f'component{ci} has unmapped point row')
 baseline_det,baseline_q=m.measurements(out_points,out_t);base_pass=np.asarray(m.passes(baseline_det,baseline_q),bool);raw_det,_=m.measurements(raw,t);before_exact=exact_dets(out_points,out_t);new_exact=before_exact
 # Reconstruct the baseline geometry with this node's original F32 point, and certify candidate star positivity.
 out_points_before=out_points.copy();out_points_before[global_row]=old_coord
 before_exact=exact_dets(out_points_before,out_t);old_det,old_q=m.measurements(out_points_before,out_t);old_pass=np.asarray(m.passes(old_det,old_q),bool);new_det,new_q=m.measurements(out_points,out_t);new_pass=np.asarray(m.passes(new_det,new_q),bool)
 star_exact=[new_exact[k] for k in incident]
 if any(x<=0 for x in star_exact):raise ValueError(f'component{ci} corrected star remains nonpositive')
 if np.any(old_pass&~new_pass):raise ValueError(f'component{ci} correction degrades formerly-good cell')
 if not np.all(np.isfinite(new_det)):raise ValueError(f'component{ci} nonfinite determinant')
 old_star_local=topo.boundary(t[incident]);new_star=topo.boundary(out_t[incident]);old_star=node_global[old_star_local]
 old_comp=node_global[topo.boundary(t)];new_comp=topo.boundary(out_t)
 if oriented(old_star)!=oriented(new_star):raise ValueError(f'component{ci} star boundary changed')
 if oriented(old_comp)!=oriented(new_comp):raise ValueError(f'component{ci} component boundary changed')
 star_flux=exact_flux(out_points,new_star);star_vol=sum(star_exact)
 if star_flux!=star_vol:raise ValueError(f'component{ci} exact star volume/flux mismatch')
 full_vol=sum(new_exact);full_flux=exact_flux(out_points,new_comp)
 if full_vol!=full_flux:raise ValueError(f'component{ci} exact component volume/flux mismatch')
 local_tree=caps.TetBVH(out_points,out_t);local_sat=caps.check_intersections(out_points,out_t,out_t[incident],incident,[],local_tree,max_pairs=2_000_000)
 if not local_sat.get('pass'):raise ValueError(f'component{ci} local SAT failed: {local_sat}')
 external_sat=caps.check_intersections(out_points,tdesc,out_t[incident],np.asarray(deleted,np.int64),[],external_tree,max_pairs=10_000_000)
 if not external_sat.get('pass'):raise ValueError(f'component{ci} d4906 retained SAT failed: {external_sat}')
 neg_before=int(sum(x<=0 for x in before_exact));neg_after=int(sum(x<=0 for x in new_exact));native_before=int(sum(~old_pass));native_after=int(sum(~new_pass))
 usedpts=np.unique(out_t);local_npz=OUT/f'component-{ci:04d}-interior-corrected.npz';np.savez_compressed(local_npz,points_world_m=out_points[usedpts],tetrahedra=np.searchsorted(usedpts,out_t),output_point_global_volume_rows=usedpts,material_indices=np.full(len(out_t),16,np.int32),tetgen_output_node_ids=ids,output_node_global_volume_rows=node_global,corrected_tetgen_node_id=np.asarray([nodeid],np.int64),corrected_parent_volume_point_row=np.asarray([global_row],np.int64),corrected_point_world_m=np.asarray([new_coord],np.float32),incident_element_ids=el[incident,0].astype(np.int64))
 results[str(ci)]={'component_id':ci,'tetgen_node_id':nodeid,'global_appended_point_row':global_row,'old_point_f32':old_coord.tolist(),'new_point_f32':new_coord.tolist(),'ulp_offsets':cand['offsets_ulp'],'displacement_m':cand['displacement_m'],'star_tets':len(incident),'star_exact_positive_after':int(len(star_exact)),'component_tets':len(out_t),'component_exact_nonpositive_before':neg_before,'component_exact_nonpositive_after':neg_after,'component_exact_positive_after':int(len(new_exact)-neg_after),'component_native_failures_before':native_before,'component_native_failures_after':native_after,'formerly_good_cells_degraded':0,'raw_tetgen_f64_nonpositive':int(np.count_nonzero(raw_det<=0)),'star_boundary_unchanged':True,'component_boundary_unchanged':True,'exact_star_volume_equals_boundary_flux':True,'exact_component_volume_equals_boundary_flux':True,'local_component_sat':local_sat,'d4906_retained_sat':external_sat,'local_candidate_path':str(local_npz),'local_candidate_sha256':sha(local_npz)}
report={'schema':'numi.fat614.inferred-interior-point-exact-star-and-sat.v1','status':'PASS_LOCAL_INTERIOR_STAR_CHECKS','physical_admission':False,'native_steps':0,'mass_mapping_status':'NOT_RUN','parent_candidate_sha256':sha(PARENT),'D_target_sha256':sha(DFINAL),'search_receipt_sha256':sha(SEARCH),'pending_source_cavities_excluded':len(deleted),'components':results,'elapsed_s':time.monotonic()-started,'comp7_boundary_only_nonpositive_remains':1}
path=OUT/'report.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
