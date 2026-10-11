from pathlib import Path
import sys,importlib.util,hashlib,json,time,math,traceback
from collections import Counter,defaultdict
import numpy as np
from scipy.optimize import linprog
from scipy import sparse
R=Path('/Users/n/numi-human-retained-delivery-20261009'); A=R/'anatomy-completion-1276'
OUT=A/'epl-active-external-clearance-005'; OUT.mkdir(exist_ok=True)
sys.path.insert(0,'/Users/n/numi-human-self-separation-partial-resume-1267/src')
from numilab_human import common_atlas_skin_clearance as ca, cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology
from numilab_human.resting_anatomy_interface_patch import normals
H=R/'fullbody-source-surface-repair-1276/gluteus-maximus-pilot-001/attempt-010/run_gluteus_union_010.py'
J=R/'fullbody-source-surface-repair-1276/gluteus-maximus-pilot-001/attempt-019/solve_terminal_self_pairs_019.py'
TISS=R/'source-seam-connectivity-1247/fhl-current-7b23-count-reconciliation-1258/attempt-002/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'; MAN=TISS.with_suffix('.manifest.json')
INV=Path('/Users/n/numi-human-resting-evidence-20261005/native-complete-skin-containment-audit-890/pair-summary-v3.csv')
CAND=A/'epl-local-self-separation-008/stable-143-unadmitted-local-self-candidate.npz'
SELF=A/'epl-local-self-separation-008/report.json'; AUDIT=A/'epl-all859-interface-audit-002/report.json'; REMAP=A/'epl-all859-target-row-remap-003/report.json'; SKIN=A/'epl-skin-target-audit-001/report.json'
PAIR=R/'skin-resting-multipose-clearance-1218/native-coupled-pair-preparation-011'; EARLY=A/'seventeen-surface-native-composition-002/baseline/native-run/accepted-geometry'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def imp(path,name):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
h=imp(H,'epl_joint_forward_owner');jowner=imp(J,'epl_existing_pose_jacobian_owner')
tissue=h.load_tissue(TISS); row=h.row_data(tissue,143)
z=np.load(CAND,allow_pickle=False); base=np.ascontiguousarray(z['vertices6'][:,:3],dtype='<f4');faces=np.ascontiguousarray(z['faces'],dtype=np.int64);origins=np.ascontiguousarray(z['face_origins'],dtype=np.int64);local=np.ascontiguousarray(z['binding_indices'],dtype='<u4');weights=np.ascontiguousarray(z['weights'],dtype='<f4')
orig_faces=np.ascontiguousarray(row['faces'],dtype=np.int64);orig_positions=np.ascontiguousarray(row['positions'],dtype='<f4');orig_local=np.ascontiguousarray(row['local'],dtype='<u4');orig_weights=np.ascontiguousarray(row['weights'],dtype='<f4')
assert len(faces)==1324 and len(orig_faces)==1326 and len(base)==660 and len(origins)==len(faces)
keys,*_=ca._load_target_inventory(INV); assert len(keys)==859 and (51005,143) in keys
TARGET_KEYS=set(keys)-{(51005,143)}; SKIN_KEY=(51007,1); assert len(TARGET_KEYS)==858
PAIR_STEPS=(0,47519,151999,152607,153215,153823,154431,155000)
pose_specs=[]
for arm in ('baseline','intervention'):
 for step in PAIR_STEPS:pose_specs.append(('pair011',arm,step,PAIR/arm/'native-run/accepted-geometry'))
for step in (0,4767,5023,5599,6207,6815,7423,8000):pose_specs.append(('early','baseline',step,EARLY))
assert len(pose_specs)==24
rem=json.load(open(REMAP)); audit=json.load(open(AUDIT)); skin=json.load(open(SKIN)); self_report=json.load(open(SELF))
assert rem['status']=='complete_all_reported_delta_pairs_remapped_and_exactly_rechecked' and rem['inputs_unchanged']
assert audit['inputs_unchanged'] and audit['pose_rows_expected']==24 and audit['pose_rows_actual']==24
assert audit['identity']['inventory_key_count']==859
assert all(int(r['inventory_total'])==859 and int(r['external_target_count_scanned'])==858 and r['self_target_excluded']==[51005,143] for r in audit['rows']), 'inventory audit is 858 external inventory targets after excluding source; it must be paired with explicit skin target audit'
assert skin['inputs_unchanged'] and skin['status']=='complete_24_skin_target_exact_checks'
assert self_report['inputs_unchanged'] and self_report['after']['all24_pose_self_clear']
assert len(rem['rows'])==24 and len(audit['rows'])==24 and len(skin['rows'])==24
rem_rows={(r['cohort'],r['arm'],int(r['step'])):r for r in rem['rows']}
audit_rows={(r['cohort'],r['arm'],int(r['step'])):r for r in audit['rows']}
skin_rows={(r['cohort'],r['arm'],int(r['step'])):r for r in skin['rows']}
posekeys={(c,a,s) for c,a,s,_ in pose_specs}; assert set(rem_rows)==set(audit_rows)==set(skin_rows)==posekeys
# Independently bind every currently retained input used to generate the added-pair screen.
pins={str(p):sha(p) for p in [Path(__file__),H,J,TISS,MAN,INV,CAND,SELF,AUDIT,REMAP,SKIN,Path(ca.__file__),Path(ci.__file__),Path(analyze_topology.__code__.co_filename),Path(normals.__code__.co_filename)]}
for report in (audit,rem,skin):
 for p,digest in report.get('inputs_after',{}).items():
  pp=Path(p); key=str(pp)
  if pp.exists():
   if key in pins: assert pins[key]==digest, f'conflicting pinned report input digest: {pp}'
   else: assert sha(pp)==digest, f'report input drift: {pp}'
  pins[key]=digest
# Existing matched target tables from audit002 contain all858 inventory external targets; skin target was independently rechecked empty.
# Do not use the report's stale prose label; reconstruct coverage explicitly from owner-returned surface sets.
added_occ=[]; counts_by_pose={}
def pair_records(points,face_rows):
 # Cross-surface exact predicate uses triangle coordinates/AABBs; shared vertex IDs are not used.
 # Extract only this face's original full-mesh vertex rows, then map face IDs into the sorted subset.
 p=np.asarray(points,dtype='<f4'); f=np.asarray(face_rows,dtype=np.int64)
 used=np.unique(f.reshape(-1))
 if used.size != 3: raise RuntimeError(f'expected one nondegenerate triangle, got {used.size} referenced vertices')
 local_faces=np.searchsorted(used,f)
 rec,deg=h.exact_rows(np.ascontiguousarray(p[used]),local_faces,ci)
 if deg: raise RuntimeError(f'degenerate exact pair witness faces {deg[:4]}')
 return rec

def exact_pair_hit(pa,fa,pb,fb):
 ra=pair_records(pa,np.asarray(fa).reshape(1,3)); rb=pair_records(pb,np.asarray(fb).reshape(1,3))
 return int(ci._audit_pair(ra,rb,same_surface=False)['count'])

def sat_axis_from_clear(a,b):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
 ea=np.roll(a,-1,axis=0)-a;eb=np.roll(b,-1,axis=0)-b
 ns=[np.cross(ea[0],ea[1]),np.cross(eb[0],eb[1])]
 axes=np.asarray(ns+[np.cross(x,y) for x in ea for y in eb],dtype=np.float64)
 lens=np.linalg.norm(axes,axis=1); axes=axes[lens>1.0e-15];lens=lens[lens>1.0e-15]
 if len(axes)==0:raise RuntimeError('clear witness triangles have no valid SAT axes')
 axes=axes/lens[:,None]; axes=np.concatenate((axes,-axes),axis=0)
 gaps=(b@axes.T).min(0)-(a@axes.T).max(0);k=int(np.argmax(gaps));gap=float(gaps[k])
 if not math.isfinite(gap) or gap<=0:raise RuntimeError(f'original clear witness has no positive SAT gap ({gap})')
 return axes[k],gap

def _separation_row(axis, source_jacobian, other_jacobian=None):
 # gap = min(axis dot other) - max(axis dot source). Moving source along +axis reduces gap.
 a=np.asarray(axis,dtype=np.float64)
 c=-(a @ np.asarray(source_jacobian,dtype=np.float64))
 if other_jacobian is not None:
  c=c+(a @ np.asarray(other_jacobian,dtype=np.float64))
 return c

def _separation_rhs_mm(floor_m, current_gap_m, reserve_mm=0.0):
 # Require updated gap >= floor_m and add a positive rounding reserve.
 return 1000.0*(float(floor_m)-float(current_gap_m))+float(reserve_mm)

# Geometric sign checks: source +axis closes an external gap; corrections in the
# opposite direction open it. The same signed row applies to source-vs-moving-other.
_axis=np.array([1.0,0.0,0.0]); _floor=0.01; _gap=-0.20
_external=_separation_row(_axis,np.array([[1.0],[0.0],[0.0]]))
assert np.allclose(_external,[-1.0])
_rhs=_separation_rhs_mm(_floor,_gap,0.002)
assert _rhs>0 and abs(_rhs-210.002)<1e-12
assert float(_external @ np.array([+210.0])) < 0.0
assert float(_external @ np.array([-210.0])) >= _rhs-0.002-1e-10
_self_source=_separation_row(_axis,np.array([[1.0],[0.0],[0.0]]),np.array([[0.0],[0.0],[0.0]]))
_self_other=_separation_row(_axis,np.array([[0.0],[0.0],[0.0]]),np.array([[1.0],[0.0],[0.0]]))
assert np.allclose(_self_source,[-1.0]) and np.allclose(_self_other,[1.0])
assert float(_self_source @ np.array([-210.0]))>0 and float(_self_other @ np.array([+210.0]))>0
assert float(_self_source @ np.array([+210.0]))<0 and float(_self_other @ np.array([-210.0]))<0
(OUT/'direction-sanity.json').write_text(json.dumps({'schema':'numi.human.epl-separation-direction-sanity.v1','gap_definition':'min(axis dot target) minus max(axis dot source)','axis':[1,0,0],'current_gap_m':_gap,'required_floor_m':_floor,'source_plus_axis_reduces_gap':True,'source_minus_axis_correction_opens_gap':True,'self_source_and_other_corrections_open_gap':True,'rhs_positive_mm_including_reserve':_rhs,'reserve_mm':0.002,'assertions':'passed'},indent=2,sort_keys=True)+'\n')

# Build 68-vertex local support from all exact added source faces; one-ring neighbors are the only free variables.
active_specs=[]
for key,r in rem_rows.items():
 rr=[]
 for x in r['delta_pairs_full_target_rows']:
  if int(x['candidate_exact_intersection_count'])==1 and int(x['original_tiss_exact_intersection_count'])==0:
   sf=int(x['candidate_source_face_row']);origin=int(x['source_face_origin']);sem=int(x['target_semantic']);sid=int(x['target_stable_id']);tf=int(x['target_face_full_mesh_row'])
   assert 0<=sf<len(faces) and int(origins[sf])==origin and 0<=origin<len(orig_faces)
   rr.append((sf,origin,sem,sid,tf))
 active_specs.append((key,rr));added_occ.extend([{'cohort':key[0],'arm':key[1],'step':key[2],'source_face_row':a,'source_face_origin':b,'target_semantic':c,'target_stable_id':d,'target_face_row':e} for a,b,c,d,e in rr])
assert len(active_specs)==24
unique_pair_occ={(x['cohort'],x['arm'],x['step'],x['source_face_row'],x['target_semantic'],x['target_stable_id'],x['target_face_row']) for x in added_occ}
assert len(unique_pair_occ)==len(added_occ), 'remap duplicate pair rows'
seed_face_rows=sorted({x['source_face_row'] for x in added_occ}); seed_vertices=np.unique(faces[seed_face_rows].reshape(-1)); seed_set=set(map(int,seed_vertices)); support=set(seed_set)
for tri in faces:
 ids=set(map(int,tri))
 if ids.intersection(seed_set):support.update(ids)
movable=np.asarray(sorted(support),dtype=np.int64); protected_route=np.flatnonzero(np.max(weights,axis=1)>=np.float32(1.0-1e-7)).astype(np.int64)
assert len(seed_vertices)==22 and len(movable)==68
assert not np.intersect1d(movable,protected_route).size
var=np.full(len(base),-1,dtype=np.int32);var[movable]=np.arange(len(movable),dtype=np.int32);nvar=3*len(movable)
# Gather exactly one positive-axis witness per added pair occurrence, from original-TISS clear geometry.
Arows=[];brows=[]; constraint_receipts=[];posecontexts=[];unconstrained=[];start=time.monotonic()
for ix,(cohort,arm,step,root) in enumerate(pose_specs,1):
 k=(cohort,arm,step);mr=rem_rows[k];assert mr['exact_witnesses_recomputed'] is True
 ap=ROOTP=Path(root); rp=ap/f'step-{step}.receipt.json';pack=ap/f'step-{step}.mrvpack';pins[str(rp)]=sha(rp);pins[str(pack)]=sha(pack)
 rec=json.loads(rp.read_text());assert int(rec['accepted_step'])==step and rec.get('pack_file_sha256')==pins[str(pack)]
 poses={int(q['body_index']):(np.asarray(q['position_m'],np.float32),np.asarray(q['quaternion_xyzw'],np.float32)) for q in rec['accepted_registered_body_poses']};assert len(poses)>=86
 packed,surfaces,_=ca._pack_surfaces(pack,keys)
 external_keys=(set(surfaces)-{(51005,143)});assert len(surfaces)==860 and external_keys==TARGET_KEYS|{SKIN_KEY} and SKIN_KEY in surfaces and (51999,1) not in surfaces
 candidate_world=h.forward(base,local,weights,row,tissue,poses); original_world=h.forward(orig_positions,orig_local,orig_weights,row,tissue,poses)
 jac=jowner.pose_jacobians(base,local,weights,row,tissue,poses)
 # Confirm all pair witnesses exactly match report at this accepted pose before building any row.
 pair_rows=[]
 for x in mr['delta_pairs_full_target_rows']:
  if int(x['candidate_exact_intersection_count'])==1 and int(x['original_tiss_exact_intersection_count'])==0:
   sf=int(x['candidate_source_face_row']);origin=int(x['source_face_origin']);keyt=(int(x['target_semantic']),int(x['target_stable_id']));tf=int(x['target_face_full_mesh_row'])
   target=surfaces[keyt];target_faces=np.asarray(target['faces'],dtype=np.int64);assert 0<=tf<len(target_faces)
   cand_hit=exact_pair_hit(candidate_world,faces[sf],packed,target_faces[tf]);orig_hit=exact_pair_hit(original_world,orig_faces[origin],packed,target_faces[tf])
   assert cand_hit==1 and orig_hit==0, f'exact pair report witness mismatch {k} sf{sf} origin{origin} target{keyt}:{tf}'
   axis,gap=sat_axis_from_clear(original_world[orig_faces[origin]],packed[target_faces[tf]])
   # Small explicitly declared positive floor; source/target meshes are F32 and the reserve is separate.
   floor_m=min(gap*0.1,0.05e-6)
   for vi in faces[sf]:
    vi=int(vi); coeff=np.zeros(nvar,dtype=np.float64);v=int(var[vi])
    if v>=0: coeff[3*v:3*v+3]=-(axis@jac[vi])
    for tj in target_faces[tf]:
     delta_project=float(np.dot(axis,packed[int(tj)].astype(np.float64)-candidate_world[vi].astype(np.float64)))
     rhs_mm=_separation_rhs_mm(floor_m,delta_project)
     reserve_mm=0.00015*float(np.abs(coeff).sum())
     rhs_mm+=reserve_mm
     if not np.any(coeff):
      if rhs_mm>1e-9: unconstrained.append({'pose':list(k),'source_face_row':sf,'source_vertex':vi,'target':list(keyt),'target_face_row':tf,'target_vertex':int(tj),'rhs_mm':rhs_mm})
      continue
     Arows.append(coeff);brows.append(rhs_mm)
     constraint_receipts.append({'cohort':cohort,'arm':arm,'step':step,'candidate_source_face_row':sf,'source_face_origin':origin,'target_semantic':keyt[0],'target_stable_id':keyt[1],'target_face_full_mesh_row':tf,'source_vertex':vi,'target_vertex':int(tj),'original_positive_sat_gap_m':gap,'required_gap_floor_m':floor_m,'axis_world':axis.tolist(),'rounding_reserve_mm':reserve_mm})
   pair_rows.append({'source_face_row':sf,'source_face_origin':origin,'target_semantic':keyt[0],'target_stable_id':keyt[1],'target_face_row':tf,'candidate_exact_hit':cand_hit,'original_exact_hit':orig_hit,'positive_axis_gap_m':gap,'floor_m':floor_m})
 posecontexts.append({'key':k,'receipt':str(rp),'pack':str(pack),'candidate_world_sha256':hashlib.sha256(np.ascontiguousarray(candidate_world,dtype='<f4').tobytes()).hexdigest(),'original_world_sha256':hashlib.sha256(np.ascontiguousarray(original_world,dtype='<f4').tobytes()).hexdigest(),'added_pair_count':len(pair_rows),'added_pair_rows':pair_rows})
 print(json.dumps({'pose_index':ix,'cohort':cohort,'arm':arm,'step':step,'added_pairs':len(pair_rows),'elapsed_s':time.monotonic()-start}),flush=True)
 del packed,surfaces,candidate_world,original_world,jac
assert len(Arows)>0 and not unconstrained, f'fixed/out-of-support target rows incompatible: {unconstrained[:3]}'
M=np.ascontiguousarray(np.vstack(Arows),dtype=np.float64); rhs=np.ascontiguousarray(brows,dtype=np.float64)
assert M.shape==(len(rhs),nvar) and np.isfinite(M).all() and np.isfinite(rhs).all()
# Minimum-L1 displacement LP with explicit per-vertex L2<=1mm conservative box.
# x is in mm, t bounds |x|, and Mx>=rhs retains the verified separating direction.
cap_mm=1.0/math.sqrt(3.0)
deadline=time.monotonic()+120.0
I=sparse.eye(nvar,format='csr',dtype=np.float64)
Ms=sparse.csr_matrix(M)
Z=sparse.csr_matrix((Ms.shape[0],nvar),dtype=np.float64)
A_ub=sparse.vstack((sparse.hstack((-Ms,Z)),sparse.hstack((I,-I)),sparse.hstack((-I,-I))),format='csr')
b_ub=np.concatenate((-rhs,np.zeros(2*nvar,dtype=np.float64)))
c=np.concatenate((np.zeros(nvar,dtype=np.float64),np.ones(nvar,dtype=np.float64)))
bounds=[(-cap_mm,cap_mm)]*nvar+[(0.0,None)]*nvar
lp=linprog(c,A_ub=A_ub,b_ub=b_ub,bounds=bounds,method='highs',options={'time_limit':max(1.0,deadline-time.monotonic())})
report={'schema':'numi.human.epl-active-external-clearance.v1','status':'infeasible_or_solver_failure','prior_attempt_003':'preserved; its SAT constraint sign was reversed and its LP solution is invalid, not reused','prior_attempt_004':'preserved; correct-sign L1 LP reached candidate serialization but failed because np.column_stack was called on a tuple; no candidate or report was emitted','scope':'Unadmitted inferred source-space correction for stable143/FJ1495. Rows target exact external pair additions relative to raw frozen TISS only; initial target cohort covers 858 inventory surfaces plus skin51007:1 (859 anatomical external targets). Bed51999:1 is separately excluded as a viewer primitive. No binding/topology/attachment-map edits, composition or native execution.','identity':{'stable_id':143,'member_id':'FJ1495','candidate_base_path':str(CAND),'candidate_base_sha256':sha(CAND),'tiss_path':str(TISS),'tiss_sha256':sha(TISS),'manifest_sha256':sha(MAN),'target_inventory_count':len(keys),'external_inventory_keys':len(TARGET_KEYS),'prior_audit_status_label':audit['status'],'prior_audit_actual_external_count_per_pose':858,'prior_audit_inventory_total_per_pose':859,'skin_extra_target':[51007,1],'skin_target_separately_rechecked_all24':True,'external_target_count':len(TARGET_KEYS|{SKIN_KEY}),'excluded_own_source':[51005,143],'bed_viewer_primitive_excluded':[51999,1],'count_reconciliation':'858 external inventory targets plus skin51007:1 equals 859 anatomical external targets; bed51999:1 is a viewer primitive and excluded'},'method':{'exact_predicate_owner':str(Path(ci.__file__)),'exact_predicate_sha256':sha(ci.__file__),'target_incremental_audit_owner':str(Path(ca.__file__)),'target_incremental_audit_sha256':sha(ca.__file__),'forward_owner':str(H),'forward_owner_sha256':sha(H),'pose_jacobian_owner':str(J),'pose_jacobian_owner_sha256':sha(J),'source_reference':'raw TISS row143 exact-clear triangle versus target full face row; positive SAT axis recomputed and exact pair hit rechecked before every linear row','gap_definition':'min(axis dot target)-max(axis dot source)','constraint_sign':'-axis@J_source (+axis@J_other for a moving other side) >= 1000*(floor_m-current_gap_m) + positive_float32_reserve_mm','floor_rule':'min(0.1 * original positive SAT gap, 0.05 micrometres)','float32_projection_reserve_mm':'0.00015 * L1 norm of each constraint coefficient','free_support':'all vertices in exact added-pair faces plus one face-neighbor ring; all other source positions fixed','movable_vertex_ids':movable.tolist(),'active_seed_face_rows':seed_face_rows,'active_seed_vertex_ids':seed_vertices.tolist(),'movable_vertex_count':len(movable),'pure_route_anchor_ids':protected_route.tolist(),'pure_route_anchors_in_patch':int(np.intersect1d(movable,protected_route).size),'binding_indices_and_weights_fixed':True,'faces_and_face_origins_fixed':True,'component_box_cap_mm':cap_mm,'l2_cap_mm':1.0,'objective':'minimum L1 displacement sum(abs(x)) in millimetres, subject to separating halfspaces and component box','optimizer_budget_seconds':120,'pose_count':24,'candidate_added_pair_occurrences':len(added_occ),'unique_pose_pair_rows':len(unique_pair_occ),'exact_witness_table_validation':{'expected_reported_added_pair_occurrences':len(added_occ),'recomputed_pair_witnesses':sum(len(p['added_pair_rows']) for p in posecontexts),'each_candidate_hit_matches_1_and_original_hit_matches_0':True,'special_skin_target_included_in_final_859_gate':True},'linear_constraints':len(rhs),'direction_sanity':'direction-sanity.json passed before pair-row construction'},'solver':{'method':'scipy.optimize.linprog(method=highs)','objective':'minimum L1 displacement with split x/t variables','success':bool(lp.success),'status':int(lp.status),'message':str(lp.message),'nit':int(getattr(lp,'nit',0) or 0),'objective_value':float(lp.fun) if lp.success else None,'wall_s':time.monotonic()-start,'optimizer_started':True,'matrix_max_abs_coefficient':float(np.max(np.abs(M))),'rhs_min_mm':float(np.min(rhs)),'rhs_max_mm':float(np.max(rhs)),'returned_x_linf_mm':float(np.max(np.abs(lp.x[:nvar]))) if lp.success and np.isfinite(lp.x).all() else None},'poses':posecontexts,'constraint_receipts':constraint_receipts,'unconstrained_incompatible_rows':unconstrained,'inputs_before':pins,'inputs_after_precheck':{p:sha(p) for p,d in pins.items()},'inputs_unchanged_precheck':all(sha(p)==d for p,d in pins.items())}
if lp.success and np.isfinite(lp.x).all():
 res=np.asarray(lp.x[:nvar],dtype=np.float64)
 report['solver']['minimum_linear_slack_mm']=float(np.min(M@res-rhs)) if np.max(np.abs(M)) < 1.0e6 else None
 if np.isfinite(res).all():
  q=base.copy();requested=np.asarray(res,dtype=np.float64).reshape(len(movable),3);q[movable]=(base[movable].astype(np.float64)+requested/1000.0).astype('<f4')
  changed=np.flatnonzero(np.any(q.view('<u4')!=base.view('<u4'),axis=1));changed_faces=np.flatnonzero(np.any(np.isin(faces,changed),axis=1))
  assert np.all(np.isin(changed,movable)) and len(changed_faces)>0 and np.array_equal(local,z['binding_indices']) and np.array_equal(weights,z['weights']) and np.array_equal(faces,z['faces']) and np.array_equal(origins,z['face_origins'])
  maxl2=float(np.linalg.norm((q.astype(np.float64)-base.astype(np.float64))*1000,axis=1).max(initial=0)); maxcomp=float(np.abs((q.astype(np.float64)-base.astype(np.float64))*1000).max(initial=0))
  npz=OUT/'stable-143-unadmitted-active-external-candidate.npz';nv=normals(q.astype(np.float64),faces);np.savez(npz,vertices6=np.column_stack((q,nv)).astype('<f4'),faces=faces.astype('<u4'),face_origins=origins,binding_indices=local,weights=weights)
  report['status']='proposal_emitted_for_exact_checks';report['proposal']={'path':str(npz),'sha256':sha(npz),'changed_vertex_ids':changed.tolist(),'changed_face_rows':changed_faces.tolist(),'max_increment_vertex_l2_mm':maxl2,'max_increment_component_mm':maxcomp,'binding_indices_unchanged':True,'weights_unchanged':True,'faces_unchanged':True,'face_origins_unchanged':True}
(OUT/'constraint-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'output':str(OUT/'constraint-report.json'),'status':report['status'],'lp_success':bool(lp.success),'lp_status':int(lp.status),'lp_message':str(lp.message),'constraints':len(rhs),'poses':len(posecontexts),'active_occurrences':len(added_occ),'active_unique':len(unique_pair_occ),'elapsed_s':time.monotonic()-start,'proposal':report.get('proposal',{})},sort_keys=True),flush=True)
# Save full post checks in a separate result only if a finite candidate was emitted.
if 'proposal' in report:
 pp=np.load(report['proposal']['path'],allow_pickle=False);q=np.ascontiguousarray(pp['vertices6'][:,:3],dtype='<f4')
 source_rec,source_deg=h.exact_rows(q,faces,ci);src=ci._audit_pair(source_rec,source_rec,same_surface=True);source_top=analyze_topology(q.astype(float).tolist(),faces.tolist())
 assert not source_deg
 check_rows=[]; target_added_total=Counter(); exact_pair_counts=Counter(); cache=set(); candidate_check_pins={str(npz):sha(npz)}
 base_bykey={}
 for ix,(cohort,arm,step,root) in enumerate(pose_specs,1):
  k=(cohort,arm,step);rp=Path(root)/f'step-{step}.receipt.json';pack=Path(root)/f'step-{step}.mrvpack';receipt=json.loads(rp.read_text());poses={int(x['body_index']):(np.asarray(x['position_m'],np.float32),np.asarray(x['quaternion_xyzw'],np.float32)) for x in receipt['accepted_registered_body_poses']}
  packed,surfaces,_=ca._pack_surfaces(pack,keys);target_keys=sorted(set(surfaces)-{(51005,143)});assert len(target_keys)==859 and SKIN_KEY in target_keys and (51999,1) not in target_keys
  tfaces={tuple(map(int,key)):np.ascontiguousarray(surfaces[key]['faces'],dtype=np.int64) for key in target_keys};assert len(tfaces)==859
  baseline_audits={f'{sem}:{sid}':{'count':0,'triangle_pairs':[],'degenerate_face_rows':[]} for sem,sid in target_keys}
  ar=audit_rows[k]
  for target in ar['target_pair_details']:
   for item in target['pairs'].get('local_self_candidate',[]):
    sf,origin,sem,sid,tf=map(int,item);keystr=f'{sem}:{sid}';assert keystr in baseline_audits and int(origins[sf])==origin
    baseline_audits[keystr]['triangle_pairs'].append([sf,tf]);baseline_audits[keystr]['count']+=1
  expected_count=int(ar['pair_occurrences_by_variant']['local_self_candidate']['pair_occurrences'])
  assert sum(v['count'] for v in baseline_audits.values())==expected_count
  # Original TISS baseline lineage multiset for final exact addition comparison.
  orig_counter=Counter()
  for target in ar['target_pair_details']:
   for item in target['pairs'].get('original_tiss',[]):
    sf,origin,sem,sid,tf=map(int,item);orig_counter[(origin,sem,sid,tf)]+=1
  base_world=h.forward(base,local,weights,row,tissue,poses);next_world=h.forward(q,local,weights,row,tissue,poses)
  selfr,deg=h.exact_rows(next_world,faces,ci);selfa=ci._audit_pair(selfr,selfr,same_surface=True)
  top=analyze_topology(next_world.astype(float).tolist(),faces.tolist())
  inc=ca.audit_incremental_skin_target_intersections(baseline_world_positions=base_world,candidate_world_positions=next_world,baseline_faces=faces,candidate_faces=faces,baseline_target_audits=baseline_audits,target_faces_by_key=tfaces,target_positions=packed,expected_baseline_world_f32_sha256=hashlib.sha256(np.ascontiguousarray(base_world,dtype='<f4').tobytes()).hexdigest(),expected_face_index_sha256=ca._face_index_sha256(faces),expected_target_geometry_f32_sha256=ca._target_geometry_f32_sha256(packed,tfaces),expected_baseline_target_audits_sha256=ca._baseline_target_pair_table_sha256(baseline_audits),_validated_target_geometry_cache=cache)
  cand_counter=Counter()
  for keystr,arow in inc['target_audits'].items():
   sem,sid=map(int,keystr.split(':'))
   for sf,tf in arow['triangle_pairs']:cand_counter[(int(origins[int(sf)]),sem,sid,int(tf))]+=1
  added=cand_counter-orig_counter;removed=orig_counter-cand_counter
  detail=[{'source_face_origin':int(k[0]),'target_semantic':int(k[1]),'target_stable_id':int(k[2]),'target_face_row':int(k[3]),'occurrences':int(v)} for k,v in sorted(added.items()) for _ in range(v)]
  target_added_total.update(added); selfpairs=[list(map(int,x)) for x in selfa.get('triangle_pairs',[])]
  check={'cohort':cohort,'arm':arm,'step':step,'pack_sha256':sha(pack),'receipt_sha256':sha(rp),'exact_self_pair_count':int(selfa['count']),'self_pair_rows':selfpairs,'degenerate_face_count':len(deg),'closed_oriented_manifold_candidate':bool(top.get('closed_oriented_manifold_candidate',False)),'candidate_full_target_surface_count':int(inc['target_surface_count']),'target_count_expected':859,'changed_face_rows':inc['changed_skin_face_rows'],'exact_added_vs_raw_tiss_occurrences':sum(v for v in added.values()),'added_vs_raw_tiss':detail,'removed_vs_raw_tiss_occurrences':sum(v for v in removed.values()),'reused_baseline_pairs':inc['reused_baseline_exact_pair_count'],'fresh_changed_face_pairs':inc['fresh_changed_face_exact_pair_count'],'fresh_changed_face_aabb_candidates':inc['fresh_changed_face_aabb_candidate_pairs'],'skin_target_pair_count':inc['target_audits']['51007:1']['count']}
  check_rows.append(check);print(json.dumps({'postcheck':ix,'cohort':cohort,'arm':arm,'step':step,'self':selfa['count'],'added':check['exact_added_vs_raw_tiss_occurrences'],'targets':inc['target_surface_count'],'skin':check['skin_target_pair_count']},sort_keys=True),flush=True)
  assert set(inc['changed_skin_face_rows']).issubset(set(np.flatnonzero(np.any(np.isin(faces,movable),axis=1)).tolist()))
  del packed,surfaces
 postpins={p:sha(p) for p in pins};unchanged=postpins==pins
 final={'schema':'numi.human.epl-active-external-clearance-postcheck.v1','status':'complete_24_pose_exact_self_and_859_target_comparison' if len(check_rows)==24 and unchanged else 'incomplete_or_inputs_changed','candidate_path':str(npz),'candidate_sha256':sha(npz),'source_exact_self_pair_count':int(src['count']),'source_degenerate_face_count':len(source_deg),'source_closed_oriented_manifold_candidate':bool(source_top.get('closed_oriented_manifold_candidate',False)),'all24_exact_self_pairs_clear':all(r['exact_self_pair_count']==0 and r['degenerate_face_count']==0 and r['closed_oriented_manifold_candidate'] for r in check_rows),'all24_full859_target_pairs_no_additions_vs_raw_tiss':all(r['exact_added_vs_raw_tiss_occurrences']==0 for r in check_rows),'all24_skin_no_pairs':all(r['skin_target_pair_count']==0 for r in check_rows),'target_surface_count_per_pose':859,'poses_expected':24,'poses_actual':len(check_rows),'poses':check_rows,'new_pair_identity_union':[{'source_face_origin':int(k[0]),'target_semantic':int(k[1]),'target_stable_id':int(k[2]),'target_face_row':int(k[3]),'occurrences':int(v)} for k,v in sorted(target_added_total.items()) for _ in range(v)],'inputs_after':postpins,'inputs_unchanged':unchanged,'limitations':['Offline row-map forward prediction, not native GPU candidate export.','No composition or whole-body admission; target-pair comparison is relative to exact raw-TISS pair lineage, which itself contains existing crossings.','Bed support/contact remains a distinct gate; 51999:1 bed viewer primitive was not treated as an anatomical surface.']}
 (OUT/'exact-postcheck.json').write_text(json.dumps(final,indent=2,sort_keys=True)+'\n');print(json.dumps({'postcheck_path':str(OUT/'exact-postcheck.json'),'postcheck_sha256':sha(OUT/'exact-postcheck.json'),'status':final['status'],'source_self':src['count'],'all24_self_clear':final['all24_exact_self_pairs_clear'],'no_added_target_pairs':final['all24_full859_target_pairs_no_additions_vs_raw_tiss'],'all24_rows':len(check_rows),'inputs_unchanged':unchanged},sort_keys=True),flush=True)
