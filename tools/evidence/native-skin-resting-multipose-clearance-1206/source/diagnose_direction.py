#!/usr/bin/env python3
import gc,hashlib,importlib.util,json,mmap,struct,sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
E=Path('/Users/n/numi-human-resting-evidence-20261005'); R=Path('/Users/n/numi-human-retained-delivery-20261009')
ROOT=Path('/Users/n/numi-human-common-skin-multipose-001'); CURRENT=Path('/Users/n/numi-human-free-apex-two-family-1178/src/numilab_human/common_atlas_skin_clearance.py'); CURRENT_CARDIAC=CURRENT.parent/'cardiac_cavity_intersections.py'; RESP=Path('/Users/n/numi-human-resting-resp-source-20261006')
OUT=R/'skin-resting-multipose-clearance-1206/direction-diagnosis-001'
SKIN=E/'native-skin-epl143-clearance-1187/attempt-006/bodyparts3d-myosim-skinned-shell-epl143-candidate.nhskin'
NHA=E/'native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-thorax.nhanatomy'
VALID=E/'cardiac-wall-native-self-audit-001/accepted_mrvpack_surface_audit.py'; FORWARD=E/'native-common-skin-multipose-forward-model-915.py'
INV=E/'native-complete-skin-containment-audit-890/pair-summary-v3.csv'
RUNS=[(R/'native-lung1178-thumb1187-smoke-1191/native-run',0),(R/'native-lung1178-thumb1187-smoke-1191/native-run',10000),(R/'native-flat-reference-40s-1201/native-run',20000)]
WIT=R/'native-flat-reference-40s-1201/skin-audit-1201-terminal/step-20000.crossing-witnesses.jsonl'
PREF=R/'skin-resting-multipose-clearance-1206/zero_delta_preflight.json'; SOL=R/'skin-resting-multipose-clearance-1206/fit-attempt-001/solve-result.json'
sys.path.insert(0,str(ROOT/'src'))
from numilab_human import common_atlas_skin_clearance as c
cardiac_spec=importlib.util.spec_from_file_location('numilab_human.cardiac_cavity_intersections',CURRENT_CARDIAC)
current_cardiac=importlib.util.module_from_spec(cardiac_spec); sys.modules[cardiac_spec.name]=current_cardiac; cardiac_spec.loader.exec_module(current_cardiac)
current_spec=importlib.util.spec_from_file_location('numilab_human.current_clearance',CURRENT)
current_c=importlib.util.module_from_spec(current_spec); sys.modules[current_spec.name]=current_c; current_spec.loader.exec_module(current_c)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
 return h.hexdigest()
def pin(p):
 p=Path(p).resolve(); return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def readj(p): return json.loads(Path(p).read_text())
def loadskin(path):
 raw=Path(path).read_bytes(); magic,abi,nb,nv,ni,fp,arch=struct.unpack_from('<8s5I32s',raw); vo=60+36*nb; io=vo+56*nv; wo=io+4*ni
 if magic!=b'NHSKIN1\0' or abi!=5 or len(raw)!=wo+4*nv*nb: raise RuntimeError('bad NHSKIN')
 vr=np.frombuffer(raw,'<f4',14*nv,vo).reshape(nv,14); faces=np.frombuffer(raw,'<u4',ni,io).reshape(-1,3).astype(np.int64)
 bind=np.frombuffer(raw,'<f4',9*nb,60).reshape(nb,9).copy()
 weights=np.frombuffer(raw,'<f4',nv*nb,wo).reshape(nv,nb).astype(float)
 return raw,{'nb':nb,'nv':nv,'ni':ni,'fp':fp,'bind':bind,'bind_u':np.frombuffer(raw,'<u4',9*nb,60).reshape(nb,9).copy(),'pos':vr[:,:3].astype(float),'faces':faces,'weights':weights}
raw,skin=loadskin(SKIN); ref=np.unique(skin['faces']); faces=skin['faces']; compact=np.searchsorted(ref,faces)
hs=importlib.util.spec_from_file_location('mrv',VALID); helper=importlib.util.module_from_spec(hs); hs.loader.exec_module(helper)
fs=importlib.util.spec_from_file_location('fwd915',FORWARD); fm=importlib.util.module_from_spec(fs); fs.loader.exec_module(fm)
allkeys,nonocular,_,_=c._load_target_inventory(INV); nonocular=set(nonocular)
initial=None; captures=[]; state_receipts=[]; pose_pins=[]
for run,step in RUNS:
 pack=run/'accepted-geometry'/f'step-{step}.mrvpack'; rec=pack.with_suffix('.receipt.json'); rd=readj(rec)
 if rd.get('accepted_step')!=step or rd.get('physical_endpoint')!='accepted' or rd.get('surface_audit_endpoint')!='passed': raise RuntimeError(f'invalid accepted state {step}')
 if initial is None: initial=rd
 elif json.dumps(rd['initial_anatomical_registration']['body_poses'],sort_keys=True)!=json.dumps(initial['initial_anatomical_registration']['body_poses'],sort_keys=True): raise RuntimeError('initial registration differs')
 if (rd['skin_source_mapping']['vertex_map']['sha256'],rd['skin_source_mapping']['anatomy_parameters']['sha256'])!=(initial['skin_source_mapping']['vertex_map']['sha256'],initial['skin_source_mapping']['anatomy_parameters']['sha256']): raise RuntimeError('runtime map/resp params differ')
 with pack.open('rb') as f:
  with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
   vo=next(x[2] for x in helper._pack_sections(mm) if x[0]==2); helper.validate_accepted_receipt(pack,rec,step,mm,vo,{})
 pos,surfs,_=c._pack_surfaces(pack,allkeys); sf=surfs[(51007,1)]['faces']; base=int(sf.min())
 if len(sf)!=len(faces) or not np.array_equal(sf-base,faces): raise RuntimeError(f'skin topology mismatch at {step}')
 captures.append(pos[base:base+skin['nv']].astype('<f4').astype(float)[ref].copy())
 state_receipts.append({'step':step,'body_poses':rd['accepted_registered_body_poses'],'respiratory_motion':rd['accepted_respiratory_motion']})
 pose_pins += [pin(pack),pin(rec)]; del pos,surfs; gc.collect()
captures=np.asarray(captures,dtype='<f4').astype(float)
forward,map_report,selectors=fm.build_forward(skin=skin,source_positions=skin['pos'],referenced_ids=ref,captured_by_pose=captures,state_receipts=state_receipts,initial_body_poses=initial['initial_anatomical_registration']['body_poses'],map_receipt=initial,anatomy_parameters_path=initial['skin_source_mapping']['anatomy_parameters']['path'],respiration_source=RESP,clearance_module=c)
base=forward(skin['pos'].astype('<f4')); pred=np.asarray(base['world_positions_by_pose'],dtype='<f4')
if base.get('diagnostics',{}).get('admissible') is not True or not np.array_equal(pred,captures.astype('<f4')): raise RuntimeError('zero-delta forward mismatch')
maps=np.asarray(base['jacobians_by_pose'],dtype=np.float64); world=captures
# Rebuild the active list exactly as the failed fit: sorted source face rows present in nonocular crossings.
active=set(); pair_count=0
with WIT.open() as f:
 for line in f:
  x=json.loads(line)
  if x.get('role')=='surface_crossing' and tuple(map(int,x['target_surface'])) in nonocular:
   active.add(int(x['skin_source_face_row'])); pair_count+=1
active_faces=sorted(active); active_compact=compact[np.asarray(active_faces,dtype=np.int64)]
if pair_count!=3112: raise RuntimeError(f'expected 3112 nonocular pairs, got {pair_count}')
pose_normals=np.stack([c._area_weighted_vertex_normals(world[p],compact) for p in range(3)])
smooth=np.stack([c._smooth_vertex_directions(pose_normals[p],compact) for p in range(3)])
source_dirs,_,_,direction_report=c._shared_source_directions_from_pose_normals(maps,smooth)
tri=world[:,active_compact]; cross=np.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0]); lengths=np.linalg.norm(cross,axis=2)
if np.any(lengths<=0) or not np.isfinite(lengths).all(): raise RuntimeError('degenerate active face')
normals=cross/lengths[:,:,None]
# Exact failing index and corner from preserved owner exception.
pose_fail=1; ai_fail=446; corner_fail=2
face_fail=active_faces[ai_fail]; compact_v=int(active_compact[ai_fail,corner_fail]); source_v=int(ref[compact_v])
incidence=np.flatnonzero(np.any(active_compact==compact_v,axis=1))
# Run the current Human owner's direction selector on this vertex's complete incident constraints only.
local_faces=active_compact[incidence]; local_normals=normals[:,incidence]
conditioned_dirs, conditioning_report=current_c._condition_shared_source_directions(maps,source_dirs.copy(),local_faces,local_normals)
zero_required=np.zeros((3,len(local_faces)),dtype=np.float64)
_, conditioned_seed_report=current_c._shared_source_seed_demands(maps,conditioned_dirs,local_faces,local_normals,zero_required)
constraints=[]
for p in range(3):
 for ai in incidence:
  face=int(active_faces[int(ai)]); corners=np.flatnonzero(active_compact[int(ai)]==compact_v); J=maps[p,compact_v]; n=normals[p,int(ai)]
  v=J@source_dirs[compact_v]; proj=float(np.dot(v/np.linalg.norm(v),n))
  for corner in corners:
   constraints.append({'pose_index':p,'step':RUNS[p][1],'active_face_index':int(ai),'source_face_row':face,'face_corner':int(corner),'source_vertex_id':source_v,'projection_current':proj,'normal_world':n.tolist(),'jacobian':J.tolist()})
current=source_dirs[compact_v]/np.linalg.norm(source_dirs[compact_v])
def project(d):
 out=[]
 for q in constraints:
  J=np.asarray(q['jacobian']); n=np.asarray(q['normal_world']); v=J@d; out.append(float(np.dot(v/np.linalg.norm(v),n)))
 return np.asarray(out)
def soc_margin(d):
 vals=[]
 for q in constraints:
  J=np.asarray(q['jacobian']); n=np.asarray(q['normal_world']); v=J@d; vals.append(float(np.dot(v,n)-.5*np.linalg.norm(v)))
 return np.asarray(vals)
# Deterministic multi-start max-min angular feasibility search over this vertex's incident face/pose constraints.
cons=[]
for q in constraints: cons.append((np.asarray(q['jacobian']),np.asarray(q['normal_world'])))
def objective(z): return -float(z[3])
def con_feas(z): return soc_margin(z[:3])-z[3]
def con_ball(z): return 1.-float(np.dot(z[:3],z[:3]))
starts=[current]
for p in range(3):
 d=smooth[p,compact_v]; starts.append(d/np.linalg.norm(d))
 for ai in incidence[:min(len(incidence),20)]:
  n=normals[p,int(ai)]; J=maps[p,compact_v]; d=np.linalg.solve(J,n); starts.append(d/np.linalg.norm(d))
solutions=[]
for si,start in enumerate(starts):
 z0=np.r_[start,float(soc_margin(start).min()-1e-8)]
 res=minimize(objective,z0,method='SLSQP',constraints=[{'type':'ineq','fun':con_feas},{'type':'ineq','fun':con_ball}],bounds=[(-1,1),(-1,1),(-1,1),(None,None)],options={'maxiter':250,'ftol':1e-12})
 d=res.x[:3]; dn=np.linalg.norm(d)
 if dn>1e-12:
  d=d/dn; pp=project(d); solutions.append({'start_index':si,'success':bool(res.success),'message':str(res.message),'unit_direction':d.tolist(),'minimum_projection':float(pp.min()),'maximum_projection':float(pp.max()),'minimum_soc_margin':float(soc_margin(d).min()),'projections':pp.tolist()})
best=max(solutions,key=lambda x:x['minimum_projection']) if solutions else None
report={'schema':'numi.human.skin-shared-direction-failure-diagnosis.v1','status':'read_only_direction_diagnostic','failed_fit':readj(SOL),'failed_face':{'fit_pose_index':pose_fail,'fit_step':RUNS[pose_fail][1],'active_face_index':ai_fail,'source_skin_face_row':face_fail,'source_skin_vertex_rows':[int(v) for v in faces[face_fail]],'offending_corner':corner_fail,'offending_source_vertex_id':source_v,'compact_vertex_index':compact_v,'source_vertex_xyz_m':skin['pos'][source_v].tolist(),'accepted_world_xyz_m_by_pose':captures[:,np.flatnonzero(ref==source_v)[0]].tolist()},'active_set':{'nonocular_crossing_pair_count':pair_count,'unique_source_face_count':len(active_faces),'active_face_at_index':face_fail,'active_face_rows':active_faces,'witnesses':pin(WIT)},'direction_method':{'module_path':str(Path(c.__file__).resolve()),'module_sha256':sha(Path(c.__file__).resolve()),'forward_model':pin(FORWARD),'forward_report':map_report,'owner_current_direction_report':direction_report,'current_source_direction':current.tolist(),'current_minimum_incident_projection':float(project(current).min()),'current_owner_conditioning_report':conditioning_report,'current_owner_conditioned_source_direction':(conditioned_dirs[compact_v]/np.linalg.norm(conditioned_dirs[compact_v])).tolist(),'current_owner_conditioned_minimum_incident_projection':float(project(conditioned_dirs[compact_v]/np.linalg.norm(conditioned_dirs[compact_v])).min()),'current_owner_incident_active_seed_guard_report':conditioned_seed_report,'current_owner_module_path':str(CURRENT.resolve()),'current_owner_module_sha256':sha(CURRENT),'incident_active_face_count':len(incidence),'incident_active_face_rows':[int(active_faces[int(i)]) for i in incidence],'incident_constraints':constraints,'feasibility_search':{'method':'deterministic multi-start SLSQP maximization of minimum exact normalized Jd·n over unit source direction; SOC cone g= n·Jd - 0.5*||Jd||','projection_floor':0.5,'best':best,'solutions':solutions,'feasible_at_floor':bool(best and best['minimum_projection']>=0.5-1e-8)}},'pose_pins':pose_pins,'inputs':[pin(SKIN),pin(NHA),pin(INV),pin(PREF),pin(SOL),pin(WIT),pin(ROOT/'src/numilab_human/common_atlas_skin_clearance.py'),pin(CURRENT),pin(CURRENT_CARDIAC),pin(ROOT/'src/numilab_human/cardiac_cavity_intersections.py'),pin(FORWARD)],'qualification':'No scan rerun, geometry edits, candidate generation, native run, or admission; only three retained accepted poses and existing terminal witnesses.'}
out=OUT/'direction-diagnosis.json'; out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'report':str(out),'sha256':sha(out),'face_row':face_fail,'source_vertex':source_v,'active_faces':len(active_faces),'incident_faces':len(incidence),'constraints':len(constraints),'current_min_projection':report['direction_method']['current_minimum_incident_projection'],'best':best},sort_keys=True),flush=True)
