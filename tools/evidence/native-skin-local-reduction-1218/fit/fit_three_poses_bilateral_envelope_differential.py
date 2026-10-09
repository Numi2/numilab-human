#!/usr/bin/env python3
from __future__ import annotations
import gc,hashlib,importlib.util,json,mmap,os,resource,struct,sys,time
from pathlib import Path
import numpy as np
E=Path('/Users/n/numi-human-resting-evidence-20261005'); R=Path('/Users/n/numi-human-retained-delivery-20261009')
BASE=R/'skin-resting-multipose-clearance-1218'; PREF=R/'skin-resting-multipose-clearance-1206'/'zero_delta_preflight.json'; PREF_SHA='cd0b2809cf4cf3bf427ad090f2fb463465cbf195dccf0ee1a66c3df052e0a809'
OUT=BASE/'fit-attempt-001'
if OUT.exists(): raise SystemExit('refuse existing fit output '+str(OUT))
OUT.mkdir(parents=True); (OUT/'attempts').mkdir()
ROOT=Path('/Users/n/numi-human-conforming-composition-source-1216'); RESP=Path('/Users/n/numi-human-resting-resp-source-20261006')
SKIN=E/'native-skin-epl143-clearance-1187/attempt-006/bodyparts3d-myosim-skinned-shell-epl143-candidate.nhskin'
OLD_SKIN=E/'common-atlas-skin-registration-003/bodyparts3d-myosim-skinned-shell.nhskin'
NHA=E/'native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-thorax.nhanatomy'
INV=E/'native-complete-skin-containment-audit-890/pair-summary-v3.csv'
SCENE=E/'common-atlas-skin-composition-907/resting-scene/resting-supine-scene.manifest.json'
ORIENT=E/'local-skin-orientation-892/local-orientation-report.json'
BONE_DIR=Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/current-bone-registration-b1b410ad')
BONE=BONE_DIR/'bodyparts3d-myosim-major-bones.nhbones'; BONE_MAN=BONE_DIR/'bodyparts3d-myosim-major-bones.manifest.json'
VALID=E/'cardiac-wall-native-self-audit-001/accepted_mrvpack_surface_audit.py'
A1191=E/'native-skin-epl143-clearance-1187'; R1191=R/'native-lung1178-thumb1187-smoke-1191/native-run'
R1201=R/'native-flat-reference-40s-1201/native-run'; A1201=R/'native-flat-reference-40s-1201/skin-audit-1201-terminal'
sys.path.insert(0,str(ROOT/'src')); from numilab_human import common_atlas_skin_clearance as c, cardiac_cavity_intersections as ci
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
 return h.hexdigest()
def pin(p):
 p=Path(p).resolve()
 if not p.is_file(): raise RuntimeError('missing input '+str(p))
 return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def j(p): return json.loads(Path(p).read_text())
def wj(p,d): Path(p).write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n')
def rss(): return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
def loadskin(path):
 raw=Path(path).read_bytes(); magic,abi,nb,nv,ni,fp,arch=struct.unpack_from('<8s5I32s',raw); vo=60+36*nb; io=vo+56*nv; wo=io+4*ni
 if magic!=b'NHSKIN1\0' or abi!=5 or len(raw)!=wo+4*nv*nb: raise RuntimeError('bad NHSKIN')
 vr=np.frombuffer(raw,'<f4',14*nv,vo).reshape(nv,14); faces=np.frombuffer(raw,'<u4',ni,io).reshape(-1,3).astype(np.int64); bind=np.frombuffer(raw,'<f4',9*nb,60).reshape(nb,9).copy()
 return raw,{'nb':nb,'nv':nv,'ni':ni,'fp':fp,'bind_u':np.frombuffer(raw,'<u4',9*nb,60).reshape(nb,9).copy(),'bind':bind,'pos':vr[:,:3].astype(float),'faces':faces,'weights':np.frombuffer(raw,'<f4',nv*nb,wo).reshape(nv,nb).astype(float)}
pre=j(PREF)
if sha(PREF)!=PREF_SHA or pre.get('status')!='zero_delta_forward_exact_fit_ready' or pre.get('all_nine_zero_delta_positions_bit_exact') is not True: raise RuntimeError('zero-delta preflight absent or stale')
# Recheck every input SHA retained in the preflight, but do not mutate or rewrite that proof.
for pinrow in pre['pose_inputs']+pre['dependencies']+[pre['source_skin']]+[p for row in pre['audits'] for p in row.values() if isinstance(p,dict) and 'path' in p]:
 if sha(pinrow['path'])!=pinrow['sha256']: raise RuntimeError('preflight input changed: '+pinrow['path'])
raw,skin=loadskin(SKIN); _,old=loadskin(OLD_SKIN)
if skin['nv']!=54949 or skin['nb']!=86 or len(skin['faces'])!=109211 or not np.array_equal(skin['faces'],old['faces']): raise RuntimeError('1187 source topology mismatch')
ref=np.unique(skin['faces']); faces=skin['faces']; compact=np.searchsorted(ref,faces)
allkeys,nonocular,_,_=c._load_target_inventory(INV); alltext={f'{a}:{b}' for a,b in allkeys}; otext={f'{a}:{b}' for a,b in c._OCULAR_MONITOR_KEYS}; nontext={f'{a}:{b}' for a,b in nonocular}
if len(allkeys)!=859: raise RuntimeError('target inventory mismatch')
hs=importlib.util.spec_from_file_location('mrv',VALID); helper=importlib.util.module_from_spec(hs); hs.loader.exec_module(helper)
fs=importlib.util.spec_from_file_location('forward915',E/'native-common-skin-multipose-forward-model-915.py'); fm=importlib.util.module_from_spec(fs); fs.loader.exec_module(fm)
NATIVE=R/'muscle-conforming-refinement-1216'/'native-refinement-1217-attempt2'
AUDIT=NATIVE/'skin-audit-1217'
FIT=[(NATIVE/'native-run',0),(NATIVE/'native-run',9983),(NATIVE/'native-run',20000)]
closure=j(NATIVE/'execution.json')
if closure.get('returncode')!=0 or closure.get('changed_inputs'): raise RuntimeError('1217 native run has not closed successfully')

captures=[]; state_receipts=[]; fit_receipts=[]; audits=[]; initial=None; closed_meshes=[]
def target_audit(run,step):
 ad=AUDIT
 r=j(ad/f'step-{step}.result.json'); targets=ad/f'step-{step}.targets.jsonl'; cross=ad/f'step-{step}.crossing-witnesses.jsonl'; selfp=ad/f'step-{step}.self-witnesses.jsonl'; invp=ad/f'step-{step}.invalid-triangles.jsonl'
 if r.get('status')!='complete_pair_coverage' or r.get('surface_target_count')!=859 or r.get('skin_self_crossing_pair_count')!=0 or r.get('invalid_target_triangle_count')!=0 or selfp.stat().st_size or invp.stat().st_size: raise RuntimeError('1217 exact baseline scan incomplete or invalid')
 rows={}
 with targets.open() as f:
  for line in f:
   x=json.loads(line); k=x['surface']; rows[f'{int(k[0])}:{int(k[1])}']={'triangle_pairs':[],'count':int(x['intersecting_triangle_pairs']),'aabb_candidate_pairs':int(x['aabb_candidate_pairs']),'degenerate_face_rows':[int(v) for v in x.get('degenerate_face_rows',[])]}
 with cross.open() as f:
  for line in f:
   x=json.loads(line); k=x['target_surface']; rows[f'{int(k[0])}:{int(k[1])}']['triangle_pairs'].append((int(x['skin_source_face_row']),int(x['target_surface_face_row'])))
 if set(rows)!=alltext or any(row['count']!=len(row['triangle_pairs']) or len(set(row['triangle_pairs']))!=row['count'] for row in rows.values()) or sum(x['count'] for x in rows.values())!=int(r['all_skin_crossing_pair_count']): raise RuntimeError('1217 baseline witness totals invalid')
 return rows,[pin(ad/f'step-{step}.result.json'),pin(targets),pin(cross),pin(selfp),pin(invp)]
for run,step in FIT:
 pack=run/'accepted-geometry'/f'step-{step}.mrvpack'; rec=pack.with_suffix('.receipt.json'); rd=j(rec)
 if rd.get('accepted_step')!=step or rd.get('physical_endpoint')!='accepted' or rd.get('surface_audit_endpoint')!='passed' or rd.get('common_field_source_anatomy_payload_sha256')!=sha(NHA): raise RuntimeError('invalid accepted fit state '+str(step))
 if initial is None: initial=rd
 elif json.dumps(rd['initial_anatomical_registration']['body_poses'],sort_keys=True)!=json.dumps(initial['initial_anatomical_registration']['body_poses'],sort_keys=True): raise RuntimeError('initial registration mismatch')
 if (rd['skin_source_mapping']['vertex_map']['sha256'],rd['skin_source_mapping']['anatomy_parameters']['sha256'])!=(initial['skin_source_mapping']['vertex_map']['sha256'],initial['skin_source_mapping']['anatomy_parameters']['sha256']): raise RuntimeError('runtime skin/resp maps changed')
 with pack.open('rb') as f:
  with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
   vo=next(x[2] for x in helper._pack_sections(mm) if x[0]==2); helper.validate_accepted_receipt(pack,rec,step,mm,vo,{})
 pos,surfs,shape=c._pack_surfaces(pack,allkeys); sf=surfs[(51007,1)]['faces']; base=int(sf.min())
 if len(sf)!=len(faces) or not np.array_equal(sf-base,faces): raise RuntimeError('fit pose skin topology differs')
 captures.append(pos[base:base+skin['nv']].astype('<f4').astype(float)[ref].copy())
 closed_row={}
 for stable,count in ((63,6688),(64,6690)):
  tf=surfs[(51005,stable)]['faces']; tv=np.unique(tf)
  if len(tf)!=count: raise RuntimeError('vastus complete captured face count changed '+str(stable))
  closed_row[f'51005:{stable}']=(pos[tv].astype('<f4').copy(),np.searchsorted(tv,tf))
 closed_meshes.append(closed_row)
 state_receipts.append({'step':step,'body_poses':rd['accepted_registered_body_poses'],'respiratory_motion':rd['accepted_respiratory_motion']})
 aud,ap=target_audit(run,step); audits.append(aud)
 fit_receipts += [pin(pack),pin(rec)]+ap
 del pos,surfs; gc.collect()
 print(json.dumps({'phase':'fit_pose_ready','step':step,'baseline_pairs':sum(x['count'] for x in aud.values())}),flush=True)
captures=np.stack(captures).astype('<f4').astype(float)
if not np.array_equal(captures[0],np.asarray(captures[0],dtype='<f4')): raise RuntimeError('fit capture not F32')
maprec=initial; body_init=maprec['initial_anatomical_registration']['body_poses']; anatomy_path=maprec['skin_source_mapping']['anatomy_parameters']['path']
forward,map_report,selector_ids=fm.build_forward(skin=skin,source_positions=skin['pos'],referenced_ids=ref,captured_by_pose=captures,state_receipts=state_receipts,initial_body_poses=body_init,map_receipt=maprec,anatomy_parameters_path=anatomy_path,respiration_source=RESP,clearance_module=c)
base_forward=forward(skin['pos'].astype('<f4'))
if base_forward.get('diagnostics',{}).get('admissible') is not True or not np.array_equal(np.asarray(base_forward['world_positions_by_pose'],dtype='<f4'),captures.astype('<f4')): raise RuntimeError('fit cohort zero-delta replay failed')
maps=np.asarray(base_forward['jacobians_by_pose'],dtype=np.float64)
orient=j(ORIENT); seeds=orient.get('per_face',[])
if orient.get('inputs',{}).get('nhskin',{}).get('sha256')!=sha(OLD_SKIN) or len(seeds)!=980: raise RuntimeError('outward orientation source mismatch')
seedids=np.asarray([int(x['face']) for x in seeds],dtype=np.int64)
if any(not np.array_equal(faces[int(x['face'])],np.asarray(x['source_vertex_ids'],dtype=np.int64)) for x in seeds): raise RuntimeError('orientation face source IDs mismatch')
signs=c._propagate_source_face_orientation(faces,seedids)
base_tri=captures[:,compact]
baseareas=np.cross(base_tri[:,:,1]-base_tri[:,:,0],base_tri[:,:,2]-base_tri[:,:,0])
winding=c._verify_source_winding_in_accepted_poses(skin['pos'][ref],compact,signs,maps,baseareas)
if winding.get('minimum_source_to_accepted_pose_normal_alignment',1)<=0: raise RuntimeError('accepted-pose winding failed')
scene=j(SCENE); bed=scene['bed']; fixed=np.asarray([int(x['vertex_index']) for x in bed['support_witnesses']],dtype=np.int64)
if len(fixed)!=32 or len(set(map(int,fixed)))!=32 or not set(map(int,fixed)).issubset(set(map(int,ref))): raise RuntimeError('support witness set invalid')
spill=np.asarray([19059,19060,19061,19743,19745,20391,20392,21186,34342,34343,34344,34992,34993,34994],dtype=np.int64)
anchors=np.unique(np.concatenate((selector_ids[np.isin(selector_ids,ref)],spill)))
if not set(map(int,anchors)).issubset(set(map(int,ref))): raise RuntimeError('thorax anchor set invalid')
anchors=anchors[~np.isin(anchors,fixed)]
orig=np.repeat(np.asarray(bed['plane_point_m'],dtype=float)[None,:],3,axis=0); norms=np.repeat(np.asarray(bed['normal'],dtype=float)[None,:],3,axis=0)
active=-1; positions=None; surfaces=None; scan_id=0
def load_pose(i):
 global active,positions,surfaces
 if active==i: return
 if positions is not None: del positions,surfaces; gc.collect()
 run,step=FIT[i]; pack=run/'accepted-geometry'/f'step-{step}.mrvpack'
 positions,surfaces,_=c._pack_surfaces(pack,allkeys)
 active=i
 print(json.dumps({'phase':'load_scan_pose','step':step,'target_count':859,'rss_peak':rss()}),flush=True)
def target_triangle(i,key,row):
 load_pose(i); k=tuple(map(int,key.split(':'))); return positions[surfaces[k]['faces'][int(row)]].astype('<f4').astype(float)
def full_scan(i,world):
 global scan_id
 scan_id+=1; load_pose(i)
 skin_records=c._exact_surface_records(np.asarray(world,dtype='<f4').astype(float),compact)
 skin_index=ci._prepare_surface_aabb(skin_records)
 out={}; pairs=0
 for ordinal,key in enumerate(sorted(allkeys),1):
  ft=surfaces[key]['faces']; unique=np.unique(ft); local=np.searchsorted(unique,ft)
  target_records=c._exact_surface_records(positions[unique].astype('<f4').astype(float),local)
  a=ci._audit_pair_prepared_first(skin_index,target_records,same_surface=False)
  out[f'{key[0]}:{key[1]}']={'triangle_pairs':[(int(x),int(y)) for x,y in a['triangle_pairs']],'count':int(a['count']),'aabb_candidate_pairs':int(a['aabb_candidate_pairs']),'degenerate_face_rows':[]}
  pairs+=int(a['count']); del target_records,unique,local,a
  if ordinal%100==0 or ordinal==859:
   with (OUT/'scan-progress.jsonl').open('a') as f: f.write(json.dumps({'phase':'exact_candidate_scan','attempt':scan_id,'pose_index':i,'step':FIT[i][1],'targets_done':ordinal,'targets_total':859,'pairs_so_far':pairs,'rss_peak':rss()},sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())
   print(json.dumps({'phase':'exact_candidate_scan','attempt':scan_id,'pose_index':i,'step':FIT[i][1],'targets_done':ordinal,'pairs_so_far':pairs,'rss_peak':rss()}),flush=True)
  gc.collect()
 return out

# Exact changed-face reuse follows the existing1187 differential audit:
# inherit only baseline pairs whose skin vertices are byte-identical, and test
# every changed face against every target. First use at each pose is independently
# checked against the unchanged complete scan implementation above.
differential_verified_poses=set()
differential_scan_count=0
def scan(i,world):
 global differential_scan_count
 differential_scan_count+=1
 t0=time.monotonic(); load_pose(i)
 packed=np.asarray(world,dtype='<f4')
 baseline=np.asarray(captures[i],dtype='<f4')
 if packed.shape!=baseline.shape or not np.isfinite(packed).all(): raise RuntimeError('differential scan geometry mismatch')
 changed_v=np.any(packed.view('<u4')!=baseline.view('<u4'),axis=1)
 changed_f=np.any(changed_v[compact],axis=1)
 changed_rows=np.flatnonzero(changed_f)
 unchanged_rows=np.flatnonzero(~changed_f)
 if not np.array_equal(packed[compact[unchanged_rows]].view('<u4'),baseline[compact[unchanged_rows]].view('<u4')): raise RuntimeError('unchanged face byte proof failed')
 records=c._exact_surface_records(packed.astype(float),compact)
 changed_records=[records[int(row)] for row in changed_rows]
 skin_index=ci._prepare_surface_aabb(changed_records)
 if len(changed_rows):
  changed_points=packed[np.unique(compact[changed_rows])]
  lower=changed_points.min(axis=0); upper=changed_points.max(axis=0)
 out={}; scanned=0; disjoint=0; retained_total=0; new_total=0
 for key in sorted(allkeys):
  textkey=f'{key[0]}:{key[1]}'
  inherited=[(int(a),int(b)) for a,b in audits[i][textkey]['triangle_pairs'] if not changed_f[int(a)]]
  ft=surfaces[key]['faces']; unique=np.unique(ft); target_pos=positions[unique].astype('<f4')
  overlap=bool(len(changed_rows)) and bool(np.all(target_pos.min(axis=0)<=upper) and np.all(target_pos.max(axis=0)>=lower))
  if overlap:
   local=np.searchsorted(unique,ft)
   target_records=c._exact_surface_records(target_pos.astype(float),local)
   result=ci._audit_pair_prepared_first(skin_index,target_records,same_surface=False)
   fresh=[(int(a),int(b)) for a,b in result['triangle_pairs']]
   if any(not changed_f[a] for a,b in fresh): raise RuntimeError('narrowphase returned an unchanged row')
   scanned+=1; del target_records
  else:
   fresh=[];result={'aabb_candidate_pairs':0};disjoint+=1
  pairs=sorted(inherited+fresh)
  if len(pairs)!=len(set(pairs)):raise RuntimeError('differential union has duplicate pairs')
  out[textkey]={'triangle_pairs':pairs,'count':len(pairs),'aabb_candidate_pairs':int(result['aabb_candidate_pairs']),'aabb_candidate_scope':'fresh_changed_skin_faces_only','degenerate_face_rows':[],'inherited_unchanged_face_pair_count':len(inherited),'fresh_changed_face_pair_count':len(fresh)}
  retained_total+=len(inherited);new_total+=len(fresh)
 if len(out)!=859 or scanned+disjoint!=859:raise RuntimeError('differential target coverage mismatch')
 dt=time.monotonic()-t0
 evidence={'scan':differential_scan_count,'pose_index':i,'accepted_step':FIT[i][1],'changed_vertices':int(changed_v.sum()),'changed_faces':int(changed_f.sum()),'unchanged_faces_byte_exact':True,'complete_target_count':859,'targets_narrowphase_scanned':scanned,'targets_with_disjoint_F32_bounds_or_no_changed_faces':disjoint,'inherited_pairs':retained_total,'fresh_pairs':new_total,'differential_wall_s':dt}
 if i not in differential_verified_poses:
  t1=time.monotonic(); reference=full_scan(i,world)
  mismatch=[k for k in out if sorted(out[k]['triangle_pairs'])!=sorted(reference[k]['triangle_pairs']) or out[k]['count']!=reference[k]['count'] or out[k]['degenerate_face_rows']!=reference[k]['degenerate_face_rows']]
  evidence.update({'independent_full_scan_wall_s':time.monotonic()-t1,'full_scan_exact_pair_mismatches':mismatch,'full_scan_verified':not mismatch})
  if mismatch:raise RuntimeError('differential/full exact pair mismatch '+str(mismatch))
  differential_verified_poses.add(i)
 with (OUT/'differential-scan-proof.jsonl').open('a') as f:f.write(json.dumps(evidence,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'phase':'differential_scan_complete',**evidence}),flush=True)
 gc.collect()
 return out
def persist(event):
 event=dict(event); event['elapsed_wall_s']=time.monotonic()-start; src=np.asarray(event.pop('source_positions_f32'),dtype='<f4'); n=int(event['attempt'])
 audits_out=event.pop('target_audits_by_pose',None)
 if audits_out is not None:
  ap=OUT/'attempts'/f'attempt-{n:04d}-target-audits.json'; ap.write_text(json.dumps(audits_out,sort_keys=True,separators=(',',':'),allow_nan=False)); event['target_audits_path']=str(ap); event['target_audits_sha256']=sha(ap)
 npy=OUT/'attempts'/f'attempt-{n:04d}-source.npy'
 with npy.open('xb') as f: np.save(f,src,allow_pickle=False); f.flush(); os.fsync(f.fileno())
 event['candidate_npy_path']=str(npy); event['candidate_npy_sha256']=sha(npy)
 with (OUT/'attempts.jsonl').open('a') as f: f.write(json.dumps(event,sort_keys=True,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
 print(json.dumps({k:event.get(k) for k in ('attempt','iteration','backtrack','status','reason','nonocular_pair_counts_before_by_pose','nonocular_pair_counts_candidate_by_pose','closed_target_interior_counts_before_by_pose','closed_target_interior_counts_candidate_by_pose','candidate_npy_path')}),flush=True)
def fit_forward(full_source):
 result=dict(forward(full_source))
 result['world_positions_by_pose']=np.asarray(result['world_positions_by_pose'])[0:3]
 result['jacobians_by_pose']=np.asarray(result['jacobians_by_pose'])[0:3]
 return result
# Receipts are ordered fit states [0,9983,20000], as required by the existing solver.
input_paths=[ROOT/'source-revision.json',ROOT/'src/numilab_human/model.py',BASE/'fit-attempt-001-differential-plan.json',Path(__file__),PREF,SKIN,OLD_SKIN,NHA,INV,SCENE,ORIENT,BONE,BONE_MAN,VALID,E/'native-common-skin-multipose-forward-model-915.py',ROOT/'src/numilab_human/common_atlas_skin_clearance.py',ROOT/'src/numilab_human/cardiac_cavity_intersections.py',RESP/'src/numilab_human/resting_respiratory_conforming_field.py',E/'common-atlas-skin-composition-907/common-atlas-skin-geometry-registration.manifest.json',E/'common-atlas-skin-composition-907/composition-report.json']
input_paths += [NATIVE/'run-declaration.json',NATIVE/'execution.json',NATIVE/'native-run'/'run-metadata.json']
pins=[pin(p) for p in input_paths]+fit_receipts
prehash={p['path']:p['sha256'] for p in pins}
preflight={'schema':'numi.human.skin-resting-multipose-clearance-fit-preflight.v1','status':'ready_for_bounded_bilateral_vastus_enclosure_fit','fit_steps':[0,9983,20000],'held_out_steps':[4991,5375,5759,6111,6495,7743],'fit_baseline_nonocular_pair_counts':[sum(v['count'] for k,v in a.items() if k in nontext) for a in audits],'fit_baseline_ocular_pair_counts':[sum(v['count'] for k,v in a.items() if k in otext) for a in audits],'fit_baseline_skin_self_pairs':[0,0,0],'candidate_forward_model':map_report,'source_winding':winding,'fixed_support_source_vertex_ids':fixed.tolist(),'preserved_source_anchor_vertex_ids':anchors.tolist(),'selected_closed_target_face_counts':{'51005:63':6688,'51005:64':6690},'probe_scope':'bounded fit from unchanged1187 using actual1217 accepted geometry; explicit bilateral vastus outer envelopes after complete self and component eligibility; no automatic admission','explicit_outer_envelope_keys':['51005:63','51005:64'],'solver_parameters':{'selected_margin_mm':0.25,'support_radius_edge_multiple':4.0,'max_iterations':12,'backtrack_count':7},'input_pins':pins,'qualification':'existing source-position solver only; no native replay/admission'}
wj(OUT/'preflight.json',preflight)
np.save(OUT/'starting-source-positions-f32.npy',skin['pos'].astype('<f4'),allow_pickle=False)
start=time.monotonic(); wj(OUT/'solve-start.json',{'status':'running','preflight_sha256':sha(OUT/'preflight.json'),'fit_steps':[0,9983,20000],'solver':'derive_shared_multipose_inferred_clearance','selected_margin_mm':0.25,'support_radius_edge_multiple':4.0,'max_iterations':12,'backtrack_count':7})
try:
 candidate,report=c.derive_shared_multipose_inferred_clearance(source_positions=skin['pos'],faces=faces,jacobians_by_pose=None,accepted_skin_world_by_pose=captures,baseline_target_audits_by_pose=audits,baseline_skin_self_pairs_by_pose=[0,0,0],source_outward_face_signs=signs,scan_candidate_targets=scan,target_triangle_by_row=target_triangle,all_target_keys=alltext,ocular_monitor_keys=otext,fixed_source_vertex_ids=fixed,preserved_source_anchor_vertex_ids=anchors,bed_plane_origins_by_pose=orig,bed_plane_normals_by_pose=norms,selected_margin_mm=0.25,support_radius_edge_multiple=4.0,max_iterations=12,backtrack_count=7,progress_callback=persist,candidate_forward=fit_forward,baseline_replay_tolerance_m=1e-6,scan_progress_callback=None,closed_target_surfaces_by_pose=closed_meshes,closed_target_face_counts={'51005:63':6688,'51005:64':6690},closed_target_outer_envelope_keys={'51005:63','51005:64'})
except BaseException as exc:
 wj(OUT/'solve-result.json',{'schema':'numi.human.skin-resting-multipose-clearance-fit.v1','status':'failed_or_incomplete_candidate_solve','error_type':type(exc).__name__,'error':str(exc),'elapsed_wall_s':time.monotonic()-start,'attempt_log':str(OUT/'attempts.jsonl'),'qualification':'no candidate admitted','target_validation_failure':getattr(exc,'target_report',None)})
 raise
finally:
 post={p:sha(p) for p in prehash}
 wj(OUT/'input-postcheck.json',{'status':'unchanged' if post==prehash else 'input_changed','input_count':len(post),'mismatches':[{'path':p,'expected':prehash[p],'observed':post[p]} for p in post if post[p]!=prehash[p]]})
 if post!=prehash: raise RuntimeError('pinned input changed during solve')

candidate=np.asarray(candidate,dtype='<f4')
npy=OUT/'candidate-source-positions-f32.npy'
with npy.open('xb') as f: np.save(f,candidate,allow_pickle=False)
report.update({'accepted_pose_steps':[0,9983,20000],'fit_pose_steps':[0,9983,20000],'held_out_pose_steps':[4991,5375,5759,6111,6495,7743],'candidate_source_positions_f32_sha256':hashlib.sha256(candidate.tobytes()).hexdigest(),'candidate_source_positions_npy':str(npy),'candidate_source_positions_npy_sha256':sha(npy),'preflight_path':str(OUT/'preflight.json'),'fit_elapsed_wall_s':time.monotonic()-start,'runtime_asset_status':'not composed or admitted; holdout six-pose audit and native replay pending','qualification':'CPU source-position fit only'})
wj(OUT/'candidate-report.json',report)
wj(OUT/'solve-result.json',{'schema':'numi.human.skin-resting-multipose-clearance-fit.v1','status':report.get('status'),'elapsed_wall_s':time.monotonic()-start,'candidate_report':str(OUT/'candidate-report.json'),'candidate_source_positions_npy':str(npy),'candidate_source_positions_f32_sha256':report['candidate_source_positions_f32_sha256'],'candidate_source_positions_npy_sha256':report['candidate_source_positions_npy_sha256'],'final_nonocular_pair_count_by_pose':report.get('final_nonocular_pair_count_by_pose'),'qualification':'candidate fit only; heldout and native checks pending'})
print(json.dumps({'phase':'fit_complete','status':report.get('status'),'fit_elapsed_wall_s':time.monotonic()-start,'initial_nonocular':report.get('initial_nonocular_pair_count_by_pose'),'final_nonocular':report.get('final_nonocular_pair_count_by_pose'),'candidate_npy':str(npy),'candidate_sha':report['candidate_source_positions_f32_sha256']}),flush=True)
