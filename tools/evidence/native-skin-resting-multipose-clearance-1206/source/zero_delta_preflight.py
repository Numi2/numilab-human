import gc,hashlib,importlib.util,json,mmap,struct,sys
from pathlib import Path
import numpy as np
E=Path('/Users/n/numi-human-resting-evidence-20261005'); R=Path('/Users/n/numi-human-retained-delivery-20261009'); O=R/'skin-resting-multipose-clearance-1206'; O.mkdir(parents=True,exist_ok=True)
OUT=O/'zero_delta_preflight.json'
if OUT.exists(): raise SystemExit('refuse existing report')
ROOT=Path('/Users/n/numi-human-common-skin-multipose-001'); RESP=Path('/Users/n/numi-human-resting-resp-source-20261006')
S=E/'native-skin-epl143-clearance-1187/attempt-006/bodyparts3d-myosim-skinned-shell-epl143-candidate.nhskin'; OLD=E/'common-atlas-skin-registration-003/bodyparts3d-myosim-skinned-shell.nhskin'; INV=E/'native-complete-skin-containment-audit-890/pair-summary-v3.csv'; VALID=E/'cardiac-wall-native-self-audit-001/accepted_mrvpack_surface_audit.py'
A1191=E/'native-skin-epl143-clearance-1187'; R1191=R/'native-lung1178-thumb1187-smoke-1191/native-run'; R1201=R/'native-flat-reference-40s-1201/native-run'; A1201=R/'native-flat-reference-40s-1201/skin-audit-1201-terminal'
sys.path.insert(0,str(ROOT/'src')); from numilab_human import common_atlas_skin_clearance as c
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):
 p=Path(p).resolve()
 if not p.is_file(): raise RuntimeError('missing '+str(p))
 return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def j(p): return json.loads(Path(p).read_text())
def loadskin(p):
 raw=Path(p).read_bytes(); magic,abi,nb,nv,ni,fp,arch=struct.unpack_from('<8s5I32s',raw); vo=60+36*nb; io=vo+56*nv; wo=io+4*ni
 if magic!=b'NHSKIN1\0' or abi!=5 or len(raw)!=wo+4*nv*nb: raise RuntimeError('bad NHSKIN')
 vr=np.frombuffer(raw,'<f4',14*nv,vo).reshape(nv,14); faces=np.frombuffer(raw,'<u4',ni,io).reshape(-1,3).astype(np.int64); bind=np.frombuffer(raw,'<f4',9*nb,60).reshape(nb,9).copy()
 return raw,{'nb':nb,'nv':nv,'bind_u':np.frombuffer(raw,'<u4',9*nb,60).reshape(nb,9).copy(),'bind':bind,'pos':vr[:,:3].astype(float),'faces':faces,'weights':np.frombuffer(raw,'<f4',nv*nb,wo).reshape(nv,nb).astype(float)}
raw,skin=loadskin(S); _,old=loadskin(OLD)
if (skin['nv'],skin['nb'],len(skin['faces']))!=(54949,86,109211) or not np.array_equal(skin['faces'],old['faces']): raise RuntimeError('1187 topology mismatch')
ref=np.unique(skin['faces']); allkeys,nonocular,_,_=c._load_target_inventory(INV)
if len(allkeys)!=859: raise RuntimeError('target inventory mismatch')
hs=importlib.util.spec_from_file_location('mrv',VALID); helper=importlib.util.module_from_spec(hs); hs.loader.exec_module(helper)
fs=importlib.util.spec_from_file_location('fwd',E/'native-common-skin-multipose-forward-model-915.py'); fm=importlib.util.module_from_spec(fs); fs.loader.exec_module(fm)
poses=[(R1191,s) for s in [0,4991,5375,5759,6111,6495,7743,10000]]+[(R1201,20000)]
captures=[]; receipts=[]; posepins=[]; auditrows=[]; initial=None
for run,step in poses:
 pack=run/'accepted-geometry'/f'step-{step}.mrvpack'; rec=pack.with_suffix('.receipt.json'); rd=j(rec)
 if rd.get('accepted_step')!=step or rd.get('physical_endpoint')!='accepted' or rd.get('surface_audit_endpoint')!='passed': raise RuntimeError('invalid accepted capture '+str(step))
 if initial is None: initial=rd
 elif json.dumps(rd['initial_anatomical_registration']['body_poses'],sort_keys=True)!=json.dumps(initial['initial_anatomical_registration']['body_poses'],sort_keys=True): raise RuntimeError('initial body pose mismatch')
 if (rd['skin_source_mapping']['vertex_map']['sha256'],rd['skin_source_mapping']['anatomy_parameters']['sha256'])!=(initial['skin_source_mapping']['vertex_map']['sha256'],initial['skin_source_mapping']['anatomy_parameters']['sha256']): raise RuntimeError('map/resp parameter mismatch')
 with pack.open('rb') as f:
  with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
   vo=next(x[2] for x in helper._pack_sections(mm) if x[0]==2); helper.validate_accepted_receipt(pack,rec,step,mm,vo,{})
 pos,surfs,shape=c._pack_surfaces(pack,allkeys); sf=surfs[(51007,1)]['faces']; base=int(sf.min())
 if len(sf)!=len(skin['faces']) or not np.array_equal(sf-base,skin['faces']): raise RuntimeError('face index mismatch '+str(step))
 captures.append(pos[base:base+skin['nv']].astype('<f4').astype(float)[ref].copy())
 receipts.append({'step':step,'body_poses':rd['accepted_registered_body_poses'],'respiratory_motion':rd['accepted_respiratory_motion']})
 if run==R1191:
  ad=A1191/('native-1191-differential-full-001' if step in [0,4991,5375,5759] else f'native-1191-differential-parallel-002-step-{step}')/f'step-{step}'
  result=j(ad/'result.json'); rows=j(ad/'target-results.json')
  if result.get('full_skin_self_audit_performed') is not True or result.get('skin_self_unallowed_intersection_count')!=0 or len(rows)!=859 or any(int(x['unallowed_intersections']) for x in rows): raise RuntimeError('1191 full 859/self proof not zero at '+str(step))
  auditrows.append({'step':step,'count':0,'result':pin(ad/'result.json'),'target_results':pin(ad/'target-results.json')})
 else:
  result=j(A1201/'step-20000.result.json'); targets=A1201/'step-20000.targets.jsonl'; crosses=A1201/'step-20000.crossing-witnesses.jsonl'; selfp=A1201/'step-20000.self-witnesses.jsonl'; invalid=A1201/'step-20000.invalid-triangles.jsonl'
  rows=[json.loads(x) for x in targets.read_text().splitlines()]
  if result.get('status')!='complete_pair_coverage' or result.get('surface_target_count')!=859 or result.get('skin_self_crossing_pair_count')!=0 or result.get('invalid_target_triangle_count')!=0 or len(rows)!=859 or sum(int(x['intersecting_triangle_pairs']) for x in rows)!=int(result['all_skin_crossing_pair_count']) or selfp.stat().st_size or invalid.stat().st_size: raise RuntimeError('1201 audit incomplete')
  auditrows.append({'step':step,'count':int(result['all_skin_crossing_pair_count']),'result':pin(A1201/'step-20000.result.json'),'targets':pin(targets),'crossings':pin(crosses),'self':pin(selfp),'invalid':pin(invalid)})
 posepins.extend([pin(pack),pin(rec)]); del pos,surfs; gc.collect(); print(json.dumps({'pose':step,'pairs':auditrows[-1]['count']}),flush=True)
captures=np.stack(captures).astype('<f4').astype(float)
if sha(R1191/'accepted-geometry/step-0.mrvpack')!=sha(R1201/'accepted-geometry/step-0.mrvpack'): raise RuntimeError('initial packs differ')
forward,mapr,selectors=fm.build_forward(skin=skin,source_positions=skin['pos'],referenced_ids=ref,captured_by_pose=captures,state_receipts=receipts,initial_body_poses=initial['initial_anatomical_registration']['body_poses'],map_receipt=initial,anatomy_parameters_path=initial['skin_source_mapping']['anatomy_parameters']['path'],respiration_source=RESP,clearance_module=c)
base=forward(skin['pos'].astype('<f4')); pred=np.asarray(base.get('world_positions_by_pose'),dtype='<f4')
if base.get('diagnostics',{}).get('admissible') is not True or pred.shape!=captures.shape or not np.array_equal(pred,captures.astype('<f4')):
 err=np.abs(pred.astype(float)-captures.astype(float)); raise RuntimeError(f'zero-delta forward mismatch max={err.max()} differing={np.count_nonzero(err)}')
doc={'schema':'numi.human.skin-resting-multipose-clearance-preflight.v1','status':'zero_delta_forward_exact_fit_ready','source_skin':pin(S),'source_counts':{'vertices':skin['nv'],'faces':len(skin['faces']),'bindings':skin['nb']},'fit_steps':[0,10000,20000],'held_out_steps':[4991,5375,5759,6111,6495,7743],'all_nine_zero_delta_positions_bit_exact':True,'max_replay_error_m':0.0,'runtime_map_sha256':initial['skin_source_mapping']['vertex_map']['sha256'],'runtime_respiration_parameters_sha256':initial['skin_source_mapping']['anatomy_parameters']['sha256'],'forward_model_report':mapr,'audits':auditrows,'pose_inputs':posepins,'dependencies':[pin(E/'native-common-skin-multipose-forward-model-915.py'),pin(ROOT/'src/numilab_human/common_atlas_skin_clearance.py'),pin(ROOT/'src/numilab_human/cardiac_cavity_intersections.py'),pin(RESP/'src/numilab_human/resting_respiratory_conforming_field.py'),pin(INV)],'qualification':'CPU preflight only; no candidate fit or native admission'}
OUT.write_text(json.dumps(doc,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'status':doc['status'],'fit_pair_counts':[a['count'] for a in auditrows if a['step'] in [0,10000,20000]],'holdout_pair_counts':[a['count'] for a in auditrows if a['step'] in [4991,5375,5759,6111,6495,7743]],'report':str(OUT)}),flush=True)
