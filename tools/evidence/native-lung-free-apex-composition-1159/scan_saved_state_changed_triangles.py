#!/usr/bin/env python3
"""Exact changed-star scan for 1159 using saved production-Metal point replay."""
from pathlib import Path
import collections, gzip, hashlib, importlib.util, json, sys, time
import numpy as np
E=Path('/Users/n/numi-human-resting-evidence-20261005')
OUT=Path(__file__).resolve().parent
CAND=E/'native-lung-free-apex-composition-1159-eightops-attempt1'
PRED=E/'native-lung-free-apex-composition-1160'
RUN=E/'final-native-scene-preflight-936/skin-927-lung-1116-viewer-018-v015-attempt1/native-run'
PRIOR=E/'native-lung-transformed-pose-audit-runner-1096/attempt-001-native1113-geometry-only'
V8=E/'native-lung-final-selected-composition-dryrun-1113/provisional-1105-1106-v8'
AUDIT=E/'native-lung-transformed-pose-audit-runner-1096/audit_lung_cycle_1096.py'
STEPS=(0,4991,5375,5759,6111,6495,7743,10000)
IDS=(305,306,307,308,309,311); CHANGED=(305,306,307,308)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
 return h.hexdigest()
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p)
 if s is None or s.loader is None: raise RuntimeError('cannot load '+str(p))
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def writej(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
t0=time.monotonic()
comp=json.loads((CAND/'composition-report.json').read_text()); pi=json.loads((PRED/'prediction-input.json').read_text()); pr=json.loads((PRED/'prediction-run.json').read_text())
cand_nha=CAND/'final/resting-thorax.nhanatomy'
if sha(cand_nha)!='c10dce4609be99fdc569801c2705e23c46120b610c8aa86c631165d38baf4713': raise RuntimeError('candidate NHA hash mismatch')
if sha(CAND/'composition-report.json')!='ce5948d2dfe6388fc64f656fa34abfd7f409e471ee272a4a8231eba38281e108': raise RuntimeError('composition report hash mismatch')
if tuple(pi['steps'])!=STEPS or not pr.get('complete'): raise RuntimeError('incomplete prediction result')
if pi['candidate_nha']['sha256']!=sha(cand_nha) or pi['candidate_receipt']['sha256']!=sha(CAND/'final/resting-anatomy-receipt.json'): raise RuntimeError('prediction candidate identity mismatch')

# Pin exact native predicates, accepted-state owner, and strict map adapters.
a=load(AUDIT,'audit1160'); base=load(a.BASE,'base1160'); adapters=a.load_native_adapters(base)
owner=a.load_module(base.OWNER,'owner1160'); _,steps,inv,meta=owner.preflight(RUN)
verified,roots,dt=owner.validate_documents(inv,meta,(RUN/'native.log').read_text())
if tuple(map(int,steps))!=STEPS or tuple(map(int,verified))!=STEPS or not meta.get('loaded_metal_runtime',{}).get('verified',False): raise RuntimeError('baseline accepted-state validation failed')
base_nha=Path(owner.option(inv['argv'],'--torso-anatomy-payload')).resolve(); base_nha_sha=inv.get('asset_sha256',{}).get(str(base_nha))
if not base_nha_sha or sha(base_nha)!=base_nha_sha: raise RuntimeError('baseline NHA is not invocation hash-bound')
parser=base.load(base.PARSER,'parser1160'); base_rows=parser.parse_payload(base_nha)[1]; cand_rows=parser.parse_payload(cand_nha)[1]
if any(s not in base_rows or s not in cand_rows for s in IDS): raise RuntimeError('missing audit row')
for sid in IDS:
 if not np.array_equal(base_rows[sid]['faces'],cand_rows[sid]['faces']): raise RuntimeError('unexpected topology change row '+str(sid))
 if len(base_rows[sid]['vertices6'])!=len(cand_rows[sid]['vertices6']): raise RuntimeError('unexpected vertex count change row '+str(sid))

# The candidate v2 bridge validates every L-L map. The owner D-map reader validates source lineage;
# additionally require every mapped source triangle to remain byte-identical in the 1159 NHA.
lineage=adapters['lineage_v2'].load_v2_bridge(base=base,report_path=CAND/'current-reciprocal-map-report-v2.json',final_nha_path=cand_nha,final_nha_sha=sha(cand_nha),final_rows=cand_rows)
if set(lineage['pairs'])!=set(base.LOBE_PAIRS): raise RuntimeError('candidate L-L map does not declare all ten pairs')
v8nha=V8/'final/resting-thorax.nhanatomy'; v8rows=parser.parse_payload(v8nha)[1]
dmap=adapters['v8_dmap'].load_v8_current_dmap(base=base,composition_report_path=V8/'composition-report.json',final_nha_path=v8nha,final_nha_sha=sha(v8nha),final_rows=v8rows)
def tri(r,fi):
 v=r.get('vertices6',r.get('v')); f=r.get('faces',r.get('f'))
 return np.ascontiguousarray(v[f[int(fi)],:3],dtype='<f4')
dmap_identity={}
for sid,pair in sorted(dmap['pairs'].items()):
 for di,li in pair['d2l'].items():
  if not np.array_equal(tri(cand_rows[311],di),tri(dmap['source_rows'][311],di)) or not np.array_equal(tri(cand_rows[sid],li),tri(dmap['source_rows'][sid],li)):
   raise RuntimeError('candidate changed an exact D-map source triangle: '+str(sid))
 dmap_identity[str(sid)]={'mapped_face_count':len(pair['d2l']),'all_candidate_source_triangles_byte_identical':True}

ranges_path=E/'native-respiratory-kernel-saved-points-1119/source-ranges.json'
ranges={int(x['stable_id']):x for x in json.loads(ranges_path.read_text())}
for sid in IDS:
 if sid not in ranges or int(ranges[sid]['count'])!=len(cand_rows[sid]['vertices6']): raise RuntimeError('source point range mismatch row '+str(sid))
changed_report=comp['changed_rows']
changed_report_faces={int(s):set(map(int,r['faces'])) for s,r in changed_report.items()}
source_changed={int(s):set(map(int,r['vertices'])) for s,r in changed_report.items()}
selected={}
for sid in CHANGED:
 actual=set(map(int,np.flatnonzero(np.any(np.isin(cand_rows[sid]['faces'],list(source_changed.get(sid,set()))),axis=1))))
 declared=set(map(int,changed_report[str(sid)]['faces']))
 if actual!=declared: raise RuntimeError('composition face list is not exact source vertex star row '+str(sid))
 selected[sid]=actual

outputs={int(x['step']):Path(x['path']).resolve() for x in pr['outputs']}
if set(outputs)!=set(STEPS): raise RuntimeError('point output state set mismatch')
for out in pr['outputs']:
 p=outputs[int(out['step'])]
 if not p.is_file() or sha(p)!=out['sha256'] or p.stat().st_size!=5318400: raise RuntimeError('point output missing/hash/size mismatch')
sys.path.insert(0,str(base.ROOT/'src'))
baseline={}
for s in STEPS: baseline[s]=base.row_pose(RUN,s,base_rows)[0]
old_unallowed={}
for s in STEPS:
 p=PRIOR/('step-%06d-cross.jsonl.gz'%s); events=[]
 with gzip.open(p,'rt') as f:
  for line in f:
   if not line.strip(): continue
   e=json.loads(line); cl=e.get('class',e.get('classification',''))
   if str(cl).startswith(('unclassified','unallowed')): events.append(e)
 old_unallowed[s]=events
if sum(map(len,old_unallowed.values()))!=115: raise RuntimeError('baseline rejected event inventory is not 115')

def event_key(e): return (tuple(map(int,e['owners'])),tuple(map(int,e.get('face_rows',e.get('faces')))))
def build_pose(s):
 raw=np.fromfile(outputs[s],dtype='<f4')
 if raw.size!=443200*3 or not np.isfinite(raw).all(): raise RuntimeError('bad point output step '+str(s))
 raw=raw.reshape(-1,3); pose={}
 for sid in IDS:
  r=ranges[sid];st=int(r['start']);n=int(r['count'])
  pose[sid]={'v':np.ascontiguousarray(raw[st:st+n],dtype='<f4'),'f':cand_rows[sid]['faces']}
 actual_changed={}; actual_faces={}
 for sid in CHANGED:
  before=baseline[s][sid]['v']; after=pose[sid]['v']
  diff=np.any(after.view('<u4')!=before.view('<u4'),axis=1)
  ids=set(map(int,np.flatnonzero(diff)))
  if not ids.issubset(source_changed.get(sid,set())): raise RuntimeError('kernel changed undeclared source vertices row %d step %d'%(sid,s))
  actual_changed[sid]=ids
  actual_faces[sid]=set(map(int,np.flatnonzero(np.any(np.isin(pose[sid]['f'],list(ids)),axis=1)))) if ids else set()
  if not actual_faces[sid].issubset(selected[sid]): raise RuntimeError('runtime changed faces outside declared source star row %d'%sid)
 if not np.array_equal(pose[311]['v'].view('<u4'),baseline[s][311]['v'].view('<u4')): raise RuntimeError('D311 changed without source operation')
 return pose,actual_changed,actual_faces

def records_for(v,f,face_rows):
 ids=sorted(face_rows)
 raw,bad=base.exact_records(v,f[np.asarray(ids,dtype=np.int64)])
 if bad: raise RuntimeError('degenerate selected changed faces '+repr(bad[:8]))
 return [(x[0],x[1],x[2],ids[int(x[3])],x[4]) for x in raw]
pred=base.pred()
def scan_pair(pose,a0,b0,changed,native_dm,native_lm):
 va,fa=pose[a0]['v'],pose[a0]['f']; vb,fb=pose[b0]['v'],pose[b0]['f']
 sel_a=changed.get(a0,set()); sel_b=changed.get(b0,set())
 if a0==b0:
  if not sel_a:return [],{'aabb_candidate_pairs':0,'unique_aabb_pairs':0,'exact_hit_pairs':0,'classification_counts':{}}
  src=records_for(va,fa,sel_a); tris=vb[fb]; low=tris.min(axis=1); high=tris.max(axis=1); pairs={}; aabb=0
  for x in src:
   i=x[3]; t=va[fa[i]]; lo=t.min(axis=0); hi=t.max(axis=0)
   matches=np.flatnonzero(np.all(high>=lo,axis=1)&np.all(low<=hi,axis=1)); aabb+=len(matches)
   ys,bad=base.exact_records(vb,fb[matches])
   if bad:raise RuntimeError('degenerate peer self triangles')
   for y0 in ys:
    j=int(matches[int(y0[3])])
    if j==i or (j in sel_a and j<i):continue
    y=(y0[0],y0[1],y0[2],j,y0[4]); key=(min(i,j),max(i,j))
    pairs[key]=(x,y) if i<j else (y,x)
  events=[]; counts=collections.Counter()
  for (i,j),(x,y) in sorted(pairs.items()):
   pts=pred.triangle_intersection_points(x[0],y[0])
   if not pts:continue
   cl='allowed_indexed_adjacency' if base.adjacency(i,j,va,fa,pts) else 'unallowed_self_intersection'
   events.append({'owners':[a0,a0],'face_rows':[i,j],'class':cl,**base.event_points(pts)}); counts[cl]+=1
  return events,{'aabb_candidate_pairs':aabb,'unique_aabb_pairs':len(pairs),'exact_hit_pairs':len(events),'classification_counts':dict(counts)}
 pairs={}; aabb=0
 def visit(src_sid,src_sel,peer_sid):
  nonlocal aabb
  if not src_sel:return
  sv,sf=pose[src_sid]['v'],pose[src_sid]['f']; pv,pf=pose[peer_sid]['v'],pose[peer_sid]['f']
  src=records_for(sv,sf,src_sel); tris=pv[pf]; low=tris.min(axis=1); high=tris.max(axis=1)
  for x in src:
   i=x[3]; t=sv[sf[i]]; lo=t.min(axis=0); hi=t.max(axis=0)
   matches=np.flatnonzero(np.all(high>=lo,axis=1)&np.all(low<=hi,axis=1)); aabb+=len(matches)
   ys,bad=base.exact_records(pv,pf[matches])
   if bad:raise RuntimeError('degenerate peer exact triangles')
   for y0 in ys:
    j=int(matches[int(y0[3])]); y=(y0[0],y0[1],y0[2],j,y0[4]); key=(i,j) if src_sid==a0 else (j,i)
    if key not in pairs:pairs[key]=(x,y) if src_sid==a0 else (y,x)
 visit(a0,sel_a,b0);visit(b0,sel_b,a0)
 events=[]; counts=collections.Counter()
 for (i,j),(x,y) in sorted(pairs.items()):
  pts=pred.triangle_intersection_points(x[0],y[0])
  if not pts:continue
  cl=base.classify(a0,b0,x,y,pts,pose,native_dm,native_lm)
  events.append({'owners':[a0,b0],'face_rows':[i,j],'class':cl,**base.event_points(pts)}); counts[cl]+=1
 return events,{'aabb_candidate_pairs':aabb,'unique_aabb_pairs':len(pairs),'exact_hit_pairs':len(events),'classification_counts':dict(counts)}

script_path=Path(__file__).resolve()
tracked=[script_path,AUDIT,base.__file__,base.OWNER,base.PARSER,base.PRED,base.ROOT/'src/numilab_human/common_atlas_skin_clearance.py',
 CAND/'final/resting-thorax.nhanatomy',CAND/'final/resting-anatomy-receipt.json',CAND/'final/resting-anatomy-manifest.json',CAND/'composition-report.json',CAND/'current-reciprocal-map-report-v2.json',CAND/'row310-face-lineage.npy',
 PRED/'prediction-input.json',PRED/'prediction-run.json',PRED/'combined-source-points.xyz-f32.bin',ranges_path]
tracked += [Path(x) for x in pi['input_sha256']]+list(outputs.values())+[PRIOR/'report.json']+[PRIOR/('step-%06d-cross.jsonl.gz'%s) for s in STEPS]
tracked=sorted(set(Path(x).resolve() for x in tracked)); before={str(x):sha(x) for x in tracked}
writej(OUT/'declaration.json',{'schema':'numi.human.lung-composed-candidate-exact-changed-star-audit.declaration.v1','candidate_nha':{'path':str(cand_nha.resolve()),'sha256':sha(cand_nha)},'candidate_receipt_sha256':sha(CAND/'final/resting-anatomy-receipt.json'),'candidate_manifest_sha256':sha(CAND/'final/resting-anatomy-manifest.json'),'candidate_config_sha256':sha(CAND/'final/resting-reference-respiration.json'),'prediction_run':{'path':str(PRED/'prediction-run.json'),'sha256':sha(PRED/'prediction-run.json')},'baseline_run':str(RUN),'baseline_nha':{'path':str(base_nha),'sha256':base_nha_sha},'steps':list(STEPS),'changed_face_scope':{str(k):sorted(v) for k,v in selected.items()},'scope':'Exact changed-face stars for source rows305-308 against all six lung/diaphragm owners, including self and changed-to-changed pairs; reconciles all115 prior 1113 unclassified cross events. Row310 is a recooked derived pleural proxy and is outside the six-owner 1096 lung/diaphragm predicate scope. Saved-state production-kernel point replay only; not a fresh native capture or full-surface scan.','tracked_inputs_before':before})
pose_summaries=[]
for s in STEPS:
 pose,actual_changed,actual_faces=build_pose(s)
 dm={sid:base.native_map(sid,dmap,pose) for sid in base.LOBES}; lm=base.native_lobe_maps(lineage,pose)
 d_ok=all(x['valid'] for x in dm.values()); l_ok=all(x['valid'] for x in lm.values())
 classifier_regression=None
 if s==0:
  fa,fb=next(iter(sorted(lm[(305,308)]['face_pairs'])))
  ar,bad_a=base.exact_records(pose[305]['v'],pose[305]['f'][[fa]])
  br,bad_b=base.exact_records(pose[308]['v'],pose[308]['f'][[fb]])
  if bad_a or bad_b or not ar or not br: raise RuntimeError('mapped-contact classifier regression fixture is degenerate')
  x=(ar[0][0],ar[0][1],ar[0][2],fa,ar[0][4]); y=(br[0][0],br[0][1],br[0][2],fb,br[0][4])
  pts=pred.triangle_intersection_points(x[0],y[0])
  if not pts or base.classify(305,308,x,y,pts,pose,dm,lm)!='exact_reciprocal_face': raise RuntimeError('native mapped-contact classifier regression failed')
  classifier_regression={'owners':[305,308],'face_rows':[fa,fb],'class':'exact_reciprocal_face','nonempty_exact_intersection':True}
 changed={sid:selected[sid] for sid in CHANGED}
 events=[]; pair_summaries=[]
 for aa in IDS:
  for bb in IDS:
   if bb<aa:continue
   if aa==bb and aa not in CHANGED:continue
   if aa!=bb and not changed.get(aa) and not changed.get(bb):continue
   ev,sm=scan_pair(pose,aa,bb,changed,dm,lm);events.extend(ev);pair_summaries.append({'owners':[aa,bb],**sm})
 idx={event_key(e):e for e in events}
 if len(idx)!=len(events):raise RuntimeError('duplicate event identities')
 ep=OUT/('step-%05d-events.jsonl.gz'%s)
 with gzip.open(ep,'wt',compresslevel=1) as f:
  for e in sorted(events,key=lambda q:(q['owners'],q['face_rows'])): f.write(json.dumps(e,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
 old=old_unallowed[s]; old_clear=[];old_resolved=[];old_still=[];missing_change=[];missing_pair=[]
 for e in old:
  key=event_key(e); own,fs=key
  if not any(fs[k] in changed_report_faces.get(own[k],set()) for k in (0,1)):missing_change.append([list(own),list(fs)])
  ce=idx.get(key)
  if ce is None:
   old_resolved.append({'owners':list(own),'face_rows':list(fs),'result':'no exact intersection on candidate replay'})
   if tuple(sorted(own)) not in {tuple(x['owners']) for x in pair_summaries}:missing_pair.append([list(own),list(fs)])
  elif ce['class'].startswith(('unclassified','unallowed')):old_still.append({'owners':list(own),'face_rows':list(fs),'candidate_class':ce['class']})
  else:old_clear.append({'owners':list(own),'face_rows':list(fs),'candidate_class':ce['class']})
 bad=[e for e in events if e['class'].startswith(('unclassified','unallowed'))]
 oldkeys={event_key(e) for e in old}; newbad=[e for e in bad if event_key(e) not in oldkeys]
 row={'accepted_step':s,'mapped_classifier_regression':classifier_regression,'all_D_maps_valid':d_ok,'all_L_L_maps_valid':l_ok,'D_map_source_face_counts':dmap_identity,'actual_kernel_changed_vertex_counts':{str(k):len(v) for k,v in actual_changed.items()},'runtime_changed_faces':{str(k):sorted(v) for k,v in actual_faces.items()},'scanned_changed_face_counts':{str(k):len(v) for k,v in changed.items()},'owner_pair_summaries':pair_summaries,'exact_changed_star_hit_count':len(events),'exact_unallowed_count':len(bad),'new_unallowed_count':len(newbad),'prior_1096_unclassified_count':len(old),'prior_event_coverage':{'all_touch_changed_faces':not missing_change,'all_owner_pair_scans_present':not missing_pair,'missing_changed_face_events':missing_change,'missing_scan_events':missing_pair},'prior_event_reconciliation':{'no_exact_intersection':old_resolved,'reclassified_registered_contact':old_clear,'still_unclassified_or_unallowed':old_still},'new_unallowed_events':newbad,'events_path':str(ep),'events_sha256':sha(ep)}
 pose_summaries.append(row);writej(OUT/('step-%05d-summary.json'%s),row)
 writej(OUT/'progress.json',{'completed_steps':[int(x['accepted_step']) for x in pose_summaries],'current_step':s,'current_unallowed_count':len(bad),'pose_summaries':pose_summaries})
 print('POSE',s,'hits',len(events),'unallowed',len(bad),'old',len(old),'resolved',len(old_resolved),'registered',len(old_clear),'maps',d_ok,l_ok,'elapsed_s',round(time.monotonic()-t0,1),flush=True)
after={str(x):sha(x) for x in tracked}
report={'schema':'numi.human.lung-composed-candidate-exact-changed-star-audit.v1','status':'complete_saved_state_production_kernel_replay','qualification':'Diagnostic saved-state production-Metal point prediction at eight already-accepted states. Not a new integrated native capture, full 6-owner scan, physiology qualification, or continuous-time proof.','candidate':{'NHA':{'path':str(cand_nha.resolve()),'sha256':sha(cand_nha)},'receipt_sha256':sha(CAND/'final/resting-anatomy-receipt.json'),'manifest_sha256':sha(CAND/'final/resting-anatomy-manifest.json'),'respiration_config_sha256':sha(CAND/'final/resting-reference-respiration.json'),'composition_report_sha256':sha(CAND/'composition-report.json')},'production_prediction':{'run_path':str(PRED),'run_sha256':sha(PRED/'prediction-run.json'),'source_points_sha256':sha(PRED/'combined-source-points.xyz-f32.bin'),'probe_sha256':sha(pi['probe_argv'][0]),'metallib_sha256':sha(pi['probe_argv'][1]),'states':pr['outputs']},'baseline':{'accepted_run':str(RUN),'NHA_path':str(base_nha),'NHA_sha256':base_nha_sha,'prior_1096_report':str(PRIOR/'report.json'),'prior_1096_report_sha256':sha(PRIOR/'report.json'),'prior_unclassified_total':115,'prior_event_files':[{'step':s,'path':str(PRIOR/('step-%06d-cross.jsonl.gz'%s)),'sha256':sha(PRIOR/('step-%06d-cross.jsonl.gz'%s))} for s in STEPS]},'maps':{'D_map_adapter':'pinned v8 source map; 47,343 D/lobe mapped triangles were independently byte-compared against all corresponding 1159 source triangles','D_map_source_triangle_identity':dmap_identity,'lobe_lineage_report':{'path':str(CAND/'current-reciprocal-map-report-v2.json'),'sha256':sha(CAND/'current-reciprocal-map-report-v2.json'),'face_map_sha256':lineage['map_sha256'],'edge_map_sha256':lineage['edge_sha256'],'explicit_pair_count':len(lineage['pairs'])}},'predicate':{'audit_driver':{'path':str(AUDIT),'sha256':sha(AUDIT)},'exact_module':{'path':str(base.__file__),'sha256':sha(base.__file__)}},'scope':{'changed_owner_rows':[305,306,307,308],'all_six_owners':list(IDS),'changed_star_scan':True,'changed_to_changed_pairs':True,'self_checks':True,'row310_recooked_but_outside_six_owner_audit':True,'all_115_old_events_touch_changed_faces':all(not x['prior_event_coverage']['missing_changed_face_events'] for x in pose_summaries),'all_115_old_pairs_scanned':all(not x['prior_event_coverage']['missing_scan_events'] for x in pose_summaries)},'poses':pose_summaries,'all_maps_valid_every_pose':all(x['all_D_maps_valid'] and x['all_L_L_maps_valid'] for x in pose_summaries),'all_115_prior_events_resolved_or_reclassified':all(not x['prior_event_reconciliation']['still_unclassified_or_unallowed'] for x in pose_summaries),'no_new_unallowed_contacts':all(x['new_unallowed_count']==0 for x in pose_summaries),'immutable_input_hashes_before':before,'immutable_input_hashes_after':after,'inputs_unchanged':before==after,'elapsed_seconds':time.monotonic()-t0}
writej(OUT/'report.json',report)
if before!=after:raise RuntimeError('an immutable input changed')
print('FINAL',json.dumps({'report':str(OUT/'report.json'),'sha256':sha(OUT/'report.json'),'maps_valid':report['all_maps_valid_every_pose'],'prior_events_clear':report['all_115_prior_events_resolved_or_reclassified'],'no_new_unallowed':report['no_new_unallowed_contacts'],'elapsed_s':report['elapsed_seconds']},sort_keys=True),flush=True)
