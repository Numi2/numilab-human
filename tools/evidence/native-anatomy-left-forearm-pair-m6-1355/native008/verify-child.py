#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,sys,time,pickle
from pathlib import Path
import numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
C=A/'forearm-shared-field-native-diagnostic-008';P=A/'forearm054-native-diagnostic-007-candidate060'
RUN=C/'baseline/native-run';BASE=P/'baseline/native-run'
T=C/'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
PT=P/'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
SOURCE=A/'left-fds-native-m6-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
MAP=Path('/Users/n/numi-human-fds132-m6-1350/attempt-061-step0-capture-remesh-diagnostic/candidate060-adapted-source-parent-map-v2.pkl')
H=A/'neck-back-retained-pose-audit-002/forward_all_positive_weights.py'
PUB=Path('/Users/n/numi-human-left-pq-publication-1344/src')
PREP=C/'preparation.json';DECL=C/'baseline/run-declaration.json';RECEIPT=C/'resting-anatomy-receipt.json'
STEPS=[0,3807,4767,5023,5599,6207,7423,8000];COMMON=set(STEPS)
PARENT_SOURCE_SHA='21c3f535ba2a301110855ab18a0c0bd24fcbc1f2ff7ce97802bb543160b44103'
CURRENT_SOURCE_SHA='746c5f27f2d19929259fc14df124c0c12a0d796470a3d068af5e42d8476e9057'
CAPTURE_IDS={132:51005,134:51005}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def json_default(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,Path):return str(x)
 raise TypeError(f'not JSON serializable: {type(x).__name__}')
def volume_signs(xyz,faces,components):
 out=[]
 for comp in components:
  fc=faces[np.asarray(comp,dtype=np.int64)];used=np.unique(fc)
  tri=xyz.astype(np.float64)[fc]-xyz.astype(np.float64)[used].mean(axis=0,dtype=np.float64)
  v=float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6.)
  out.append(v)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args()
 tag='native-audit-progress-001' if args.partial else 'native-audit-001';OUT=C/tag;OUT.mkdir(exist_ok=False)
 report={'schema':'numi.human.left-fds-fdp-shared-field-008-native-diagnostic-audit.v1','mode':'partial_0_3807' if args.partial else 'final_8_capture_grid',
  'scope':'Experimental shared-field diagnostic only; no anatomical admission or all25 qualification. All eight native 008 captures are compared against corresponding accepted 007 candidate060 captures; step3807 remains diagnostic evidence, not production qualification.',
  'checks':{},'captures':{},'pins':{},'failures':[],'complete':False}
 RP=OUT/'report.json'
 def save(): RP.write_text(json.dumps(report,indent=2,sort_keys=True,default=json_default)+'\n')
 def check(name,ok,**fields):
  report['checks'][name]=dict(passed=bool(ok),**fields)
  if not ok:report['failures'].append(name)
  save()
 def req(x,msg):
  if not x:raise RuntimeError(msg)
 sys.path.insert(0,str(PUB))
 from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca,passive_attachment_composition as pc
 from numilab_human.cardiac_cavity_geometry import analyze_topology
 h=module('forearm054_native007_forward',H)
 prep=json.loads(PREP.read_text());decl=json.loads(DECL.read_text());receipt=json.loads(RECEIPT.read_text())
 parent_decl=json.loads((P/'baseline/run-declaration.json').read_text())
 ex_path=C/'baseline/execution.json';inv_path=RUN/'invocation.json';meta_path=RUN/'run-metadata.json'
 ex=json.loads(ex_path.read_text());inv=json.loads(inv_path.read_text());meta=json.loads(meta_path.read_text())
 immutable_bad=[]
 for name,digest in decl['immutable_assets'].items():
  p=Path(name);observed=sha(p) if p.is_file() else None
  if observed!=digest:immutable_bad.append({'path':name,'expected':digest,'observed':observed})
 prep_bad=[]
 for name,digest in prep.get('input_sha256',{}).items():
  p=Path(name);observed=sha(p) if p.is_file() else None
  if observed!=digest:prep_bad.append({'path':name,'expected':digest,'observed':observed})
 check('declaration_immutable_asset_pins',not immutable_bad,pin_count=len(decl['immutable_assets']),mismatches=immutable_bad)
 check('preparation_input_pins',not prep_bad,pin_count=len(prep.get('input_sha256',{})),mismatches=prep_bad)
 decl_hash=sha(DECL)
 check('prepared_declaration_and_capture_grid',prep.get('run_declaration_sha256')==decl_hash and decl.get('capture_steps')==STEPS and 'NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS=0,3807,4767,5023,5599,6207,7423,8000' in decl['argv'],
       preparation_declared_hash=prep.get('run_declaration_sha256'),observed_decl_hash=decl_hash,capture_steps=decl.get('capture_steps'))
 check('native_invocation_matches_prepared_owner_preview',inv.get('argv')==decl['owner_cli_preview']['native_argv'] and inv.get('environment')==decl['owner_cli_preview']['recorded_invocation_environment'] and ex.get('native_argv_matches_prepared_cli_preview') is True and ex.get('native_environment_matches_prepared_cli_preview') is True and not ex.get('changed_inputs') and not meta.get('source_files_changed_during_run'),
       argv_matches=ex.get('native_argv_matches_prepared_cli_preview'),environment_matches=ex.get('native_environment_matches_prepared_cli_preview'),changed_inputs=ex.get('changed_inputs'),source_files_changed=meta.get('source_files_changed_during_run'),returncode=ex.get('returncode'),timed_out=ex.get('timed_out'))
 # Verify the two possible source payloads and the unchanged raw source-row tables directly.
 check('source_payload_identity_and_raw_rows',sha(T)==CURRENT_SOURCE_SHA and sha(PT)==CURRENT_SOURCE_SHA and sha(SOURCE)==PARENT_SOURCE_SHA and receipt['passive_tissue_parent_bindings']['source_payloads']==[{'path':'assets/shared-field-carriers.nhtissue','sha256':'7f99dc0c1e4a390543b3c880a898330741f615d14fa30bb21d8e2fb1532d4a9b'}],
       candidate_tiss_sha256=sha(T),parent_007_tiss_sha256=sha(PT),original_map_source_tiss_sha256=sha(SOURCE),bound_shared_field_carrier_payloads=receipt['passive_tissue_parent_bindings']['source_payloads'])
 current=pc._read_nhtiss4(PT);old=pc._read_nhtiss4(SOURCE)
 rows_equal={}
 for sid in (132,134):
  a=next(r for r in current['records'] if int(r[6])==sid);b=next(r for r in old['records'] if int(r[6])==sid)
  rows_equal[str(sid)]=pc._row_local_equal(pc._row_slices(current,a),pc._row_slices(old,b))
 check('raw_132_134_source_vertex_and_binding_rows_exact',all(rows_equal.values()),rows_equal=rows_equal)
 # NHTISS non-target source records should remain exact, independent of the native surface readback.
 cand=pc._read_nhtiss4(T);child_tissue=h.load_tissue(T);unchanged=[];changed=[]
 for pr in current['records']:
  sid=int(pr[6]);cr=[r for r in cand['records'] if int(r[6])==sid]
  if len(cr)!=1:changed.append({'stable_id':sid,'reason':'missing_or_duplicate'});continue
  if sid in (132,134):continue
  if pc._row_local_equal(pc._row_slices(current,pr),pc._row_slices(cand,cr[0])):unchanged.append(sid)
  else:changed.append({'stable_id':sid,'reason':'payload_row_diff'})
 check('all_non_target_nhtiss_rows_exact',len(unchanged)==len(current['records'])-2 and not changed,exact_row_count=len(unchanged),expected=len(current['records'])-2,differences=changed[:30])
 base_trace_path=BASE/'resting-coupled.csv';run_trace_path=RUN/'resting-coupled.csv'
 with run_trace_path.open(newline='') as f:trace=list(csv.DictReader(f))
 with base_trace_path.open(newline='') as f:base_trace=list(csv.DictReader(f))
 prefix=len(trace)<=len(base_trace) and trace==base_trace[:len(trace)]
 check('physiology_trace_is_exact_parent_prefix',prefix,current_rows=len(trace),parent_rows=len(base_trace),full_trace_equal=trace==base_trace,last_current_time_s=float(trace[-1]['time_s']) if trace else None,last_current_step=int(float(trace[-1]['step'])) if trace else None)
 # Source map terms and canonical seam IDs must be shared across both output children.
 source_map=pickle.loads(MAP.read_bytes());surface_map=source_map['surfaces'];sm={int(k):v for k,v in surface_map.items()}
 gids={sid:list(map(int,sm[sid]['global_vertex_ids'])) for sid in (132,134)}
 gix={sid:{g:i for i,g in enumerate(gids[sid])} for sid in (132,134)}
 shared=sorted(set(gids[132])&set(gids[134]))
 canonical=True
 for g in shared:
  if sm[132]['source_parent_terms_exact'][gix[132][g]]!=sm[134]['source_parent_terms_exact'][gix[134][g]]:canonical=False;break
 check('canonical_parent_map_shared_vertex_identity',canonical and len(shared)==1986,shared_global_ids=len(shared),terms_identical=canonical,map_sha256=sha(MAP))
 # Capture reads use the same owner parser for native and parent packs.
 keys,*_=ca._load_target_inventory(h.INV);allkeys=set(keys)|{(51005,s) for s in (151,152,159,160,161,162,163,164)}
 expected_inventory=allkeys|{(51007,1)}
 target_keys={(51005,132),(51005,134)}
 expected_steps=STEPS[:2] if args.partial else STEPS
 existing=[];missing=[]
 for step in expected_steps:
  pack=RUN/'accepted-geometry'/f'step-{step}.mrvpack';rcpt=pack.with_suffix('.receipt.json')
  if not pack.is_file() or not rcpt.is_file():missing.append(step);continue
  existing.append(step)
 check('requested_capture_availability',not missing and (args.partial or existing==STEPS),requested_steps=expected_steps,available_steps=existing,missing_steps=missing)
 for step in existing:
  pack=RUN/'accepted-geometry'/f'step-{step}.mrvpack';rcpt=pack.with_suffix('.receipt.json');pinfo=json.loads(rcpt.read_text())
  parent_pack=BASE/'accepted-geometry'/f'step-{step}.mrvpack';parent_rcpt=parent_pack.with_suffix('.receipt.json')
  pos,surfs,pmeta=ca._pack_surfaces(pack,allkeys)
  inventory_missing=sorted(expected_inventory-set(surfs));inventory_extra=sorted(set(surfs)-expected_inventory)
  inventory_ok=set(surfs)==expected_inventory and len(surfs)==868
  pose_list=pinfo.get('accepted_body_poses',[])
  pose_ok=len(pose_list)==157
  pose_equal=None;unmodified_exact=None;unmodified_count=0;unmodified_bad=[]
  parent_surface_count=None
  if step in COMMON and parent_pack.is_file() and parent_rcpt.is_file():
   bp=json.loads(parent_rcpt.read_text());bpos,bsurfs,bmeta=ca._pack_surfaces(parent_pack,allkeys)
   parent_surface_count=len(bsurfs);pose_equal=pose_list==bp.get('accepted_body_poses',[])
   untouched=set(surfs)-target_keys
   for key in untouched:
    if key not in bsurfs:unmodified_bad.append({'key':list(key),'reason':'missing_parent_surface'});continue
    fa=np.asarray(surfs[key]['faces'],dtype=np.int64);fb=np.asarray(bsurfs[key]['faces'],dtype=np.int64)
    if fa.shape!=fb.shape or not np.array_equal(pos[fa],bpos[fb]):unmodified_bad.append({'key':list(key),'reason':'geometry_mismatch'})
   unmodified_count=len(untouched)-len(unmodified_bad);unmodified_exact=(len(untouched)==866 and len(bsurfs)==868 and not unmodified_bad)
  entry={'pack_path':str(pack),'pack_sha256':sha(pack),'receipt_path':str(rcpt),'receipt_sha256':sha(rcpt),'native_inventory_count':len(surfs),'requested_key_count':len(allkeys),'expected_inventory_count':len(expected_inventory),'inventory_missing_keys':[list(x) for x in inventory_missing],'inventory_extra_keys':[list(x) for x in inventory_extra],'inventory_ok':inventory_ok,
         'accepted_body_pose_count':len(pose_list),'poses_equal_parent':pose_equal,'unmodified_surface_count':unmodified_count,'unmodified_surfaces_exact_parent':unmodified_exact,'unmodified_mismatches':unmodified_bad[:30]}
  # Confirm native pack endpoint and payload identity before geometry checks.
  entry['capture_receipt_valid']=pinfo.get('physical_endpoint')=='accepted' and pinfo.get('pack_file_sha256')==sha(pack)
  row_results={};world_by_sid={}
  for sid in (132,134):
   key=(51005,sid);rowrec=next(r for r in cand['records'] if int(r[6])==sid);row=pc._row_slices(cand,rowrec)
   local_faces=np.asarray(row['local_faces'],dtype=np.int64)
   nrow=h.row_data(child_tissue,sid)
   native_face=np.asarray(surfs[key]['faces'],dtype=np.int64)
   offset_ok=native_face.shape==local_faces.shape and native_face.size>0 and np.all((native_face-local_faces)==(native_face-local_faces).flat[0])
   start=int((native_face-local_faces).flat[0]) if offset_ok else -1
   if offset_ok:
    world=pos[start:start+len(gids[sid])].astype('<f4',copy=True);world_by_sid[sid]=world
    topo=analyze_topology(world.astype(float).tolist(),local_faces.tolist())
    rec,dg=h.exact_rows(world,local_faces,ci)
    if rec is None or dg:selfcount=None;pairlist=[]
    else:
     audit=ci._audit_pair(rec,rec,same_surface=True);selfcount=int(audit['count']);pairlist=audit.get('triangle_pairs',[])
    component_volumes=volume_signs(world,local_faces,topo.get('face_components',[]))
    restxyz=np.asarray([[np.float32(float(x)) for x in rowxyz] for rowxyz in sm[sid]['rest_vertices_exact_m']],dtype='<f4')
    restvol=volume_signs(restxyz,local_faces,topo.get('face_components',[]))
    signs=lambda vs:['positive' if v>0 else 'negative' if v<0 else 'zero' for v in vs]
    signs_ok=len(component_volumes)==len(restvol) and signs(component_volumes)==signs(restvol) and all(v!=0 for v in component_volumes)
    topology_ok=topo.get('closed_oriented_manifold_candidate') is True and len(topo.get('face_components',[]))==({132:1,134:4}[sid])
    geometry_ok=topology_ok and not dg and selfcount==0 and signs_ok
    row_results[str(sid)]={'vertex_count':len(world),'triangle_count':len(local_faces),'offset':start,'face_order_matches_candidate':offset_ok,
      'closed_oriented_manifold':topology_ok,'topology':topo,'degenerate_face_count':len(dg),'self_pair_count':selfcount,'self_pairs':[list(map(int,x)) for x in pairlist[:20]],
      'component_signed_volumes_m3':component_volumes,'rest_component_signed_volumes_m3':restvol,'component_signs':signs(component_volumes),'rest_component_signs':signs(restvol),'winding_matches_rest':signs_ok,'geometry_pass':geometry_ok}
   else:row_results[str(sid)]={'face_order_matches_candidate':False,'geometry_pass':False,'reason':'native face index layout does not match candidate row'}
  seam_ok=False;seam_exact=0;max_gap=None
  if set(world_by_sid)=={132,134}:
   aa=world_by_sid[132][[gix[132][g] for g in shared]];bb=world_by_sid[134][[gix[134][g] for g in shared]]
   seam_exact=int(np.count_nonzero(np.all(aa==bb,axis=1)));gaps=np.linalg.norm(aa.astype(np.float64)-bb.astype(np.float64),axis=1)*1e6
   max_gap=float(gaps.max()) if len(gaps) else None;seam_ok=len(shared)==1986 and seam_exact==1986
  entry.update({'capture_receipt_valid':entry['capture_receipt_valid'],'step0_or_step3807_standalone':step not in COMMON,
    'poses_157':pose_ok,'surface_inventory_868':inventory_ok,'seam_vertices':len(shared),'seam_bitwise_equal_count':seam_exact,'seam_max_gap_um':max_gap,'seam_pass':seam_ok,'rows':row_results})
  required=entry['capture_receipt_valid'] and pose_ok and inventory_ok and seam_ok and all(x.get('geometry_pass') for x in row_results.values())
  if step in COMMON:required=required and pose_equal is True and unmodified_exact is True
  report['captures'][str(step)]=entry;check(f'capture_step_{step}_local_and_parent_checks' if step in COMMON else f'capture_step_{step}_standalone_local_checks',required,**entry)
 # Stable native execution and exact common anatomy state are distinct from failure-free full duration.
 full=ex.get('returncode')==0 and not ex.get('timed_out') and meta.get('exit_code')==0 and len(trace)==1000 and trace==base_trace and existing==STEPS
 report['execution']={'execution_json':str(ex_path),'execution_sha256':sha(ex_path),'returncode':ex.get('returncode'),'wall_seconds':ex.get('wall_seconds'),
   'native_log_sha256':sha(RUN/'native.log'),'run_metadata_sha256':sha(meta_path),'accepted_trace_rows':len(trace),'parent_trace_rows':len(base_trace),'full_16s_native_run':full}
 report['available_capture_steps']=existing;report['missing_final_grid_steps']=[s for s in STEPS if s not in existing]
 report['all_capture_local_checks_passed']=all(v['passed'] for k,v in report['checks'].items() if k.startswith('capture_step_'))
 report['complete']=not args.partial or len(existing)==2
 report['status']='experimental_shared_field_008_full_grid_verified' if full and not report['failures'] else ('partial_native_audit' if args.partial and not report['failures'] else 'native_diagnostic_incomplete_or_failed')
 report['pins'].update({str(p):sha(p) for p in (PREP,DECL,RECEIPT,T,T.with_suffix('.manifest.json'),PT,PT.with_suffix('.manifest.json'),SOURCE,MAP,ex_path,inv_path,meta_path,RUN/'native.log',run_trace_path,base_trace_path) if p.is_file()})
 report['pins'].update({str(p):sha(p) for step in existing for p in (BASE/'accepted-geometry'/f'step-{step}.mrvpack',BASE/'accepted-geometry'/f'step-{step}.receipt.json') if p.is_file()})
 save();print(json.dumps({'report':str(RP),'status':report['status'],'failed_checks':report['failures'],'available_steps':existing,'full_native_run':full},indent=2),flush=True)

main()
