
"""Freeze jointly checked current skin/fat construction; no native physical admission."""
from pathlib import Path
import numpy as np,json,hashlib,importlib.util,datetime,os
H=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
s=importlib.util.spec_from_file_location('snapshot541',H/'transfer_surface_contractions_indexed_379_root.py');T=importlib.util.module_from_spec(s);s.loader.exec_module(T)
srp=H/'skin-canonical-fan-chain-538-root/receipt.json';frp=H/'fat-canonical-current-target-540-root/receipt.json';sr=read(srp);fr=read(frp)
assert sr['status']=='PASS_CANONICAL_FIXED_BOUNDARY_SKIN_FAN_CURRENT_VOLUME_AND_ANCESTRY'
assert fr['status']=='PASS_CANONICAL_CURRENT_FAT_TARGET_METADATA_AND_PARENT_UNIONS'
assert not sr['physical_admission'] and not fr['physical_admission'] and sr['native_steps']==fr['native_steps']==0
context=Path(fr['context']['path']);assert sha(context)==fr['context']['sha256'];ar=read(context);assert ar['status']=='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
assert ar['changed_OID_joint_self_pairs']['count']==ar['changed_vs_complete_retained_OID_both_walls_all185']['count']==0
sources=fr['source_surfaces']
for role,e in sources.items():assert sha(e['path'])==e['sha256']
geoms={}
for role,r,rp,bad in [('Skin',sr,srp,988),('Fat',fr,frp,14366)]:
 p=Path(r['candidate']);assert sha(p)==r['candidate_sha256']
 with np.load(p) as z:
  xyz=z['points_world_m'];t=z['tetrahedra'];mat=z['material_indices'];bd=T.boundary(t);target=z['desired_boundary_faces']
  assert np.array_equal(xyz.view('u4'),z['points_world_m_f32'].view('u4'))
  assert np.array_equal(T.boundary_identity(bd),T.boundary_identity(target))
  d,q=T.m.measurements(xyz,t);assert np.all(d>0) and int((~T.m.passes(d,q)).sum())==bad
  assert len(z['output_cell_to_input_cell'])==len(t)
  for x in ('I','O'):assert str(z['typed_'+x+'_candidate_sha256'][0])==sources[x]['sha256']
 geoms[role]={**pin(p),'nodes':len(xyz),'cells':len(t),'native_quality_failures':bad,'nonpositive':0,'complete_target_boundary':True,'target_boundary_faces':len(bd),'missing_target_faces':0,'extra_actual_faces':0,'all_point_coordinate_views_synchronized':True,'qualification':pin(rp),'mass_mapping':'COMPLETE_CURRENT_SOURCE330_REBIND_REQUIRED','source_geometry':sources}
mp=H.parent/'fractional-semantic-support-090-sparse-graph/fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001/attempt-061-exact-volume-060-source-exhaustion/verified-chain-manifest.json'
mr=read(mp);assert sha(mp)=='039188096d9234fbc1340c5cf5404af28569e1c2fde49ab60e6316d1e4bcc012' and not mr['complete_allocation']
last=mr['attempts'][-1];ip=Path(last['independent']['path']);assert sha(ip)==last['independent']['sha256'];assert read(ip)['status'].startswith('PASS')
cp=H/'checkpoint.json';before=cp.read_bytes();assert hashlib.sha256(before).hexdigest()=='594ff2ec173ee22e8d0f2987329499f19fa916d68a8347e7d46eb6c96c49adcd'
archive=H/'checkpoint-before-current541-root.json';assert not archive.exists();archive.write_bytes(before)
data=json.loads(before);state=data['current_verified_state_20261031'];state['Skin_current']=geoms['Skin'];state['Fat_current']=geoms['Fat'];state['source_surfaces']=sources
state['source_context_proofs']=[pin(context),pin(srp),pin(frp),fr['D697'],sr['chain_receipt']]
state['current406_partial_mass_reference']={'chain_receipt':pin(mp),'independent_final_receipt':last['independent'],'held_kg':mr['chain_totals_after_061']['transferred_mass_kg'],'unallocated_source_kg':mr['chain_totals_after_061']['unallocated_residual_kg'],'density_floor_deficit_kg':mr['chain_totals_after_061']['floor_deficit_kg'],'exact_density_overage_rows':0,'complete_allocation':False,'qualified_for_current_541_cells':False,'recipient_geometry':'Legacy current406 only; conservative current541 rebase and complete paired allocation required','M4_relative_diagnostic':'Original owner admission retained; 061 manifest relative-residual label under separate numerical audit before stronger precision claims'}
state.update(status='QUALIFIED_ZERO_CROSSING_REFERENCE_SURFACES_AND_CLOSED_LAYER_TARGETS_NATIVE_ADMISSION_INCOMPLETE',source_reference_pose_only=True,surface_crossings=0,back412_unchanged=True,source_mass_immutable=True,physical_admission=False,native_steps=0,motion_verification='NOT_RUN',mass_mapping='NOT_ADMITTED_COMPLETE_CURRENT_PAIRED_SOURCE330_ALLOCATION_REQUIRED')
state['remaining']=['Remove 988 Skin and 14366 Fat native-quality failures without changing physical thresholds.','Complete conservative source330 mass allocation and density checks on the current paired cells.','Keep at least12GiB available before native qualification.','Integrate accepted physical tissue state and verify16s then60s native motion, coupled physiology and rejected-step rollback.']
state['construction_current_closed_Fat']={'D480_operations':904,'D500_operations':1277,'I528_paired_operations':267,'comp0_kernel_fan_operations':487,'latest_native_quality_failures':14366,'geometric_target_faces':459940,'missing':0,'extra':0,'declared_D_added_point_count':1,'parent_exact_comp0_SAT_pairs':2315388,'full_parent_SAT_inherited_through_exact_cavity_replay':True,'native_thresholds_unchanged':True}
state['construction_current_closed_Skin']={'O506_operations':21,'O530_operations':14,'I528_paired_operations':267,'latest_fixed_boundary_inferred_points_added':30,'latest_native_quality_failures':988,'geometric_target_faces':391918,'missing':0,'extra':0,'native_thresholds_unchanged':True}
state['publication_main']='aed0a821ab90c7fc24679d2fdf8157c289b627d8';state['publication_scope']='Through current502; current541 pending scoped tested publication.'
state['latest_checkpoint_archive']=pin(archive);state['latest_update_source']=pin(Path(__file__));state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
fs=os.statvfs(H);avail=fs.f_bavail*fs.f_frsize;state['storage']={'available_bytes':avail,'qualification_floor_bytes':12*1024**3,'qualification_floor_met':avail>=12*1024**3}
state['active_future_proposals_excluded']=['Next D1–2mm quotient batch.','New paired I cavity fallback.','Current541 source mass mapping preparations.']
snap=H/'checkpoint-current541-root.json';assert not snap.exists();raw=(json.dumps(data,indent=2,sort_keys=True)+'\n').encode();snap.write_bytes(raw);tmp=cp.with_name('checkpoint541.tmp');assert not tmp.exists();tmp.write_bytes(raw);assert cp.read_bytes()==before;os.replace(tmp,cp)
receipt={'status':'PASS_CURRENT541_JOINT_CLOSED_PHYSICAL_LAYER_CONSTRUCTION_CHECKPOINT','checkpoint':pin(snap),'Skin':geoms['Skin'],'Fat':geoms['Fat'],'source_context':pin(context),'physical_admission':False,'native_steps':0,'mass_mapping':'INCOMPLETE_LEGACY406_REFERENCE_ONLY_CURRENT_REBASE_REQUIRED','storage':state['storage']}
out=H/'checkpoint-current541-root-receipt.json';assert not out.exists();out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
