"""Atomically record fully joint-audited D414/I466 current construction; no physical admission."""
from pathlib import Path
import os,json,hashlib,datetime
H=Path(__file__).resolve().parent
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
cp=H/'checkpoint.json';old=cp.read_bytes();data=json.loads(old);s=data['current_verified_state_20261031']
pr=H/'joint-D414-I466-target-475-root/receipt.json';r=json.loads(pr.read_text());assert r['status']=='PASS_JOINT_D414_I466_COMPLETE_CONTEXT_AND_CURRENT_FAT_TARGET'
fr=Path(r['Fat_target']['path']);f=json.loads(fr.read_text());fp=f['output_candidate'];assert sha(fp['path'])==fp['sha256']
sr=H/'paired-I-quality-exchange-466-root/qualification467001/skin-target369/receipt.json';skin=json.loads(sr.read_text());assert sha(skin['candidate'])==skin['candidate_sha256']
s['source_surfaces']={'O':r['source_O'],'I':r['typed_I'],'D':r['typed_D']};s['source_context_proofs']=[r['source_context']]
s['Skin_current']={'path':skin['candidate'],'sha256':skin['candidate_sha256'],'cells':None,'native_quality_failures':1135,'nonpositive':0,'complete_target_boundary':True,'target_boundary_faces':392556,'qualification':{'path':str(sr),'sha256':sha(sr)},'joint_context':r['source_context'],'mass_mapping':'COMPLETE_CURRENT_SOURCE330_REBIND_REQUIRED'}
with __import__('numpy').load(skin['candidate']) as z:s['Skin_current']['cells']=len(z['tetrahedra'])
s['Fat_current']={'path':fp['path'],'sha256':fp['sha256'],'cells':810991,'nodes':253223,'native_quality_failures':20443,'nonpositive':0,'complete_target_boundary':False,'target_boundary_faces':468526,'missing_target_faces':f['actual_vs_typed_target']['missing_oriented_records'],'extra_actual_faces':f['actual_vs_typed_target']['extra_oriented_records'],'qualification':{'path':str(pr),'sha256':sha(pr)},'independent_D_transfer':{'path':str(H/'d-shortedge-quotient-414-batch001-volume-transfer/run002/independent-replay-417.json'),'sha256':sha(H/'d-shortedge-quotient-414-batch001-volume-transfer/run002/independent-replay-417.json')},'I_volume_composition':{'path':str(H/'combined-D414-I466-volume-469-root/report.json'),'sha256':sha(H/'combined-D414-I466-volume-469-root/report.json')}}
s.pop('next_Fat_construction_candidate',None)
mass=H.parent/'fractional-semantic-support-090-sparse-graph/fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001/attempt-021-source-exhaustion-batch/independent-replay-012.json';m=json.loads(mass.read_text());assert m['status']=='PASS_INDEPENDENT_PARTIAL_CURRENT406_SOURCE_EXHAUSTION_BATCH_REPLAY'
s['current406_partial_mass_reference']={'complete_allocation':False,'qualified_for_current_475_cells':False,'receipt':{'path':str(mass),'sha256':sha(mass)},'held_kg':m['mass_kg']['all_flow_total'],'unallocated_source_kg':m['mass_kg']['source_residual_after'],'density_floor_deficit_kg':m['mass_kg']['lower_deficit_after'],'maximum_source_corner_ledger_closure_error_kg':m['metrics']['max_source_corner_residual_error_kg'],'actual_density_direct_volume_overages':{'count':m['metrics']['direct_volume_upper_overage_count'],'maximum_kg_m3':m['metrics']['direct_volume_upper_overage_max_kg_m3']},'recipient_geometry':'current406; must rebase to current475 after construction'}
s['construction_I_quality_exchange']={'new_retained_quality_failures':{'Skin':2,'Fat':10},'combined_quality_gain':191,'per_tissue_bad_count_nonincrease':True,'exact_combined_volume_delta':0,'native_thresholds_unchanged':True,'source330_mass_immutable':True}
s['mass_mapping']='NOT_ADMITTED_COMPLETE_CURRENT_SOURCE330_M_OVER_4_ALLOCATION_REQUIRED'
s['remaining']=['complete127missing/191extraFatboundarytransfer','remove1135Skinand20443Fatnative-qualityfailures','rebaseandcompleteconservativecurrent-cellsource330M/4massallocation','restore12GiBfreebeforenativequalification','nativeintegration,motionandaccepted-staterollbackqualification']
s['publication_main']='fe7d08599b12fcbbaff823907c71f5376a081350'
s['publication_scope']='ThroughO433-448,Skin452/455,D411/comp1; D414/I466/current406mass021 pending next publication.'
s['physical_admission']=False;s['native_steps']=0;s['motion_verification']='NOT_RUN';s['source_reference_pose_only']=True
backup=H/'checkpoint-before-joint475-476-root.json';assert not backup.exists();backup.write_bytes(old)
s['latest_checkpoint_archive']={'path':str(backup),'sha256':sha(backup)};s['latest_update_source']={'path':str(Path(__file__)),'sha256':sha(__file__)}
s['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();v=os.statvfs(H);s['storage']={'available_bytes':v.f_bavail*v.f_frsize,'qualification_floor_bytes':12*1024**3,'qualification_floor_met':v.f_bavail*v.f_frsize>=12*1024**3}
tmp=cp.with_name('checkpoint.json.476.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,cp)
proof={'status':'PASS_CURRENT_JOINT_SOURCE_AND_POSITIVE_VOLUME_CHECKPOINT','checkpoint_sha256':sha(cp),'current_Skin_native_bad':1135,'current_Fat_native_bad':20443,'source_crossings':0,'physical_admission':False,'native_steps':0,'backup_sha256':sha(backup),'producer_sha256':sha(__file__)}
(H/'checkpoint-joint475-476-root.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
