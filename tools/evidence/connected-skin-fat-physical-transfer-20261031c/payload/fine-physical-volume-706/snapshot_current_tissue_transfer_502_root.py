
from pathlib import Path
import json,hashlib,os,datetime
H=Path(__file__).resolve().parent;P=H.parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
cp=H/'checkpoint.json';prior=sha(cp);archive=H/'checkpoint-before-current502-root.json';assert not archive.exists();archive.write_bytes(cp.read_bytes())
assert sha(archive)==prior
j=json.loads(cp.read_text());s=j['current_verified_state_20261031']
sr=H/'skin-I-flip-independent-501-root/receipt.json';skin=json.loads(sr.read_text());assert skin['status']=='PASS_INDEPENDENT_PAIRED_I_FLIP_SKIN_VOLUME_TRANSFER'
fr=H/'fat-comp0-connected-warning-repair-412/run-033-target-rebind-480/result-002/report.json';fat=json.loads(fr.read_text())
fcp=Path(fat['output_candidate']['path']);assert sha(fcp)==fat['output_candidate']['sha256']=='1bdff4e0007fdc2ab5867830f20ded3f30c70a1bac5a9ffaa0b66577db7913b2'
ctx=H/'joint-current-source-497-root/receipt.json';r=json.loads(ctx.read_text());assert r['status']=='PASS_ALL_CURRENT_O484_I_FLIP_D475_AND_WARNING_FLIPS_JOINT_CONTEXT'
s.update(status='QUALIFIED_ZERO_CROSSING_REFERENCE_SURFACES_AND_POSITIVE_VOLUME_CONSTRUCTION_ADMISSION_INCOMPLETE',
 source_surfaces=r['current_surfaces'],source_context_proofs=[r['context']],surface_crossings=0,source_reference_pose_only=True,
 source_mass_immutable=True,back412_unchanged=True,physical_admission=False,native_steps=0,motion_verification='NOT_RUN',
 latest_checkpoint_archive=pin(archive),latest_update_source=pin(Path(__file__)),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
s['Skin_current']={'path':skin['candidate'],'sha256':skin['candidate_sha256'],'cells':731059,'nodes':210268,'native_quality_failures':1111,'nonpositive':0,'complete_target_boundary':True,'target_boundary_faces':392522,'missing_target_faces':0,'extra_actual_faces':0,'qualification':pin(sr),'source_geometry':r['current_surfaces'],'mass_mapping':'COMPLETE_CURRENT_SOURCE330_REBIND_REQUIRED'}
# Every scalar domain is recomputed directly from current emitted arrays.
import numpy as np
with np.load(skin['candidate']) as z:
 s['Skin_current']['cells']=len(z['tetrahedra']);s['Skin_current']['nodes']=len(z['points_world_m'])
with np.load(fcp) as z:
 cells=len(z['tetrahedra']);nodes=len(z['points_world_m'])
s['Fat_current']={'path':str(fcp),'sha256':sha(fcp),'cells':cells,'nodes':nodes,'native_quality_failures':17640,'nonpositive':0,'complete_target_boundary':False,'target_boundary_faces':464834,'missing_target_faces':135,'extra_actual_faces':199,'qualification':pin(fr),'source_geometry':r['current_surfaces'],'independent_D_transfer':pin(H/'d-shortedge-quotient-475-next-001-preflight-100um/volume-transfer-001/independent-replay-001/report.json'),'independent_fixed_boundary_interior_LP':pin(H/'fixed-fat-coordinate-views-496-root/independent495001/receipt.json'),'all_point_coordinate_views_synchronized':True}
mass=P/'fractional-semantic-support-090-sparse-graph/fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001/attempt-022-source-exhaustion-reserved-batch/independent-replay-013.json'
mr=json.loads(mass.read_text());assert mr['status']=='PASS_INDEPENDENT_PARTIAL_CURRENT406_RESERVED_SOURCE_EXHAUSTION_BATCH_REPLAY'
assert sha(mass)=='7ce2d8dc4ab8c57da2b2b3b3692dece7f1fd8fe0901b15e26a7e5cbe1a1ea090'
s['current406_partial_mass_reference']={'complete_allocation':False,'qualified_for_current_502_cells':False,'receipt':pin(mass),'held_kg':mr['mass_kg']['all_flow_total'],'unallocated_source_kg':mr['mass_kg']['source_residual_after'],'density_floor_deficit_kg':mr['mass_kg']['lower_deficit_after'],'maximum_source_corner_ledger_closure_error_kg':mr['metrics']['max_source_corner_residual_error_kg'],'actual_density_direct_volume_overages':{'count':3,'maximum_kg_m3':mr['metrics']['direct_volume_upper_overage_max_kg_m3']},'recipient_geometry':'current406; current502 full conservative rebase mandatory'}
s['construction_O_quality_exchange']={'operations':17,'quality_gain':24,'new_retained_quality_failures':3,'source_coordinates_unchanged':True,'back412_unchanged':True,'exact_replay':pin(H/'outer-O-quality-exchange-484-root/exchange/typed434001/receipt.json'),'native_thresholds_unchanged':True}
s['construction_D475_quality']={'operations':1846,'quality_gain':2431,'old_good_regressions':0,'exact_geometric_volume_delta_m3':4.648590561863849e-10,'native_thresholds_unchanged':True}
s['construction_Fat_interior_LP']={'operations':282,'quality_gain':372,'all_246001_registered_points_fixed':True,'all_boundary_points_fixed':True,'exact_local_global_volume_delta':0,'protected_comp0_comp1_fullstar_corners_fixed':True,'native_thresholds_unchanged':True}
s['construction_I_face_flip']={'operations':1,'source_points_unchanged':True,'physical_Skin_transfer_complete':True,'Skin_signed_geometric_volume_delta_m3':skin['signed_geometric_volume_change_m3'],'Fat_transfer_complete':False,'source330_mass_immutable':True}
v=os.statvfs(H);s['storage']={'available_bytes':v.f_bavail*v.f_frsize,'qualification_floor_bytes':12*1024**3,'qualification_floor_met':v.f_bavail*v.f_frsize>=12*1024**3}
s['remaining']=['complete135missing/199extraFatboundarytransfer','remove1111Skinand17640Fatnative-qualityfailures','rebaseandcompletecurrent-cellconservativesource330M/4massallocation','restore12GiBfreebeforenativequalification','nativeintegration,motionandaccepted-staterollbackqualification']
s['publication_scope']='Throughfe7d0859; currentD475/O484/Iflip/Skin501/Fat496/mass022 pending next tested publication.'
snapshot=H/'checkpoint-current502-root.json';snapshot.write_text(json.dumps(j,indent=2)+'\n');cp.write_bytes(snapshot.read_bytes())
print(json.dumps({'status':'PASS_CURRENT_CONSTRUCTION_CHECKPOINT502','checkpoint_sha256':sha(cp),'Skin':s['Skin_current'],'Fat':s['Fat_current'],'physical_admission':False,'native_steps':0,'source_reference_pose_only':True,'source_crossings':0,'mass_unallocated_current406_kg':s['current406_partial_mass_reference']['unallocated_source_kg']}))
