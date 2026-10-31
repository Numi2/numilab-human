
"""Freeze joint reference surfaces with closed physical layer targets; native admission remains false."""
from pathlib import Path
import json,hashlib,datetime,os,importlib.util
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def read(p,want):
 assert sha(p)==want,(p,want,sha(p))
 return json.loads(p.read_text())
s=importlib.util.spec_from_file_location('snap523',H/'transfer_surface_contractions_indexed_379_root.py');T=importlib.util.module_from_spec(s);s.loader.exec_module(T)
canonical=H/'fat-comp0-connected-warning-repair-412/run-042-current-54381-normalized-rebind/target-normalization-receipt.json'
fr=read(canonical,'0e5cbe3c412de50e99ae3f5955784db7d97a515cc931d8d89a62d22faa65b5ee')
assert fr['status']=='PASS_METADATA_ONLY_CURRENT_TARGET_NORMALIZATION_AND_EXACT_F32_CROSSWALK'
fat=Path(fr['output_candidate_path']);assert sha(fat)==fr['output_candidate_sha256']
sp=H/'skin-all-views-interior-LP-515-root/receipt.json';sr=json.loads(sp.read_text());assert sr['status'].startswith('PASS_CURRENT_SKIN_LP')
skin=Path(sr['candidate']);assert sha(skin)==sr['candidate_sha256']=='bbc30d8efe7538694df629a1a739365dd13ff3d85dfe38984bf4d511260bb274'
root=H/'d-shortedge-quotient-480-next-001-preflight-100um/volume-transfer-001'
context=root/'comp0-boundary-split-001/exact-audit-197-005/report.json'
cr=read(context,'083c88aaae74455a9c5084f4097cd942bb8031d743e2cb34cd10e269463b42f4')
assert cr['status']=='PASS_CHANGED_O_D_VS_BASELINE_ALL185_WALLS_AND_APEX'
lineage=root/'comp0-boundary-split-001/metadata-lineage-correction-002/metadata-lineage-correction-receipt.json'
lr=read(lineage,'146d3c38065b5254f5fb2d9efbbaf35f92664bd3c3bf9ee77eb50fcb5b7e0625')
dproof=root/'independent-replay-005/report.json'
dr=read(dproof,'fc08155a99940beb6b09ced5f26f103e2db1cab7e3dec1a8fe59e24f21651ded');assert dr['status'].startswith('PASS')
mass=P/'fractional-semantic-support-090-sparse-graph/fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001/attempt-058-exact-nominal-upper-m4-band-exchange/run-001/independent-replay/independent-replay.json'
mr=read(mass,'5432e43b2c7c4ca1dcbdb1295e46173146a7740e4c8079a447fe8b9739a7df79');assert mr['status']=='PASS_INDEPENDENT_FULL_LEDGER_REPLAY_PARTIAL_EXCHANGE'
proofs=[pin(context),pin(lineage),pin(dproof),pin(canonical),pin(sp)]
geoms={}
for role,p,bad in [('Skin',skin,1060),('Fat',fat,16929)]:
 with np.load(p) as z:v={k:z[k] for k in z.files}
 pp=v['points_world_m'];t=v['tetrahedra'];b=T.boundary(t);d,q=T.m.measurements(pp,t)
 assert np.all(d>0) and int((~T.m.passes(d,q)).sum())==bad
 assert np.array_equal(pp.view('u4'),v['points_world_m_f32'].view('u4'))
 assert np.array_equal(T.boundary_identity(b),T.boundary_identity(v['desired_boundary_faces']))
 geoms[role]={**pin(p),'nodes':len(pp),'cells':len(t),'native_quality_failures':bad,'nonpositive':0,'complete_target_boundary':True,'target_boundary_faces':len(b),'missing_target_faces':0,'extra_actual_faces':0,'all_point_coordinate_views_synchronized':True,'qualification':pin(sp if role=='Skin' else canonical),'mass_mapping':'COMPLETE_CURRENT_SOURCE330_REBIND_REQUIRED'}
sources={}
for role,e in fr['role_pins'].items():
 p=Path(e['path']);assert sha(p)==e['sha256']
 sources[role]={'path':str(p),'sha256':e['sha256']}
geoms['Skin']['source_geometry']=sources;geoms['Fat']['source_geometry']=sources
cp=H/'checkpoint.json';archive=H/'checkpoint-before-current523-root.json';assert not archive.exists();before=cp.read_bytes();archive.write_bytes(before)
data=json.loads(before);state=data['current_verified_state_20261031']
state['Skin_current']=geoms['Skin'];state['Fat_current']=geoms['Fat'];state['source_surfaces']=sources
state['source_context_proofs']=proofs
state['current406_partial_mass_reference']={'receipt':pin(mass),'held_kg':mr['mass_kg']['merged_flow_total'],'unallocated_source_kg':mr['mass_kg']['unallocated_source_residual'],'density_floor_deficit_kg':mr['mass_kg']['global_floor_deficit_after'],'exact_density_overage_rows':0,'density_evaluator':'existing081 exact dyadic Float32 determinant','complete_allocation':False,'qualified_for_current_523_cells':False,'recipient_geometry':'current406; conservative current523 rebase mandatory'}
state.update(surface_crossings=0,source_reference_pose_only=True,back412_unchanged=True,source_mass_immutable=True,physical_admission=False,native_steps=0,motion_verification='NOT_RUN',mass_mapping='NOT_ADMITTED_COMPLETE_CURRENT_SOURCE330_M_OVER_4_ALLOCATION_REQUIRED')
state['status']='QUALIFIED_ZERO_CROSSING_REFERENCE_SURFACES_AND_COMPLETE_CLOSED_LAYER_TARGETS_NATIVE_ADMISSION_INCOMPLETE'
state['remaining']=['remove1060Skinand16929Fatnative-qualityfailures','rebaseandcompletecurrent-cellconservativesource330M/4massallocation','restore12GiBfreebeforenativequalification','nativeintegration,motionandaccepted-staterollbackqualification']
state['publication_main']='aed0a821ab90c7fc24679d2fdf8157c289b627d8';state['publication_scope']='Throughcurrent502; closedcurrent523andlaterworkerproposalsnotyetpublished'
state['latest_checkpoint_archive']=pin(archive);state['latest_update_source']=pin(Path(__file__))
state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
st=os.statvfs(H);available=st.f_bavail*st.f_frsize
state['storage']={'available_bytes':available,'qualification_floor_bytes':12*1024**3,'qualification_floor_met':available>=12*1024**3}
state['construction_current_closed_Fat']={'D480_operations':904,'native_quality_before':17640,'D480_after':16668,'comp0_reused_mesh_quality_debt':739,'local_connectivity_quality_gain':478,'geometric_target_faces':463028,'missing':0,'extra':0,'declared_D_added_point_count':1,'exact_comp0_SAT_parent_pairs':2315388,'native_thresholds_unchanged':True}
data['current_verified_state_20261031']=state
snapshot=H/'checkpoint-current523-root.json';assert not snapshot.exists();raw=(json.dumps(data,indent=2,sort_keys=True)+'\n').encode();snapshot.write_bytes(raw)
tmp=cp.with_name('checkpoint523.tmp');assert not tmp.exists();tmp.write_bytes(raw);assert cp.read_bytes()==before;os.replace(tmp,cp)
res={'status':'PASS_CURRENT523_CLOSED_PHYSICAL_LAYER_CONSTRUCTION_CHECKPOINT','checkpoint':pin(snapshot),'Skin':geoms['Skin'],'Fat':geoms['Fat'],'physical_admission':False,'native_steps':0,'mass_current406_only':state['current406_partial_mass_reference'],'active_future_fan_and_D500_proposals_excluded':True}
rp=H/'checkpoint-current523-root-receipt.json';assert not rp.exists();rp.write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps({k:v for k,v in res.items() if k not in ('Skin','Fat')}),flush=True)
