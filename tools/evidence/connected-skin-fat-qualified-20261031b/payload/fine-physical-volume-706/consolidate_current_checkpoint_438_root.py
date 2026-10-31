"""Consolidate the existing checkpoint's current state; preserve all previous history."""
from pathlib import Path
import json,hashlib,time,shutil,os
H=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):
 p=Path(p);return {'path':str(p),'sha256':sha(p)}
cp=H/'checkpoint.json';backup=H/'checkpoint-before-current-state-consolidation-438-root.json'
assert not backup.exists();shutil.copy2(cp,backup);r=json.loads(cp.read_text())
prev=r['current_verified_state_20261031'];o=H/'outer-O-quotient-433-root'
oq=o/'qualification435001/receipt.json';qr=json.loads(oq.read_text())
assert qr['status']=='PASS_BOUNDED_OUTER_SKIN_TYPED_GEOMETRY_VOLUME_CONTEXT_AND_TARGET'
fr=H/'fat614-yy-volume-replay-004/run-041-D407-comp1-remesh-budget24'
fp=pin(fr/'connected-fixed-surface-volume-proposal.npz')
assert fp['sha256']=='e11249bd297a7e08ae8d74fe406439289dfaf2f1f3069d1ebf83876e3d170a15'
proof=pin(fr/'independent-full-audit-419.json')
assert proof['sha256']=='40314094d4eb66505236e4ec712f1528dbd63b83d0af40469f47134e1e5da6dd'
stat=os.statvfs(H)
s={'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 'status':'QUALIFIED_SOURCE_AND_POSITIVE_VOLUME_CONSTRUCTION_PHYSICAL_ADMISSION_INCOMPLETE',
 'previous_state_archive':pin(backup),
 'source_surfaces':{'O':pin(o/'typed434001/candidate-O-typed-current.npz'),
 'I':pin(H/'paired-I-interface-399-root/iteration002/typed365/candidate-I-typed-current.npz'),
 'D':pin(H/'fat614-comp1-target-D-refinement-407-004/candidate-D-comp1-current-target-refinement-typed.npz')},
 'surface_crossings':0,'source_reference_pose_only':True,'back412_unchanged':True,
 'source_context_proofs':[pin(o/'qualification435001/audit354/report.json'),pin(H/'fat614-comp1-target-D-refinement-407-004/exact-audit-001/report.json')],
 'Skin_current':dict(pin(qr['candidate']),native_quality_failures=1336,nonpositive=0,complete_target_boundary=True,target_boundary_faces=393098,qualification=pin(oq)),
 'Fat_current':dict(fp,native_quality_failures=28966,nonpositive=0,complete_target_boundary=False,missing_target_faces=117,extra_actual_faces=181,cells=826836,target_boundary_faces=477584,independent_volume_replay=proof),
 'construction_only_comp1_quality_debt':{'old':620,'new':644,'delta':24,'native_thresholds_unchanged':True},
 'source_mass_immutable':True,'mass_mapping':'NOT_ADMITTED_REBIND_AND_COMPLETE_M_OVER_4_ALLOCATION_REQUIRED',
 'current406_partial_mass_reference':prev['current406_partial_mass_reference'],
 'physical_admission':False,'native_steps':0,'motion_verification':'NOT_RUN',
 'publication_main':'89e0665ba9e41886df98ba082d45551cea67c8b5',
 'publication_scope':'392-406,421 and current406 partialmass; comp1/O433 pending next publication',
 'storage':{'available_bytes':stat.f_bavail*stat.f_frsize,'qualification_floor_bytes':12*1024**3,'qualification_floor_met':stat.f_bavail*stat.f_frsize>=12*1024**3},
 'remaining':['complete117missing/181extraFatboundary transfer','remove native-quality failures from both tissues','finish conservative source330 M/4 current-cell mass allocation','native integration, accepted-state motion and rollback qualification'],
 'producer':pin(__file__)}
r['current_verified_state_20261031']=s
r['updated_utc']=s['updated_utc'];r['updated_unix_s']=time.time()
r['historical_fields_notice']='Only current_verified_state_20261031 is authoritative; other top-level progress keys preserve historical checkpoints.'
cp.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
out=H/'checkpoint-consolidation-438-root.json';assert not out.exists()
out.write_text(json.dumps({'status':'PASS_SINGLE_CURRENT_STATE_PIN_CONSOLIDATION','checkpoint':pin(cp),'previous_archive':pin(backup),'physical_admission':False,'native_steps':0},indent=2)+'\n')
print(out.read_text())
