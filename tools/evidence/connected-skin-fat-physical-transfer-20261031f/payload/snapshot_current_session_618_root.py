"""Freeze already-verified geometry at session end; no simulation or new geometry."""
from pathlib import Path
import json,hashlib,datetime,os,shutil
H=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def main():
 cp=H/'checkpoint.json';prior=H/'checkpoint-current604-root.json'
 assert sha(cp)==sha(prior)=='07637bf441b37edadc446697ed69b0b1ebd83197bfdac91f1e26159dc2aea6ba'
 r=json.loads(cp.read_text());s=r['current_verified_state_20261031']
 prev=H/'checkpoint-before-current618-root.json';shutil.copyfile(cp,prev)
 ctx=H/'joint-current-O612-D597-context-613-root/audit354/report.json';assert sha(ctx)=='4f457bbbab1d858061ac141b383cff5648ff5d2080a2a39cc4b39853b6fdb73c'
 cr=json.loads(ctx.read_text());assert cr['status']=='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
 cfg=json.loads(Path(cr['config']['path']).read_text());surfaces=cfg['candidate_surfaces']
 sr=H/'outer-O-current612-canonical615-root/receipt.json';skin=json.loads(sr.read_text())
 fr=H/'fat-current-D-546-597-root/run011-target397-current/canonical-v3/receipt.json';fat=json.loads(fr.read_text())
 assert skin['status']=='PASS_CANONICAL_CURRENT_O612_IMMEDIATE_PARENT_UNIONS'
 assert fat['status']=='PASS_CANONICAL_O612_D597_I50_FAT_PARENT_UNIONS_SOURCE_ONLY'
 for role,rr,proof,cells,nodes,bad,target in [('Skin',skin,sr,729600,210298,873,391726),('Fat',fat,fr,788760,253712,10345,454314)]:
  e=s[role+'_current'];candidate=rr['candidate'];assert sha(candidate['path'])==candidate['sha256']
  e.update(candidate,cells=cells,nodes=nodes,native_quality_failures=bad,target_boundary_faces=target,nonpositive=0,
   current_global_surface_context=surfaces,qualification=pin(proof),mass_mapping='Complete source330 allocation to these exact cells remains NOT_RUN.',
   complete_target_boundary=True,missing_target_faces=0,extra_actual_faces=0,all_point_coordinate_views_synchronized=True)
  if role=='Fat':
   e['protected_construction_region_previous_census']=e.pop('protected_construction_region',None)
   e['protected_construction_region_previous_census']['current_row_domain']=False
   e['lineage_parent']=fat['canonical_immediate_parent']
 s.update(source_surfaces=surfaces,surface_crossings=0,physical_admission=False,native_steps=0,motion_verification='NOT_RUN',
  source_reference_pose_only=True,source_mass_immutable=True,back412_unchanged=True,
  latest_checkpoint_archive=pin(prev),latest_update_source=pin(Path(__file__)),
  updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  publication_main='442b854aea4e903971ae39bec0fef66617d797d0',
  publication_scope='Prior current604 package is on main; current618 geometry is frozen for session-end narrow publication.',
  mass_mapping='Current Skin615 P1/143 support independently replayed. Fat597 cache is older topology; final Fat canonical support/cell carriers and complete source330 allocation remain NOT_RUN.',
  remaining=['Remove 873 Skin and 10,345 Fat native-quality failures with unchanged native thresholds.',
   'Finish current typed authoring-source adapter without changing actual230 admission gates.',
   'Complete exact current-cell P1/143/carrier support and source330 allocation with independent source/corner/child replay.',
   'Restore at least 12 GiB available before native qualification.',
   'Import admitted tissues and verify 16 s then 60 s native motion and rollback.'],
  active_future_proposals_excluded=['Next D search frontier and next paired-I batch not run.',
   'Authoring adapter v9 incomplete and untested for parent62000200.',
   'No current paired mass allocation or native motion run.'])
 s['source_context_proofs']+=list(map(pin,[ctx,sr,fr]))
 st=os.statvfs(H);s['storage']={'available_bytes':st.f_bavail*st.f_frsize,'qualification_floor_bytes':12*1024**3,'qualification_floor_met':st.f_bavail*st.f_frsize>=12*1024**3}
 s['current406_partial_mass_reference']['qualified_for_current_618_cells']=False
 frozen=H/'checkpoint-current618-root.json';frozen.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
 shutil.copyfile(frozen,cp);assert sha(frozen)==sha(cp)
 print(json.dumps({'checkpoint':pin(frozen),'Skin_quality_failures':873,'Fat_quality_failures':10345,'physical_admission':False,'native_steps':0}))
if __name__=='__main__':main()
