
from pathlib import Path
import numpy as np,json,hashlib,subprocess,time
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
out=H/'fixed-fat-coordinate-views-496-root';out.mkdir(exist_ok=False);start=time.monotonic()
report=H/'fixed-fat-identity-LP-492-root/run001/report.json';r=json.loads(report.read_text())
raw=Path(r['candidate']);assert sha(raw)==r['candidate_sha256']=='a0eb62490b5f24a9cab6bead6cafc8a1dc1eda0b3b4e2e1f239ad5b5ca475269'
proof=H/'fixed-fat-identity-LP-492-root/independent487001/receipt.json'
assert sha(proof)=='b247c0151f9df3572d2b848040b010e083e4ee96cb23a1f50965ca6c02af373c'
assert json.loads(proof.read_text())['status']=='PASS_EXACT_FIXED_BOUNDARY_INTERIOR_LP_REPLAY'
with np.load(raw) as z:a={k:z[k] for k in z.files}
with np.load(r['input']) as z:assert np.array_equal(a['points_world_m_f32'].view(np.uint32),z['points_world_m_f32'].view(np.uint32))
a['points_world_m_f32']=a['points_world_m'].copy()
candidate=out/'fat-all-coordinate-views-volume-proposal.npz';np.savez_compressed(candidate,**a)
with np.load(candidate) as z:
 for k in a:assert np.array_equal(a[k],z[k])
 assert np.array_equal(z['points_world_m'],z['points_world_m_f32'])
r.update(candidate=str(candidate),candidate_sha256=sha(candidate),producer_sha256=sha(__file__),all_point_coordinate_views_synchronized=True,coordinate_view_reconciliation={'raw_report':str(report),'raw_report_sha256':sha(report),'raw_candidate':str(raw),'raw_candidate_sha256':sha(raw),'raw_independent_geometric_replay':str(proof),'raw_independent_sha256':sha(proof),'changed_fields':['points_world_m_f32'],'physical_positions_changed_by_reconciliation':False,'previous_shadow_view_stale':True,'future_constructor':str(H/'condition_fixed_fat_all_views_LP_494_root.py'),'future_constructor_sha256':sha(H/'condition_fixed_fat_all_views_LP_494_root.py')})
rp=out/'report.json';rp.write_text(json.dumps(r,indent=2)+'\n')
with (out/'verify495.log').open('w') as f:subprocess.run([PY,str(H/'verify_fixed_fat_all_views_LP_495_root.py'),'--report',str(rp),'--output',str(out/'independent495001')],stdout=f,stderr=subprocess.STDOUT,check=True)
vr=out/'independent495001/receipt.json';rr=json.loads(vr.read_text());assert rr['all_point_coordinate_views_synchronized']
result={'status':'PASS_REPLAY_AND_SYNCHRONIZED_F32_PHYSICAL_COORDINATE_VIEWS','physical_admission':False,'native_steps':0,'candidate':str(candidate),'candidate_sha256':sha(candidate),'report':str(rp),'report_sha256':sha(rp),'proof':str(vr),'proof_sha256':sha(vr),'bad_before':r['bad_before'],'bad_after':r['bad_after'],'interior_operations':r['accepted_operations'],'source330_mass_immutable':True,'boundary_unchanged':True,'inherited_target_mismatch_unchanged':True,'elapsed_s':time.monotonic()-start,'producer_sha256':sha(__file__)}
(out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
