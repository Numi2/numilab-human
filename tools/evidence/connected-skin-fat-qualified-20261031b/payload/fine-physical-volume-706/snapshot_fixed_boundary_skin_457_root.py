"""Record independently replayed fixed-boundary Skin; retain pending Fat candidate separately."""
from pathlib import Path
import json,hashlib,datetime,os
H=Path(__file__).resolve().parent
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
cp=H/'checkpoint.json';old=cp.read_bytes();out=H/'checkpoint-fixed-skin457-root.json';backup=H/'checkpoint-before-fixed-skin457-root.json';assert not out.exists() and not backup.exists()
pr=H/'skin-interior-LP-455-root/independent456001/receipt.json';r=json.loads(pr.read_text());assert r['status']=='PASS_EXACT_FIXED_BOUNDARY_INTERIOR_LP_REPLAY';assert sha(r['candidate'])==r['candidate_sha256'];assert r['bad_after']==1149
data=json.loads(old);s=data['current_verified_state_20261031'];assert s['Skin_current']['sha256']=='7822cd0cd55c346d0c3005cc4f3675bc10f85d51c137893bb1b28aec594da54f'
previous=s['Skin_current'].copy();s['Skin_current'].update(path=r['candidate'],sha256=r['candidate_sha256'],native_quality_failures=1149,nonpositive=0,qualification={'path':str(pr),'sha256':sha(pr)},complete_target_boundary=True,target_boundary_faces=392858,source_surface_qualification=previous['qualification'],fixed_boundary_parent={'path':previous['path'],'sha256':previous['sha256']},mass_mapping='CURRENT_SOURCE330_REBIND_REQUIRED_AFTER_INTERIOR_RELOCATION')
source=H/'d-shortedge-quotient-411-batch-final/volume-transfer-414/run001/independent-415/independent-verification.json';f=json.loads(source.read_text());assert f['status'].startswith('PASS')
s['next_Fat_construction_candidate']={'path':str(H/'d-shortedge-quotient-411-batch-final/volume-transfer-414/run001/fat-D411-379-plus-comp1-run041-composed.npz'),'sha256':'14f06beaf20a2f78396585e7868df831cff4b9f27e3401e271d7643175073a77','native_quality_failures':24174,'nonpositive':0,'independent_replay':{'path':str(source),'sha256':sha(source)},'status':'POSITIVE_QUOTIENT_PLUS_COMP1_PENDING_FINAL_THREE_FLIP_BOUNDARY_TRANSFER','target_residual_before_three_flips':{'missing':117,'extra':181},'current_final_target_residual':'NOT_YET_REBOUND','physical_admission':False,'native_steps':0}
s['latest_checkpoint_archive']={'path':str(backup),'sha256':hashlib.sha256(old).hexdigest()};s['latest_update_source']={'path':str(Path(__file__)),'sha256':sha(__file__)};s['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
s['publication_scope']='Through89e0665; O433-448,comp1,D411physical construction,Skin452/455 pending next publication'
sv=os.statvfs(H);s['storage']={'available_bytes':sv.f_bavail*sv.f_frsize,'qualification_floor_bytes':12*1024**3,'qualification_floor_met':sv.f_bavail*sv.f_frsize>=12*1024**3}
backup.write_bytes(old);tmp=cp.with_name('checkpoint.json.457.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,cp)
receipt={'status':'PASS_CURRENT_FIXED_BOUNDARY_SKIN_CHECKPOINT','checkpoint_sha256':sha(cp),'backup_sha256':sha(backup),'skin_quality_failures':1149,'next_Fat_candidate_quality_failures':24174,'physical_admission':False,'native_steps':0,'producer_sha256':sha(__file__)};out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
