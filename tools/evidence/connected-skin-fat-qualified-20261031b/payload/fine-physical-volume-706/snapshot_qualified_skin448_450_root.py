from pathlib import Path
import json,hashlib,time,shutil
H=Path(__file__).resolve().parent
def pin(p):
 p=Path(p)
 with p.open('rb') as f:s=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':s}
root=H/'outer-O-witness-driven-sequence-448-root/iteration002'
qp=root/'qualification435001/receipt.json';qr=json.loads(qp.read_text())
assert qr['status']=='PASS_BOUNDED_OUTER_SKIN_TYPED_GEOMETRY_VOLUME_CONTEXT_AND_TARGET' and qr['Skin_native_quality_after']==1176
target=json.loads(Path(qr['target_receipt']).read_text())
tr=json.loads((root/'typed434001/receipt.json').read_text())
assert pin(qr['candidate'])['sha256']==qr['candidate_sha256']=='7822cd0cd55c346d0c3005cc4f3675bc10f85d51c137893bb1b28aec594da54f'
assert pin(tr['candidate'])['sha256']==tr['candidate_sha256']=='7a3954dc5b655e6ac0601645a1fcaa809710b89b142bed63e13d477c89c66dc5'
cp=H/'checkpoint.json';backup=H/'checkpoint-before-skin448-450-root.json';assert not backup.exists();shutil.copy2(cp,backup)
r=json.loads(cp.read_text());s=r['current_verified_state_20261031']
s['source_surfaces']['O']=pin(tr['candidate'])
s['Skin_current']=dict(pin(qr['candidate']),native_quality_failures=1176,nonpositive=0,complete_target_boundary=True,target_boundary_faces=target['desired_boundary_face_count'],qualification=pin(qp))
s['source_context_proofs'][0]=pin(qr['context_report'])
s['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
s['publication_scope']='392-406,421,current406partialmass published; O433-448 and comp1 pending next publication'
s['latest_checkpoint_archive']=pin(backup);s['latest_update_source']=pin(__file__)
r['updated_utc']=s['updated_utc'];r['updated_unix_s']=time.time()
cp.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
receipt={'status':'PASS_CHECKPOINT_QUALIFIED_SKIN448_REFRESH','checkpoint':pin(cp),'previous':pin(backup),'physical_admission':False,'native_steps':0}
p=H/'checkpoint-skin448-450-root.json';assert not p.exists();p.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
