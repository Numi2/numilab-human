"""Bounded continuation of the qualified433 outer-Skin batch through identical owners."""
from pathlib import Path
import json,subprocess,hashlib,time
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
base=H/'outer-O-quotient-433-root';proof=base/'qualification435001/receipt.json'
r=json.loads(proof.read_text());assert r['status']=='PASS_BOUNDED_OUTER_SKIN_TYPED_GEOMETRY_VOLUME_CONTEXT_AND_TARGET'
skin=Path(r['candidate']);skinhash=r['candidate_sha256']
tp=base/'typed434001/receipt.json';tr=json.loads(tp.read_text());surface=Path(tr['candidate']);ohash=tr['candidate_sha256']
assert sha(skin)==skinhash and sha(surface)==ohash
out=H/'outer-O-connected-sequence-440-root';out.mkdir(exist_ok=False)
start=time.monotonic();history=[];bad=1336;absolute_delta=3.475875875880513e-11
for cycle in range(3):
 root=out/('iteration%03d'%(cycle+1));root.mkdir(exist_ok=False)
 def call(tag,argv):
  command=[PY,*map(str,argv)]
  with (root/(tag+'.log')).open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
  print(tag,flush=True)
  return command
 calls=[]
 calls.append(call('construct433',[H/'repair_bounded_outer_O_433_root.py','--skin',skin,'--skin-sha256',skinhash,'--o',surface,'--o-sha256',ohash,'--output',root/'run001','--maximum-edge-m','.001','--maximum-operations','1024','--seconds','40']))
 rp=root/'run001/report.json';rr=json.loads(rp.read_text())
 if rr['passing_operations']==0:
  history.append({'iteration':cycle+1,'status':'NO_PROGRESS_STOP','report':str(rp),'sha256':sha(rp)});break
 calls.append(call('typed434',[H/'bind_and_replay_bounded_outer_O_434_root.py','--report',rp,'--output',root/'typed434001']))
 calls.append(call('qualify439',[H/'qualify_bounded_outer_skin_generic_439_root.py','--root',root]))
 qp=root/'qualification435001/receipt.json';qr=json.loads(qp.read_text())
 assert qr['status']=='PASS_BOUNDED_OUTER_SKIN_TYPED_GEOMETRY_VOLUME_CONTEXT_AND_TARGET'
 nr=json.loads((root/'typed434001/receipt.json').read_text())
 skin=Path(qr['candidate']);skinhash=qr['candidate_sha256'];surface=Path(nr['candidate']);ohash=nr['candidate_sha256']
 assert sha(skin)==skinhash and sha(surface)==ohash
 newbad=qr['Skin_native_quality_after'];gain=bad-newbad;assert gain>0
 absolute_delta+=rr['accumulated_absolute_volume_change_m3']
 history.append({'iteration':cycle+1,'status':'PASS_TYPED_COMPLETE_CONTEXT_AND_PHYSICAL_TARGET_CONSTRUCTION','qualification':str(qp),'qualification_sha256':sha(qp),'bad_before':bad,'bad_after':newbad,'operations':rr['passing_operations'],'commands':calls})
 bad=newbad
 progress={'status':'BOUNDED_OUTER_SKIN_SEQUENCE_IN_PROGRESS','physical_admission':False,'native_steps':0,'history':history,'final_skin':str(skin),'final_skin_sha256':skinhash,'final_O':str(surface),'final_O_sha256':ohash,'quality_after':bad}
 (out/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
 if gain<12:break
res={'status':'PASS_BOUNDED_CONNECTED_OUTER_SKIN_SEQUENCE','physical_admission':False,'native_steps':0,'input_qualification':str(proof),'input_qualification_sha256':sha(proof),'quality_before':1336,'quality_after':bad,'history':history,'final_skin':str(skin),'final_skin_sha256':skinhash,'final_O':str(surface),'final_O_sha256':ohash,'cumulative_absolute_geometry_volume_change_from_421_m3':absolute_delta,'mass_mapping':'NOT_ADMITTED_SOURCE330_CONSERVATIVE_REBIND_REQUIRED','back412_unchanged':True,'native_motion':'NOT_RUN','producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start}
(out/'receipt.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res),flush=True)
