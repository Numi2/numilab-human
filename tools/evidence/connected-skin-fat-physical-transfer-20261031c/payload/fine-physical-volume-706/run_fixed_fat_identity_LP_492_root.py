
from pathlib import Path
import hashlib,json,subprocess,time
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
out=H/'fixed-fat-identity-LP-492-root';out.mkdir(exist_ok=False);start=time.monotonic();commands=[]
ip=H/'d-shortedge-quotient-475-next-001-preflight-100um/volume-transfer-001/run001/fat-D475-surface-quotient-volume-proposal.npz';assert sha(ip)=='703011aba8e8bbc5590dc1cef61140eed5abe7f2eb92727f416608d1b521067d'
ref=H/'joint-D414-I466-target-475-root/fat-target397/fat-current-typed-target-rebound.npz';assert sha(ref)=='c1903e60b03aeb3a53592c8ecab020abae59eb8829899dece614d4a946f6fb04'
def call(tag,args):
 cmd=[PY,*map(str,args)];commands.append(cmd)
 with (out/(tag+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 call('controls',[H/'condition_fixed_fat_identity_LP_490_root.py','--fixture','--output',out/'controls001'])
 opts=[]
 for name in ('comp0-current-protection.npz','comp1-current-protection.npz'):
  pp=H/'d-shortedge-quotient-475-next-001-preflight-100um/transfer-inputs-001'/name
  opts+=['--protected-cell-map',pp,sha(pp)]
 call('construct',[H/'condition_fixed_fat_identity_LP_490_root.py','--input',ip,'--input-sha256',sha(ip),'--output',out/'run001','--seconds','45','--maximum-operations','10000','--protected-reference-volume',ref,'--protected-reference-sha256',sha(ref),*opts])
 report=out/'run001/report.json';r=json.loads(report.read_text())
 call('verify487',[H/'verify_fixed_fat_identity_LP_491_root.py','--report',report,'--output',out/'independent487001'])
 result={'status':'PASS_FIXED_ACTUAL_FAT_BOUNDARY_LP_AND_INDEPENDENT_REPLAY','physical_admission':False,'native_steps':0,'report':str(report),'report_sha256':sha(report),'proof':str(out/'independent487001/receipt.json'),'proof_sha256':sha(out/'independent487001/receipt.json'),'bad_before':r['bad_before'],'bad_after':r['bad_after'],'candidate':r['candidate'],'candidate_sha256':r['candidate_sha256'],'complete_target_boundary':False,'inherited_target_mismatch_unchanged':True,'mass_mapping':'CURRENT_SOURCE330_REBIND_REQUIRED','commands':commands,'producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start}
 (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
except Exception as e:
 (out/'failure.json').write_text(json.dumps({'status':'FAIL_CLOSED_FIXED_FAT_LP_TRIAL','error':repr(e),'commands':commands,'physical_admission':False,'native_steps':0},indent=2)+'\n');raise
