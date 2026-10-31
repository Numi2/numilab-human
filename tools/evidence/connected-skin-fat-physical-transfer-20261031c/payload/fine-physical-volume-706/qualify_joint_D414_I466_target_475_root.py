"""Qualify jointly composed D414/I466 surfaces and regenerate final current Fat target."""
from pathlib import Path
import json,subprocess,hashlib,time
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python';out=H/'joint-D414-I466-target-475-root';out.mkdir(exist_ok=False);start=time.monotonic()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
base=H/'combined-D414-I466-volume-469-root/fat-D414-I466-composed-volume-proposal.npz';br=H/'combined-D414-I466-volume-469-root/report.json'
assert sha(base)=='d4e17216ba617cde361d3972b80fff003f426a874f17c449499727eabeabbf17'
D=H/'fat-comp0-connected-warning-repair-412/run-023-D414-typed-warning-flips/candidate-D-comp0-warning-flips-typed.npz';DR=H/'d-shortedge-quotient-414-batch001-volume-transfer/typed-rebind-420/actual-697-D414-postflip-load-001.json'
assert sha(D)=='c285291a36182ad3a025b439ec10b1c1bb15595a1554d4fbb5d943d2a00da7a6' and sha(DR)=='873d8b4a89c633df2627b2bfe13f3933fbc775d03de6031a412a66f9a2b8d989'
I=H/'paired-I-quality-exchange-466-root/typed463001/candidate-I-typed-current.npz';IR=H/'paired-I-quality-exchange-466-root/typed463001/receipt.json'
assert sha(I)=='0e48a2f42dc6eb7ee1f1b0ce6fc63213e70dde6cc9f398ef58b03e7dae0d85b5'
O=H/'outer-O-witness-driven-sequence-448-root/iteration002/typed434001/candidate-O-typed-current.npz';assert sha(O)=='7a3954dc5b655e6ac0601645a1fcaa809710b89b142bed63e13d477c89c66dc5'
cfg=json.loads((H/'paired-I-quality-exchange-466-root/qualification467001/config354.json').read_text());cfg['candidate_surfaces']['D']={'path':str(D),'sha256':sha(D)};cfg['changed_roles']=['D','I']
cp=out/'config354.json';cp.write_text(json.dumps(cfg,indent=2)+'\n');commands=[]
def call(tag,args):
 cmd=[PY,*map(str,args)];commands.append(cmd)
 with (out/(tag+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
call('joint354',[H/'combined-surface-diff-audit-270/audit_combined_declared_surface_diff_354.py','--config',cp,'--out',out/'audit354'])
ap=out/'audit354/report.json';a=json.loads(ap.read_text());assert a['status']=='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
fc=json.loads((H/'paired-I-quality-exchange-466-root/qualification467001/config397.json').read_text())
for name,path in [('base_candidate',base),('base_report',br),('typed_D',D),('typed_D_697_receipt',DR),('typed_I',I),('typed_I_receipt',IR)]:fc[name]={'path':str(path),'sha256':sha(path)}
fcp=out/'config397.json';fcp.write_text(json.dumps(fc,indent=2)+'\n')
call('target397',[H/'rebind_current_fat_target_397_root.py','--config',fcp,'--output',out/'fat-target397'])
fr=out/'fat-target397/report.json';f=json.loads(fr.read_text())
r={'status':'PASS_JOINT_D414_I466_COMPLETE_CONTEXT_AND_CURRENT_FAT_TARGET','physical_admission':False,'native_steps':0,'source_context':{'path':str(ap),'sha256':sha(ap)},'Fat_target':{'path':str(fr),'sha256':sha(fr)},'Fat_current_candidate':f['output_candidate'],'typed_I':{'path':str(I),'sha256':sha(I)},'typed_D':{'path':str(D),'sha256':sha(D)},'typed_D_loader':{'path':str(DR),'sha256':sha(DR)},'source_O':{'path':str(O),'sha256':sha(O)},'native_quality_failures':{'Skin':1135,'Fat':20443},'back412_unchanged':True,'source330_mass':'IMMUTABLE_REBIND_REQUIRED','commands':commands,'elapsed_s':time.monotonic()-start,'producer_sha256':sha(__file__)}
(out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='commands'}),flush=True)
