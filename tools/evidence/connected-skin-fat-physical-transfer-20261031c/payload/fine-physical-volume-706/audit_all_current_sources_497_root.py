
from pathlib import Path
import hashlib,json,subprocess,time
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
out=H/'joint-current-source-497-root';out.mkdir(exist_ok=False);start=time.monotonic()
cfg=json.loads((H/'joint-D414-I466-target-475-root/config354.json').read_text())
for role in ('O','I','D'):cfg['baseline_candidates'][role]=dict(cfg['candidate_surfaces'][role])
O=H/'outer-O-quality-exchange-484-root/exchange/typed434001/candidate-O-typed-current.npz'
D=H/'fat-comp0-connected-warning-repair-412/run-031-threeD-oneI-safe-flips/prep-003/candidate-D-warning-flips-typed.npz'
assert sha(D)=='77793b0878d8c76f97661a8e61adffa1b49d2be2dac7e155f7b67f6c4d423267'
directory=D.parent
matches=[p for p in directory.glob('*.npz') if sha(p)=='81b285a4e12f11dfd4156bb2545bb86f5ccee900ef6021a11beff36954b00774']
assert len(matches)==1;I=matches[0]
for role,path in [('O',O),('I',I),('D',D)]:cfg['candidate_surfaces'][role]={'path':str(path),'sha256':sha(path)}
cfg['changed_roles']=['O','I','D'];cp=out/'config354.json';cp.write_text(json.dumps(cfg,indent=2)+'\n')
cmd=[PY,str(H/'combined-surface-diff-audit-270/audit_combined_declared_surface_diff_354.py'),'--config',str(cp),'--out',str(out/'audit354')]
with (out/'audit354.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
ap=out/'audit354/report.json';r=json.loads(ap.read_text());assert r['status']=='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
res={'status':'PASS_ALL_CURRENT_O484_I_FLIP_D475_AND_WARNING_FLIPS_JOINT_CONTEXT','physical_admission':False,'native_steps':0,'context':{'path':str(ap),'sha256':sha(ap)},'config':{'path':str(cp),'sha256':sha(cp)},'baseline':'current475 before O484 and D475 and warning flips','current_surfaces':cfg['candidate_surfaces'],'source_reference_pose_only':True,'back412_unchanged':True,'producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start}
(out/'receipt.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res),flush=True)
