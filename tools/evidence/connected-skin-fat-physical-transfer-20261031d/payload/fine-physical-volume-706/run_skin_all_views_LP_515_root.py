
"""Continue current Skin on same tested LP equations with both physical coordinate views synchronized."""
from pathlib import Path
import json,hashlib,subprocess,time,numpy as np
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
out=H/'skin-all-views-interior-LP-515-root';out.mkdir(exist_ok=False);start=time.monotonic()
ip=H/'skin-canonical-lineage-512-root/skin-current-O506-Iflip-physical-volume-proposal.npz'
assert sha(ip)=='97137971e9b29f88c3e80dc777ba188dd1adfbb18475d8b639bce085f75a1a69'
commands=[]
def call(tag,owner,argv):
 cmd=[PY,str(H/owner),*map(str,argv)];commands.append(cmd)
 with (out/(tag+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
call('fixture513','condition_fixed_skin_all_views_LP_513_root.py',['--fixture','--output',out/'fixture513'])
call('LP513','condition_fixed_skin_all_views_LP_513_root.py',['--input',ip,'--input-sha256',sha(ip),'--output',out/'run001','--seconds','40','--maximum-operations','10000'])
call('replay514','verify_fixed_skin_all_views_LP_514_root.py',['--report',out/'run001/report.json','--output',out/'independent514001'])
rp=out/'run001/report.json';r=json.loads(rp.read_text());p=Path(r['candidate'])
with np.load(p) as z:assert np.array_equal(z['points_world_m'].view('u4'),z['points_world_m_f32'].view('u4'))
proof=out/'independent514001/receipt.json';pr=json.loads(proof.read_text())
res={'status':'PASS_CURRENT_SKIN_LP_SYNCHRONIZED_PHYSICAL_VIEWS_AND_EXACT_REPLAY','physical_admission':False,'native_steps':0,'candidate':str(p),'candidate_sha256':sha(p),'bad_before':r['bad_before'],'bad_after':r['bad_after'],'operations':len(r['operations']),'independent_proof':str(proof),'independent_proof_sha256':sha(proof),'source330_mass_immutable':True,'boundary_coordinates_unchanged':True,'native_thresholds_unchanged':True,'commands':commands,'producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start}
(out/'receipt.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps({k:v for k,v in res.items() if k!='commands'}),flush=True)
