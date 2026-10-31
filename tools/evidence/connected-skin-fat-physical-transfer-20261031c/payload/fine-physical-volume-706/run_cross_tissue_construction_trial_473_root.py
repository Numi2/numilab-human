"""Bounded joint construction-quality exchange, exact physics admission unchanged."""
from pathlib import Path
import json,hashlib,subprocess,time
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python';out=H/'paired-I-cross-tissue-473-root';assert (out/'controls001/report.json').is_file()
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
pins={'skin':(H/'skin-interior-LP-455-root/run001/conditioned-volume-proposal.npz','cdeccad0b3aa6b5e6171dce2f62691a08b62dbf136950d1abb140d2c1004ddf3'),'fat':(H/'fat-comp0-connected-warning-repair-412/run-012-final-D-current-target-rebind/fat-current-typed-target-rebound.npz','72c851692580a639f89a4916bc4bf5c38331be9b6def89322eccf02ee66ea6ec'),'i':(H/'paired-I-interface-399-root/iteration002/typed365/candidate-I-typed-current.npz','fec15d6b59e9094952845cbeaaf418b91a4b2f5d0405d1c58b7e3e2e752ec406')}
extra=H/'paired-I-quality-exchange-466-root/protected-comp0-comp1-all-next-D-orientation-stars.npz';assert extra.is_file()
for path,pin in pins.values():assert sha(path)==pin
plan={'status':'PREDECLARED_BOUNDED_CONSTRUCTION_TRIAL','physical_admission':False,'native_steps':0,'strict_combined_bad_count_descent_required_each_operation':True,'role_quality_debt_factor_maximum':1.2,'native_thresholds_unchanged':True,'exact_combined_Skin_Fat_volume_delta_required':0,'source330_masses':'IMMUTABLE','maximum_edge_m':.001,'wall_budget_seconds':45,'reference_net_role_quality_trial':{'Skin':1135,'Fat':23997},'protection':{'path':str(extra),'sha256':sha(extra)},'qualification':'Only retain as new construction candidate if better combined count; typed/full185/nativequality/mass still mandatory.'}
(out/'trial-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
cmd=[PY,str(H/'repair_paired_I_cross_tissue_471_root.py'),'--output',str(out/'run001'),'--maximum-edge-m','.001','--maximum-operations','2048','--seconds','45','--protect-fat-points',str(extra),'--protect-fat-points-sha256',sha(extra),'--allow-cross-tissue-quality-exchange']
for role,(path,pin) in pins.items():cmd+=['--'+role,str(path),'--'+role+'-sha256',pin]
start=time.monotonic()
with (out/'construction.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
r=json.loads((out/'run001/report.json').read_text());qual={x['role']:x['bad_after'] for x in r['roles']}
result={'status':'CONSTRUCTED_PENDING_INDEPENDENT_TYPED_AND_COMPLETE_CONTEXT','physical_admission':False,'native_steps':0,'raw_report':str(out/'run001/report.json'),'raw_report_sha256':sha(out/'run001/report.json'),'operations':r['passing_operations'],'quality':qual,'combined_gain_vs_input':1149+24174-sum(qual.values()),'combined_gain_vs_verified_per_role_trial':1135+23997-sum(qual.values()),'new_retained_quality_failures_by_role':{x['role']:sum(o['new_retained_quality_failures'] for o in x['operations']) for x in r['roles']},'producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start,'command':cmd}
(out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='command'}),flush=True)
