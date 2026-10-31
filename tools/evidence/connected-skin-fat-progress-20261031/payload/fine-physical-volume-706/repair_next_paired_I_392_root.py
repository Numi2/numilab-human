"""Run the next bounded paired I quotient with disjoint concurrent D/cavity ownership."""
from pathlib import Path
import numpy as np,json,hashlib,subprocess,time,os
H=Path(__file__).resolve().parent
PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
out=H/'paired-I-interface-392-root';out.mkdir(exist_ok=False)
pre=H/'d-shortedge-quotient-385-run006/preflight.json'
r=json.loads(pre.read_text())
fat=Path(r['source_pins']['current_Fat_rebound_8915']['path'])
assert sha(fat)==r['source_pins']['current_Fat_rebound_8915']['sha256']=='8915c3c7a58f8622d0568da5006bf4b7b0f2832f802b65e29bd144e95f11001d'
with np.load(fat) as z:
 p=z['points_world_m'];t=z['tetrahedra']
drows=np.unique(np.concatenate([np.asarray(x['affected_cell_rows'],np.int64) for op in r['preflight']['selected_operations'] for x in op['volume_trials']]))
shell=H/'fat614-yy-volume-replay-004/run-021-current-target-cavities-bounded-closure/connected-cavity-shells.npz'
assert sha(shell)=='34dfe3eb2a49f56e036bc5c03ae4e4e72a17ab669dc9bdaf6a299cb905204a85'
with np.load(shell) as z:crows=z['input_cell_rows']
old=json.loads((H/'paired-I-interface-377-root/run002/report.json').read_text())
with np.load(old['extra_protected_fat_points']) as z:previous=z['affected_volume_point_rows']
assert sha(old['extra_protected_fat_points'])==old['extra_protected_fat_points_sha256']
rows=np.unique(np.r_[drows,crows]);protected=np.unique(np.r_[t[rows].ravel(),previous])
mask=out/'protected-D006-cavity021-and-prior-stars.npz'
np.savez_compressed(mask,affected_volume_point_rows=protected)
prep={'status':'PASS_DISJOINT_CURRENT_ROW_PROTECTION_PREPARATION','physical_admission':False,'native_steps':0,
 'input_fat':str(fat),'input_fat_sha256':sha(fat),'D_preflight':str(pre),'D_preflight_sha256':sha(pre),'D_candidate_operation_count':len(r['preflight']['selected_operations']),
 'D_trial_cell_rows':len(drows),'cavity_old_cell_rows':len(crows),'union_cell_rows':len(rows),'protected_point_rows':len(protected),
 'mask':str(mask),'mask_sha256':sha(mask),'cavity_shell_sha256':sha(shell),'producer_sha256':sha(__file__)}
(out/'protection-receipt.json').write_text(json.dumps(prep,indent=2)+'\n');print(json.dumps(prep),flush=True)
skin=H/'current-skin-neighborhood-remesh-387-root/remesh305/connected-fixed-surface-volume-proposal.npz'
I=Path(r['source_pins']['current_I']['path'])
assert sha(skin)=='0b0aa6e97d1ea2624c8a0a8a177d737260ccf189fccc867f4bc7f85ae183aa38'
assert sha(I)=='8e04da9ad1e9c04216dae1fde854a3d69c8e5bb4c7a7c4a4796754675ce7b747'
cmd=[PY,str(H/'repair_paired_I_interfaces_indexed_384_root.py'),'--skin',str(skin),'--skin-sha256',sha(skin),'--fat',str(fat),'--fat-sha256',sha(fat),'--i',str(I),'--i-sha256',sha(I),'--protect-fat-points',str(mask),'--protect-fat-points-sha256',sha(mask),'--output',str(out/'run001'),'--maximum-edge-m','0.0005','--maximum-operations','4096','--seconds','90']
(out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (out/'execution.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
result=json.loads((out/'run001/report.json').read_text());print(json.dumps({k:result[k] for k in ['status','passing_operations','elapsed_s','rejections']}),flush=True)
print(json.dumps([{k:a[k] for k in ['role','bad_before','bad_after','output','output_sha256']} for a in result['roles']]),flush=True)
