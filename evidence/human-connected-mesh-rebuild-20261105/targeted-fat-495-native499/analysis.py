from pathlib import Path
import json,csv,sys,math
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');D=W/'targeted-fat-native-499'
sys.path.insert(0,'/Users/numi/Work/human-checkpoint-owner-20261105/tools')
from compare_human_native_mesh_trials import summarize,control_differences
C=W/'endpoint-potential-native-481';a=summarize(C);b=summarize(D)
diffs=control_differences(json.loads((D/'install-motion.plan.json').read_text()),json.loads((C/'install-motion.plan.json').read_text()));assert not diffs,diffs
ar=list(csv.DictReader((C/'install-motion/resting-coupled.csv').open()));br=list(csv.DictReader((D/'install-motion/resting-coupled.csv').open()));n=min(len(ar),len(br));assert n>0
phys={}
for k in ar[0]:
 av=[float(r[k]) for r in ar[:n]];bv=[float(r[k]) for r in br[:n]];assert all(math.isfinite(x) for x in av+bv)
 delta=max(abs(x-y) for x,y in zip(av,bv));scale=max(map(abs,av));relative=(delta/scale if scale else 0 if not delta else None)
 phys[k]=dict(maximum_absolute_change=delta,maximum_reference_magnitude=scale,relative_maximum_change=relative)
rows=[]
for line in (D/'install-motion.stderr.log').read_text().splitlines():
 if not line.startswith('fem_newton_terminal_certificate='):continue
 r=json.loads(line.split('=',1)[1]);assert len(r['status'])==1
 state=r['status'][0];objs=r['objects'];norms=next(x for x in b['per_root']['matter_solver_domain_norms'] if x['root']==r['root'])
 active=next(x for x in b['per_root']['matter_fgmres_active_work'] if x['root']==r['root'])
 row=dict(root=r['root'],accepted=r['root']<b['accepted_roots'],status_code=state['code'],mechanical_norm=norms['nonvascular_norm'],mechanical_threshold=norms['mechanical_threshold'],vascular_norm=norms['vascular_norm'],vascular_threshold=norms['vascular_threshold'],minimum_cell_J=min(x['validity'][0] for x in objs),active_columns=active['used_columns_sum'],active_solves=active['cycles_with_work'],budget_exhausted_solves=active['budget_exhausted_solves'])
 row['fresh_certificate_nonlinear_norm']=max(x['nonlinear'][0] for x in objs)
 if row['accepted']:
  assert row['status_code']==0 and row['minimum_cell_J']>0 and row['fresh_certificate_nonlinear_norm']<=row['mechanical_threshold']
  row['prelinearization_domain_norms_within_printed_thresholds']=row['mechanical_norm']<=row['mechanical_threshold'] and row['vascular_norm']<=row['vascular_threshold']
 rows.append(row)
with (D/'compact-certificates.csv').open('w') as f:
 out=csv.DictWriter(f,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)
r=dict(candidate=b,control={k:v for k,v in a.items() if k!='per_root'},matched_non_geometry_controls=diffs,accepted_roots_change=b['accepted_roots']-a['accepted_roots'],shared_physiology_rows=n,shared_physiology_exact=ar[:n]==br[:n],physiology=phys,physiological_1percent_failures=[k for k,v in phys.items() if v['relative_maximum_change'] is None or v['relative_maximum_change']>.01],all_accepted_status_codes_zero=True,prelinearization_diagnostic_exceeds_final_threshold_roots=[x['root'] for x in rows if x['accepted'] and not x.get('prelinearization_domain_norms_within_printed_thresholds',True)],domain_norm_phase='Last encoded recordFGMRESBegin (runtime.mm21138), preceding final fresh certificate; not accepted-state residual.',all_accepted_fresh_certificates_below_threshold=True,physical_admission=False,scope='Whole native scene initialization test only. No force/energy mesh-fidelity promotion, supported rest, sustained motion or0.1x claim.')
(D/'summary.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in r.items() if k not in ['candidate','control','physiology']},indent=2))
print('candidate',{k:v for k,v in b.items() if k not in ['per_root','execution','body','plan']})

import gzip
def profiles(directory):
 out={}
 with gzip.open(directory/'install-motion.stdout.log.gz','rt') as f:
  for line in f:
   if line.startswith('resting_native_profile '):
    fields=dict(x.split('=',1) for x in line.split()[1:]);step=int(fields['step']);assert step not in out
    out[step]={k:float(fields[k]) for k in ['command_ms','gpu_ms','host_copy_ms']}
 return out
pa,pb=profiles(C),profiles(D);steps=range(1,n+1)
assert all(i in pa and i in pb for i in steps)
pair={k:{'control_ms':sum(pa[i][k] for i in steps),'candidate_ms':sum(pb[i][k] for i in steps)} for k in ['command_ms','gpu_ms','host_copy_ms']}
for value in pair.values():value['candidate_over_control']=value['candidate_ms']/value['control_ms']
r['matched_accepted_prefix_profile']={'shared_accepted_steps':n,'metrics':pair,'scope':'One matched initialization pair, native command-span profiles only. Not steady-state or complete presentation throughput; no statistical speedup qualification.'}
r['accepted_native_profile']={'command_ms':sum(pb[i]['command_ms'] for i in range(1,b['accepted_roots']+1)),'gpu_ms':sum(pb[i]['gpu_ms'] for i in range(1,b['accepted_roots']+1))}
(D/'summary.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print(json.dumps(r['matched_accepted_prefix_profile'],indent=2))
