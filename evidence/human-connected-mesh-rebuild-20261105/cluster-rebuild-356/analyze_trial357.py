"""Extract authoritative acceptance and active-work counters; no physical promotion."""
from pathlib import Path
import csv,gzip,hashlib,json,re
W=Path('/Users/numi/Work/human-integrated-gaps-20261104')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def summarize(name):
 d=W/name;p=d/'install-motion.plan.json';ex=d/'install-motion.execution.json'
 plan=json.loads(p.read_text());execution=json.loads(ex.read_text())
 trace=d/'install-motion/resting-coupled.csv'
 rows=list(csv.DictReader(trace.open())) if trace.exists() else []
 events={}
 for line in (d/'install-motion.stderr.log').read_text().splitlines():
  key,sep,value=line.partition('=')
  if sep and key in ['matter_fgmres_active_work','matter_solver_domain_norms','matter_contact_work']:
   events.setdefault(key,[]).append(json.loads(value))
 active=events.get('matter_fgmres_active_work',[])
 args=plan['stages'][0]['argv'];dt=float(args[args.index('--muscle-step-seconds')+1])
 count=int(args[args.index('--muscle-step-count')+1]);accepted=len(rows)
 body=None
 with gzip.open(d/'install-motion.stdout.log.gz','rt') as f:
  for line in f:
   if 'resting_integrated_body=completed ' in line:
    body=dict(v.split('=',1) for v in line.split('resting_integrated_body=completed ',1)[1].split() if '=' in v)
 failure=d/'install-motion/diagnostics/common-field-failure.json'
 if failure.exists():
  f=json.loads(failure.read_text());assert f['segment_start']+f['stand_completed_steps']==accepted
  assert f['respiratory_accepted_status'][0]==accepted
 assert all(v['root']<count for v in active)
 r=dict(directory=str(d),plan=ref(p),execution=execution,accepted_roots=accepted,requested_roots=count,dt_s=dt,accepted_time_s=accepted*dt,launcher_accepted_time_factor=accepted*dt/execution['elapsed_s'],full_requested_horizon_pass=execution['exit_code']==0 and accepted==count and execution['resource_stop_reason'] is None,active_work_all_attempts={key:sum(v[key] for v in active) for key in ['used_columns_sum','cycles_with_work','budget_exhausted_solves']},active_work_accepted_roots={key:sum(v[key] for v in active if v['root']<accepted) for key in ['used_columns_sum','cycles_with_work','budget_exhausted_solves']},per_root=events,body=body,coupled_trace=ref(trace) if trace.exists() else None,failure=ref(failure) if failure.exists() else None,scope='Short initialization transient, no supported-rest, sustained throughput or visual motion admission.')
 if body:
  r['body_real_time_factor']=accepted*dt/float(body['wall_s'])
 return r
candidate=summarize('witness-cluster-native-357')
control=summarize('skin-backtracking-witness-351')
out=W/'witness-cluster-native-357/compact-comparison.json'
assert not out.exists()
r=dict(candidate=candidate,matched_solver_control=control,accepted_roots_change=candidate['accepted_roots']-control['accepted_roots'],candidate_exceeds_six_root_rejection=candidate['accepted_roots']>6,force_strain_energy_fidelity_admitted=False,physical_admission=False,analysis_owner=ref(__file__))
out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in candidate.items() if k not in ['per_root','execution']},indent=2))
