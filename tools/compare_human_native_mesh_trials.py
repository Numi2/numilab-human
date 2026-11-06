"""Extract authoritative acceptance and active-work counters; no physical promotion."""
from pathlib import Path
import argparse,csv,gzip,hashlib,json
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def summarize(directory):
 d=Path(directory).resolve();p=d/'install-motion.plan.json';ex=d/'install-motion.execution.json'
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
def control_differences(candidate_plan,control_plan):
 # Geometry and output paths may differ. Numerical controls and loaded code may not.
 a=candidate_plan['stages'][0];b=control_plan['stages'][0]
 def canonical(stage):
  root=stage['argv'][4]
  geometry=lambda k:k.startswith('NUMI_H706_FRACTIONAL_') or k.startswith('NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_') or k in {'NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ','NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256'}
  def value(v):return '<OUTPUT>'+v[len(root):] if v.startswith(root) else v
  env={k:value(v) for k,v in stage['environment'].items() if not geometry(k)}
  argv=[value(v) for v in stage['argv']]
  return dict(argv=argv,environment=env)
 aa,bb=canonical(a),canonical(b);different=[]
 if aa['argv']!=bb['argv']:different.append('argv')
 for k in sorted(aa['environment'].keys()|bb['environment'].keys()):
  if aa['environment'].get(k)!=bb['environment'].get(k):different.append('environment:'+k)
 if candidate_plan['native_binary']['sha256']!=control_plan['native_binary']['sha256']:different.append('native_binary')
 def deps(p):return {k:v['sha256'] for k,v in p['runtime_dependencies'].items()}
 if deps(candidate_plan)!=deps(control_plan):different.append('runtime_dependencies')
 return different

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--candidate-dir',required=True,type=Path)
 ap.add_argument('--control-dir',required=True,type=Path)
 ap.add_argument('--output',required=True,type=Path)
 args=ap.parse_args()
 assert not args.output.exists(),'retain previous analysis; choose an unused output'
 candidate=summarize(args.candidate_dir);control=summarize(args.control_dir)
 cp=json.loads(Path(candidate['plan']['path']).read_text());bp=json.loads(Path(control['plan']['path']).read_text())
 differences=control_differences(cp,bp)
 if differences:raise ValueError('Unmatched native controls: '+', '.join(differences))
 r=dict(candidate=candidate,matched_solver_control=control,
        accepted_roots_change=candidate['accepted_roots']-control['accepted_roots'],
        candidate_exceeds_control_accepted_horizon=candidate['accepted_roots']>control['accepted_roots'],
        force_strain_energy_fidelity_admitted=False,physical_admission=False,
        analysis_owner=ref(__file__),control_compatibility=dict(matched=True,geometry_and_output_paths_may_differ=True))
 args.output.parent.mkdir(parents=True,exist_ok=True)
 args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in candidate.items() if k not in ['per_root','execution']},indent=2))
if __name__=='__main__':main()
