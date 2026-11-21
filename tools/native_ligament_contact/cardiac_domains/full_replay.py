import csv,hashlib,json,shutil,subprocess,sys
from pathlib import Path
owner,config,baseline,scratch,out=map(Path,sys.argv[1:])
py=sys.executable
cfg=json.loads(config.read_text()); assert cfg['argv'][cfg['argv'].index('--muscle-step-count')+1]=='8'
# Capture the accepted physical trace. This parity test does not need a second native mesh export.
cfg['environment'].pop('NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS')
cfg['hashes'][str(config)]=hashlib.sha256(config.read_bytes()).hexdigest()
run=scratch/'full';out.mkdir(exist_ok=True);launch=out/'package.json';launch.write_text(json.dumps(cfg,indent=2)+'\n')
subprocess.run([py,owner/'run_scene.py',launch,run],check=True)
for name in ('result.json','stdout.log','stderr.log','launch.json'):shutil.copyfile(run/name,out/name)
for name in ('resting-coupled.csv','resting-com-support-impulses.csv','resting-surface-audit.csv'):
    if (run/'native'/name).exists():shutil.copyfile(run/'native'/name,out/name)
a=list(csv.DictReader((out/'resting-coupled.csv').open()));b=list(csv.DictReader((baseline/'resting-coupled.csv').open()))
report=json.loads((out/'result.json').read_text())
assert report['exit_code']==0 and len(a)==len(b)==8
report['exact_matched_columns']=[k for k in a[0] if all(x[k]==y[k] for x,y in zip(a,b,strict=True))]
report['changed_columns']=[k for k in a[0] if k not in report['exact_matched_columns']]
report['reference_launch_sha256']=hashlib.sha256((baseline/'launch.json').read_bytes()).hexdigest()
report['accepted_seconds']=float(a[-1]['time_s'])
(out/'physical-parity.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['accepted_seconds'],report['changed_columns'])
