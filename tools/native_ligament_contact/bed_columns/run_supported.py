"""Run the native supported scene with the existing GPU contact and circulation owners."""
import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

config_path, binary, scratch, out = map(Path, sys.argv[1:5])
steps=int(sys.argv[5]) if len(sys.argv)>5 else 8000
if steps not in {8000,30000}: raise ValueError('Expected 16 or 60 seconds')
cfg=json.loads(config_path.read_text())
def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
assert cfg['argv'][cfg['argv'].index('--muscle-step-count')+1]=='8000'
assert cfg['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']=='1' and not cfg['headless']
if cfg['argv'][cfg['argv'].index('--muscle-step-seconds')+1]!='0.002': raise ValueError('Mismatched native step')
cfg['argv'][cfg['argv'].index('--muscle-step-count')+1]=str(steps)
out.mkdir(exist_ok=True)
old_binary=cfg['argv'][0]
cfg['argv'][0]=str(binary)
cfg['argv'][4]=str(scratch/'native')
env=cfg['environment']
# Rebuild the initial native visual pack so source admission also executes on a cold launch.
cfg['pins'].pop(env.pop('NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE'))
env.pop('NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256')
if old_binary in cfg['pins']:
    assert digest(Path(old_binary))==cfg['pins'].pop(old_binary)
cfg['pins'][str(binary)]=digest(binary)
for p,h in cfg['pins'].items():
    if digest(Path(p))!=h: raise ValueError('Source identity mismatch: '+p)
if 'NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT' in env:
    env['NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT']=str(out/'failure.json')
cfg['source_launch_sha256']=digest(config_path)
(out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
t=time.monotonic()
base={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}
with (out/'stdout.log').open('wb') as so,(out/'stderr.log').open('wb') as se:
    child=subprocess.Popen(cfg['argv'],cwd=cfg['cwd'],env=base|env,stdout=so,stderr=se)
    reason=None
    while child.poll() is None:
        if shutil.disk_usage('/System/Volumes/Data').free<1024**3: reason='system disk below 1 GiB'
        if time.monotonic()-t>60+steps*.040: reason='native wall budget'
        if reason:
            child.terminate()
            try:child.wait(timeout=12)
            except subprocess.TimeoutExpired:child.kill()
            break
        time.sleep(.25)
    code=child.wait()
wall=time.monotonic()-t
(out/'execution.json').write_text(json.dumps(dict(native_exit_code=code,wall_seconds=wall,watchdog=reason),indent=2)+'\n')
for p in (scratch/'native').glob('*.csv'):
    if p.name in {'resting-coupled.csv','resting-local-bed-sections.csv','resting-com-momentum-diagnostic.csv','resting-com-support-impulses.csv','resting-support-point-motion.csv','resting-support-selected-skin-witness.csv','resting-surface-audit.csv'}:
        shutil.copyfile(p,out/p.name)
if code or reason:sys.exit(f'Native execution failed {code} {reason}')
