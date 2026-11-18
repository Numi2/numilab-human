import argparse, hashlib, json, os, shutil, subprocess, time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('arm',choices=('reference','drive-0p6'))
args=p.parse_args()
today=Path.cwd()
base=today.parent/'local-bed-layers-002'
reference=base/'loaded-24s/launch.json'
root=json.loads(reference.read_text())
for path, sha in root['pins'].items():
    with Path(path).open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==sha, path
out=today/args.arm
assert shutil.disk_usage(today).free > 10*1024**3
out.mkdir()
argv=root['argv']
env=root['environment']
argv[4]=str(out/'native')
argv[argv.index('--muscle-step-count')+1]='30000'
if args.arm=='drive-0p6':
    argv += ['--resting-drive-intervention','8','20','0.6']
env['NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT']=str(out/'common-failure.json')
(root['design'])['qualification_plan']='qualification.json'
(out/'launch.json').write_text(json.dumps(root,indent=2)+'\n')
environ={k:v for k,v in os.environ.items() if not k.startswith(('NUMI_','NM_'))}
with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
    launch_time=time.monotonic()
    with subprocess.Popen(argv,cwd=root['cwd'],env=environ|env,stdout=stdout,stderr=stderr) as process:
        reason=None
        while process.poll() is None:
            if time.monotonic()-launch_time>720:
                reason='wall budget'
            if shutil.disk_usage(today).free<9*1024**3:
                reason='disk floor'
            if reason:
                process.terminate()
                break
            time.sleep(.5)
        code=process.wait(timeout=20)
    receipt={'exit_code':code,'seconds_process':time.monotonic()-launch_time,'stopped':reason}
    (out/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt,flush=True)
raise SystemExit(code)
