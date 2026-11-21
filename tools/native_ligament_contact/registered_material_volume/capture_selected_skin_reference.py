"""Capture the current NHSKIN in the same native skeletal frame as the frozen skin."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
old_launch,new_launch,reference,out=map(Path,sys.argv[1:])
a,b=[json.loads(x.read_text()) for x in [old_launch,new_launch]]
def asset(x):return Path(x['argv'][x['argv'].index('--skin-payload')+1])
old_frozen=json.loads((reference.parent/'skin-source-rebind-002/receipt.json').read_text())['inputs_sha256']
old=next(Path(x) for x,h in old_frozen.items() if h.startswith('4fd80f449b'))
assert sha(reference)==old_frozen[reference.name] and sha(old)==old_frozen[str(old)]
out.mkdir(exist_ok=True); argv=a['argv'].copy();argv[4]=str(out/'native')
argv[argv.index('--skin-payload')+1]=str(asset(b))
env=a['environment'].copy();dump=out/'source-frame.csv';env['NUMI_HUMAN_SKIN_REFERENCE_DUMP']=str(dump)
assert not dump.exists()
(out/'capture-launch.json').write_text(json.dumps(dict(argv=argv,environment=env,cwd=a['cwd'],source_launch_sha256=sha(old_launch),selected_launch_sha256=sha(new_launch)),indent=2)+'\n')
with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
    run=subprocess.run(argv,env={**os.environ,**env},cwd=a['cwd'],stdout=stdout,stderr=stderr,timeout=100)
if not dump.exists():raise RuntimeError((out/'stderr.log').read_text()[-500:])
(out/'identity.json').write_text(json.dumps(dict(frozen_source=dict(path=str(old),sha256=sha(old)),current=dict(path=str(asset(b)),sha256=sha(asset(b))),capture_exit=run.returncode,reference_sha256=sha(reference),current_capture_sha256=sha(dump)),indent=2)+'\n')
