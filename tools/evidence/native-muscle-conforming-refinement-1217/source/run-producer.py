from pathlib import Path
import os,json,subprocess,hashlib,time
R=Path(__file__).resolve().parent
d=json.loads((R/'producer-declaration.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(sha(p)==v for p,v in d['immutable_inputs'].items())
env=dict(os.environ);env['PYTHONPATH']=d['PYTHONPATH'];start=time.monotonic()
with (R/'producer-stdout.log').open('x') as log:
 result=subprocess.run(d['argv'],cwd=d['source_snapshot'],env=env,stdout=log,stderr=subprocess.STDOUT)
changes=[p for p,v in d['immutable_inputs'].items() if sha(p)!=v]
(R/'producer-execution.json').write_text(json.dumps({'returncode':result.returncode,'wall_seconds':time.monotonic()-start,'changed_inputs':changes,'declaration_sha256':sha(R/'producer-declaration.json')},indent=2)+'\n')
print((R/'producer-execution.json').read_text())
raise SystemExit(result.returncode or (1 if changes else 0))
