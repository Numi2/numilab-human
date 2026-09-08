import hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
base=Path(__file__).parent
label=sys.argv[1] if len(sys.argv)>1 else 'transaction'
request=json.loads((base/('describe-request.json' if label.startswith('describe') else 'legacy-request.json' if label.startswith('legacy') else 'transaction-request.json')).read_text())
env=dict(os.environ, **request['environment'])
env['PATH']='/opt/homebrew/bin:'+env.get('PATH','')
request['source_state']={}
for name,path in [('native','/Users/n/MetalRobo-human-completion-20260907'),('brain',request['cwd'])]:
 request['source_state'][name]={'revision':subprocess.check_output(['git','-C',path,'rev-parse','HEAD'],text=True).strip(),
  'status':subprocess.check_output(['git','-C',path,'status','--short'],text=True)}
request['competing_workloads'] = '\n'.join(l for l in subprocess.check_output(['ps','-axo','pid,etime,%cpu,command'],text=True).splitlines() if ('numivivo md-run ' in l or 'numivivo md-benchmark ' in l))
request['artifact_sha256']={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in request['environment'].items() if Path(v).is_file()}
(base/(label+'-launch.json')).write_text(json.dumps(request,indent=2)+'\n')
binary = Path(request['cwd']) / '.build/release/NumiBrainPackageTests.xctest/Contents/MacOS/NumiBrainPackageTests'
request['brain_test_binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
request['brain_test_binary_bytes'] = binary.stat().st_size
request['authoring_revision'] = 'f71d98c6464fd136e905cf232b20afd08f53aa28'
request['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(base/(label+'-launch.json')).write_text(json.dumps(request,indent=2)+'\n')
t=time.monotonic()
with (base/(label+'.log')).open('w') as log:
 p=subprocess.Popen(request['command'],cwd=request['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try: code=p.wait(timeout=300)
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  code=124
request.update(returncode=code,elapsed_seconds=time.monotonic()-t)
(base/(label+'-launch.json')).write_text(json.dumps(request,indent=2)+'\n')
print(json.dumps({'returncode':code,'elapsed_seconds':request['elapsed_seconds']}))
sys.exit(code)
