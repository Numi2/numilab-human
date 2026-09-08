import hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
base=Path(__file__).parent
label=sys.argv[1] if len(sys.argv)>1 else 'transaction'
request=json.loads((base/('costal-request.json' if label.startswith('costal') else 'transaction-request.json')).read_text())
env=dict(os.environ, **request['environment'])
env['PATH']='/opt/homebrew/bin:'+env.get('PATH','')
request['source_state']={}
for name,path in [('native','/Users/n/MetalRobo-human-completion-20260907'),('brain',request['cwd'])]:
 request['source_state'][name]={'revision':subprocess.check_output(['git','-C',path,'rev-parse','HEAD'],text=True).strip(),
  'status':subprocess.check_output(['git','-C',path,'status','--short'],text=True)}
request['artifact_sha256']={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in request['environment'].items() if Path(v).is_file()}
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
