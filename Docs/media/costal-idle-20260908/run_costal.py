import hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
base=Path(__file__).parent
label=sys.argv[1] if len(sys.argv)>1 else 'transaction'
request=json.loads((base/'costal-request.json').read_text())
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
request['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(base/(label+'-launch.json')).write_text(json.dumps(request,indent=2)+'\n')
if request['competing_workloads']:
 raise SystemExit('GPU workload present; costal run not started')
def descendants(processes, root):
 tree = {int(row.split()[0]):int(row.split()[1]) for row in processes.splitlines() if row.split() and row.split()[0].isdigit()}
 owned = {root}
 while True:
  expanded = owned | {pid for pid,parent in tree.items() if parent in owned}
  if expanded == owned: return owned
  owned = expanded

t=time.monotonic()
with (base/(label+'.log')).open('w') as log:
 p=subprocess.Popen(request['command'],cwd=request['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 code = None
 samples = set()
 observations = []
 while code is None:
  try: code = p.wait(timeout=2)
  except subprocess.TimeoutExpired: pass
  elapsed = time.monotonic()-t
  processes = subprocess.check_output(['ps','-axo','pid,ppid,etime,%cpu,command'],text=True)
  active = [line for line in processes.splitlines() if 'numivivo md-run ' in line or 'numivivo md-benchmark ' in line]
  observations.append({'elapsed_seconds':elapsed,'competing_gpu':active})
  (base/(label+'-workload.json')).write_text(json.dumps(observations,indent=2)+'\n')
  owned = descendants(processes, p.pid)
  if code is None and (active or elapsed > 300):
   request['interrupted_reason'] = 'competing_gpu_workload' if active else '300_second_runner_deadline'
   for pid in sorted(owned, reverse=True):
    try: os.kill(pid,signal.SIGTERM)
    except ProcessLookupError: pass
   try: p.wait(timeout=5)
   except subprocess.TimeoutExpired:
    for pid in sorted(owned, reverse=True):
     try: os.kill(pid,signal.SIGKILL)
     except ProcessLookupError: pass
    p.wait()
   code=125 if active else 124
  if code is None:
   for checkpoint in [10,35,65]:
    if elapsed < checkpoint or checkpoint in samples: continue
    samples.add(checkpoint)
    for line in processes.splitlines():
     if 'NumiBrainPackageTests.xctest' not in line: continue
     try: pid=int(line.split()[0])
     except ValueError: continue
     if pid not in owned: continue
     subprocess.run(['sample',str(pid),'1','-file',str(base/(label+'-sample-'+str(checkpoint)+'.txt'))],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
     break
request.update(returncode=code,elapsed_seconds=time.monotonic()-t)
(base/(label+'-launch.json')).write_text(json.dumps(request,indent=2)+'\n')
print(json.dumps({'returncode':code,'elapsed_seconds':request['elapsed_seconds']}))
sys.exit(code)
