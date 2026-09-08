import hashlib, json, os, signal, subprocess, time
from pathlib import Path
base=Path(__file__).parent
request=json.loads(Path('/Users/n/human-costal-idle-20260908/costal-request.json').read_text())
env=dict(os.environ, **request['environment'])
env['NUMI_MATTER_PROFILE_BROADPHASE']='1'
request['environment']['NUMI_MATTER_PROFILE_BROADPHASE']='1'
env['PATH']='/opt/homebrew/bin:'+env.get('PATH','')
native='/Users/n/MetalRobo-human-completion-20260907'
request['native_revision']=subprocess.check_output(['git','-C',native,'rev-parse','HEAD'],text=True).strip()
diff=subprocess.check_output(['git','-C',native,'diff','--','matter/src/runtime.mm','matter/src/metal/contact.metalinc'])
(base/'measured.patch').write_bytes(diff)
request['patch_sha256']=hashlib.sha256(diff).hexdigest()
request['purpose']='bounded kernel attribution; deliberately stops after first physical command completes'
def rows(): return subprocess.check_output(['ps','-axo','pid,ppid,etime,%cpu,command'],text=True).splitlines()[1:]
def competing(data): return [r for r in data if 'numivivo md-run ' in r or 'numivivo md-benchmark ' in r]
if competing(rows()): raise SystemExit('Competing GPU workload; measurement not started')
start=time.monotonic(); stopped=False
with (base/'kernel-profile.log').open('w') as log:
 p=subprocess.Popen(request['command'],cwd=request['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 while p.poll() is None:
  try:p.wait(timeout=1)
  except subprocess.TimeoutExpired:pass
  data=rows();elapsed=time.monotonic()-start
  content=(base/'kernel-profile.log').read_text()
  if 'nm_kernel_profile name=' not in content and elapsed<90 and not competing(data):continue
  owned={p.pid};parents={int(r.split()[0]):int(r.split()[1]) for r in data}
  while True:
   more=owned|{pid for pid,parent in parents.items() if parent in owned}
   if more==owned:break
   owned=more
  request['stop_reason']='first_profiled_command_complete' if 'seconds=' in content else 'unsupported_or_deadline_or_competition'
  for pid in sorted(owned,reverse=True):
   try:os.kill(pid,signal.SIGTERM)
   except ProcessLookupError:pass
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:
   for pid in sorted(owned,reverse=True):
    try:os.kill(pid,signal.SIGKILL)
    except ProcessLookupError:pass
   p.wait()
  stopped=True
request.update(returncode=p.returncode,elapsed_seconds=time.monotonic()-start,deliberately_stopped=stopped)
(base/'kernel-profile-launch.json').write_text(json.dumps(request,indent=2)+'\n')
print(json.dumps({k:request.get(k) for k in ['returncode','elapsed_seconds','stop_reason']}))
