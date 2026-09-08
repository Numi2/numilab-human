from pathlib import Path
import json,subprocess,os,time,hashlib,signal
root=Path('/Users/n/MetalRobo-human-completion-20260907')
build=Path('/Users/n/MetalRobo-human-completion-build-20260907')
out=Path('/Users/n/numi-human-balance-solver-20260908')
source=root/'src/core/NumiHumanMuscleEquilibrium.cpp';correct=source.read_text()
rows=json.loads((out/'regressions.json').read_text())
def run(label,command,cwd=None,env=None,timeout=480):
 start=time.monotonic()
 with (out/(label+'.log')).open('w') as log:
  p=subprocess.Popen([str(x) for x in command],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);code=124
 row={'label':label,'command':[str(x) for x in command],'cwd':str(cwd) if cwd else None,'exit_code':code,'elapsed_seconds':time.monotonic()-start}
 rows.append(row);(out/'regressions.json').write_text(json.dumps(rows,indent=2)+'\n');print(label,code,round(row['elapsed_seconds'],3),flush=True);return code
assert run('human-io-corrected',[build/'bin/metalrobo_numanx_human_io_probe',build/'shaders/MetalRobo.metallib'],env=dict(os.environ,MTL_DEBUG_LAYER='1'))==0
# Existing source-compliant, mass-conserving v5 transaction regression.
prior=json.loads(Path('/Users/n/numi-human-tissue-ownership-20260908/brain-costal-e2e-launch.json').read_text())
brain=Path('/Users/n/numi-brain-human-completion-20260907')
env=dict(os.environ,**prior['environment']);env['PATH']='/opt/homebrew/bin:'+env.get('PATH','')
(out/'costal-regression-environment.json').write_text(json.dumps(prior['environment'],indent=2)+'\n')
assert run('costal-regression',prior['command'],cwd=brain,env=env)==0
