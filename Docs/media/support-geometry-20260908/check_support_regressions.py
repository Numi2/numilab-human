from pathlib import Path
import json,subprocess,os,time,hashlib,signal
root=Path('/Users/n/MetalRobo-human-completion-20260907')
build=Path('/Users/n/MetalRobo-human-completion-build-20260907')
out=Path('/Users/n/numi-human-balance-solver-20260908')
source=root/'src/core/NumiHumanMuscleEquilibrium.cpp';correct=source.read_text()
rows=[]
def run(label,command,cwd=None,env=None,timeout=480):
 start=time.monotonic()
 with (out/(label+'.log')).open('w') as log:
  p=subprocess.Popen([str(x) for x in command],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);code=124
 row={'label':label,'command':[str(x) for x in command],'cwd':str(cwd) if cwd else None,'exit_code':code,'elapsed_seconds':time.monotonic()-start}
 rows.append(row);(out/'regressions.json').write_text(json.dumps(rows,indent=2)+'\n');print(label,code,round(row['elapsed_seconds'],3),flush=True);return code
cmake=['/opt/homebrew/bin/cmake','--build',build,'--target','metalrobo_numi_human_static_support_test','-j','4']
assert run('analytic-correct',[build/'bin/metalrobo_numi_human_static_support_test'])==0
try:
 # Restore the old unsupported-force semantics for the negative control only.
 mutant=correct.replace('if (gap > gapTolerance) continue;', '/* negative control: admit force across a gap */')
 assert mutant!=correct;source.write_text(mutant)
 assert run('analytic-mutant-build',cmake)==0
 assert run('analytic-mutant',[build/'bin/metalrobo_numi_human_static_support_test'])!=0
finally:
 source.write_text(correct)
 restore=run('analytic-restore-build',cmake)
assert restore==0
assert run('analytic-restored',[build/'bin/metalrobo_numi_human_static_support_test'])==0
assert run('native-final-build',['/opt/homebrew/bin/cmake','--build',build,'--target','metalrobo_numilab_human_myosim_reference_probe','metalrobo_numanx_human_io_probe','metalrobo_numilab_human_myosim_visual_probe','-j','4'])==0
inputs=Path('/Users/n/human-completion-20260907/input')
assert run('muscle-reference',[build/'bin/metalrobo_numilab_human_myosim_reference_probe',inputs/'myosim-fullbody-core-reference.nhrigid',inputs/'myosim-fullbody-muscle-reference.nhmyo',inputs/'numi-human-tendon-attachments.nhtendon','--metal',inputs/'myosim-fullbody-joint-equalities.nheq','--equilibrium'],env=dict(os.environ,MTL_DEBUG_LAYER='1'))==0
assert run('human-io',[build/'bin/metalrobo_numanx_human_io_probe'],env=dict(os.environ,MTL_DEBUG_LAYER='1'))==0
# Existing source-compliant, mass-conserving v5 transaction regression.
prior=json.loads(Path('/Users/n/numi-human-tissue-ownership-20260908/brain-costal-e2e-launch.json').read_text())
brain=Path('/Users/n/numi-brain-human-completion-20260907')
env=dict(os.environ,**prior['environment']);env['PATH']='/opt/homebrew/bin:'+env.get('PATH','')
(out/'costal-regression-environment.json').write_text(json.dumps(prior['environment'],indent=2)+'\n')
assert run('costal-regression',prior['command'],cwd=brain,env=env)==0
