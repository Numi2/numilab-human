import hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
base=Path(__file__).parent
label=sys.argv[1] if len(sys.argv)>1 else 'transaction'
request=json.loads((Path('/Users/n/human-compliant-equilibrium-20260910')/('horizon-request.json' if label.startswith('horizon') else 'describe-request.json' if label.startswith('describe') else 'legacy-request.json' if label.startswith('legacy') else 'transaction-request.json')).read_text())
if label.startswith('refine-'):
 dt=int(label.split('-')[1].removesuffix('us'))
 folder=base/(f'fixture-exact-{dt}us'+('-iterations32' if 'iterations32' in label else ''))
 identity=json.loads((folder/'prepared.json').read_text())
 request['command'][-1]='testPreparedNativeTimestepTrajectory'
 request['environment'].update(NUMANX_PREPARED_TIMESTEP_US=str(dt),NUMANX_PREPARED_DURATION_US='1600',
  NUMANX_MATTER_WORLD_PACKAGE=str(folder/f'prepared-{dt}us.nmatterpack'),
  NUMANX_MATTER_WORLD_FP=identity['world_fp'],NUMANX_INITIAL_STATE=str(folder/'prepared.nhinit'),
  NUMANX_INITIAL_STATE_FP=identity['initial_state_fp'])
request['environment']['MRNX_CANDIDATE_GPU_TIMING']=os.environ.get('MRNX_CANDIDATE_GPU_TIMING','0')
if 'NM_HUMAN_SUPPORT_TRACE_ROOT' in os.environ:
 request['environment']['NM_HUMAN_SUPPORT_TRACE_ROOT']=os.environ['NM_HUMAN_SUPPORT_TRACE_ROOT']
env=dict(os.environ, **request['environment'])
env['PATH']='/opt/homebrew/bin:'+env.get('PATH','')
request['source_state']={}
for name,path in [('native','/Users/n/MetalRobo-human-completion-20260907'),('brain',request['cwd'])]:
 request['source_state'][name]={'revision':subprocess.check_output(['git','-C',path,'rev-parse','HEAD'],text=True).strip(),
  'status':subprocess.check_output(['git','-C',path,'status','--short'],text=True)}
request['source_sha256']={}
for root,files in [('/Users/n/MetalRobo-human-completion-20260907',[
 'apps/numanx_fullbody_bridge_probe.mm','apps/numilab_human_myosim_reference_probe.cpp',
 'include/metalrobo/MetalArticulatedOperator.hpp','matter/src/metal/fgmres.metalinc',
 'matter/src/runtime.mm','src/metal/ArticulatedOperator.metal','src/metal/MetalArticulatedOperator.mm']),
 (request['cwd'],['Tests/NumiBrainMetalTests/MetalNumanXBridgeV1EndToEndTests.swift'])]:
 for file in files:
  source=Path(root)/file;request['source_sha256'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
request['competing_workloads'] = '\n'.join(l for l in subprocess.check_output(['ps','-axo','pid,etime,%cpu,command'],text=True).splitlines() if ('numivivo md-run ' in l or 'numivivo md-benchmark ' in l))
if request['competing_workloads']: raise SystemExit('competing workload')
request['artifact_sha256']={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in request['environment'].items() if Path(v).is_file()}
(base/(label+'-launch.json')).write_text(json.dumps(request,indent=2)+'\n')
binary = Path(request['command'][0]) if label.startswith('describe') else Path(request['cwd']) / '.build/release/NumiBrainPackageTests.xctest/Contents/MacOS/NumiBrainPackageTests'
request['brain_test_binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
request['brain_test_binary_bytes'] = binary.stat().st_size
request['authoring_revision'] = '113a82682db9a234aa85aa225a8b89283f7c52e3'
request['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(base/(label+'-launch.json')).write_text(json.dumps(request,indent=2)+'\n')
request['runner_timeout_seconds']=600 if label.startswith('horizon') else 300
t=time.monotonic()
with (base/(label+'.log')).open('w') as log:
 p=subprocess.Popen(request['command'],cwd=request['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try: code=p.wait(timeout=request['runner_timeout_seconds'])
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  code=124
request.update(returncode=code,elapsed_seconds=time.monotonic()-t)
(base/(label+'-launch.json')).write_text(json.dumps(request,indent=2)+'\n')
print(json.dumps({'returncode':code,'elapsed_seconds':request['elapsed_seconds']}))
sys.exit(code)
