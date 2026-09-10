from pathlib import Path
import json,hashlib,subprocess,os,time,sys
b=Path(__file__).parent;label=sys.argv[1]
source=Path('/Users/n/MetalRobo-human-completion-20260907');build=Path('/Users/n/MetalRobo-human-completion-build-20260907')
command=[str(build/'bin/metalrobo_numilab_human_myosim_reference_probe'),'/Users/n/human-completion-20260907/input/myosim-fullbody-core-reference.nhrigid','/Users/n/human-completion-20260907/input/myosim-fullbody-muscle-reference.nhmyo']
command+=['--metal'] if label.startswith('default') else ['--prepared-paths','/Users/n/human-compliant-equilibrium-20260910/fixture/prepared.nhinit']
env=dict(os.environ,MTL_DEBUG_LAYER='1')
files=[Path(command[0]),Path(command[1]),Path(command[2]),build/'lib/libmetalrobo.dylib',build/'shaders/MetalRobo.metallib',source/'src/metal/MujocoMuscleReference.metal',source/'apps/numilab_human_myosim_reference_probe.cpp']
if not label.startswith('default'): files.append(Path(command[-1]))
if len(sys.argv) > 2: command += ['--timestep-us',sys.argv[2]]
files.append(Path(__file__))
r={'command':command,'environment':{'MTL_DEBUG_LAYER':'1'},'source_revision':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),'source_status':subprocess.check_output(['git','-C',str(source),'status','--short'],text=True),'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
r['competing_workloads']='\n'.join(l for l in subprocess.check_output(['ps','-axo','pid,etime,%cpu,command'],text=True).splitlines() if 'numivivo md-run ' in l or 'numivivo md-benchmark ' in l)
if r['competing_workloads']:raise SystemExit('competing workload')
t=time.monotonic()
with (b/(label+'.log')).open('w') as f: r['returncode']=subprocess.run(command,cwd=build,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180).returncode
r['elapsed_seconds']=time.monotonic()-t
(b/(label+'-launch.json')).write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'label':label,'returncode':r['returncode'],'elapsed_seconds':r['elapsed_seconds']}))
sys.exit(r['returncode'])
