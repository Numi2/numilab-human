import hashlib,json,pathlib,subprocess,time,sys
root=pathlib.Path('/Users/n/human-compliant-equilibrium-20260910')
repo=pathlib.Path('/Users/n/MetalRobo-human-completion-20260907')
build=pathlib.Path('/Users/n/MetalRobo-human-completion-build-20260907')
label=sys.argv[1]; iterations=sys.argv[2]
command=[str(build/'bin/metalrobo_numilab_human_myosim_visual_probe'),'--source-compliant-certificate',
'/Users/n/human-completion-20260907/input/myosim-fullbody-core-reference.nhrigid',
'/Users/n/human-completion-20260907/input/myosim-fullbody-muscle-reference.nhmyo',
'/Users/n/human-capsule-20260908/input/myosim-fullbody-support-primitives.nhcnt',
'/Users/n/human-prepared-state-20260908/fixture/prepared.nhinit',
'/Users/n/human-completion-20260907/input/myosim-fullbody-joint-equalities-source-compliance.nheq',
'/Users/n/human-dynamic-limits-20260908/myosim-fullbody-joint-limits.nhlim',iterations]
if len(sys.argv)>3 and sys.argv[3]=='recruit':command.append('--recruit')
if len(sys.argv)>3 and sys.argv[3]=='support':command.append('--support-reactions-only')
if len(sys.argv)>4:command[5]=sys.argv[4]
workloads=subprocess.check_output(['ps','-axo','pid,pcpu,etime,command'],text=True)
if any('numivivo md-run' in l or 'numivivo md-benchmark' in l for l in workloads.splitlines()):raise SystemExit('competing workload')
meta={'command':command,'native_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
'native_status':subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True),
'authority':'offline_native_initialization_only','hashes':{}}
for p in [pathlib.Path(x) for x in command if pathlib.Path(x).is_file()]+[build/'lib/libmetalrobo.dylib',repo/'src/core/NumiHumanMuscleEquilibrium.cpp',repo/'include/metalrobo/NumiHumanCompliantEquilibrium.hpp',repo/'apps/numilab_human_myosim_visual_probe.mm']:
 meta['hashes'][str(p)]=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
(root/f'{label}-launch.json').write_text(json.dumps(meta,indent=2)+'\n')
start=time.monotonic()
with (root/f'{label}.log').open('w') as log:
 try:run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600);meta['returncode']=run.returncode
 except subprocess.TimeoutExpired:meta['returncode']='timeout'
meta['elapsed_seconds']=time.monotonic()-start
(root/f'{label}-completion.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({k:meta[k] for k in ('returncode','elapsed_seconds')}),flush=True)
