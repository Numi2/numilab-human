import gzip,hashlib,json,os,pathlib,subprocess,sys,time
repo=pathlib.Path('/Users/n/MetalRobo-human-completion-20260907')
build=pathlib.Path('/Users/n/MetalRobo-human-completion-build-20260907')
root=repo/'Build/absolute-blood-20260912/legacy-cardiac-requalification'
root.mkdir(parents=True,exist_ok=True)
runs=[('refinement_2ms','.002',500),('refinement_1ms','.001',1000),('refinement_05ms','.0005',2000),('ten_cycles_2ms','.002',5000)]
source_paths=[pathlib.Path(p) for p in subprocess.check_output(['git','ls-files','matter/include','matter/src','matter/tools/cardiac_check.mm','matter/tools/shi_hose_reference'],cwd=repo,text=True).splitlines() if pathlib.Path(p).suffix in {'.h','.hpp','.c','.cpp','.mm','.metal','.metalinc'}]
source_paths += [pathlib.Path('matter/tools/fixtures/shi-hose.native.v2.json')]
binaries=['matter/numi-matter-cardiac-check','matter/shaders/NumiMatter.metallib','lib/libmetalrobo.dylib']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def identity():
 return {'native_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
  'sources':{str(p):sha(repo/p) for p in sorted(set(source_paths))},
  'binaries':{p:sha(build/p) for p in binaries}}
for name,dt,steps in runs:
 prefix=root/name
 if prefix.with_suffix('.execution.json').exists():raise RuntimeError('refuse overwrite existing run '+name)
 command=[str(build/'matter/numi-matter-cardiac-check'),str(repo/'matter/tools/fixtures/shi-hose.native.v2.json'),'--steps',str(steps),'--dt',dt,'--trace',str(prefix.with_suffix('.csv')),'--checkpoint',str(prefix.with_suffix('.failure.bin')),'--newton','12','--krylov','64']
 before=identity();record={'schema':'NumiHuman.LegacyCardiacExecution.v1','id':name,'argv':command,'cwd':str(repo),'environment':{'MTL_DEBUG_LAYER':'1'},'before':before}
 prefix.with_suffix('.started.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
 with prefix.with_suffix('.log').open('wb') as output:
  result=subprocess.run(command,cwd=repo,env={**os.environ,'MTL_DEBUG_LAYER':'1'},stdout=output,stderr=subprocess.STDOUT)
 record['exit_code']=result.returncode;record['after']=identity();record['source_and_binary_identity_unchanged']=record['after']==before
 raw=prefix.with_suffix('.csv')
 if raw.exists():
  data=raw.read_bytes();prefix.with_suffix('.csv.gz').write_bytes(gzip.compress(data,mtime=0));record['trace_raw_sha256']=hashlib.sha256(data).hexdigest();raw.unlink()
 record['artifacts']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in root.glob(name+'.*') if p.suffix in {'.log','.gz','.bin'}}
 prefix.with_suffix('.execution.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
 print(name,'exit='+str(result.returncode),'identity_unchanged='+str(record['source_and_binary_identity_unchanged']),flush=True)
 if result.returncode or not record['source_and_binary_identity_unchanged']:sys.exit(1)
