"""Run the real registered-force scene up to the material site admission."""
import hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
original,exe,output=map(Path,sys.argv[1:])
output.mkdir()
cfg=json.loads(original.read_text())
def digest(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
previous_binary_sha256=cfg['pins'].pop(cfg['argv'][0])
for path,sha in cfg['pins'].items():assert digest(path)==sha
cfg['pins'][str(exe)]=digest(exe)
cfg['previous_binary_sha256']=previous_binary_sha256
old=str(original.parent); new=str(output)
argv=[s.replace(old+'/',new+'/') for s in cfg['argv']]
argv[0]=str(exe); assert '--resting-registered-ligaments' in argv and argv[4]==str(output/'native')
env={k:v.replace(old+'/',new+'/') for k,v in cfg['environment'].items()}
cfg.update(argv=argv,environment=env,original_launch_sha256=digest(original),candidate_binary_sha256=digest(exe))
(output/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
start=time.monotonic()
with (output/'stdout.log').open('w') as out,(output/'stderr.log').open('w') as err:
    child=subprocess.run(argv,env={**os.environ,**env},cwd=cfg['cwd'],stdout=out,stderr=err,timeout=150)
s=(output/'stderr.log').read_text()
rows=[line for line in s.splitlines() if 'registered ligament span lies inside a closed bone' in line]
result=dict(exit_code=child.returncode,wall_s=time.monotonic()-start,errors=rows)
(output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
assert child.returncode and all('side='+str(i)+' stable_id='+str(sid) in s for i,sid in ((0,13),(1,12))),s[-2000:]
