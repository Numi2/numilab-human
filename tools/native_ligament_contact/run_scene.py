"""Launch the complete scene with the matched ligament adapter and shader images."""
import hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
from verify_trace import verify_trace
h=Path(__file__).resolve().parent;b=h/'build'
prior,out=map(Path,sys.argv[1:]);out.mkdir(exist_ok=False)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
cfg=json.loads(prior.read_text())
for path,digest in cfg['hashes'].items():assert sha(path)==digest,path
origin=cfg.get('launch_output_parent',str(prior.parent))
args=[x.replace(origin+'/native',str(out)+'/native') for x in cfg['argv']]
args[0]=str(b/'numi-human-native')
env={k:v.replace(origin+'/',str(out)+'/') for k,v in cfg['environment'].items()}
if 'input_parent' not in cfg: raise ValueError('The complete composed native package is required')
env['DYLD_LIBRARY_PATH']=str(b)+':'+env['DYLD_LIBRARY_PATH']
env['NUMI_MATTER_METALLIB_OVERRIDE']=str(b/'NumiMatter.metallib')
pins={str(x):sha(x) for x in (prior,Path(args[0]),b/'libmetalrobo.dylib',b/'NumiMatter.metallib',h/'strip.json')}
(out/'launch.json').write_text(json.dumps(dict(argv=args,environment=env,cwd=cfg['cwd'],hashes=pins),indent=2)+'\n')
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
start=time.monotonic()
with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
    proc=subprocess.run(args,cwd=cfg['cwd'],env={**os.environ,**env},stdout=stdout,stderr=stderr)
wall=time.monotonic()-start
csv=out/'native/resting-coupled.csv';rows=sum(1 for _ in csv.open())-1 if csv.exists() else 0
report=dict(exit_code=proc.returncode,wall_seconds=wall,trace_rows=rows)
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(report)

if proc.returncode:
    raise SystemExit(proc.returncode)
verify_trace(csv, int(args[args.index('--muscle-step-count')+1]), float(args[args.index('--muscle-step-seconds')+1]))
