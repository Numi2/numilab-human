from pathlib import Path
import hashlib,json,os,subprocess,time,signal
out=Path('/Users/n/numi-human-active-locomotion-20260908')
base=json.loads((out/'costal-native-arguments.json').read_text());base[1]='capture'
program=out/'costal-standing.json';sha=hashlib.sha256(program.read_bytes()).hexdigest()
# These are explicit research coordinates, not a frozen behavior TaskPack.
def fnv(data):
 value=0xcbf29ce484222325
 for byte in data:value=((value^byte)*0x100000001b3)&0xffffffffffffffff
 return f'{value:016x}'
common=['--muscle-locomotor-program',str(program),'--roots','4','--episode','1','--seed','17',
 '--source-revision',subprocess.check_output(['git','rev-parse','HEAD'],cwd='/Users/n/numi-brain-human-completion-20260907',text=True).strip(),
 '--dataset-id','human-costal-locomotor-research-20260908','--dataset-revision',sha,
 '--task-fp',fnv(b'four-root-standing-recruitment-research-v1'),
 '--scene-fp','cc354c5cf1bcb81c','--object-fp','4b26cde26f8a228a','--embodiment-fp','4b26cde26f8a228a']
for mode,extra in [('active',[]),('replay',[]),('unavailable',['--sensor-intervention','invalidate-all']),('emergency',['--hard-safety-intervention','emergency-stop'])]:
 dest=out/('costal-'+mode);dest.mkdir(exist_ok=True)
 command=base+common+['--run-id','costal-locomotor-'+mode,'--artifact-dir',str(dest)]+extra
 start=time.monotonic()
 with (out/('costal-'+mode+'.log')).open('w') as log:
  p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,MTL_DEBUG_LAYER='1'),start_new_session=True)
  try:code=p.wait(timeout=240)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);code=124
 receipt={'command':command,'returncode':code,'elapsed_seconds':time.monotonic()-start,'program_sha256':sha,'binary_sha256':hashlib.sha256(Path(base[0]).read_bytes()).hexdigest()}
 (out/('costal-'+mode+'-launch.json')).write_text(json.dumps(receipt,indent=2)+'\n')
 print(mode,code,receipt['elapsed_seconds'],flush=True)
 if code:raise SystemExit(code)
