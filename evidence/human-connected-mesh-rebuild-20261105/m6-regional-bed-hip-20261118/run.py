import hashlib, json, os, shutil, subprocess, time
from pathlib import Path
import sys

here=Path.cwd()
baseline=here.parent/'local-bed-layers-002'
out=here/sys.argv[1]
mode=sys.argv[2]
assert out.name==sys.argv[1] and mode in ('uniform','regional','hip')
deck=json.loads((baseline/'loaded-24s/launch.json').read_text())
build=json.loads((here/'build-verification.json').read_text())
def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
for key,value in deck['pins'].items():
    assert digest(key)==value,key
pins=deck['pins']
pins[build['binary']]=build['sha256']
local_shaders=[here/'metalrobo'/name for name in ('MetalRobo.metallib','HumanRestingSkin.metallib')]
for file in local_shaders:
    pins[str(file)]=digest(file)
assert digest(build['binary'])==build['sha256']
env=deck['environment']
argv=deck['argv']
argv[0]=build['binary']
argv[4]=str(out/'native')
argv[argv.index('--muscle-step-count')+1]=sys.argv[3] if len(sys.argv)>3 else '8000'
env['NUMI_HUMAN_LOCAL_BED_REGIONAL_THICKNESS']='0' if mode=='uniform' else '1'
if mode=='hip':
    argv.append('--resting-hip-capsule-reference')
env['NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT']=str(out/'common-failure.json')
deck['mode']=mode
deck['design']['measurements']='Area-weighted exact-source reference construction thickness, support load with regional geometry, full viewer validity, unchanged mass. Hip uses the existing four coupled soft nonlinear capsule rows.'
assert shutil.disk_usage(here).free > 10*1024**3
out.mkdir()
(out/'launch.json').write_text(json.dumps(deck,indent=2)+'\n')
base_env={k:v for k,v in os.environ.items() if not k.startswith(('NUMI_','NM_'))}
with (out/'stdout.log').open('wb') as so,(out/'stderr.log').open('wb') as se:
    start=time.monotonic()
    with subprocess.Popen(argv,cwd=deck['cwd'],env=base_env|env,stdout=so,stderr=se) as process:
        stop=None
        while process.poll() is None:
            if time.monotonic()-start > 300:
                stop='wall guard'
            free=shutil.disk_usage(here).free
            if free < 9*1024**3:
                stop=f'disk guard ({free / 2**30:.3f} GiB)'
            if stop:
                process.terminate()
                break
            time.sleep(.5)
        status=process.wait(timeout=20)
end={'exit_code':status,'seconds_process':time.monotonic()-start,'stopped':stop}
(out/'execution.json').write_text(json.dumps(end,indent=2)+'\n')
print(end,flush=True)
raise SystemExit(status)
