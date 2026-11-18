import argparse, hashlib, json, os, shutil, subprocess, time
from pathlib import Path

here=Path.cwd()
baseline=here.parent/'local-bed-layers-002'
p=argparse.ArgumentParser()
p.add_argument('output')
p.add_argument('--steps',type=int,default=6000)
p.add_argument('--pressure',action='store_true')
p.add_argument('--intervention',action='store_true')
p.add_argument('--capture',default='')
p.add_argument('--passive-upper',action='store_true')
p.add_argument('--thumb',action='store_true')
args=p.parse_args()
out=here/args.output
assert out.name==args.output and out.parent==here and args.steps>0
deck=json.loads((baseline/'loaded-24s/launch.json').read_text())
build=json.loads((here/'build-verification.json').read_text())
def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
for key,value in deck['pins'].items():
    assert digest(key)==value,key
pins=deck['pins']
pins[build['binary']]=build['sha256']
pins[build['runtime_object']]=build['runtime_sha256']
assert digest(build['runtime_object'])==build['runtime_sha256']
local_shaders=[here/'metalrobo'/name for name in ('MetalRobo.metallib','HumanRestingSkin.metallib')]
for file in local_shaders:
    pins[str(file)]=digest(file)
assert digest(build['binary'])==build['sha256']
env=deck['environment']
argv=deck['argv']
argv[0]=build['binary']
argv[4]=str(out/'native')
argv[argv.index('--muscle-step-count')+1]=str(args.steps)
env['NUMI_HUMAN_LOCAL_BED_REGIONAL_THICKNESS']='1'
env['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']='1' if args.pressure else '0'
if args.capture:
    env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']=args.capture
    env['NUMI_HUMAN_RESTING_CAPTURE_REUSE_SOURCE_DIR']=str(here.parent/'regional-bed-layers-001/capture-4s/native')
argv.append('--resting-hip-capsule-reference')
if args.passive_upper:
    argv.append('--persistent-source-passive-joint-tissue')
if args.thumb:
    assert args.passive_upper
    env['NUMI_HUMAN_RESTING_THUMB_PASSIVE_REFERENCE']='1'
if args.intervention:
    argv.extend(['--resting-drive-intervention','4','8','0.6'])
env['NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT']=str(out/'common-failure.json')
deck['mode']=vars(args)
deck['design']['measurements']='Absolute pleural gauge pressure boundary A/B with matched drive and mechanical startup. Timing includes accepted viewer cadence.'
assert shutil.disk_usage(here).free > (9.25 if args.capture else 9.10)*1024**3
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
            if free < 8.75*1024**3:
                stop=f'disk guard ({free / 2**30:.3f} GiB)'
            if stop:
                process.terminate()
                break
            time.sleep(.5)
        status=process.wait(timeout=20)
end={'exit_code':status,'seconds_process':time.monotonic()-start,'stopped':stop}
(out/'execution.json').write_text(json.dumps(end,indent=2)+'\n')
pack=out/'native'/'resting-human.mrvpack'
shared=baseline.parent/'local-bed-layers-001'/'loaded-pilot'/'native'/'resting-human.mrvpack'
if pack.is_file() and shared.is_file() and not pack.samefile(shared) and digest(pack)==digest(shared):
    link=pack.with_suffix('.shared')
    os.link(shared,link)
    os.replace(link,pack)
print(end,flush=True)
raise SystemExit(status)
