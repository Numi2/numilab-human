"""Build the existing native Human executable for the emitted bone hash and run the complete scene."""
from pathlib import Path
import hashlib,json,os,resource,shutil,subprocess,sys,time

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def run(prior,source,catalog,bone,tendon,output):
    output.mkdir(exist_ok=False);build=output/'build';build.mkdir()
    cfg=json.loads(prior.read_text());base=source/'candidate'
    assert sha(base/'numi-human-native')==cfg['candidate_binary_sha256']
    assert all(sha(p)==s for p,s in cfg['pins'].items())
    code='numilab_human_myosim_visual_probe.mm';shutil.copyfile(base/code,build/code)
    shutil.copyfile(catalog/'NumiHumanRegisteredLigaments.hpp',build/'NumiHumanRegisteredLigaments.hpp')
    commands=[[(str(build/Path(s).name) if Path(s).parent==base else s) for s in cmd] for cmd in json.loads((base/'build-commands.json').read_text())]
    for cmd in commands:subprocess.run(cmd,cwd=Path.home()/'Work/skin-fat-native-integration-209/build-native',check=True)
    exe=build/'numi-human-native';old=str(prior.parent)
    original_b=Path(cfg['argv'][3])
    assert json.loads(bone.with_suffix('.manifest.json').read_text())['source']['connected_girdle_parent']['bone_sha256']==sha(original_b)
    td=json.loads(tendon.with_suffix('.manifest.json').read_text())
    assert sha(tendon)==td['payload']['sha256'] and td['source']['regional_recook']['compiled_bone_sha256']==sha(bone)
    assert Path(cfg['argv'][4])==prior.parent/'native'
    args=[v.replace(old+'/',str(output)+'/') for v in cfg['argv']]
    args[0]=str(exe);args[3]=str(bone);args[args.index('--tendon-payload')+1]=str(tendon)
    assert '--resting-registered-ligaments' in args and '--resting-scene' in args
    env={k:v.replace(old+'/',str(output)+'/') for k,v in cfg['environment'].items()}
    step_json={'parent_launch_sha256':sha(prior),'argv':args,'environment':env,'cwd':cfg['cwd'],'commands':commands,
               'hashes':{str(p):sha(p) for p in (exe,bone,tendon,build/code,build/'NumiHumanRegisteredLigaments.hpp')}}
    (output/'launch.json').write_text(json.dumps(step_json,indent=2)+'\n')
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));start=time.monotonic()
    with (output/'stdout.log').open('w') as o,(output/'stderr.log').open('w') as e:
        child=subprocess.run(args,cwd=cfg['cwd'],env={**os.environ,**env},stdout=o,stderr=e,timeout=150)
    err=(output/'stderr.log').read_text()
    result=dict(exit_code=child.returncode,wall_seconds=time.monotonic()-start,errors=err.splitlines()[:12])
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)
if __name__=='__main__':run(*[Path(p).resolve() for p in sys.argv[1:]])
