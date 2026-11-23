"""Run one native scene with pinned registration replacements."""
import json,os,sys
from pathlib import Path
from compile_connected_joint import digest
from run_cavity_motion import execute_native

def run(base,dest,changes):
    cfg=json.loads((base/'launch.json').read_bytes());dest.mkdir()
    env={k:v.replace(str(base)+'/',str(dest)+'/') for k,v in cfg['environment'].items()};pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    expanded=[]
    for spec in changes:
        if spec.startswith('--tables='):
            report=json.loads(Path(spec.split('=',1)[1]).read_bytes())
            for key,record in report['files'].items():
                if digest(Path(record['path']))!=record['sha256']:raise ValueError('Unpinned input '+key)
                expanded.append(key+'='+record['path'])
        elif spec.startswith('--build='):
            build=Path(spec.split('=',1)[1]).resolve();app=build/'app/numi-human-native'
            if digest(app)!=json.loads((build/'build.json').read_bytes())['executable']:raise ValueError('Stale scene binary')
            cfg['pins'].pop(cfg['argv'][0],None);cfg['argv'][0]=str(app);cfg['pins'][str(app)]=digest(app)
        else:expanded.append(spec)
    for spec in expanded:
        key,value=spec.split('=',1);path=Path(value).resolve();old=env[pre+key+'_CSV']
        cfg['pins'].pop(old,None);env[pre+key+'_CSV']=str(path);env[pre+key+'_SHA256']=digest(path);cfg['pins'][str(path)]=digest(path)
    cfg['argv'][4]=str(dest/'native-traces');cfg['environment']=env;cfg['parent_launch']=digest(base/'launch.json')
    (dest/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(execute_native(cfg['argv'],cfg['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|env,dest,minimum_free_bytes=4*2**30,timeout_s=400),flush=True)
if __name__=='__main__':run(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve(),sys.argv[3:])
