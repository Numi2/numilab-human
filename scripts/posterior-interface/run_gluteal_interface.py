"""Replay the pinned body with the optional constitutive coupling on its Metal force owner."""
import json
import os
from pathlib import Path
import shutil
import sys
from compile_connected_joint import digest
from run_cavity_motion import execute_native


def run(parent, build, dest, steps):
    cfg=json.loads((parent/'launch.json').read_bytes())
    owner=json.loads((build/'build.json').read_bytes())
    for f in ('material.f32','libmetalrobo.dylib','NumiMatter.metallib'):
        if digest(build/f)!=owner['pins'][f]:
            raise ValueError(f'Changed built owner {f}')
    env=cfg['environment']
    for k,s in list(env.items()):
        if s.startswith(str(parent)+'/') and s not in cfg['pins']:
            env[k]=str(dest/Path(s).relative_to(parent))
    if int(cfg['argv'][cfg['argv'].index('--muscle-step-count')+1])!=steps:
        raise ValueError('Use the existing paired control with matching horizon and capture cadence')
    cfg['argv'][4]=str(dest/'native-traces')
    for key in set(cfg['argv']+list(env.values()))&set(cfg['pins']):
        if digest(Path(key))!=cfg['pins'][key]:
            raise ValueError(f'Unmatched native input {key}')
    env.update(NUMI_MATTER_METALLIB_OVERRIDE=str(build/'NumiMatter.metallib'),
               DYLD_LIBRARY_PATH=str(build)+':'+env['DYLD_LIBRARY_PATH'],
               NUMI_HUMAN_GLUTEAL_OPERATOR=owner['operator'],
               NUMI_HUMAN_GLUTEAL_MATRIX=str(build/'material.f32'),
               NUMI_HUMAN_GLUTEAL_OBSERVATION=str(dest/'force-observer.f32'))
    cfg['pins'].update({str(build/f):owner['pins'][f] for f in ('material.f32','libmetalrobo.dylib','NumiMatter.metallib')})
    cfg['compartment_build']=digest(build/'build.json')
    cfg['control_launch']=digest(parent/'launch.json')
    if shutil.disk_usage(dest.parent).free<4*2**30:
        raise ValueError('Insufficient storage for the matched prefix')
    dest.mkdir()
    (dest/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    clean={k:v for k,v in os.environ.items() if not k.startswith(('NUMI_','NM_'))}
    result=execute_native(cfg['argv'],cfg['cwd'],clean|env,dest,minimum_free_bytes=4*2**30,timeout_s=1000)
    (dest/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)

if __name__=='__main__':
    run(*(Path(s).resolve() for s in sys.argv[1:4]),int(sys.argv[4]))
