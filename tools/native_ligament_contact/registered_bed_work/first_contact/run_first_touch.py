"""Run the complete native scene with the first-contact assembly and normal gravity."""
import json
import os
import shutil
import sys
from pathlib import Path
from run_cavity_motion import digest, execute_native


def run(parent, installed, out, count, shader_build=None):
    cfg=json.loads((parent/'launch.json').read_text())
    install=json.loads((installed/'installation.json').read_text())
    if install['parent_launch']!=digest(parent/'launch.json') or not install['material']['passed'] or any(any(r) for r in install['layer_pairs']):
        raise ValueError('First-contact preparation has no matching certificate')
    source_env=cfg['environment']
    for field in ('NUMI_HUMAN_FEM_SKIN_SUPPORT','NUMI_HUMAN_RESTING_PASSIVE_FASCIA','NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM'):
        if source_env[field]!='1':raise ValueError('Missing tissue mechanics')
    for filename,expected in cfg['pins'].items():
        if digest(filename)!=expected:raise ValueError('Input changed: '+filename)
    argv=list(cfg['argv'])
    old_dir=Path(argv[4]).parent
    exe=installed/'owner/numi-human-native'
    pos=installed/'current.node'
    if digest(exe)!=install['binary'] or digest(pos)!=install['loaded_pose']:
        raise ValueError('Binary or material pose changed')
    out.mkdir()
    support_i=argv.index('--support-contact-payload')+1
    if 'support' in install:
        if digest(argv[support_i])!=install['original_support_sha256'] or digest(install['support'])!=install['support_sha256']:
            raise ValueError('Fixed support identity changed')
        argv[support_i]=install['support']
    argv[0]=str(exe)
    argv[4]=str(out/'native-traces')
    argv[argv.index('--root-pose')+3]=str(install['root_z_m'])
    argv[argv.index('--muscle-step-count')+1]=str(count)
    env={k:v.replace(str(old_dir)+'/',str(out)+'/') for k,v in source_env.items()}
    env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']=str(pos)
    env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']=str(count)
    pins=cfg['pins'].copy()
    pins.update({str(exe):digest(exe),str(pos):digest(pos)})
    pins[argv[support_i]]=digest(argv[support_i])
    if shader_build is not None:
        ident=json.loads((shader_build/'identities.json').read_text())
        old=env['NUMI_MATTER_METALLIB_OVERRIDE']
        new=str(shader_build/'NumiMatter.metallib')
        if digest(old)!=ident['parent_metallib'] or digest(new)!=ident['metallib'] or pins.get(old)!=digest(old):
            raise ValueError('Unmatched experimental contact shader')
        env['NUMI_MATTER_METALLIB_OVERRIDE']=new
        pins[new]=ident['metallib']
        if 'library' in ident:
            previous=Path(env['DYLD_LIBRARY_PATH'])/'libmetalrobo.dylib'
            lib=shader_build/'libmetalrobo.dylib'
            binary=shader_build/'numi-human-native'
            if (digest(previous)!=ident['parent_library'] or digest(lib)!=ident['library'] or
                    digest(exe)!=ident['parent_binary'] or digest(binary)!=ident['binary']):
                raise ValueError('Mismatched candidate compiler, library or application')
            env['DYLD_LIBRARY_PATH']=str(shader_build)
            pins[str(lib)]=ident['library'];pins[str(binary)]=ident['binary']
            argv[0]=str(binary)
    record=dict(cfg,argv=argv,environment=env,pins=pins,parent_launch=install['parent_launch'],first_contact=digest(installed/'installation.json'),compiled_material_owner=str(exe.parent))
    (out/'launch.json').write_text(json.dumps(record, indent=2)+'\n')
    actual={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|env
    result=execute_native(argv,cfg['cwd'],actual,out)
    return result['exit']


if __name__=='__main__':
    sys.exit(run(*(Path(v).resolve() for v in sys.argv[1:4]),int(sys.argv[4]),Path(sys.argv[5]).resolve() if len(sys.argv)==6 else None))
