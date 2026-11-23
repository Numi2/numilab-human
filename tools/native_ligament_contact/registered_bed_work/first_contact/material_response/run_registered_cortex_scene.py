"""Run the complete native scene with all registered physical shoulder rows."""
import json,os,sys
from pathlib import Path
from compile_connected_joint import digest
from run_cavity_motion import execute_native

def run(base,build,out,registration=None):
    config=json.loads((base/'launch.json').read_bytes());receipt=json.loads((build/'build.json').read_bytes())
    app,lib,metal=[build/k for k in ['app/numi-human-native','libmetalrobo.dylib','metal/NumiMatter.metallib']]
    for path in (app,lib,metal):
        if receipt['binaries'][str(path)]!=digest(path):raise ValueError('Binary generation mismatch')
    out.mkdir();env={k:v.replace(str(base)+'/',str(out)+'/') for k,v in config['environment'].items()}
    old_shader=env['NUMI_MATTER_METALLIB_OVERRIDE'];old_app=config['argv'][0]
    config['pins'].pop(old_shader);config['pins'].pop(old_app)
    env['NUMI_MATTER_METALLIB_OVERRIDE']=str(metal);env['DYLD_LIBRARY_PATH']=str(build)+':'+env.get('DYLD_LIBRARY_PATH','')
    if registration is not None:
        changes=json.loads(registration.read_bytes())['environment']
        for key,path in changes.items():
            if key.endswith('_CSV'):
                expected=changes[key[:-4]+'_SHA256']
                if digest(Path(path))!=expected:raise ValueError('Unverified surface '+key)
                config['pins'][path]=expected
        env.update(changes);config['surface_rebinding']=digest(registration)
    config['environment']=env;config['argv'][0]=str(app);config['argv'][4]=str(out/'native-traces')
    config['pins'].update(receipt['binaries']);config['cortical_force_build']=digest(build/'build.json');config['parent_launch']=digest(base/'launch.json')
    (out/'launch.json').write_text(json.dumps(config,indent=2)+'\n')
    print(execute_native(config['argv'],config['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NUMI_','NM_'))}|env,out,minimum_free_bytes=4*2**30,timeout_s=550),flush=True)
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
