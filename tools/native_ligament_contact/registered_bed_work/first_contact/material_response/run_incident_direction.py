"""Replay the closed native launch with the tested incident-volume host/shader pair."""
import json,os,sys
from pathlib import Path
from compile_connected_joint import digest
from run_cavity_motion import execute_native
def run(parent,build,out,app=None):
    cfg=json.loads((parent/'launch.json').read_bytes());b=json.loads((build/'build.json').read_bytes())
    env=cfg['environment'];args=cfg['argv'];pins=cfg['pins']
    for val in set(args+list(env.values()))&set(pins):
        if digest(Path(val))!=pins[val]:raise ValueError('Changed physical input '+val)
    libs=env['DYLD_LIBRARY_PATH'].split(':');rt='libmetalrobo.dylib'
    old=Path(libs[0])/rt;new=build/'runtime'/rt
    sk='NUMI_MATTER_METALLIB_OVERRIDE';existing=Path(env[sk]);shader=build/'metal/NumiMatter.metallib'
    for key,f in [('parent_runtime',old),('parent_shader',existing),('runtime',new),('shader',shader)]:
        if digest(f)!=b[key] or (key.startswith('parent_') and pins[str(f)]!=b[key]):raise ValueError('Changed dependency '+key)
    if app is not None:
        owner=json.loads((app/'build.json').read_bytes());binary=app/'numi-human-native'
        if owner['parent_app']!=pins[args[0]] or owner['runtime']!=b['runtime'] or digest(binary)!=owner['app']:raise ValueError('Static solver binding changed')
        args[0]=str(binary);pins[str(binary)]=owner['app']
    out.mkdir();env['DYLD_LIBRARY_PATH']=':'.join([str(new.parent)]+libs[1:]);env[sk]=str(shader)
    pins[str(new)]=b['runtime'];pins[str(shader)]=b['shader'];args[4]=str(out/'native-traces')
    for k,v in list(env.items()):
        if v.startswith(str(parent)+'/') and v not in pins:env[k]=str(out/Path(v).relative_to(parent))
    cfg['parent_launch']=digest(parent/'launch.json');cfg['incident_direction_build']=digest(build/'build.json')
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    clean={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}
    result=execute_native(args,cfg['cwd'],clean|env,out,minimum_free_bytes=4*2**30,timeout_s=600)
    if not any(str(new) in l for l in (out/'stderr.log').open() if l.startswith('dyld[')):
        raise ValueError('Test did not load the intended physical runtime')
    print(result,flush=True)
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
