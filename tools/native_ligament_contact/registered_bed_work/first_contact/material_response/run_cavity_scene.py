"""Run the native body with the current cortical faces, material ledger and ordered connectivity."""
import json, os, sys
from pathlib import Path
from compile_connected_joint import digest
from run_cavity_motion import execute_native

def run(job,out):
    prior=job/'verified-connected-material/scene'; binding=out/'binding';scene=out/'scene';scene.mkdir()
    read=lambda p:json.loads(p.read_bytes())
    cfg=read(prior/'launch.json');env=cfg['environment'];changes={}
    stage=read(out/'scene-material-p1/launch.json')['environment']
    for k,v in stage.items():
        if env.get(k)!=v:changes[k]=v
    for rec in [binding/'tables.json',out/'packets/result.json',out/'native-replay/result.json']:
        changes.update(read(rec)['environment'])
    exe=binding/'app/numi-human-native';build=read(binding/'build.json');dylib=binding/'runtime/libmetalrobo.dylib'
    if digest(exe)!=build['model'] or digest(dylib)!=build['library']:raise ValueError('Unknown executable')
    for k,v in changes.items():
        if v.startswith('/'):
            options=[k+'_SHA256',k.rsplit('_',1)[0]+'_SHA256']
            if k.endswith('_RECEIPT_JSON'):options.append(k[:-5]+'_SHA256')
            hashes={changes[i] if i in changes else stage[i] for i in options if i in changes or i in stage}
            if hashes!={digest(Path(v))}:raise ValueError('Invalid scene input '+k)
            cfg['pins'][v]=hashes.pop()
    env.update(changes);env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']=str(out/'volume/current.node')
    cfg['pins'][str(out/'volume/current.node')]=digest(out/'volume/current.node')
    env['DYLD_LIBRARY_PATH']=str(binding/'runtime')+':'+env['DYLD_LIBRARY_PATH']
    cfg['argv'][0]=str(exe);cfg['pins'][str(exe)]=build['model'];cfg['pins'][str(dylib)]=build['library']
    cfg['argv'][4]=str(scene/'native-traces')
    for k,v in list(env.items()):
        if v.startswith(str(prior)+'/') and v not in cfg['pins']:env[k]=str(scene/Path(v).relative_to(prior))
    for k in ('NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT','NM_FEM_NEWTON_TRACE_ROOT','NM_FEM_BACKTRACKING_TRACE_ROOT'):
        env[k]='0'
    cfg['parent_launch']=digest(prior/'launch.json')
    for v in set(cfg['argv']+list(env.values()))&set(cfg['pins']):
        if digest(Path(v))!=cfg['pins'][v]:raise ValueError('Input changed: '+v)
    (scene/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    clean={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}
    print(execute_native(cfg['argv'],cfg['cwd'],clean|env,scene,minimum_free_bytes=4*2**30,timeout_s=600),flush=True)
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
