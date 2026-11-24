"""Run the unchanged coupled native scene with the exact material registration and replay."""
from pathlib import Path
import json,os,sys
from compile_connected_joint import digest
from run_cavity_motion import execute_native

def run(work,out):
    q=work.parent
    source=q/'gravity-scene'
    read=lambda p:json.loads(p.read_bytes())
    owner=read(work/'binding/build.json')
    cfg=read(source/'launch.json')
    env=cfg['environment']
    args=cfg['argv']
    pins=cfg['pins']
    app=work/'binding/app/numi-human-native'
    rt=work/'binding/runtime'
    vol=q/'qualified-material'
    updates={}
    for f in ['current-p1/result.json','recorded-mass/packets/result.json','recorded-mass/native-replay/result.json','binding/tables/result.json']:
        updates.update(read(work/f)['environment'])
    if read(work/'recorded-mass/native-replay/result.json')['exit'] or read(work/'joint-physical-check/verification.json')['wall_crossings']:
        raise ValueError('Failed material preparation')
    pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    for k,name in [('TETGEN_NODE','mesh.node'),('TETGEN_ELEMENTS','mesh.ele'),('CHILD_LABELS','current-child.labels.csv')]:
        updates[pre+k]=str(vol/name)
        updates[pre+k+'_SHA256']=digest(vol/name)
    updates['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']=str(work/'selected-pose/current.node')
    # Match redundant owner-side hash variables by identity; every substituted file is re-hashed.
    immutable={}
    for k,new in updates.items():
        if k in env and env[k] in pins and new.startswith('/') and Path(new).is_file():
            old=env[k]
            if old in immutable and immutable[old]!=new:raise ValueError('Ambiguous input alias')
            immutable[old]=new
    hashes={pins[k]:digest(Path(val)) for k,val in immutable.items()}
    for k,v in list(env.items()):
        if v in immutable:env[k]=immutable[v]
        elif v in hashes:env[k]=hashes[v]
    env.update(updates)
    args[0]=str(app)
    pins[str(app)]=owner['app'];pins[str(rt/'libmetalrobo.dylib')]=owner['library']
    for old in immutable:pins[immutable[old]]=hashes[pins[old]]
    env['DYLD_LIBRARY_PATH']=str(rt)+':'+env['DYLD_LIBRARY_PATH']
    joined=work/'selected-pose/current.node'
    if digest(joined)!=owner['loaded'] or digest(app)!=owner['app'] or digest(rt/'libmetalrobo.dylib')!=owner['library']:raise ValueError('Native build revision diverged')
    out.mkdir()
    args[4]=str(out/'native-traces')
    for k,v in list(env.items()):
        if v.startswith(str(source)+'/') and v not in pins:env[k]=str(out/Path(v).relative_to(source))
    for v in set(args+list(env.values()))&set(pins):
        if digest(Path(v))!=pins[v]:raise ValueError('Unmatched launch input '+v)
    cfg.update(parent_launch=digest(source/'launch.json'),carrier_replay=digest(work/'recorded-mass/native-replay/result.json'),current_material_registration=digest(work/'binding/tables/result.json'))
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    clean={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}
    r=execute_native(args,cfg['cwd'],clean|env,out,minimum_free_bytes=4*2**30,timeout_s=600)
    print(r,flush=True)
if __name__=='__main__':run(Path(__file__).resolve().parent,Path(sys.argv[1]).resolve())
