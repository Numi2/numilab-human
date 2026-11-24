"""Run the actual native application with matching material, transfer and source-registration receipts."""
import sys,json,os
from pathlib import Path
from compile_connected_joint import digest
from run_cavity_motion import execute_native
def run(job,material,ledger,p1,tables,build,out):
    rd=lambda p:json.loads(p.read_bytes());source=job/'registered-donor-cook/scene';cfg=rd(source/'launch.json');out.mkdir()
    env={k:v.replace(str(source)+'/',str(out)+'/') for k,v in cfg['environment'].items()};geo=rd(material/'result.json');ev=rd(ledger/'native-replay/result.json')
    if ev['exit'] or any(any(any(r) for r in cut) for cut in geo['layer_intersections'].values()):raise ValueError('Incomplete native transfer or layer audit')
    changes={}
    for data in [rd(p1/'result.json'),rd(tables/'result.json'),rd(ledger/'packets/result.json'),ev]:changes.update(data['environment'])
    pr='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    for k,fn in [('TETGEN_NODE','mesh.node'),('TETGEN_ELEMENTS','mesh.ele')]:changes[pr+k]=str(material/fn);changes[pr+k+'_SHA256']=digest(material/fn)
    for k,v in changes.items():
        if v.startswith('/'):
            key=[k+'_SHA256',k.rsplit('_',1)[0]+'_SHA256']
            if k.endswith('_RECEIPT_JSON'):key.append(k[:-5]+'_SHA256')
            expected={changes[i] for i in key if i in changes}
            if expected!={digest(Path(v))}:raise ValueError('Stale '+k)
            cfg['pins'][v]=expected.pop()
    pose=material/'current.node';exe=build/'app/numi-human-native'
    if digest(exe)!=rd(build/'build.json')['binary'] or digest(pose)!=geo['initial']:raise ValueError('Incorrect build')
    cfg['pins'][str(exe)]=digest(exe);cfg['pins'][str(pose)]=geo['initial'];cfg['parent_launch']=digest(source/'launch.json');env.update(changes)
    env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']=str(pose);cfg['argv'][0]=str(exe);cfg['argv'][4]=str(out/'native-traces');cfg['environment']=env
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n');print(execute_native(cfg['argv'],cfg['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|env,out,minimum_free_bytes=4*2**30,timeout_s=400),flush=True)
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
