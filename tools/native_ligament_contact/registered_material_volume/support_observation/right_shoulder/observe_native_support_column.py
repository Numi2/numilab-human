"""Use the existing native material observer to look for continuum support on a measured load normal."""
import json,os,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from run_cavity_motion import execute_native

def run(case,measured,corridor,out):
    src=json.loads((case/'launch.json').read_bytes());q=json.loads(corridor.read_bytes());s=json.loads(measured.read_bytes())
    if q['native']!=digest(case/'launch.json') or s['launch_sha256']!=q['native'] or not q['fraction']>.5:raise ValueError('Unmatched loaded footprint')
    a=np.loadtxt(case/'physical-accepted.node',skiprows=1,usecols=(1,2,3))[q['node']]
    b=a+np.array(s['bed_normal'])*q['samples']['native'][0]['depth_mm']*.001
    out.mkdir();roi=out/'loaded-scapular-bounds.txt';np.savetxt(roi,np.stack((np.minimum(a,b)-.003,np.maximum(a,b)+.003)),fmt='%.17g')
    env=src['environment'];args=src['argv'];pins=src['pins']
    for k,v in list(env.items()):
        if v.startswith(str(case)+'/') and v not in pins:env[k]=str(out/Path(v).relative_to(case))
    args[4]=str(out/'native-traces');env['NUMI_HUMAN_GLUTEAL_FAT_TISSUE_AUDIT']=str(roi);pins[str(roi)]=digest(roi)
    for v in set(args+list(env.values()))&set(pins):
        if digest(Path(v))!=pins[v]:raise ValueError('Input drift '+Path(v).name)
    src.update(observer_control=digest(case/'launch.json'),column_target=digest(corridor))
    (out/'launch.json').write_text(json.dumps(src,indent=2)+'\n')
    clean={k:v for k,v in os.environ.items()if not k.startswith(('NM_','NUMI_'))}
    result=execute_native(args,src['cwd'],clean|env,out,timeout_s=900,minimum_free_bytes=4*2**30)
    (out/'execution.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
if __name__=='__main__':run(*(Path(s).resolve()for s in sys.argv[1:]))
