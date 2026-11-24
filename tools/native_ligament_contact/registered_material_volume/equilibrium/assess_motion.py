"""Run the existing shape, registered wall, and cortical solid checks on an accepted endpoint."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from check_full_motion import run as audit
from measure_matched_bones import run as bones
from compile_connected_joint import digest

def run(fit,source,case,parent,result):
    record=json.loads((fit/'fit.json').read_bytes())
    stage=fit/'analysis';vol=stage/'witness-material/volume';vol.mkdir(parents=True,exist_ok=True)
    for name in ('mesh.node','mesh.ele','material-walls.npz','current.node'):
        p=vol/name;target=fit/name if name=='current.node' else source/'volume'/name
        if not p.exists():p.symlink_to(target.resolve())
        if p.resolve()!=target.resolve():raise ValueError('Stale material audit input')
    changed=np.any(np.loadtxt(vol/'current.node',skiprows=1,usecols=(1,2,3))!=np.loadtxt(source/'volume/current.node',skiprows=1,usecols=(1,2,3)),axis=1)
    if changed.sum()!=record['changed_nodes']:raise ValueError('Different fitted material pose')
    (vol/'result.json').write_text(json.dumps(dict(updated_sites=[dict(vertex=int(n)) for n in np.flatnonzero(changed)]))+'\n')
    result.mkdir()
    audit(case,stage,result/'motion.json')
    bones(parent,stage,case,result/'motion.json',result/'bones')
    result_record=json.loads((result/'motion.json').read_bytes())
    print('accepted',result_record['accepted_s'],'skin/fat shape failures',[(k,r['below_shape']) for k,r in result_record['roles'].items()],'wall pairs',result_record['exact_material_surface_pair_count'])
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('fit','source','case','parent','result'):p.add_argument(n,type=lambda x:Path(x).resolve())
    run(**vars(p.parse_args()))
