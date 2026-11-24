"""Capture the active contact floor at the final correction on the matched physical scene."""
import argparse
import json
import os
from pathlib import Path
from compile_connected_joint import digest
from run_cavity_motion import execute_native

def run(parent, out, slot=None):
    source = parent / 'launch.json'
    cfg = json.loads(source.read_bytes())
    environment = cfg['environment']
    argv = cfg['argv']
    for filename in set(argv + list(environment.values())).intersection(cfg['pins']):
        if digest(Path(filename)) != cfg['pins'][filename]:
            raise ValueError(f'Changed input {filename}')
    out.mkdir()
    argv[4] = str(out / 'native-traces')
    for key, value in list(environment.items()):
        if value.startswith(str(parent) + '/') and value not in cfg['pins']:
            environment[key] = str(out / Path(value).relative_to(parent))
    environment.update(NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT='0',
                       NM_FEM_DEFORMABLE_LIMIT_TRACE_MINIMA='1',
                       NM_FEM_DEFORMABLE_LIMIT_TRACE_FAILURES='1',
                       NM_FEM_CONTACT_TRACE_ITERATION='15')
    if slot is not None:
        environment['NM_FEM_DEFORMABLE_LIMIT_TRACE_SLOT'] = str(slot)
        environment.pop('NM_FEM_DEFORMABLE_LIMIT_TRACE_FAILURES')
    cfg['parent_launch'] = digest(source)
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    clean = {k:v for k,v in os.environ.items() if not k.startswith(('NM_', 'NUMI_'))}
    print(execute_native(argv,cfg['cwd'],clean|environment,out,minimum_free_bytes=4*2**30,timeout_s=600),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('parent',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--slot',type=int)
    args=ap.parse_args()
    run(args.parent.resolve(),args.out.resolve(),args.slot)
