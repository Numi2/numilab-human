"""Capture globalization of the first physically rejected coupled step on the authenticated scene."""
import json, os, sys
from pathlib import Path
from compile_connected_joint import digest
from run_cavity_motion import execute_native

def run(previous,out):
    cfg=json.loads((previous/'launch.json').read_bytes());env=cfg['environment'];args=cfg['argv']
    for v in sorted(set(args+list(env.values()))&set(cfg['pins'])):
        if digest(Path(v))!=cfg['pins'][v]:
            raise ValueError('Input changed: '+v)
    out.mkdir()
    args[4]=str(out/'native-traces')
    for k,v in list(env.items()):
        if v.startswith(str(previous)+'/') and v not in cfg['pins']:
            env[k]=str(out/Path(v).relative_to(previous))
    for k in ('NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT','NM_FEM_NEWTON_TRACE_ROOT','NM_FEM_BACKTRACKING_TRACE_ROOT'):
        env[k]='0'
    cfg['parent_launch']=digest(previous/'launch.json')
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    clean={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}
    print(execute_native(args,cfg['cwd'],clean|env,out,minimum_free_bytes=4*2**30,timeout_s=600),flush=True)
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
