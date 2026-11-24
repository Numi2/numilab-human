"""Replay the matched full tissue scene with identical physics and a changed GPU search direction."""
import json,os,sys
from pathlib import Path
from compile_connected_joint import digest
from run_cavity_motion import execute_native
def run(parent,build,out):
    cfg=json.loads((parent/'launch.json').read_bytes());lib=build/'NumiMatter.metallib'
    if digest(lib)!=json.loads((build/'build.json').read_bytes())['library']:raise ValueError('Changed search shader')
    env=cfg['environment'];cfg['search_control']=digest(parent/'launch.json')
    out.mkdir();cfg['argv'][4]=str(out/'native-traces')
    for k,v in list(env.items()):
        if v.startswith(str(parent)+'/') and v not in cfg['pins']:env[k]=str(out/Path(v).relative_to(parent))
    env['NUMI_MATTER_METALLIB_OVERRIDE']=str(lib);cfg['pins'][str(lib)]=digest(lib)
    env['NM_FEM_BACKTRACKING_TRACE_ROOT']='2';env['NM_FEM_BACKTRACKING_TRACE_NEWTON']='10'
    for v in set(cfg['argv']+list(env.values()))&set(cfg['pins']):
        if digest(Path(v))!=cfg['pins'][v]:raise ValueError('Changed native input')
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    clean={k:v for k,v in os.environ.items()if not k.startswith(('NM_','NUMI_'))}
    r=execute_native(cfg['argv'],cfg['cwd'],clean|env,out,minimum_free_bytes=4*2**30,timeout_s=800)
    (out/'execution.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
if __name__=='__main__':run(*(Path(a).resolve()for a in sys.argv[1:]))
