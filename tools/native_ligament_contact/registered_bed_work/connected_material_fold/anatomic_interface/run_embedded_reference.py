"""Take an independent native full-order endpoint at half the release interval."""
from pathlib import Path
import json,os
from run_cavity_motion import execute_native
from compile_connected_joint import digest

def run(work):
    src=work/'accepted-material-07'
    out=work/'half-step-material-08'
    cfg=json.loads((src/'launch.json').read_bytes())
    current=work/'witness-material/source-embedded-physical-rows.npz'
    frozen=json.loads((work/'witness-material/source-embedding-motion.json').read_bytes())['refined']
    if digest(current)!=frozen['sparse_matrix'] or not frozen['passes_mass_pivot_check']:
        raise ValueError('Require the frozen, nonsingular candidate before the holdout')
    out.mkdir()
    env=cfg['environment'];args=cfg['argv'];pins=cfg['pins']
    if env['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1' or args[args.index('--muscle-step-count')+1]!='1':
        raise ValueError('Incorrect native reference')
    i=args.index('--muscle-step-seconds')+1
    args[i]=str(float(args[i])/2)
    args[4]=str(out/'native-traces')
    for k,v in list(env.items()):
        if v.startswith(str(src)+'/') and v not in pins:
            env[k]=str(out/Path(v).relative_to(src))
    cfg['full_order_parent_launch']=digest(src/'launch.json')
    cfg['held_out_basis_sha256']=digest(current)
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    clean={k:v for k,v in os.environ.items()if not k.startswith(('NM_','NUMI_'))}
    print(execute_native(args,cfg['cwd'],clean|env,out,minimum_free_bytes=4*2**30,timeout_s=600))

if __name__=='__main__':run(Path(__file__).resolve().parent)
