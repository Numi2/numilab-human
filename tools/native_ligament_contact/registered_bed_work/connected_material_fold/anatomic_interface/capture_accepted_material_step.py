"""End at the last proven native step so the existing acceptance exporter saves the physical volume."""
from pathlib import Path
import json,os
from compile_connected_joint import digest
from run_cavity_motion import execute_native

def run(work):
    parent=work/'native-witness-05';out=work/'accepted-material-07';out.mkdir()
    read=lambda f:json.loads(f.read_bytes())
    assert read(parent/'failure.json')['segment_start']==1
    cfg=read(parent/'launch.json');args=cfg['argv'];env=cfg['environment'];pins=cfg['pins']
    args[args.index('--muscle-step-count')+1]='1';args[4]=str(out/'native-traces')
    env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']='1'
    env['NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP']='1'
    for k,v in list(env.items()):
        if v.startswith(str(parent)+'/') and v not in pins:env[k]=str(out/Path(v).relative_to(parent))
    cfg['parent_launch']=digest(parent/'launch.json')
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    clean={k:v for k,v in os.environ.items()if not k.startswith(('NM_','NUMI_'))}
    print(execute_native(args,cfg['cwd'],clean|env,out,minimum_free_bytes=4*2**30,timeout_s=600))
if __name__=='__main__':run(Path(__file__).resolve().parent)
