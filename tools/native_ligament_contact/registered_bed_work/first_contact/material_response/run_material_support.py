"""Run the native human with matched material support and its authored receiver maps."""
import json,os,sys
from pathlib import Path
from compile_connected_joint import digest
from run_cavity_motion import execute_native

def run(base,out,*patches):
    out.mkdir();cfg=json.loads((base/'launch.json').read_bytes());env={k:v.replace(str(base)+'/',str(out)+'/') for k,v in cfg['environment'].items()};cfg['parent_launch']=digest(base/'launch.json')
    for patch in patches:
        changed=json.loads(patch.read_bytes())['environment'];env.update(changed)
        for k,v in changed.items():
            if v.startswith('/'):
                f=Path(v)
                choices=[k+'_SHA256',k.rsplit('_',1)[0]+'_SHA256']
                hashes={changed[s] for s in choices if s in changed}
                if len(hashes)!=1 or digest(f) not in hashes:raise ValueError(k+' identity changed')
                cfg['pins'][str(f)]=hashes.pop()
    cfg['argv'][4]=str(out/'native-traces');cfg['environment']=env
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(execute_native(cfg['argv'],cfg['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|env,out,minimum_free_bytes=4*2**30,timeout_s=400),flush=True)
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
