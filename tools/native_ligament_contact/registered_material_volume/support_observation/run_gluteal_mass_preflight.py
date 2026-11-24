"""Exercise the registered gluteal debit against the fully composed native candidate."""
import json
import os
import sys
from pathlib import Path
from bind_gluteal_receivers import digest, require
from run_cavity_motion import execute_native


def run(parent, built, ledger_dir, out):
    cfg=json.loads((parent/'launch.json').read_bytes())
    initial=digest(parent/'launch.json')
    argv, env, pins=cfg['argv'],cfg['environment'],cfg['pins']
    compiled=json.loads((built/'build.json').read_bytes())
    require(digest(Path(argv[0]))==compiled['parent'],'Changed parent executable')
    require(env['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']=='1' and
            env['NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM']=='1','Incomplete native scene')
    ledger=ledger_dir/'mass-reference.txt'
    file_hash=json.loads((ledger_dir/'mass.json').read_bytes())['ledger']
    require(digest(ledger)==file_hash,'Changed inferred material mass')
    exe=built/'numi-human-native'
    require(digest(exe)==compiled['executable'],'Bad native build')
    argv[0],argv[4]=str(exe),str(out/'native-traces')
    for k,v in list(env.items()):
        if v.startswith(str(parent)+'/') and v not in pins:
            env[k]=str(out/Path(v).relative_to(parent))
    env['NUMI_HUMAN_GLUTEAL_FAT_MASS_PREFLIGHT']=str(ledger)
    limits=ledger_dir/'accepted-region-bounds.txt'
    if limits.is_file():
        env['NUMI_HUMAN_GLUTEAL_FAT_TISSUE_AUDIT']=str(limits)
        pins[str(limits)]=digest(limits)
    pins[str(ledger)]=file_hash
    pins[str(exe)]=compiled['executable']
    for path in set(argv+list(env.values())) & pins.keys():
        require(digest(Path(path))==pins[path], 'Changed live input '+Path(path).name)
    cfg['parent_launch']=initial
    shell={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}
    out.mkdir()
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    result=execute_native(argv,cfg['cwd'],shell|env,out,minimum_free_bytes=4*2**30,timeout_s=650)
    print(result)
    require(result['exit']==0,'Native preflight or step failed')


if __name__=='__main__':run(*(Path(p).resolve()for p in sys.argv[1:]))
