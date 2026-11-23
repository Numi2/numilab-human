"""Run the existing integrated native owner with the paired registered shoulder assets."""
from pathlib import Path
import json,sys,os
from recount_registered_crossings import read,digest
from run_cavity_motion import execute_native
from bind_registered_shoulder import bind

def run(control,registration,out,steps):
    launch=bind(read(control/'launch.json'),registration)
    a=list(launch['argv'])
    for path,pin in launch['pins'].items():
        if digest(Path(path))!=pin:raise ValueError('Changed launch input '+path)
    out.mkdir()
    a[4]=str(out/'native-traces');a[a.index('--muscle-step-count')+1]=str(steps)
    env={k:v.replace(str(control)+'/',str(out)+'/') for k,v in launch['environment'].items()}
    env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']=','.join(str(i) for i in (3999,steps) if i<=steps) if steps!=3999 else '3999'
    launch.update(argv=a,environment=env,parent_launch=digest(control/'launch.json'),registered_shoulder=digest(registration/'registration.json'))
    (out/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    real={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|env
    result=execute_native(a,launch['cwd'],real,out)
    print(result,flush=True)
    return result['exit']

if __name__=='__main__':
    sys.exit(run(*(Path(p).resolve() for p in sys.argv[1:4]),int(sys.argv[4])))
