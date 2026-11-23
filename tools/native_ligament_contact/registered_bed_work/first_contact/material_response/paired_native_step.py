"""Repeat a native release over the identical physical duration with two temporal subdivisions."""
import json
import os
from pathlib import Path
import sys

from register_material_surfaces import sha
from run_cavity_motion import execute_native


def rerun(parent,dest):
    base=parent/'launch.json'
    j=json.loads(base.read_bytes())
    argv=[s.replace(str(parent)+'/',str(dest)+'/') for s in j['argv']]
    env={k:v.replace(str(parent)+'/',str(dest)+'/') for k,v in j['environment'].items()}
    at=argv.index('--muscle-step-seconds')+1
    count=argv.index('--muscle-step-count')+1
    if argv[count]!='1' or env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']!='1':
        raise ValueError('Expected one accepted control step')
    argv[at]=str(float(argv[at])/2)
    argv[count]='2'
    del env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']  # Terminal material coordinates and the ordinary native replay are sufficient.
    env['NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP']='2'
    if any(str(parent)+'/' in s for s in [*argv,*env.values()]):
        raise ValueError('Output path still references paired control')
    if sha(Path(argv[0]))!=json.loads((parent/'material-motion.json').read_bytes())['binary']:
        raise ValueError('Native executable differs from measured control')
    dest.mkdir()
    (dest/'native-traces').mkdir()
    (dest/'launch.json').write_text(json.dumps(dict(j,argv=argv,environment=env,paired_native_launch=sha(base)),indent=2)+'\n')
    result=execute_native(argv,j['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NUMI_','NM_'))}|env,dest,minimum_free_bytes=7*2**30,timeout_s=500)
    print(result,flush=True)
    return result['exit']


if __name__=='__main__':
    sys.exit(rerun(*(Path(s).resolve() for s in sys.argv[1:])))
