"""Enter native preparation with the registered material donor executable and identical physical settings."""
from pathlib import Path
import json,os,sys
from compile_connected_joint import digest
from run_cavity_motion import execute_native

def run(base,build,out,control=None):
    cfg=json.loads((base/'launch.json').read_bytes());argv=cfg['argv']
    ids=json.loads((build/'registration.json').read_bytes());exe=build/'numi-human-native'
    if ids['executable']!=digest(exe) or ids['physical_source']!=digest(Path(argv[1])):raise ValueError('Physical source generation mismatch')
    old=argv[0];argv[0]=str(exe);argv[4]=str(out/'native-traces');env={k:v.replace(str(base)+'/',str(out)+'/') for k,v in cfg['environment'].items()}
    if control is not None:
        original=control/'launch.json'
        if cfg['connected_shoulder_parent_launch']!=digest(original):raise ValueError('Incorrect physical baseline')
        args=json.loads(original.read_bytes())['argv'];step='--muscle-step-seconds';count='--muscle-step-count'
        if args[args.index(step)+1]!=argv[argv.index(step)+1]:raise ValueError('Different physical timestep')
        argv[argv.index(count)+1]=args[args.index(count)+1]
        cfg['matched_native_horizon']=digest(original)
    if 'NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS' in env:
        env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']=str(int(argv[argv.index('--muscle-step-count')+1]))
    cfg['pins'].pop(old,None);cfg['pins'][str(exe)]=ids['executable'];cfg['environment']=env;cfg['material_donor_registration']=digest(build/'registration.json')
    out.mkdir();(out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    result=execute_native(argv,cfg['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|env,out,minimum_free_bytes=4*2**30,timeout_s=300)
    print(result)
if __name__=='__main__':run(*(Path(k).resolve() for k in sys.argv[1:]))
