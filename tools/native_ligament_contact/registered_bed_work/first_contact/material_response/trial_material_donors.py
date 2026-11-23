"""Enter native preparation with the registered material donor executable and identical physical settings."""
from pathlib import Path
import json,os,sys
from compile_connected_joint import digest
from run_cavity_motion import execute_native

def run(base,build,out):
    cfg=json.loads((base/'launch.json').read_bytes());argv=cfg['argv']
    ids=json.loads((build/'registration.json').read_bytes());exe=build/'numi-human-native'
    if ids['executable']!=digest(exe) or ids['physical_source']!=digest(Path(argv[1])):raise ValueError('Physical source generation mismatch')
    old=argv[0];argv[0]=str(exe);argv[4]=str(out/'native-traces');env={k:v.replace(str(base)+'/',str(out)+'/') for k,v in cfg['environment'].items()}
    cfg['pins'].pop(old,None);cfg['pins'][str(exe)]=ids['executable'];cfg['environment']=env;cfg['material_donor_registration']=digest(build/'registration.json')
    out.mkdir();(out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    result=execute_native(argv,cfg['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|env,out,minimum_free_bytes=4*2**30,timeout_s=300)
    print(result)
if __name__=='__main__':run(*(Path(k).resolve() for k in sys.argv[1:]))
