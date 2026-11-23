from pathlib import Path
import json,os,sys
from run_cavity_motion import execute_native
from compile_connected_joint import digest
def run(base,fit,out):
    inp=json.loads((base/'launch.json').read_text());arms=json.loads(fit.read_text());argv=inp['argv']
    if arms['source_rigid']!=digest(Path(argv[1])):raise ValueError('Mismatched rigid registration')
    updates={k:str(v) for r in arms['arms'] for k,v in r['q'].items()}
    for i,arg in enumerate(argv):
        if arg=='--pose-q' and argv[i+1] in updates:argv[i+2]=updates.pop(argv[i+1])
    for k,v in updates.items():argv.extend(['--pose-q',k,v])
    out.mkdir();argv[4]=str(out/'native-traces');env={k:v.replace(str(base)+'/',str(out)+'/') for k,v in inp['environment'].items()}
    inp['environment']=env;inp['joint_posture_fit']=digest(fit);inp['parent_launch']=digest(base/'launch.json')
    (out/'launch.json').write_text(json.dumps(inp,indent=2)+'\n')
    print(execute_native(argv,inp['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|env,out,minimum_free_bytes=4*2**30,timeout_s=180))
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
