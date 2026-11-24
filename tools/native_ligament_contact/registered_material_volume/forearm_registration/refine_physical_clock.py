"""Compare two resolutions at the physical time where the original scene rejected."""
import json,os,sys
from pathlib import Path
from build_bed_audit import sha
from run_cavity_motion import execute_native
def run(case,out):
    d=json.loads((case/'launch.json').read_bytes());args=d['argv'];env=d['environment'];pins=d['pins']
    idx=args.index('--muscle-step-seconds')+1
    if float(args[idx])!=7.8125e-6 or env['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1' or env['NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM']!='1':raise ValueError('Unexpected parent operating point')
    for p in set(args+list(env.values())).intersection(pins):
        if sha(Path(p))!=pins[p]:raise ValueError('Unmatched active input '+p)
    args[idx]=str(float(args[idx])/2);args[args.index('--muscle-step-count')+1]='4';args[4]=str(out/'native-traces')
    env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']='4';env['NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP']='4'
    for k,p in list(env.items()):
        if p.startswith(str(case)+'/') and p not in pins:env[k]=str(out/Path(p).relative_to(case))
    out.mkdir();d.update(parent_launch=sha(case/'launch.json'),matched_physical_time_s=2*7.8125e-6)
    (out/'launch.json').write_text(json.dumps(d,indent=2)+'\n')
    clean={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}
    print(execute_native(args,d['cwd'],clean|env,out,minimum_free_bytes=4*2**30,timeout_s=650),flush=True)
if __name__=='__main__':run(*(Path(x).resolve()for x in sys.argv[1:]))
