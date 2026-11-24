"""Repeat one immutable native launch with explicit diagnostic cadence and captures."""
import json
import os
import sys
from pathlib import Path
from compile_connected_joint import digest
from run_cavity_motion import execute_native


def run(reference,build,output,steps,cadence,captures):
    reference=reference.resolve()
    cfg=json.loads((reference/'launch.json').read_bytes())
    argv,env,pins=cfg['argv'],cfg['environment'],cfg['pins']
    output=output.resolve();output.mkdir()
    oldroot=Path(argv[4]).parent
    for k,v in list(env.items()):
        if v.startswith(str(oldroot)+'/') and v not in pins:
            env[k]=str(output/Path(v).relative_to(oldroot))
    if env['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1':raise ValueError('Mismatched pressure mode')
    argv[0]=str(build.resolve()/'numi-human-native')
    argv[4]=str(output/'native-traces')
    argv[argv.index('--muscle-step-count')+1]=str(steps)
    env['NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS']=str(cadence)
    env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']=captures
    for f in [Path(argv[0]),*(build.resolve()/'metalrobo').glob('*.metallib')]:
        pins[str(f)]=digest(f)
    for v in set(argv+list(env.values()))&set(pins):
        if not Path(v).is_file() or digest(Path(v))!=pins[v]:raise ValueError('Incorrect input '+v)
    registered=build/'registered-live.json'
    if registered.exists():
        manifest=json.loads(registered.read_bytes())
        for name,expected in manifest['outputs'].items():
            if digest(build/name)!=expected:raise ValueError('Altered registered GPU bundle')
        env['NUMI_HUMAN_RESOLVED_BED_WITNESS']=str(output/'material-witness.json')
    cfg['diagnostic_source_launch']=digest(reference/'launch.json')
    (output/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    clean={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}
    print(execute_native(argv,cfg['cwd'],clean|env,output,minimum_free_bytes=4*2**30,timeout_s=600),flush=True)

if __name__=='__main__':
    run(*(Path(s) for s in sys.argv[1:4]),int(sys.argv[4]),int(sys.argv[5]),sys.argv[6])
