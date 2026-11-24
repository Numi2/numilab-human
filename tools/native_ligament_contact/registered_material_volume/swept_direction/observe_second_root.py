"""Track the existing Newton and contact decisions of the second physical root."""
import json
import os
import sys
from pathlib import Path
from bind_gluteal_receivers import digest, require
from run_cavity_motion import execute_native


def run(source, out, shader=None, rows=None):
    cfg=json.loads((source/'launch.json').read_bytes())
    argv, env, pins=cfg['argv'],cfg['environment'],cfg['pins']
    require(env['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']=='1' and
            env['NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM']=='1','Incomplete current scene')
    for value in set(argv+list(env.values())).intersection(pins):
        require(digest(Path(value))==pins[value], 'Changed scene input '+Path(value).name)
    argv[4]=str(out/'native-traces')
    argv[argv.index('--muscle-step-count')+1]='2'
    env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']='2'
    env['NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP']='2'
    for key,value in list(env.items()):
        if value.startswith(str(source)+'/') and value not in pins:
            env[key]=str(out/Path(value).relative_to(source))
    if shader is not None:
        build=json.loads((shader/'build.json').read_bytes())
        original=Path(env['NUMI_MATTER_METALLIB_OVERRIDE'])
        library=shader/'NumiMatter.metallib'
        require(digest(original)==build['original_metallib'] and digest(library)==build['metallib'],
                'Unmatched mechanics shader or source')
        env['NUMI_MATTER_METALLIB_OVERRIDE']=str(library)
        pins[str(library)]=build['metallib']
    for key in ('NM_FEM_BACKTRACKING_TRACE_ROOT','NM_FEM_NEWTON_TRACE_ROOT',
                'NM_FEM_TRUE_RESIDUAL_ROOT','NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT',
                'NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT'):
        env[key]='1'
    if rows is not None:
        observed=json.loads((rows/'pins.json').read_bytes())
        for name,sha in observed.items():
            require(digest(Path(name))==sha,'Changed certificate observer')
        argv[0]=str(rows/'app/numi-human-native')
        env['NUMI_DIAG_CERTIFICATE_METALLIB']=str(rows/'rows.metallib')
        env['DYLD_LIBRARY_PATH']=str(rows/'runtime')+':'+env['DYLD_LIBRARY_PATH']
        pins.update(observed)
    cfg['parent_launch']=digest(source/'launch.json')
    out.mkdir()
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    passed={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}
    print(execute_native(argv,cfg['cwd'],passed|env,out,minimum_free_bytes=4*2**30,timeout_s=650),flush=True)


if __name__=='__main__':run(*(Path(arg).resolve() for arg in sys.argv[1:]))
