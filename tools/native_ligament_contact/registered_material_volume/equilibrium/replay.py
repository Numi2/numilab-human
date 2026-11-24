"""Replay the failing certificate with a read-only objectwise GPU reduction."""
import json
import os
import shutil
import sys
from pathlib import Path
from compile_connected_joint import digest
from run_cavity_motion import execute_native

def run(source, build, out, trace_linesearch=False):
    cfg = json.loads((source/'launch.json').read_bytes())
    if cfg['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE'] != '1':
        raise ValueError('Variable pleural pressure required')
    pins = json.loads((build/'pins.json').read_bytes())
    for name, sha in pins.items():
        if digest(Path(name)) != sha:
            raise ValueError(name)
    argv, env = cfg['argv'], cfg['environment']
    if argv[argv.index('--muscle-step-count')+1] != '4':
        raise ValueError('Expected matching four-step trial')
    argv[0] = str(build/'app/numi-human-native')
    argv[4] = str(out/'native-traces')
    for key, value in list(env.items()):
        if value.startswith(str(source)+'/') and value not in cfg['pins']:
            env[key] = str(out/Path(value).relative_to(source))
    env['DYLD_LIBRARY_PATH'] = str(build/'runtime')+':'+env['DYLD_LIBRARY_PATH']
    env['NUMI_DIAG_CERTIFICATE_METALLIB'] = str(build/'rows.metallib')
    if trace_linesearch:
        env['NM_FEM_NEWTON_TRACE_ROOT'] = '1'
        env['NM_FEM_BACKTRACKING_TRACE_ROOT'] = '1'
    cfg['pins'].update(pins)
    for val in set(argv+list(env.values())) & cfg['pins'].keys():
        if digest(Path(val)) != cfg['pins'][val]:
            raise ValueError('Changed direct input '+val)
    cfg.update(parent_launch=digest(source/'launch.json'),free_bytes_before=shutil.disk_usage(out.parent).free)
    out.mkdir()
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    clean={k:v for k,v in os.environ.items() if not k.startswith(('NUMI_','NM_'))}
    print(execute_native(argv,cfg['cwd'],clean|env,out,minimum_free_bytes=4*2**30,timeout_s=600),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    for n in ('source','build','out'):p.add_argument(n,type=lambda v:Path(v).resolve())
    p.add_argument('--trace-linesearch',action='store_true')
    run(**vars(p.parse_args()))
