"""Replay a pinned full-scene run, substituting only its paired solver owners."""
import json,os,shutil,sys
from pathlib import Path
from compile_connected_joint import digest
from run_cavity_motion import execute_native
def run(case,build,out):
    cfg=json.loads((case/'launch.json').read_bytes());m=json.loads((build/'build.json').read_bytes())
    argv,env=cfg['argv'],cfg['environment'];lib=Path(env['DYLD_LIBRARY_PATH'].split(':')[0])/'libmetalrobo.dylib'
    if (digest(Path(argv[0])),digest(lib),digest(Path(env['NUMI_MATTER_METALLIB_OVERRIDE'])))!=(m['app_parent'],m['library_parent'],m['physical_parent']):raise ValueError('Different physical control')
    if env['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1' or env['NM_FEM_REGISTERED_RELEASE_BOUNDS']!='2':raise ValueError('Wrong solver')
    for n,h in m['pins'].items():
        if digest(Path(n))!=h:raise ValueError('Changed solver '+n)
    for k,s in list(env.items()):
        if s.startswith(str(case)+'/') and s not in cfg['pins']:env[k]=str(out/Path(s).relative_to(case))
    argv[0]=str(build/'app/numi-human-native');argv[4]=str(out/'native-traces')
    env['NUMI_MATTER_METALLIB_OVERRIDE']=str(build/'metal/NumiMatter.metallib')
    env['DYLD_LIBRARY_PATH']=str(build/'runtime')+':'+env['DYLD_LIBRARY_PATH']
    cfg['pins'].update(m['pins'])
    for n in set(argv+list(env.values())) & cfg['pins'].keys():
        if digest(Path(n))!=cfg['pins'][n]:raise ValueError('Unpinned direct input '+n)
    out.mkdir();cfg.update(parent_launch=digest(case/'launch.json'),solver_build=digest(build/'build.json'),free_bytes_before=shutil.disk_usage(out).free)
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    clean={k:v for k,v in os.environ.items() if not k.startswith(('NUMI_','NM_'))}
    print(execute_native(argv,cfg['cwd'],clean|env,out,minimum_free_bytes=4*2**30,timeout_s=750),flush=True)
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
