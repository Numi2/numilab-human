"""Run the identical registered body with the paired native contact kernels and loader."""
from pathlib import Path
import sys,json,os,subprocess
from compile_connected_joint import digest
from run_cavity_motion import execute_native
def run(job,base,build,out):
    def read(p):return json.loads(p.read_bytes())
    cfg=read(base/'launch.json');out.mkdir()
    old=job/'integrated';info=read(build/'build.json')
    executable=build/'app/numi-human-native';library=build/'libmetalrobo.dylib';metal=build/'metal/NumiMatter.metallib'
    for key,path in [('binary',executable),('library',library),('metal',metal)]:
        if digest(path)!=info[key]:raise ValueError('Incorrect '+key)
        cfg['pins'][str(path)]=info[key]
    if str(library) not in subprocess.check_output(['otool','-L',str(executable)],text=True):raise ValueError('Incorrect linked mechanics runtime')
    env={k:v.replace(str(base)+'/',str(out)+'/') for k,v in cfg['environment'].items()}
    dirs=env['DYLD_LIBRARY_PATH'].split(':')
    if dirs[0]!=str(old) or env['NUMI_MATTER_METALLIB_OVERRIDE']!=str(old/'metal/NumiMatter.metallib'):raise ValueError('Unexpected parent runtime')
    env['DYLD_LIBRARY_PATH']=':'.join([str(build)]+dirs[1:]);env['NUMI_MATTER_METALLIB_OVERRIDE']=str(metal)
    cfg['argv'][0]=str(executable);cfg['argv'][4]=str(out/'native-traces');cfg['environment']=env;cfg['parent_launch']=digest(base/'launch.json')
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(execute_native(cfg['argv'],cfg['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|env,out,minimum_free_bytes=4*2**30,timeout_s=400),flush=True)
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
