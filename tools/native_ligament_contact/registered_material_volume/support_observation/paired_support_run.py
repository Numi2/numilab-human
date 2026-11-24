"""Run a physics-identical native comparison, relocating only the observer and its outputs."""
import json,os,sys
from pathlib import Path
from build_bed_audit import sha
from run_cavity_motion import execute_native

def run(parent,build,out):
    cfg=json.loads((parent/'launch.json').read_bytes());cert=json.loads((build/'build.json').read_bytes())
    argv,env,pins=cfg['argv'],cfg['environment'],cfg['pins']
    if env['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1' or env['NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM']!='1':raise ValueError('Incomplete reference scene')
    orig=Path(argv[0]); exe=build/'app/numi-human-native';argv[0]=str(exe);argv[4]=str(out/'native-traces')
    for k,v in list(env.items()):
        if v.startswith(str(parent)+'/') and v not in pins:env[k]=str(out/Path(v).relative_to(parent))
    pins.update(cert['pins'])
    for name in (set(argv+list(env.values())) & set(pins)) | set(cert['pins']) | {str(orig)}:
        if sha(Path(name))!=pins[name]:raise ValueError('Changed input '+name)
    lib=next(Path(d)/'libmetalrobo.dylib' for d in env['DYLD_LIBRARY_PATH'].split(':') if (Path(d)/'libmetalrobo.dylib').is_file())
    if sha(lib)!=cert['library']:raise ValueError('Different mechanics runtime')
    out.mkdir();cfg['parent_launch']=sha(parent/'launch.json');(out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    shell={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))};print(execute_native(argv,cfg['cwd'],shell|env,out,minimum_free_bytes=4*2**30,timeout_s=600),flush=True)
if __name__=='__main__':run(*(Path(x).resolve()for x in sys.argv[1:]))
