"""Link the registered anatomy to the tested coupled-bound runtime and replay its native control."""
from pathlib import Path
import json,shutil,subprocess,sys
from compile_connected_joint import digest
from advance_interior_scene import run
def replay(case,solver,stage,out,steps):
    app=stage/'witness-material/binding';v=stage/'witness-material/volume'
    rd=lambda p:json.loads(p.read_bytes())
    if rd(v/'interval.json')['failed'] or sum(map(sum,rd(v/'wall-crossings.json')['pair_counts'])):raise ValueError('Invalid starting geometry')
    cert=rd(solver/'build.json')
    for n,h in cert['pins'].items():
        if digest(Path(n))!=h:raise ValueError('Stale solver')
    lib=solver/'runtime/libmetalrobo.dylib';shader=solver/'metal/NumiMatter.metallib'
    meta=rd(app/'build.json')
    if meta['library']!=digest(lib):
        cmd=[str(solver/'app/app-runtime.mm.o') if arg.endswith('/app-runtime.mm.o') else str(lib) if arg.endswith('/libmetalrobo.dylib') else arg for arg in meta['commands'][-1]]
        if str(lib) not in cmd or str(solver/'app/app-runtime.mm.o') not in cmd:raise ValueError('Missing physical owner')
        subprocess.run(cmd,check=True);meta.update(library=digest(lib),app=digest(app/'app/numi-human-native'),paired_link=cmd)
        (app/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
    control=stage/'paired-control';control.mkdir(exist_ok=True)
    cfg=rd(case/'launch.json');env=cfg['environment'];pins=cfg['pins']
    if digest(Path(env['NUMI_MATTER_METALLIB_OVERRIDE']))!=cert['physical_parent']:raise ValueError('Unmatched material program')
    for k,val in list(env.items()):
        if val.startswith(str(case)+'/') and val not in pins:env[k]=str(control/Path(val).relative_to(case))
    env['NUMI_MATTER_METALLIB_OVERRIDE']=str(shader)
    env['DYLD_LIBRARY_PATH']=str(lib.parent)+':'+env['DYLD_LIBRARY_PATH'];pins.update(cert['pins'])
    cfg['parent_launch']=digest(case/'launch.json')
    (control/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    run(control,stage,out,steps)
if __name__=='__main__':replay(*(Path(n).resolve()for n in sys.argv[1:-1]),int(sys.argv[-1]))
