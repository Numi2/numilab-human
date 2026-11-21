"""Recompile the same human executable, force adapter and Matter library as a matched set."""
from pathlib import Path
import json,sys,subprocess,concurrent.futures
root=Path(__file__).resolve().parent;b=root/'build';p,e,c,app=map(Path,sys.argv[1:5])
cwd=app/'build-native';lib=b/'libmetalrobo.dylib';inc='-I'+str(b/'include')
cfg=json.loads((c/'emitted-native/launch.json').read_text())
pressure=p/'pressure-with-abdominal-cavity-build';orig=json.loads((pressure/'build-commands.json').read_text())
def replace(cmd,repls):
    return [str(repls.get(a,a)) for a in cmd]
host=replace(orig[1],{str(pressure/n):b/n for n in ['numi_human.mm','numi_human.mm.o']})
host[1:1]=[inc]
link=replace(orig[-1],{str(pressure/'libmetalrobo.dylib'):lib,str(pressure/'runtime.mm.o'):p/'guarded-runtime-admission-build/runtime.mm.o',str(pressure/'numi_human.mm.o'):b/'numi_human.mm.o'})
appcmd,applink=([str(b/Path(s).name) if Path(s).parent==c/'emitted-native/build' else s for s in cmd] for cmd in cfg['commands'])
appcmd[1:1]=[inc]
shader=['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math',inc,'-I',str(app/'matter/include'),'-I',str(app/'include'),'-c',str(b/'src/matter_kernels.metal'),'-o',str(b/'NumiMatter.air')]
shaderlink=['xcrun','-sdk','macosx','metallib',str(b/'NumiMatter.air'),'-o',str(b/'NumiMatter.metallib')]
groups={'adapter':[host,link],'shader':[shader,shaderlink],'human':[appcmd,applink]}
(root/'commands.json').write_text(json.dumps(groups,indent=2)+'\n')
def build(name):
    with (root/(name+'.log')).open('w') as log:
        for cmd in groups[name]:subprocess.run(cmd,cwd=cwd,check=True,stdout=log,stderr=log)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(build,sys.argv[5:] or groups))
