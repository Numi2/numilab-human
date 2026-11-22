import json,subprocess,sys
from pathlib import Path
from material_context import sha
def build(root,library):
    old=root/'limit-witness';dst=root/'normal-row';out=dst/'build';out.mkdir()
    import shutil
    original=json.loads((library/'identities.json').read_text())['commands']
    for i,name in [(2,'runtime.mm'),(3,'app-runtime.mm')]:
        s=(old/'build'/name).read_text();marker='            if (!encodeMPMNewtonRHSRowsTrace()) {'
        if s.count(marker)!=1:raise ValueError('Changed assembled RHS schedule')
        s=s.replace(marker,(dst/'capture.inc').read_text()+marker)
        (out/name).write_text('#include <charconv>\n'+s)
        cmd=[str(out/name) if a==str(library/name) else str(out/(name+'.o')) if a==str(library/(name+'.o')) else a for a in original[i]]
        cmd[1:1]=['-iquote',str(library)];subprocess.run(cmd,check=True)
    cmd=[s.replace(str(library/'libmetalrobo.dylib'),str(out/'libmetalrobo.dylib')).replace(str(library/'runtime.mm.o'),str(out/'runtime.mm.o')) for s in original[4]]
    subprocess.run(cmd,check=True)
    o=dst/'owner';subprocess.run(['cp','-cR',str(old/'owner'),str(o)],check=True)
    j=json.loads((old/'owner/identities.json').read_text())
    cmd=[s.replace(str(old/'owner')+'/',str(o)+'/').replace(str(old/'build')+'/',str(out)+'/') for s in j['commands'][1]]
    cfg=json.loads((old/'parent/launch.json').read_text());subprocess.run(cmd,check=True,cwd=cfg['cwd'])
    exe=o/'numi-human-native';j.update(binary=sha(exe),binary_sha256=sha(exe),binary_path=str(exe));j['commands'][1]=cmd
    (o/'identities.json').write_text(json.dumps(j,indent=2)+'\n')
    e=cfg['environment'];del cfg['pins'][str(Path(e['DYLD_LIBRARY_PATH'])/'libmetalrobo.dylib')]
    e['DYLD_LIBRARY_PATH']=str(out);cfg['pins'][str(out/'libmetalrobo.dylib')]=sha(out/'libmetalrobo.dylib')
    measured=json.loads((root/'limiting-material.json').read_text());e['NM_FEM_INITIAL_CONTACT_ROW']=str(measured['native']['node'])
    parent=dst/'parent';parent.mkdir();(parent/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
if __name__=='__main__':build(*(Path(s).resolve() for s in sys.argv[1:]))
