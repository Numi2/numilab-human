"""Link the same solver into both native owners used by this scene."""
import json,subprocess,sys
from pathlib import Path
from compile_connected_joint import digest
from build_coupled_volume_direction import replace_once

def build(parent,base,build,app):
    app.mkdir(); prior=parent/'binding/runtime'; rt=build/'runtime'
    name='app-runtime.mm'; source=(prior/name).read_text()
    for part in ['hints','claim','active']:
        old='"nm_fem_registered_volume_'+part+'"'
        if source.count(old)!=2:raise ValueError('Unexpected app solver list')
        source=source.replace(old,old[:-1]+'_incident"')
    source=replace_once(source,'candidate->device,candidate->volumeTileCount,valid,candidate->residentBytes);',
                        'candidate->device,candidate->volumeTileCount+33u,valid,candidate->residentBytes);')
    source=replace_once(source,'candidate->volumeDirection=true;','candidate->volumeDirection=true;\n                candidate->executionFingerprint=mixFingerprint(candidate->executionFingerprint,0x636f6c6f72737472ull);')
    (app/name).write_text(source)
    commands=json.loads((parent/'binding/build.json').read_bytes())['commands']
    # Build a new object. The prior build holds a hard link to the control object.
    compile=[arg.replace(str(prior/name),str(app/name)) for arg in commands[1]]
    compile.extend(['-iquote',str(prior)])
    cfg=json.loads((base/'build.json').read_bytes())
    control=base/'app/numi-human-native'
    replace={str(control):str(app/control.name),str(prior/'app-runtime.mm.o'):str(app/'app-runtime.mm.o'),
             str(prior/'libmetalrobo.dylib'):str(rt/'libmetalrobo.dylib')}
    link=[replace.get(a,a)for a in cfg['commands'][-1]]
    if not all(v in link for v in replace.values()):raise ValueError('Native linkage mismatch')
    with (app/'build.log').open('w') as log:
        for c in (compile,link):subprocess.run(c,stdout=log,stderr=subprocess.STDOUT,check=True)
    (app/'build.json').write_text(json.dumps(dict(commands=[compile,link],parent_app=digest(control),
        app=digest(app/control.name),source=digest(app/name),runtime=digest(rt/'libmetalrobo.dylib')),indent=2)+'\n')
if __name__=='__main__':build(*(Path(v).resolve() for v in sys.argv[1:]))
