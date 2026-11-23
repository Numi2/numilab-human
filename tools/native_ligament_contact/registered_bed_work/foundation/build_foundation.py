"""Build the external foundation in the existing native force and endpoint owners."""
import json,shutil,subprocess,sys
from pathlib import Path
from material_context import sha
from export_contact_columns import change

def build(parent,source,out,experiment):
    ids=json.loads((parent/'identities.json').read_text())
    if any(sha(parent/k)!=v for k,v in ids.items()):raise ValueError('parent changed')
    if out.exists():raise FileExistsError(out)
    subprocess.run(['cp','-cR',parent,out],check=True)
    name='NumiHumanFoamFoundation.metalinc';shutil.copy2(experiment/name,out/name)
    f=out/'NumiHumanResolvedBed.metalinc';s=f.read_text()
    s=change(s,'#include "NumiHumanBedPressureTable.metalinc"','#include "'+name+'"' + '''
inline float bedClosure(float2 tissue, uint i) {
    return dot(tissue,seriesBed[i].xy)+seriesBed[i].z;
}
''')
    s=change(s,'float depth=-nm_', 'float depth=foamSurfaceZ-nm_')
    s=change(s,'const float depth=-(z.high+z.low);','const float depth=foamSurfaceZ-(z.high+z.low);')
    for idx in ['lo','hi','m','256']:
        old=f'dot(c.yz,bedPressureTable[{idx}].xy)'
        if s.count(old)!=(2 if idx=='256' else 1):raise ValueError('unrecognized closure')
        s=s.replace(old,f'bedClosure(c.yz,{idx})')
    s=change(s,'bedPressureStep/(b-a)','(seriesBed[hi].w-seriesBed[lo].w)/(b-a)')
    s=change(s,'float(lo)*bedPressureStep','seriesBed[lo].w')
    f.write_text(s)
    commands=json.loads((parent/'build-commands.json').read_text())
    old=Path(commands[0][commands[0].index('-c')+1]).parent
    link=json.loads((parent/'endpoint-articulated-build.json').read_text())['commands'][1]
    def remap(cmd):
        return [v.replace(str(old)+'/',str(out)+'/').replace(str(parent)+'/',str(out)+'/') for v in cmd]
    cmds=[remap(c) for c in commands[:3]]+[remap(link)]
    cwd=Path(json.loads((source/'launch.json').read_text())['environment']['DYLD_LIBRARY_PATH']).parent
    for cmd in cmds:subprocess.run(cmd,cwd=cwd,check=True)
    (out/'build-commands.json').write_text(json.dumps(cmds,indent=2)+'\n')
    ids[name]=''
    (out/'identities.json').write_text(json.dumps({k:sha(out/k) for k in sorted(ids)},indent=2)+'\n')
if __name__=='__main__':build(*(Path(x).resolve() for x in sys.argv[1:]))
