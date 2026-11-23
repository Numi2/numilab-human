"""Rebind the unchanged volume-direction solver to its surviving physical witness cells after remeshing."""
import concurrent.futures
import json
import os
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
from register_material_surfaces import pinned,sha,tetgen_positions


def run(control,mesh,release,base,out):
    env=json.loads(control.read_bytes())['environment']
    prep=json.loads((release/'build.json').read_bytes())
    old=pinned(env,'TETGEN_ELEMENTS','TETGEN_ELEMENTS_SHA256')
    oldx=tetgen_positions(pinned(env,'TETGEN_NODE','TETGEN_NODE_SHA256'))
    x=tetgen_positions(mesh/'mesh.node')
    elements=mesh/'mesh.ele';mesh_key=sha(elements)
    if len(x)!=prep['current_nodes'] or sha(mesh/'mesh.node')!=prep['release_pose']['sha256']:raise ValueError('Different release')
    def read(path):return np.loadtxt(path,skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    a=read(old);b=read(elements)
    origin=json.loads((base/'build.json').read_bytes())
    if sha(base/'libmetalrobo.dylib')!=origin['library']:raise ValueError('Changed registered solver')
    runtime=(base/'runtime.mm').read_text()
    begin=runtime.index('                const bool matches=[&]{',runtime.index('if(const char* enabled=std::getenv("NM_FEM_REGISTERED_VOLUME_DIRECTION")'))
    end=runtime.index('                const auto& object=world.objects[candidate->volumeOwner];',begin)
    original=runtime[begin:end]
    guards=re.findall(r'if\(n.x!=(\d+)u\|\|n.y!=(\d+)u\|\|n.z!=(\d+)u\|\|n.w!=(\d+)u\)',original)
    if len(guards)!=2:raise ValueError('Unknown registered volume-direction version')
    proof=[]
    for values in guards:
        v=np.array(list(map(int,values)))
        pair=[]
        for t in (a,b):
            ix=np.flatnonzero(np.all(t[:,:4]-t[:,:1]==v-v[0],axis=1))
            if len(ix)!=1:raise ValueError('Nonunique physical tissue witness')
            pair.append(int(ix[0]))
        i,j=pair;cell=a[i,:4]-1
        if not np.array_equal(a[i],b[j]) or not np.array_equal(oldx[cell],x[cell]):
            raise ValueError('Volume direction witness changed physical geometry or material')
        proof.append(dict(source_row=i,current_row=j,material=int(a[i,4]),nodes=cell.tolist()))
    # Keep the original registration and numerical algorithm intact. For the current
    # certified mesh use the actual emitting object's range and its own node offset.
    # Both exact local tetrahedra must have the same registered continuum owner.
    lines=[]
    for w in proof:
        i=w['current_row']
        y=' && '.join(f'n.{axis}==object.stateOffset+{v}u' for axis,v in zip('xyzw',w['nodes']))
        lines.append(f'''                        {{
                            const auto& t=world.fem.tetrahedra[object.elementOffset+{i}u];
                            const auto n=t.nodes;
                            if(t.identity.y!=owner || !({y}))continue;
                        }}''')
    proof_code='\n'.join(lines)
    fallback=original[original.index('                const bool matches='):original.index('                if(!candidate->unilateralBed')]
    acquired=f'''                auto registeredVolumeOwner=NM_INVALID_INDEX;
                const char* geometry=std::getenv("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_ELEMENTS_SHA256");
                const bool rebuilt=geometry && std::string_view(geometry)=="{mesh_key}";
                if(rebuilt){{
                    for(uint32_t owner=0;owner<world.objects.size();++owner){{
                        const auto& object=world.objects[owner];
                        if(object.elementCount!={len(b)}u || object.stateCount!={len(x)}u ||
                           size_t(object.elementOffset)+object.elementCount>world.fem.tetrahedra.size())continue;
{proof_code}
                        if(registeredVolumeOwner!=NM_INVALID_INDEX){{diagnostics.message="ambiguous registered tissue volume";return diagnostics;}}
                        registeredVolumeOwner=owner;
                    }}
                }}else{{
{fallback}
                    if(matches)registeredVolumeOwner=world.fem.tetrahedra[950449u].identity.y;
                }}
                if(!candidate->unilateralBed || registeredVolumeOwner==NM_INVALID_INDEX){{
                    diagnostics.message="volume-direction proposal requires its measured registered tissue topology and bed";return diagnostics;
                }}
                candidate->volumeDirection=true;
                candidate->executionFingerprint=mixFingerprint(candidate->executionFingerprint,11270225402661047665ull);
                if(rebuilt)candidate->executionFingerprint=mixFingerprint(candidate->executionFingerprint,0x{mesh_key[:16]}ull);
                candidate->volumeOwner=registeredVolumeOwner;
'''
    out.mkdir()
    os.link(base/'bed_response.inc',out/'bed_response.inc')
    for n in ('runtime.mm','app-runtime.mm'):
        code=(base/n).read_text()
        if code.count(original)!=1:raise ValueError('Different runtime volume owner')
        (out/n).write_text(code.replace(original,acquired))
    commands=[]
    for i,seq in enumerate(origin['commands']):
        cmd=[]
        for arg in seq:
            if arg.startswith(str(base)+'/'):
                arg=str(out/Path(arg).name)
            elif i==3 and arg.endswith('/numi-human-native.o'):
                arg=str(release/'numi-human-native.o')
            cmd.append(arg)
        commands.append(cmd)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        jobs=[pool.submit(subprocess.run,c,check=True) for c in commands[:2]]
        for job in jobs:job.result()
    for c in commands[2:]:subprocess.run(c,check=True)
    info=lambda p:dict(path=str(p),sha256=sha(p))
    prep.update(commands=commands,executable=info(out/'numi-human-native'),library=info(out/'libmetalrobo.dylib'),
                registered_witnesses=proof,topology_sha256=mesh_key,parent_release_build=sha(release/'build.json'))
    (out/'build.json').write_text(json.dumps(prep,indent=2)+'\n')


if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
