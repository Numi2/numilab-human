"""Recompile the registered pose header for a fixed-topology material update, reusing the matching Metal library."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from register_material_surfaces import sha,tetgen_positions


def build(app,release,volume,mass,mesh,dest):
    a=json.loads((app/'build.json').read_bytes());r=json.loads((release/'build.json').read_bytes())
    binding=json.loads((volume/'native-acceptance.json').read_bytes())
    if not r['original_release_equals_reference'] or binding['rebuilt']['volume_cells']!=r['material_elements']:
        raise ValueError('Fixed topology required')
    source=Path(r['release_pose']['path']);target=mesh/'mesh.node'
    if sha(source)!=r['release_pose']['sha256'] or sha(mesh/'mesh.ele')!=r['topology_sha256']:
        raise ValueError('Changed release source or connectivity')
    node_sha=sha(target);cert=json.loads((mesh/'native-import.json').read_bytes())
    if cert['node_sha256']!=node_sha or not cert['native_imported'] or json.loads((mass/'result.json').read_bytes())['exit']:
        raise ValueError('Incomplete material or native mass import')
    for f in ('executable','library'):
        if sha(Path(r[f]['path']))!=r[f]['sha256']:raise ValueError('Mismatched native control '+f)
    header=Path(a['generated_header']['path'])
    if sha(header)!=a['generated_header']['sha256']:raise ValueError('Changed header')
    text=header.read_text()
    old=tetgen_positions(source);new=tetgen_positions(target)
    if old.shape!=new.shape or not np.array_equal(new.astype('f4').astype(float),new):
        raise ValueError('Different node layout or precision')
    points=set(r['first_contact_node_ids'])
    points.update(int(s) for s in re.findall(r'\{\d+u,(\d+)u,\d+u,\{',text))
    for w in r['registered_witnesses']:
        points.update(w['nodes'])
    loc=np.array(sorted(points))
    if len(loc)<15 or not np.array_equal(old[loc],new[loc]):
        raise ValueError('Changed mechanical witness, initial contact or attachment')
    for sym in ('nodeSHA','referenceSHA'):
        pattern='constexpr char '+sym+'[]="'+sha(source)+'";'
        if text.count(pattern)!=1:raise ValueError('Missing exact pose registration '+sym)
        text=text.replace(pattern,pattern.replace(sha(source),node_sha))
    dest.mkdir()
    for f in app.iterdir():
        if f.name!=header.name and f.suffix in {'.h','.hpp','.inc','.mm','.metalinc','.metal'}:
            os.link(f,dest/f.name)
    new_header=dest/header.name;new_header.write_text(text)
    compile=[v.replace(str(app)+'/',str(dest)+'/') if v.startswith(str(app)+'/') else v for v in a['commands'][0]]
    link=r['commands'][-1].copy();link[link.index(str(app/'numi-human-native.o'))]=str(dest/'numi-human-native.o')
    exe=dest/'numi-human-native';link[link.index('-o')+1]=str(exe)
    for command in (compile,link):subprocess.run(command,check=True)
    def ident(p):return dict(path=str(p),sha256=sha(p))
    result=dict(r,commands=[compile,link],executable=ident(exe),release_pose=ident(target),source_header=ident(header),
                generated_header=ident(new_header),parent_release_build=sha(release/'build.json'),
                native_mass_result=sha(mass/'result.json'),registered_volume=sha(volume/'native-acceptance.json'))
    (dest/'build.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':build(*(Path(v).resolve() for v in sys.argv[1:]))
