"""Emit the conditioned material through the existing native registration and mass reader."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from register_material_surfaces import sha


def build(parent,proposal,transfer,owner,dest):
    verdict=json.loads((transfer/'result.json').read_bytes())
    if not all(verdict[tag]['passed'] for tag in ('SKIN','FAT')):
        raise ValueError('Incomplete conservative source transfers')
    source=json.loads((parent/'receipt.json').read_bytes())
    record=json.loads((proposal/'result.json').read_bytes())
    for k in ('reference','transported_displacement','subdivided_displacement'):
        if not record[k]['passed']:raise ValueError('Unqualified material replay')
    if (record['source_pins']['reference_sha256']!=sha(parent/'mesh.node') or
        record['patch_sha256']!=sha(proposal/'changed-coordinates.npz') or
        source['cell_mass_sha256']!=sha(parent/'cell-mass.f64')):
        raise ValueError('Wrong reference or source mass')
    dest.mkdir(exist_ok=False)
    patch=np.load(proposal/'changed-coordinates.npz')
    ids=patch['ids'];change=dict(zip(ids,patch['points']))
    for name in ('mesh.ele','current-child.labels.csv','material-source-indices.npz','ownership-source.csv'):
        os.link(parent/name,dest/name)
    with (parent/'mesh.node').open() as a,(dest/'mesh.node').open('w') as b:
        b.write(next(a))
        for i,line in enumerate(a):
            if i not in change:b.write(line);continue
            if int(line.split()[0])!=i+1:raise ValueError('Wrong node layout')
            b.write(f'{i+1} '+' '.join(format(float(f),'.17g') for f in change[i])+'\n')
    m=np.fromfile(parent/'cell-mass.f64',dtype='<f8')
    for tag in ('SKIN','FAT'):
        delta=transfer/(tag+'-delta.npz')
        if not delta.exists():continue
        d=np.load(delta);m[d['cells']]=d['mass_kg']
    t=np.loadtxt(dest/'mesh.ele',skiprows=1,dtype='i4',usecols=(1,2,3,4))-1
    nodal=np.zeros(source['nodes'])
    # Accumulate in the same cell/corner order as the runtime reader.
    np.add.at(nodal,t.ravel(),np.repeat(.25*m,4))
    m.tofile(dest/'cell-mass.f64');nodal.tofile(dest/'node-mass.f64')
    report=json.loads(owner.read_bytes())
    cmd=report['launch'].copy();cmd[1]=str(dest)
    if sha(Path(cmd[0]))!=report['binary']:raise ValueError('Native importer changed')
    child=subprocess.run(cmd,capture_output=True,text=True)
    (dest/'native-import.json').write_text(child.stdout)
    if child.returncode:raise ValueError(child.stderr or child.stdout)
    accepted=json.loads(child.stdout)
    if not accepted['native_imported'] or accepted['density_outliers']:raise ValueError('Native material import failed')
    for item in source['entries']:
        item['sha256']=sha(dest/item['file'])
    source.update(conditioned_from=sha(parent/'receipt.json'),cell_mass_sha256=sha(dest/'cell-mass.f64'),
                  node_mass_sha256=sha(dest/'node-mass.f64'),minimum_nodal_mass_kg=float(nodal.min()),
                  node_mass_closure_kg=float(abs(m.sum()-nodal.sum())),
                  conditioned_coordinates=record['patch_sha256'],local_conservative_transfer=sha(transfer/'result.json'))
    (dest/'receipt.json').write_text(json.dumps(source,indent=2)+'\n')
    write_build_receipt(report,cmd,dest)
    print(child.stdout)


def write_build_receipt(report,cmd,dest):
    previous=Path(report['launch'][0])
    if sha(previous)!=report['binary']:raise ValueError('Changed native reader')
    here=dest/'verify-native-material'
    if not here.exists():os.link(previous,here)
    compile=list(report['command'])
    if str(previous) not in compile:raise ValueError('Missing original linker output')
    compile[compile.index(str(previous))]=str(here)
    cmd=[str(here),*cmd[1:]]
    (dest/'native-import-build.json').write_text(json.dumps(dict(report,command=compile,launch=cmd),indent=2)+'\n')


if __name__=='__main__':build(*(Path(a).resolve() for a in sys.argv[1:]))
