"""Verify actual solved cell masses with the same production geometry/quality reader and nodal reference."""
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from register_material_surfaces import sha


def run(mesh,flows,out):
    out.mkdir()
    config=json.loads((mesh/'native-import-build.json').read_bytes())
    cmd=config['launch'].copy()
    if sha(Path(cmd[0]))!=config['binary']:raise ValueError('Changed native mass reader')
    receipt=json.loads((mesh/'receipt.json').read_bytes())
    for f in receipt['entries']:
        if sha(mesh/f['file'])!=f['sha256']:raise ValueError('Changed mesh')
        os.link(mesh/f['file'],out/f['file'])
    if sha(mesh/'node-mass.f64')!=receipt['node_mass_sha256']:raise ValueError('Changed reference node mass')
    os.link(mesh/'node-mass.f64',out/'node-mass.f64')
    mat=np.loadtxt(mesh/'mesh.ele',skiprows=1,usecols=5,dtype='i4')
    if not np.isin(mat,[15,16]).all():raise ValueError('Unknown material')
    mass=np.empty(len(mat))
    audit=json.loads((flows/'exact-audit.json').read_bytes())
    for role,tag in ((15,'SKIN'),(16,'FAT')):
        file=flows/(tag+'-physical-flow.npz')
        if not audit[tag]['verified'] or sha(file)!=audit[tag]['flow_sha256']:raise ValueError('Unverified solved transfer')
        edge=np.load(file)
        mass[mat==role]=np.bincount(edge['role_cell'],weights=edge['mass'],minlength=int((mat==role).sum()))
    mass.tofile(out/'cell-mass.f64')
    cmd[1]=str(out)
    result=subprocess.run(cmd,capture_output=True,text=True)
    report=dict(native=json.loads(result.stdout),exit=result.returncode,cmd=cmd,binary=config['binary'],
                actual_cell_mass_sha256=sha(out/'cell-mass.f64'),nodal_reference_sha256=receipt['node_mass_sha256'])
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    if result.returncode:raise ValueError(result.stderr or report['native'])
    print(report['native'])


if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
