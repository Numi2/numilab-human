"""Apply a local conservative correction to the accepted native carrier allocation."""
import json
import os
from pathlib import Path
import sys

import numpy as np
from register_material_surfaces import sha


def emit(control,transfer,dest):
    audit=json.loads((control/'exact-audit.json').read_bytes())
    solved=json.loads((transfer/'result.json').read_bytes())
    dest.mkdir()
    for tag in ('SKIN','FAT'):
        src=control/(tag+'-physical-flow.npz');result=dest/src.name
        if not solved[tag]['passed'] or not audit[tag]['verified'] or sha(src)!=audit[tag]['flow_sha256']:
            raise ValueError('Unverified source allocation')
        patch=transfer/(tag+'-delta.npz')
        if not solved[tag]['fixed_mass_outside_density']:
            os.link(src,result)
            continue
        d=np.load(patch)
        if d['base_flow_sha256'].item()!=sha(src):raise ValueError('Wrong carrier row identities')
        f=np.load(src)
        w=f['mass'];idx=d['original_edges'];new=d['new_kg']
        if len(new)!=len(idx) or len(np.unique(idx))!=len(idx) or np.any((idx<0)|(idx>=len(w))) or not np.isfinite(new).all() or np.any(new<0):
            raise ValueError('Invalid edited flow')
        w[idx]=new
        keep=w>0
        np.savez_compressed(result,carrier=f['carrier'][keep],role_cell=f['role_cell'][keep],mass=w[keep])


if __name__=='__main__':emit(*(Path(a).resolve() for a in sys.argv[1:]))
