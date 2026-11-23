"""Classify actual deep-fat intersection witnesses by physical facet adjacency."""
import collections
import json
import sys
from pathlib import Path
import numpy as np
from material_context import sha


def classify(run, out):
    e=json.loads((run/'launch.json').read_text())['environment']
    prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_SURFACE_D_'
    src=Path(e[prefix+'FACES_CSV'])
    if sha(src)!=e[prefix+'FACES_SHA256']:raise ValueError('Changed anatomical surface')
    tri=np.loadtxt(src,delimiter=',',skiprows=1,usecols=(1,2,3),dtype=int)
    audit=json.loads((run/'accepted-layers.json').read_text())
    pairs=audit['witnesses'].get('2-2',[])
    if len(pairs)!=audit['pair_counts'][2][2]:raise ValueError('Incomplete intersection witnesses')
    hist=collections.Counter();witnesses=[]
    for a,b in pairs:
        if not 0<=a<b<len(tri):raise ValueError('Unregistered or headwall witness')
        common=sorted(set(tri[a])&set(tri[b]));hist[len(common)]+=1
        witnesses.append(dict(faces=[a,b],shared_nodes=[int(i) for i in common]))
    out.write_text(json.dumps(dict(frame=sha(run/'physical-accepted.node'),facet_topology=sha(src),
        adjacency=dict(hist),witnesses=witnesses),indent=2)+'\n')


if __name__=='__main__':classify(*(Path(v).resolve() for v in sys.argv[1:]))
