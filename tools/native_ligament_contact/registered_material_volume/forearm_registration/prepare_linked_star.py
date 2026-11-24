"""Assemble a second bounded material region from the measured right-ulna residual in the bilateral scene."""
from pathlib import Path
from itertools import combinations
import json,sys
import numpy as np
from compile_connected_joint import digest

def run(v,fit,motion,out):
    d=json.loads((out/'region.json').read_bytes());selected=d['stages']['initial']['largest_J_cell'];raw=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');t=raw[:,:4]-1
    if d['peak'] not in t[selected] or raw[selected,4]!=16 or d['stages']['initial']['largest_J']<2.4:raise ValueError('Missing recruited material-volume witness')
    n=t[selected];near=np.flatnonzero(np.isin(t,n).any(1));ids1=np.unique(t[near]);ring=np.flatnonzero(np.isin(t,ids1).any(1));ids=np.unique(t[ring])
    coords=[np.loadtxt(p,skiprows=1,usecols=(1,2,3))[ids]for p in (v/'mesh.node',fit/'current.node',motion/'physical-accepted.node')]
    L=np.loadtxt(v/'current-child.labels.csv',skiprows=1,delimiter=',',dtype='i8')[ring];w=np.load(v/'material-walls.npz')
    wall={k:w[k][np.isin(w[k],ids1).any(1)]for k in 'OIDC'}
    np.savez_compressed(out/'cavity.npz',tet=t[ring],rows=ring,local=np.searchsorted(ids,t[ring]),ids=ids,mass=np.fromfile(v/'cell-mass.f64','<f8')[ring],xyz=coords,labels=L,selected=selected,neighbors=near,**wall)
    # Keep the same high-force contributor selection rule as the left-side fit.
    r=d['stages']['initial']; vals=[dict(row=p['row'],along_net_N=p['signed_N'])for p in r['cells']]
    (out/'force-owners.json').write_text(json.dumps(dict(initial=dict(force_N=r['net_N'],cells=vals)),indent=2)+'\n')
    src=out/'parent-volume';src.mkdir(exist_ok=True)
    for name in ('mesh.node','mesh.ele','current.node','node-mass.f64','cell-mass.f64','material-walls.npz','current-child.labels.csv'):
        p=(fit if name=='current.node' else v)/name
        if not (src/name).exists():(src/name).symlink_to(p.resolve())
if __name__=='__main__':run(*(Path(s).resolve()for s in sys.argv[1:]))
