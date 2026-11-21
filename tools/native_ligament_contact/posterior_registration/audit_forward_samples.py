"""Independently measure every registered forward site on the captured native triangles."""
from pathlib import Path
import json,sys
import numpy as np
from audit_finite_cortical_crossings import load_vertices
from cortical_distance import CorticalSurface
from material_context import sha
def run(assets,cap,lib,out):
    d=json.loads((assets/'manifest.json').read_text());g=d['closed_reverse'];regs=g['sides']
    step,v=load_vertices(cap,{'material_vertex':sum(r['vertices'] for r in regs),'cortical_vertex':g['unique_vertices']})
    maps=np.fromfile(assets/'forward-surface-samples.bin','f4').reshape(-1,2,4)
    ids=maps[:,0,:3].copy().view('u4');b=maps[:,1,:3].astype(float);pts=v['material_vertex'][ids]
    pts=pts[:,0]+b[:,1:2]*(pts[:,1]-pts[:,0])+b[:,2:3]*(pts[:,2]-pts[:,0])
    skin=np.fromfile(assets/'pool-material-points.bin','f4').reshape(-1,7,4)[:len(maps),3].copy().view('u4')
    faces=np.fromfile(assets/'cortical-certificate-faces.bin','u4').reshape(-1,4)
    result={}
    for bank in d['cortical_certificate']['banks']:
        owner=bank['cortical_body'];m=bank['muscle'];part=faces[bank['first']:bank['first']+bank['count'],:3];vi,fi=np.unique(part,return_inverse=True)
        surf=CorticalSurface(lib,v['cortical_vertex'][vi],fi.reshape(-1,3))
        idx=np.flatnonzero((skin[:,1]==(131,145)[m-29])&(skin[:,2]==owner));gap,*_=surf.query(pts[idx])
        j=int(np.argmin(gap))
        result[f'{m}/{owner}']=dict(points=len(idx),minimum_signed_m=float(gap[j]),negative_sites=int(sum(gap<0)),worst_forward_row=int(idx[j]))
    out.write_text(json.dumps({'accepted_step':step,'capture':sha(cap),'assets':sha(assets/'manifest.json'),'regions':result},indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
