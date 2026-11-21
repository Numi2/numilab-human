"""Register finite material pressure corners to the actual, compiled contact vertices."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial import cKDTree
from material_context import sha
def prepare(part,base,out):
    out.mkdir(exist_ok=False)
    d=json.loads((base/'manifest.json').read_text());spec=d['finite_facets']
    pairs=np.fromfile(base/'pelvic-finite-pairs.bin','f4').reshape(-1,17,4)
    id=pairs[:,0].copy().view('u4')
    banks={(r['muscle'],r['cortical_body']):r for r in d['cortical_certificate']['banks']}
    cortex=np.fromfile(base/'cortical-certificate-faces.bin','u4').reshape(-1,4)
    first=d['closed_reverse']['first'];pool=np.fromfile(base/'pool-material-points.bin','f4').reshape(-1,7,4)
    original=np.fromfile(base/'cortical-triangles.bin','f4').reshape(-1,10,4)
    mat=np.empty((len(pairs),4),'u4');summary=[]
    for side,reg in enumerate(d['closed_reverse']['sides']):
        sid=29+side;ii=np.flatnonzero(id[:,2]==side);src=id[ii,3]
        vol=np.load(part/'projected'/f'{sid}.npz');ports=np.load(part/'ports'/f'{sid}.npz');boundary=np.load(part/'conforming-boundary'/f'{sid}.npz')
        distance,loc=cKDTree(boundary['vertices']).query(vol['port_vertices'][src].reshape(-1,3))
        if max(distance)>2.e-12:raise ValueError(('finite material corner missing',sid,max(distance)))
        if not np.array_equal(vol['port_shape'][src].astype('f4'),pairs[ii,1:4]):raise ValueError('Changed force interpolation')
        target=ports['target_face'][src];b=ports['target_body'][src]
        if not np.array_equal(b,id[ii,1]):raise ValueError('Cortical owner mismatch')
        row=np.array([banks[(sid,int(body))]['first']+int(t) for body,t in zip(b,target)])
        if not np.array_equal(pool[first+cortex[row,:3],2,:3],original[id[ii,0],:3,:3]):raise ValueError('Cortical corner mismatch')
        mat[ii,:3]=loc.reshape(-1,3)+reg['first_vertex'];mat[ii,3]=row
        summary.append(dict(muscle=sid,registered_faces=len(ii),maximum_corner_residual_m=float(max(distance))))
    for p in base.iterdir():
        if p.name!='manifest.json':(out/p.name).hardlink_to(p)
    target=out/'shared-finite-faces.bin';mat.tofile(target);d['files'][target.name]=sha(target)
    d['shared_finite_faces']={'parent':sha(base/'manifest.json'),'regions':summary}
    (out/'manifest.json').write_text(json.dumps(d,indent=2)+'\n')
if __name__=='__main__':prepare(*map(Path,sys.argv[1:]))
