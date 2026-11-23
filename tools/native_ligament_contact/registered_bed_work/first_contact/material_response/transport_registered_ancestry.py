"""Transport a labeled parent incidence through a continuous positive element deformation."""
from pathlib import Path
import sys,json
import numpy as np
from compile_connected_joint import digest
from fit_carpal_clearance import geom


def run(before,after,flow,output):
    old,new=np.load(before),np.load(after);a=np.load(flow/'ancestry/overlap.npz')
    manifest=json.loads((flow/'ancestry/result.json').read_text())
    if manifest['mesh']!=digest(before) or manifest['mapping']!=digest(flow/'ancestry/overlap.npz'):
        raise ValueError('Unexpected donor ancestry generation')
    for k in ('role','t','locked','fence','facets','markers'):
        if not np.array_equal(old[k],new[k]):raise ValueError(f'Changed material {k}')
    if not np.array_equal(old['x'][old['locked']],new['x'][old['locked']]):raise ValueError('Moved joining vertices')
    t=old['t'];p,q=(f['x'].astype(float)[t] for f in (old,new))
    A=p[:,1:]-p[:,:1];delta=q[:,1:]-q[:,:1]-A
    def det(i):
        v=np.stack([delta[:,k] if i&(1<<k) else A[:,k] for k in range(3)],axis=1)
        return np.einsum('ij,ij->i',v[:,0],np.cross(v[:,1],v[:,2]))/6
    c=np.array([det(0),sum(det(i) for i in (1,2,4)),sum(det(i) for i in (3,5,6)),det(7)])
    lower=np.minimum(c[0],c.sum(axis=0))
    for i in np.flatnonzero((delta!=0.).any((1,2))):
        for root in np.roots([3*c[3,i],2*c[2,i],c[1,i]]):
            if abs(root.imag)<1e-12 and 0<root.real<1:
                lower[i]=min(lower[i],np.polynomial.polynomial.polyval(root.real,c[:,i]))
    if lower.min()<=0:raise ValueError('The material continuation inverts cells')
    src,tar=(geom(v)[0]*1e9 for v in (p,q));ch=a['child']
    if not np.allclose(a['child_volume_mm3'],src,rtol=1e-9,atol=1e-13):raise ValueError('Wrong source cell volumes')
    out=output/'ancestry';out.mkdir(parents=True)
    vals={k:a[k] for k in a.files}
    vals['volume_mm3']=a['volume_mm3']*(tar/src)[ch];vals['child_volume_mm3']=tar
    vals['edge_is_advected']=np.asarray(a['edge_is_advected'],bool)|np.any(delta[ch]!=0,axis=(1,2))
    dest=out/'overlap.npz';np.savez_compressed(dest,**vals)
    doc=dict(mesh=digest(after),mapping=digest(dest),parent_mesh=digest(before),parent_mapping=manifest['mapping'],
             minimum_path_volume_m3=float(lower.min()),advected_children=int((delta!=0).any((1,2)).sum()),source='unchanged material tetrahedra along continuously positive piecewise-affine deformation')
    (out/'result.json').write_text(json.dumps(doc,indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
