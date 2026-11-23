from pathlib import Path
import sys,json
import numpy as np
from compile_connected_joint import digest
def measure(x,t):
    geom=x[t];e=geom[:,1:]-geom[:,:1];det=np.einsum('ij,ij->i',e[:,0],np.cross(e[:,1],e[:,2]));mx=np.zeros(len(t))
    for i in range(3):
        for j in range(i+1,4):mx=np.maximum(mx,np.einsum('ij,ij->i',geom[:,i]-geom[:,j],geom[:,i]-geom[:,j]))
    return det, mx*np.sqrt(mx)
def run(h,field,out):
    root=h/'closed-native/native';x=np.loadtxt(root/'mesh.node',skiprows=1,usecols=(1,2,3));d=np.load(field/'material-transport.npz');y=x.copy();y[d['nodes']]=d['positions']
    data=np.loadtxt(root/'mesh.ele',skiprows=1,dtype='i4');t=data[:,1:5]-1;mat=data[:,5];mass=np.fromfile(root/'cell-mass.f64','<f8')
    det,L=measure(y,t);v=det/6;lo=np.where(mat==15,900.,850.);hi=np.where(mat==15,1100.,950.);q=det/np.maximum(L,1e-120)
    # Report raw density misses separately from on-bound roundoff. The native
    # importer uses its exact per-mass allocation tolerance for promotion.
    invalid=(v<=1e-18)|(q<1e-5);density=~((mass<=v*hi*(1+1e-10))&(mass>=v*lo*(1-1e-10)))
    out.mkdir();np.savez_compressed(out/'witness.npz',geometry=np.flatnonzero(invalid),density=np.flatnonzero(density))
    r=dict(transport=digest(field/'material-transport.npz'),inverted=int((det<=0).sum()),volume=int((v<=1e-18).sum()),quality=int((q<1e-5).sum()),min_quality=float(q.min()),density_misses=int(density.sum()),skin_mass_kg=float(mass[mat==15].sum()),fat_mass_kg=float(mass[mat==16].sum()),min_z_m=float(y[:,2].min()))
    (out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
if __name__=='__main__':run(*(Path(v) for v in sys.argv[1:]))
