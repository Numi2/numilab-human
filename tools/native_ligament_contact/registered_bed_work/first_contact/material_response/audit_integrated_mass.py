"""Reconstruct the full cell and nodal mass arrays and independently check both physical layers."""
from pathlib import Path
import json,sys
import numpy as np
from supported_material_admission import geom
from registered_density import density_bounds
from audit_registered_skeleton import digest

def run(job,mesh,domain,cavity,flow,transfer,out):
    z=np.load(domain);O=np.load(cavity)['source_tetrahedra'];src=np.loadtxt(mesh/'mesh.ele',skiprows=1,dtype='i4');x=np.loadtxt(mesh/'mesh.node',skiprows=1,usecols=(1,2,3))
    t=src[:,1:5]-1;m=src[:,5];mass=np.fromfile(mesh/'cell-mass.f64',dtype='<f8');nm=np.fromfile(mesh/'node-mass.f64',dtype='<f8');n=len(x)
    oldkeep=np.ones(len(t),bool);oldkeep[O]=False;newmass=mass.copy();child=np.zeros(len(z['t']));oldpoints=np.unique(t[O])
    lookup={x[i].astype('f4').tobytes():i for i in oldpoints};node=np.full(len(z['x']),-1,int);newpts=[]
    for j in np.unique(z['t']):
        old=lookup.get(z['x'][j].tobytes())
        if old is None:old=n+len(newpts);newpts.append(z['x'][j].astype(float))
        node[j]=old
    points=np.vstack((x,newpts));tt=np.vstack((t[oldkeep],node[z['t']]));role=np.r_[m[oldkeep],z['role']];owner=Path(json.loads((job/'launch.json').read_bytes())['compiled_material_owner']);report={}
    for tag,material in [('SKIN',15),('FAT',16)]:
        verdict=json.loads((transfer/(tag+'-transport.json')).read_bytes())['final']
        if not verdict['passed']:raise ValueError('Missing source-level transfer certificate')
        d=np.load(transfer/(tag+'-transport.npz'));ext=d['external'];rec=d['child_ids']
        if not np.all(oldkeep[ext]) or not np.all(m[ext]==material) or not np.array_equal(rec,np.flatnonzero(z['role']==material)):raise ValueError('Wrong tissue owner')
        child[rec]=d['child_mass_kg'];newmass[ext]=d['external_mass_kg']
    joined=np.r_[newmass[oldkeep],child];vol=np.empty(len(tt));qual=np.empty(len(tt))
    for i in range(0,len(tt),32768):vol[i:i+32768],qual[i:i+32768]=geom(points,tt[i:i+32768])
    nodal=np.bincount(tt.ravel(),weights=np.repeat(joined*.25,4),minlength=len(points));active=np.unique(tt)
    if np.any(vol<=0) or np.any(joined<=0) or np.any(nodal[active]<=0):raise ValueError('Nonpositive tissue mass or volume')
    for tag,material in [('SKIN',15),('FAT',16)]:
        use=role==material;rho=joined[use]/vol[use];lo,hi=density_bounds(owner,material);old=mass[m==material].sum();new=joined[use].sum();nn=np.bincount(tt[use].ravel(),weights=np.repeat(joined[use]*.25,4),minlength=len(points))
        oldc=t[m==material];bef=np.zeros(3);aft=np.zeros(3)
        for i in range(4):
            bef+=np.einsum('i,ij->j',mass[m==material],x[oldc[:,i]])/4
            aft+=np.einsum('i,ij->j',joined[use],points[tt[use,i]])/4
        residual=aft-bef;bad=((vol[use]<=1e-18)|(qual[use]<1e-5))
        valid=abs(new-old)<1e-11 and abs(nn.sum()-new)<1e-11 and abs(residual).max()<1.e-11 and rho.min()>=lo-1e-8 and rho.max()<=hi+1e-8
        report[tag]=dict(source_kg=float(old),cells_kg=float(new),nodal_kg=float(nn.sum()),mass_error_kg=float(new-old),density_range=[float(rho.min()),float(rho.max())],first_moment_error_kg_m=residual.tolist(),invalid_cells=int(bad.sum()),inverted=0,passed_mass=bool(valid))
    out.mkdir();cols=dict(SKIN=report['SKIN'],FAT=report['FAT'],source_cell_mass=digest(mesh/'cell-mass.f64'),domain=digest(domain),quadrature_topology=digest(flow/'result.json'),emitted_cells=len(tt),emitted_active_nodes=len(active),total_nodal_kg=float(nodal.sum()),source_total_nodal_kg=float(nm.sum()),scope='assembled mass arrays prior to runtime geometry admission')
    (out/'result.json').write_text(json.dumps(cols,indent=2)+'\n');print(report)
if __name__=='__main__':run(*(Path(k).resolve() for k in sys.argv[1:]))
