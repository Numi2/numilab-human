"""Integrate the existing finite-strain muscle law on each occupied volume at an accepted native pose."""
import csv,json,sys
from pathlib import Path
import numpy as np
from material_context import sha

def material(x,t,inv,vol,mu,lam,relative=False):
    e=x[t[:,1:]]-x[t[:,:1]]
    F=(inv@e).transpose(0,2,1)
    if relative:
        H=F
        tr=np.trace(H,axis1=1,axis2=2)
        hess=.5*(tr*tr-np.trace(H@H,axis1=1,axis2=2))+np.linalg.det(H)
        q=tr+hess;J=1+q;ln=np.log1p(q);F=H+np.eye(3)
        # Independent double stress product and a cancellation-safe strain energy.
        thermal=(q-ln)-hess
        density=mu*(.5*(H*H).sum((1,2))+thermal)+.5*lam*ln**2
    else:
        J=np.linalg.det(F);ln=np.log(J)
        density=.5*mu*((F*F).sum((1,2))-3-2*ln)+.5*lam*ln**2
    if min(J)<=0:raise ValueError('Nonpositive carrier')
    P=mu*F+(lam*ln-mu)[:,None,None]*np.linalg.inv(F).transpose(0,2,1)
    grad=inv.transpose(0,2,1)
    grad=np.concatenate((-grad.sum(1)[:,None],grad),axis=1)
    cell=vol[:,None,None]*(grad@P.transpose(0,2,1))
    internal=np.zeros_like(x);np.add.at(internal,t,cell)
    energy=np.sum(vol*density)
    return internal,energy,J

def verify(part,resp,assets,capture,out):
    with capture.open() as f:rows=list(csv.DictReader(f))
    if len({r['accepted_step'] for r in rows})!=1:raise ValueError('Mixed state')
    def field(k):return {int(r['index']):np.array([r[a] for a in 'xyzw'],dtype='f4').astype('f8') for r in rows if r['field']==k}
    pts=field('material_node');x=np.stack([pts[i][:3] for i in range(len(pts))]);disp=field('node_displacement');u=np.stack([disp[i][:3] for i in range(len(pts))])
    contacts=field('normal_and_force');reverse=field('reverse_shape');parents=field('reverse_parent_nodes')
    pool=np.fromfile(assets/'pool-material-points.bin','f4').reshape(-1,7,4);c=np.fromfile(assets/'material-cells.bin','u4').reshape(-1,4)
    total=field('material_delta');relative=bool(total)
    if relative:x=np.stack([total[i][:3] for i in range(len(pts))])
    invall=np.fromfile(assets/('pelvic-reference-inverse.bin' if relative else 'material-inverse.bin'),'f4').reshape(-1,3,4)[:,:,:3].astype(float)
    d=json.loads((assets/'manifest.json').read_text());mu=d['specimen']['mu'];lam=d['specimen']['lam'];external=np.zeros_like(x)
    for i,r in contacts.items():
        if not r[3]:continue
        if i>=d['closed_reverse']['first'] and (i not in parents or i not in reverse):raise ValueError('Missing current contact interpolation')
        nodes=parents[i].astype(int) if i in parents else pool[i,5].copy().view('u4')
        w=reverse.get(i,pool[i,6])
        np.add.at(external,nodes,w[:,None]*r[:3]*r[3])
    result={};start=0;first=0;rng=np.random.default_rng(21)
    for sid in (29,30):
        p=np.load(part/'projected'/f'{sid}.npz');red=np.load(resp/f'{sid}.npz');v=p['cell_volumes'];valid=v>0;t=p['cells'][valid];v=v[valid];count=len(p['vertices']);inv=invall[first:first+len(t)];sl=slice(start,start+count)
        if not np.array_equal(t+start,c[first:first+len(t)]):raise ValueError('Different cell ordering')
        free=np.setdiff1d(np.unique(t),red['fixed']);y=x[sl];f,E,J=material(y,t,inv,v,mu,lam,relative);ext=external[sl]
        # Derivative of integrated finite-strain energy, perturbing free nodes only.
        direction=np.zeros_like(y);direction[free]=rng.normal(size=(len(free),3));direction/=np.linalg.norm(direction)
        h=2.e-8;Ep=material(y+h*direction,t,inv,v,mu,lam,relative)[1];Em=material(y-h*direction,t,inv,v,mu,lam,relative)[1]
        analytic=np.sum(direction*f);numeric=(Ep-Em)/(2*h);error=abs(analytic-numeric)
        if error>3.e-9+2.e-6*abs(analytic):raise ValueError(('energy gradient disagreement',analytic,numeric,error))
        res=np.linalg.norm((f-ext)[free],axis=1);raw,_,_=material(y-u[sl],t,inv,v,mu,lam,relative)
        result[sid]=dict(max_free_bulk_force_residual_N=float(res.max()),worst_free_node=int(free[np.argmax(res)]),max_free_contact_force_N=float(np.linalg.norm(ext[free],axis=1).max()),max_free_force_in_donor_pose_N=float(np.linalg.norm(raw[free],axis=1).max()),min_J=float(min(J)),max_J=float(max(J)),total_energy_J=float(E),force_energy_directional_error_N=float(error),net_internal_force_N=np.linalg.norm(f.sum(0)))
        start+=count;first+=len(t)
    assert start==len(x) and first==len(c)
    out.write_text(json.dumps(dict(accepted_step=int(rows[0]['accepted_step']),capture_sha256=sha(capture),deformation_representation='reference plus material delta' if relative else 'absolute material vertices',assets_sha256=sha(assets/'manifest.json'),sides=result),indent=2)+'\n')
if __name__=='__main__':verify(*map(Path,sys.argv[1:]))
