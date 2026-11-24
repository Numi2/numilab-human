"""Relax the measured bonded-interface star with the registered constitutive law and fixed surrounding anatomy."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.optimize import minimize
from prepare_indentation import soft_response
from import_conditioned_interior import write_nodes
from compile_connected_joint import digest
from supported_material_admission import geom

def run(volume, material, out, connected=False, guard=False):
    data=np.load('cavity.npz'); rows=data['rows'];source=int(data['selected']);idx=np.searchsorted(rows,data['neighbors'])
    ids=data['ids']; x,y,z=data['xyz']
    free=data['local'][np.flatnonzero(rows==source)[0]]
    if connected:
        ranks=json.loads(Path('force-owners.json').read_bytes())['initial']
        force_cells=[r['row'] for r in ranks['cells'] if abs(r['along_net_N'])>.05*ranks['force_N']]
        add=np.unique(data['local'][np.searchsorted(rows,force_cells)])
        # A complete star is present for every vertex in the one-ring. Keep the
        # external skin, deep fascia, cortical interface and all anchors fixed.
        add=add[np.isin(ids[add],data['I']) & ~np.isin(ids[add],np.concatenate([data[k].ravel() for k in 'ODC']))]
        free=np.unique(np.r_[free,add])
        idx=np.flatnonzero(np.isin(data['local'],free).any(1))
        if not np.isin(free,np.unique(data['local'][np.searchsorted(rows,data['neighbors'])])).all():
            raise ValueError('Missing source incidence')
    t=data['local'][idx];n=ids[free]
    if any(np.isin(n,data[k]).any() for k in 'ODC'):raise ValueError('Free nodes intersect a protected wall')
    if not np.isin(n,data['I']).all():raise ValueError('No bonded interface provenance')
    coeff=json.loads(material.read_bytes())['materials']; mat=data['labels'][idx,1]
    mu,bulk=np.array([[coeff[str(m)]['registered'][s] for s in ('source_mu_pa','source_logJ_coefficient_pa')]for m in mat]).T
    law=soft_response(x,y,t,mu,bulk);edges=lambda p:p[t[:,1:]]-p[t[:,:1]]
    volume0=np.linalg.det(edges(x))/6; inv=np.linalg.inv(edges(x));qprior=geom(z,t)[1]
    def state(u):
        p=y.copy();p[free]+=u.reshape(-1,3)*1e-6;return p
    def energy(p):
        F=inv@edges(p);J=np.linalg.det(F)
        l=np.log(np.maximum(J,1e-14))
        return volume0@(.5*mu*((F*F).sum((1,2))-3)-mu*l+.5*bulk*l*l)
    def force(p):
        f,_=law(p);out=np.zeros_like(p)
        np.add.at(out,t.ravel(),f.reshape(-1,3));return out
    def report(p):
        f=force(p);_,j=law(p);ij=np.flatnonzero(rows[idx]==source)[0]
        return dict(witness_J=j[ij],free_internal_force_N=np.linalg.norm(f[free],axis=1).tolist(),
                    energy_J=energy(p),J_interval=[float(j.min()),float(j.max())])
    initial=energy(y)
    def objective(u):
        p=state(u);return (energy(p)-initial)*1e6, force(p)[free].ravel()
    def stretches(pos):
        return np.linalg.svd(inv@edges(pos), compute_uv=False)
    upper_strain=[1.005*np.maximum(stretches(pos)[:,0],1.1) for pos in (y,z)]
    lower_strain=[np.minimum(stretches(pos)[:,-1],.9)/1.005 for pos in (y,z)]
    def constraints(u):
        p=state(u);e=z+(p-y);j,q=geom(p,t); je,qe=geom(e,t);j/=volume0;je/=volume0
        c=np.r_[j-.4,2.5-j,je-.35,2.5-je,(q-1e-5)*1e4,(qe-np.minimum(qprior,1e-5))*1e4]
        if guard:
            a,b=stretches(p),stretches(e)
            # Matched per-cell distortion envelope, with a small elastic window near identity.
            # Allow at most 0.5 percent worsening beyond that window for Float32 emission.
            c=np.r_[c,upper_strain[0]-a[:,0],a[:,-1]-lower_strain[0],upper_strain[1]-b[:,0],b[:,-1]-lower_strain[1]]
        return c
    width=200. if connected else 50.
    lower,upper=-np.ones(3*len(n))*width,np.ones(3*len(n))*width
    res=minimize(objective,np.zeros(3*len(n)),jac=True,bounds=list(zip(lower,upper)),constraints=[dict(type='ineq',fun=constraints)],method='SLSQP',options=dict(maxiter=240 if guard else 120,ftol=1e-10))
    trials=[];attempts=[]
    for s in (1.,.875,.75,.5,.25,.125):
        p=state(res.x*s).astype('f4').astype(float);u=(p[free]-y[free]).ravel()*1e6
        attempts.append([s,float(min(constraints(u)))])
        if min(constraints(u)) < -1e-9:continue
        r=report(p);r['witness_stretches']=stretches(p)[np.searchsorted(rows[idx],source)].tolist();trials.append((r['energy_J'],p,r,s))
    if not trials:raise ValueError(f'No admissible emitted material: {res.message} {attempts}')
    e,p,r,s=min(trials,key=lambda a:a[0]);prior=report(y)
    if not e<initial or max(r['free_internal_force_N'])>=max(prior['free_internal_force_N']):raise ValueError('No material force improvement')
    out.mkdir(exist_ok=True)
    write_nodes(volume/'current.node',out/'current.node',{int(i):v for i,v in zip(n,p[free])})
    d=dict(source_reference=digest(volume/'mesh.node'),control_loaded=digest(volume/'current.node'),elements=digest(volume/'mesh.ele'),source_mass=digest(volume/'node-mass.f64'),registered=digest(out/'current.node'),changed_nodes=len(n),nodes_0_based=n.tolist(),maximum_shift_um=float(1e6*np.linalg.norm(p[free]-y[free],axis=1).max()),before=prior,after=r,fraction=s,
           fit_message=res.message,material=digest(material),surrounding_force_change_N=float(np.linalg.norm((force(p)-force(y))[~np.isin(np.arange(len(y)),free)],axis=1).max()))
    (out/'fit.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
if __name__=='__main__':run(*(Path(s).resolve()for s in sys.argv[1:4]), connected='--connected' in sys.argv[4:],guard='--guard' in sys.argv[4:])
