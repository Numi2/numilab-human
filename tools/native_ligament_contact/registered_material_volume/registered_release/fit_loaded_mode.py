"""Prepare an internal bonded-interface position by balancing the elastic acceleration of a measured volume mode."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.optimize import minimize
from prepare_indentation import soft_response
from supported_material_admission import geom
from import_conditioned_interior import write_nodes
from compile_connected_joint import digest
from measure_mode import jac_volume

def run(h,v,material,out,traction_fit=False):
    a=np.load(h/'right-ulna/cavity.npz');rows=a['rows'];T=a['local'];ids=a['ids'];x,y,z=a['xyz']
    sel=int(a['selected']);target=T[np.searchsorted(rows,sel)]
    dy=z-y;dt=json.loads((h/'motion-check/motion.json').read_bytes())['accepted_s'];mass=np.fromfile(v/'node-mass.f64','<f8')[ids[target]]
    coeff=json.loads(material.read_bytes())['materials']
    mu,bulk=np.array([[coeff[str(m)]['registered'][k] for k in ('source_mu_pa','source_logJ_coefficient_pa')]for m in a['labels'][:,1]]).T
    star=np.flatnonzero(np.isin(T,target).any(1))
    baseforce=soft_response(x,y,T[star],mu[star],bulk[star])(y)[0]
    grav=jac_volume(x,y,target)/mass[:,None];weights=np.zeros_like(x);weights[target]=grav
    contributions=-(baseforce*weights[T[star]]).sum((1,2))*dt**2
    order=np.argsort(abs(contributions))[::-1];selected=order[:np.searchsorted(np.cumsum(abs(contributions[order])),.90*sum(abs(contributions)))+1]
    candidates=np.unique(np.r_[target,T[star[selected]].ravel()]);free=candidates[np.isin(ids[candidates],a['I']) & ~np.isin(ids[candidates],np.concatenate([a[k].ravel()for k in 'ODC']))]
    if not np.isin(target,free).all():raise ValueError('Witness contacts a protected interface')
    active=np.flatnonzero(np.isin(T,free).any(1));t=T[active];n=len(free);law=soft_response(x,y,t,mu[active],bulk[active]);witness=np.searchsorted(active,np.searchsorted(rows,sel))
    edges=lambda p:p[t[:,1:]]-p[t[:,:1]]
    inv=np.linalg.inv(edges(x));V=np.linalg.det(edges(x))/6
    F=lambda p:inv@edges(p)
    def modes(p):
        force,J=law(p); f=np.zeros_like(x);np.add.at(f,t.ravel(),-force.reshape(-1,3))
        acc=dt**2*np.sum(jac_volume(x,p,target)*f[target]/mass[:,None])
        return acc,float(J[witness]),np.linalg.norm(f[free],axis=1)
    def energy(p):
        f=F(p);l=np.log(np.maximum(np.linalg.det(f),1.e-20));return V@(.5*mu[active]*((f*f).sum((1,2))-3)-mu[active]*l+.5*bulk[active]*l*l)
    s0=[np.linalg.svd(F(p),compute_uv=False)for p in (y,z)];q0=geom(z,t)[1];E=energy(y);acc0,J0,_=modes(y)
    def pose(u):
        p=y.copy();p[free]+=u.reshape(-1,3)*1e-6;return p
    def domain(p):
        vals=[]
        for j,Y in enumerate((p,p+dy)):
            s=np.linalg.svd(F(Y),compute_uv=False);J,q=geom(Y,t);J/=V
            vals.extend([J-(.4 if j==0 else .35),2.5-J,(q-(1e-5 if j==0 else np.minimum(q0,1e-5)))*1e4,
                         1.0045*np.maximum(s0[j][:,0],1.1)-s[:,0],s[:,-1]-np.minimum(s0[j][:,-1],.9)/1.0045])
        return np.concatenate(vals)
    # Scale coordinates in micrometres. Prefer the smallest repair that reduces
    # the initial volume-mode acceleration and does not add patch strain energy.
    def objective(u):return .5*u@u, u
    g0=domain(y)
    from interface_traction import traction_measure
    traction=traction_measure(x,t,witness,mu[active],bulk[active],a['labels'][active,1])
    jump0,j0,_,areas,neighbors=traction(y)
    # Neighboring same-donor fat on source facets provides a local reference for log volume.
    near=np.average(np.log(j0[1:][neighbors]),weights=areas[neighbors])
    targetJ=min(J0,np.exp((np.log(J0)+near)*.5))
    def fit_bounds(p):
        acc,J,_=modes(p)
        if not traction_fit:return np.asarray([J0-J])
        jump=traction(p)[0]
        return np.asarray([targetJ-J,.85-np.sqrt(jump/jump0)])
    def lattice(p):
        def error(q):
            acc,J,_=modes(q)
            gaps=np.r_[domain(q),fit_bounds(q),(E-energy(q))/E,abs(acc0)-abs(acc)]
            return np.linalg.norm(np.minimum(gaps,0))
        cur=p.copy();v=error(cur);moves=0
        for _ in range(48):
            if v<1e-9:break
            best=v;pick=None
            for k in free:
                for c in range(3):
                    old=cur[k,c]
                    for way in (-1,1):
                        cur[k,c]=np.nextafter(np.float32(old),np.float32(way*np.inf),dtype='f4')
                        e=error(cur)
                        if e<best-1e-15:best=e;pick=(k,c,cur[k,c])
                    cur[k,c]=old
            if pick is None:break
            k,c,val=pick;cur[k,c]=val;v=best;moves+=1
        return cur,moves
    attempts=[];candidates=[]
    for desired in ((.10,.35,.75) if traction_fit else (.05,.25,.60)):
        def feasible(u):
            p=pose(u);acc,J,_=modes(p)
            return np.r_[domain(p),desired*abs(acc0)-acc,desired*abs(acc0)+acc,(E-energy(p))/E,fit_bounds(p)+1e-6]
        sol=minimize(objective,np.zeros(3*n),jac=True,bounds=[(-200.,200.)]*(3*n),
                     constraints=[dict(type='ineq',fun=feasible)],method='SLSQP',options=dict(ftol=1e-8,maxiter=100))
        for factor in (1.,.99,.95,.875,.75,.5,.25):
            p=pose(factor*sol.x).astype('f4').astype(float);p,moves=lattice(p);acc,J,forces=modes(p);delta=p[free]-y[free]
            good=min(domain(p))>=-1e-9 and min(fit_bounds(p))>=-1e-9 and energy(p)<=E and abs(acc)<abs(acc0)
            print(desired,factor,sol.message,round(float(min(domain(p))),7),round(J-J0,6),energy(p)-E,acc,flush=True)
            if good:candidates.append((abs(acc),np.linalg.norm(delta),p,acc,J,energy(p),forces))
        attempts.append(dict(desired_fraction=desired,status=sol.message,objective=sol.fun,emitted_candidates=len(candidates)))
        if candidates and min(r[0]for r in candidates)<.12*abs(acc0):break
    if not candidates:raise ValueError(f'No admissible modal correction: {attempts}')
    _,_,p,acc,J,e,fs=min(candidates,key=lambda r:r[:2])
    out.mkdir(exist_ok=True)
    write_nodes(h/'guarded-fit/current.node',out/'current.node',{int(i):p[k]for k,i in zip(free,ids[free]) if np.any(p[k]!=y[k])})
    prior=json.loads((h/'guarded-fit/fit.json').read_bytes())
    newids=[int(i)for k,i in zip(free,ids[free]) if np.any(p[k]!=y[k])]
    if set(newids)&set(prior['nodes_0_based']):raise ValueError('Overlap with prior registration')
    result={k:prior[k]for k in ['source_reference','control_loaded','elements','source_mass']}
    result.update(registered=digest(out/'current.node'),nodes_0_based=sorted(prior['nodes_0_based']+newids),changed_nodes=len(prior['nodes_0_based'])+len(newids),added_nodes=newids,source_registration=prior['registered'],
        traction_RMS_Pa=[jump0**.5,traction(p)[0]**.5],neighbor_reference_J=float(np.exp(near)),traction_fit=traction_fit,witness_cell=sel,contributing_rows=rows[star[selected]].tolist(),added_incident_cells=len(t),initial_elastic_modal_J_change=[acc0,acc],
        registered_J=[J0,J],patch_energy_J=[E,e],max_initial_assembled_material_N=[max(modes(y)[2]),max(fs)],max_added_shift_um=float(np.linalg.norm(p[free]-y[free],axis=1).max()*1e6),attempts=attempts)
    (out/'fit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':run(*(Path(s).resolve()for s in sys.argv[1:5]),traction_fit='--traction' in sys.argv[5:])
