"""Conserve each existing source carrier and regional first moments on a locally repaired material volume."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,hstack,vstack
from certify_transported_material import measure
from compile_connected_joint import digest

def run(job,h,field,out):
    s=np.load(field/'candidate.npz');base=h/'closed-native/native'
    x=np.loadtxt(base/'mesh.node',skiprows=1,usecols=(1,2,3)).astype('f4').astype(float);y=x.copy();y[s['nodes']]=s['positions']
    e=np.loadtxt(base/'mesh.ele',skiprows=1,dtype='i4');tet=e[:,1:5]-1;mat=e[:,5];cellmass=np.fromfile(base/'cell-mass.f64','<f8')
    path=job/'feasible-carriers/FAT-physical-flow.npz';f=np.load(path);q=f['carrier'];c=f['role_cell'];weight=f['mass'];roles=np.flatnonzero(mat==16);t=tet[roles]
    before=np.bincount(c,weights=weight,minlength=len(t));assert np.max(abs(before-cellmass[roles]))<1e-16
    incident=np.flatnonzero(np.isin(t,s['nodes']).any(1));current=incident.copy();out.mkdir()
    floor=850.;ceiling=950.;unit=1e-9;history=[]
    for ring in range(4):
        target=current;edges=np.flatnonzero(np.isin(c,target));donor,qrow=np.unique(q[edges],return_inverse=True);dest=np.searchsorted(target,c[edges]);n=len(edges)
        v=measure(y,t[target])[0]/6.;cg=y[t[target]].mean(1);old=x[t[target]].mean(1);offset=cg.mean(0)
        # First moments are measured at the receiving tet centroid, exactly the center of its lumped nodal masses.
        C=coo_matrix((np.ones(n),(dest,np.arange(n))),shape=(len(target),n)).tocsr()
        Q=coo_matrix((np.ones(n),(qrow,np.arange(n))),shape=(len(donor),n)).tocsr()
        rhs_moment=-np.einsum('i,ij->j',before[target]/unit,(cg-old)*1000.)
        A=vstack([Q,(cg-offset).T*1000.@C]).tocsr();b=np.r_[np.zeros(len(donor)),rhs_moment]
        # Preserve exact source identity and allow transfers only on its already qualified routes.
        # Positive and negative changes have equal cost. A negative change cannot exceed its source credit.
        inside=np.isin(target,incident)
        high=v*(ceiling-5e-7);low=v*(floor+5e-7)
        high[~inside]=np.maximum(high[~inside],before[target[~inside]])
        low[~inside]=np.minimum(low[~inside],before[target[~inside]])
        ub=np.r_[(high-before[target])/unit,(before[target]-low)/unit]
        cons=vstack([C,-C]);u=np.r_[np.full(n,np.inf),weight[edges]/unit]
        r=linprog(np.ones(2*n),A_ub=hstack((cons,-cons)),b_ub=ub,A_eq=hstack((A,-A)),b_eq=b,bounds=list(zip(np.zeros(2*n),u)),method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
        history.append(dict(ring=ring,cells=len(target),carriers=len(donor),edges=n,result=r.message));print(history[-1],flush=True)
        if r.success:
            new=weight.copy();new[edges]+=(r.x[:n]-r.x[n:])*unit
            if min(new)<-1e-20:raise ValueError('Negative source credit')
            new=np.maximum(new,0);after=np.bincount(c,weights=new,minlength=len(t))
            errors=np.bincount(q,weights=new)-np.bincount(q,weights=weight);density=after[target]/v
            residual=np.einsum('i,ij->j',after[target]-before[target],cg-offset)+np.einsum('i,ij->j',before[target],cg-old)
            if (np.any(density<floor-1e-7) or np.any(density>ceiling+1e-7) or max(abs(errors))>1e-16 or max(abs(residual))>1e-14):raise ValueError('Material mass certificate failed')
            delta=np.flatnonzero(new!=weight);cc=np.flatnonzero(before!=after);np.savez_compressed(out/'repaired-credits.npz',edge=delta,mass=new[delta],role_cells=cc,cell_mass=after[cc])
            result=dict(trials=history,geometry=digest(field/'candidate.npz'),source_flow=digest(path),source_body_mass=cellmass[roles].sum(),affected_cells=len(incident),changed_cells=len(cc),changed_credits=len(delta),mass_reassigned_kg=float(.5*abs(after-before).sum()),maximum_carrier_error_kg=float(abs(errors).max()),first_moment_error_kg_m=residual.tolist(),density_range_kg_m3=[density.min(),density.max()],updated=digest(out/'repaired-credits.npz'))
            (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True);return
        current=np.flatnonzero(np.isin(t,np.unique(t[current])).any(1))
    (out/'unresolved.json').write_text(json.dumps(history,indent=2)+'\n');raise ValueError('Insufficient regional donor support')
if __name__=='__main__':run(*(Path(a).resolve() for a in sys.argv[1:]))
