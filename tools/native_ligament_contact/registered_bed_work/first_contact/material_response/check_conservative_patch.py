"""Test local redistribution on previously certified carrier-to-cell edges, keeping each source supply."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack, hstack
from fit_dorsal_material import inputs
from registered_density import density_bounds
from register_material_surfaces import sha
from supported_material_admission import geom


def check(trial,mesh,proposal,flows,prior,out):
    ref,_,cells,roles,pins=inputs(trial)
    ref=ref.astype(float);points=ref.copy()
    change=np.load(proposal/'changed-coordinates.npz')
    if sha(proposal/'changed-coordinates.npz')!=json.loads((proposal/'result.json').read_bytes())['patch_sha256']:
        raise ValueError('Wrong repair')
    nodes=change['ids'];points[nodes]=change['points']
    incident=np.flatnonzero(np.any(np.isin(cells,nodes),axis=1))
    mass=np.fromfile(mesh/'cell-mass.f64',dtype='<f8')
    info=json.loads((mesh/'receipt.json').read_bytes())
    if sha(mesh/'cell-mass.f64')!=info['cell_mass_sha256'] or sha(mesh/'mesh.ele')!=pins['elements_sha256']:
        raise ValueError('Unpinned material')
    owner=Path(json.loads((trial/'launch.json').read_bytes())['compiled_material_owner'])
    out.mkdir(exist_ok=True)
    report={}
    for mat,tag in [(15,'SKIN'),(16,'FAT')]:
        low,high=density_bounds(owner,mat)
        file=flows/(tag+'-physical-flow.npz')
        previous=json.loads((flows/'exact-audit.json').read_bytes())[tag]
        if not previous['verified'] or previous['flow_sha256']!=sha(file):
            raise ValueError('Unsigned carrier control')
        f=np.load(file); q,g,w=(f[k] for k in ('carrier','role_cell','mass'))
        role_rows=np.flatnonzero(roles==mat)
        changed=incident[roles[incident]==mat]
        if not len(changed):
            report[tag]=dict(changed_cells=0,fixed_mass_outside_density=0,passed=True)
            continue
        vol=geom(points,cells[changed])[0]
        density=mass[changed]/vol
        bad=changed[(density<low)|(density>high)]
        payload=dict(changed_cells=len(changed),fixed_mass_outside_density=int(len(bad)),minimum_rho=float(density.min()),maximum_rho=float(density.max()))
        report[tag]=payload
        if not len(bad):
            payload['passed']=True
            continue
        fixed=np.load(prior/(tag+'-candidate-flow.npz'))
        exceptional=~fixed['eligible'];nq=int(q.max())+1
        protected_key=fixed['role_cell'][exceptional].astype('i8')*nq+fixed['carrier'][exceptional]
        allowed=~np.isin(g.astype('i8')*nq+q,protected_key)
        source=np.unique(q[allowed & np.isin(g,np.searchsorted(role_rows,changed))])
        sel=np.flatnonzero(allowed & np.isin(q,source))
        qi,iq=np.unique(q[sel],return_inverse=True);gi,ig=np.unique(g[sel],return_inverse=True)
        t=role_rows[gi]
        volume=geom(points,cells[t])[0]
        original=np.bincount(ig,weights=w[sel])
        reserved=mass[t]-original
        quota=np.bincount(iq,weights=w[sel])
        L=np.maximum(0,low*volume-reserved)
        H=high*volume-reserved
        payload.update(local_carriers=len(qi),candidate_receivers=len(t),adjustable_positive_routes=len(sel))
        if np.any(H<0) or np.setdiff1d(bad,t).size:
            payload.update(passed=False,reason='Existing fixed contributions exceed a receiver capacity or leave it unfunded');continue
        # Scale every equation to its mass; this resolves small sliver cells without
        # giving the LP an absolute tolerance comparable with a cell's entire mass.
        scale=np.minimum(quota[iq],np.maximum(original,np.maximum(L,H))[ig])
        k=np.arange(len(sel))
        supply=coo_matrix((scale/quota[iq],(iq,k)),shape=(len(qi),len(k))).tocsr()
        denom=np.maximum(H,original)
        capacity=coo_matrix((scale/denom[ig],(ig,k)),shape=(len(gi),len(k))).tocsr()
        # Solve in transported increments and keep a density margin on any moved
        # recipient. Untouched original routes need no numerical reconstruction.
        high_pad=np.maximum(np.minimum((H-L)*2e-6,H-original),0)
        low_pad=np.maximum(np.minimum((H-L)*2e-6,original-L),0)
        cost_scale=np.maximum(scale/scale.max(),1e-8)
        eq=hstack((supply,-supply))
        cap=hstack((capacity,-capacity))
        center=points[cells[t]].mean(axis=1)
        ref_center=ref[cells[t]].mean(axis=1)
        # An equal and opposite change on any carrier conserves mass. Apply
        # three additional rows for preservation of the physical first moment.
        lever=center-center.mean(axis=0)
        moment=(lever[ig]*scale[:,None]).T
        span=np.max(abs(moment),axis=1)
        moment/=span[:,None]
        eq=vstack((eq,coo_matrix(np.hstack((moment,-moment)))))
        rhs=np.r_[np.zeros(len(qi)),(mass[t]@(ref_center-center))/span]
        fit=linprog(np.r_[cost_scale,cost_scale],A_eq=eq,b_eq=rhs,A_ub=vstack((cap,-cap)),
                    b_ub=np.r_[(H-high_pad-original)/denom,(original-L-low_pad)/denom],
                    bounds=[(0,None)]*len(k)+list(zip(np.zeros(len(k)),w[sel]/scale)),method='highs',
                    options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
        payload['solver']=fit.message
        if not fit.success:
            payload['passed']=False;continue
        new=w[sel]+(fit.x[:len(k)]-fit.x[len(k):])*scale
        if np.min(new)<-1e-18:raise ValueError('Negative transfer')
        new=np.maximum(new,0.)
        # Close each donor exactly on a large retained transfer into a receiver
        # with available density headroom, then certify both ledgers again.
        rr=np.bincount(ig,weights=new)
        for src,resid in enumerate(quota-np.bincount(iq,weights=new)):
            if not resid:continue
            ix=np.flatnonzero(iq==src)
            capacity_left=(H-rr)[ig[ix]] if resid>0 else np.minimum(new[ix],(rr-L)[ig[ix]])
            ie=ix[int(np.argmax(capacity_left))]
            if capacity_left.max()+1e-23>=abs(resid):
                new[ie]+=resid;rr[ig[ie]]+=resid
        received=np.bincount(ig,weights=new)+reserved
        density_new=received/volume;supply_error=float(np.max(abs(np.bincount(iq,weights=new)/quota-1)))
        ok=not np.any(new<0) and supply_error<=1e-9 and np.all(density_new>=low-1e-8) and np.all(density_new<=high+1e-8)
        delta=received-mass[t]
        moment_error=received@points[cells[t]].mean(axis=1)-mass[t]@ref[cells[t]].mean(axis=1)
        ok=ok and np.abs(moment_error).max()<1e-11
        payload.update(passed=bool(ok),source_max_relative_error=supply_error,
              mass_error_kg=float(delta.sum()),maximum_density_error_kg_m3=float(np.maximum.reduce([low-density_new,density_new-high,np.zeros(len(t))]).max()),
              local_first_moment_change_kg_m=(received@points[cells[t]].mean(axis=1)-mass[t]@ref[cells[t]].mean(axis=1)).tolist())
        np.savez_compressed(out/(tag+'-delta.npz'),original_edges=sel,new_kg=new,cells=t,mass_kg=received,base_flow_sha256=sha(file))
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':check(*(Path(arg).resolve() for arg in sys.argv[1:]))
