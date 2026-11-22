"""Reparameterize interior reference nodes and test the original mass groups and weak forces."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize
from audit_cavity_response import elastic
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha
from torso_cell_geometry import geom


def project_mass(states, changed, t, mass, groups):
    # Preserve each original source allocation group and the first mass
    # moment at all three recorded placements. Scale coordinates about a local
    # origin so conversion from metres to millimetres cannot change admission.
    group_ids,inv=np.unique(groups,return_inverse=True)
    identity=np.eye(len(group_ids))[inv].T
    old=[identity]
    new=[identity]
    for q,p in zip(states,changed):
        u=q[t].mean(axis=1)
        v=p[t].mean(axis=1)
        origin=u.mean(0)
        scale=np.linalg.norm(u-origin,axis=1).max()
        old.append(((u-origin)/scale).T)
        new.append(((v-origin)/scale).T)
    total=mass.sum()
    A=np.vstack(new); b=np.vstack(old)@(mass/total)
    vol=geom(changed[0],t)[0]
    lo,hi=850*vol/total,950*vol/total
    eye=np.eye(len(t))
    target=mass/total
    sol=linprog(np.r_[np.zeros(len(t)),np.ones(len(t))],A_ub=np.block([[eye,-eye],[-eye,-eye]]),
                b_ub=np.r_[target,-target],A_eq=np.c_[A,np.zeros_like(A)],b_eq=b,
                bounds=[*zip(lo,hi),*[(0,None)]*len(t)],method='highs',
                options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
    if not sol.success:return None
    m=sol.x[:len(t)]*total
    if np.max(abs(A@(m/total)-b))>2e-10:return None
    return m


def analyze(native,audit,proposal,ledger,owner,out):
    x,y,t,tag,pins=inputs(native)
    j=json.loads(audit.read_text())
    if pins!=j['source'] or sha(proposal)!=j['proposal_sha256']:
        raise ValueError('Unmatched physical registration')
    er=native/'physical-accepted.node'
    if sha(er)!=j['matched_motion']['physical_nodes_sha256']:raise ValueError('Wrong motion')
    z=np.loadtxt(er,skiprows=1,usecols=(1,2,3))
    flow=ledger/'current-density-closed-flow.npz'
    if sha(flow)!=j['mass_sha256']:raise ValueError('Wrong mass')
    with np.load(flow) as f:m=f['element_mass_kg'];groups=f['group_for_cell']
    with np.load(proposal) as c: kept=c['retained_input_cell_rows'];new=c['replacement_tetrahedra']
    witnesses=j['matched_motion']['quality_witnesses']
    original=np.unique([n for w in witnesses for n in w['nodes']])
    roles={}
    env=json.loads((native/'launch.json').read_text())['environment']
    mp=Path(env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_CSV'])
    if sha(mp)!=j['map_sha256']:raise ValueError('Node map changed')
    ids=set(original.tolist())
    with mp.open() as stream:
        for r in csv.DictReader(stream):
            n=int(r['joint_node_id'])-1
            if n in ids:roles[n]=r['registered_surface_layer']
    results=[]
    mu,K=materials(owner)[0][1]
    for w in witnesses:
        row=w['source_row']
        if row is None or np.any(t[row]!=w['nodes']):raise ValueError('Unexpected cavity ancestry')
        candidates=[n for n in t[row] if roles[n]=='']
        if len(candidates)!=1:
            results.append(dict(source_row=row,status='no unique interior node'))
            continue
        n=candidates[0]
        active=np.flatnonzero((t==n).any(1))
        if np.any(tag[active]!=16) or not np.isin(active,kept).all() or (new==n).any():
            raise ValueError('Candidate is not wholly in its source fat star')
        nodes=np.unique(t[active])
        local=np.searchsorted(nodes,t[active])
        root=np.searchsorted(nodes,n)
        central=int(np.flatnonzero(active==row)[0])
        states=[q[nodes].astype(float) for q in (x,y,z)]
        allowed=np.flatnonzero(np.isin(groups,np.unique(groups[active]))&np.isin(np.arange(len(t)),kept)&(tag==16))
        # Retain contiguous material from the same allocation groups. The extra
        # carriers keep their original geometry and constitutive energy.
        halo=allowed[np.isin(t[allowed],nodes).any(1)]
        halo_nodes=np.unique(t[halo])
        hc=np.searchsorted(halo_nodes,t[halo])
        hs=[q[halo_nodes].astype(float) for q in (x,y,z)]
        hi=np.searchsorted(halo_nodes,n)
        g0,gi=np.unique(groups[active],return_inverse=True)
        refvol=geom(states[0],local)[0]
        density=np.bincount(gi,weights=m[active])/np.bincount(gi,weights=refvol)
        at_bound=(density<850.0001)|(density>949.9999)
        base_force=[elastic(states[0],s,local,np.arange(len(nodes)),mu,K)[:2] for s in states[1:]]
        best=None
        geometric=conservative=0
        for tet in local:
            neighbors=tet[tet!=root]
            def place(s,b):
                q=s.copy();q[root]+=b@(s[neighbors]-s[root])
                return q
            def validity(b):
                res=[]
                for s in states:
                    v,quality=geom(place(s,b),local)
                    qfloor=np.minimum(1.02e-5,geom(s,local)[1])
                    qfloor[central]=1.02e-5
                    res.extend((quality/qfloor-1,v/np.maximum(1e-18,.90*geom(s,local)[0])-1))
                return np.concatenate(res)
            b0=np.full(3,1e-5)
            cons=[dict(type='ineq',fun=validity),dict(type='ineq',fun=lambda b:.4-b.sum())]
            if at_bound.any():
                D=np.stack([np.bincount(gi,weights=geom(place(states[0],unit),local)[0]-refvol)[at_bound]
                            for unit in np.eye(3)],axis=1)
                scale=np.max(refvol)
                _,sv,vh=np.linalg.svd(D/scale,full_matrices=False)
                C=vh[sv>1e-9]
                if len(C):cons.append(dict(type='eq',fun=lambda b:C@b,jac=lambda b:C))
            opt=minimize(lambda b:(b@b,2*b),b0,jac=True,method='SLSQP',bounds=[(0,.35)]*3,
                         constraints=cons,
                         options=dict(maxiter=35,ftol=1e-11))
            if not opt.success:continue
            b=opt.x
            changed=[place(s,b) for s in states]
            for s in changed[:2]:s[root]=s[root].astype('f4')
            qs=[geom(s,local) for s in changed]
            if any(v.min()<=1e-18 or q.min()<1e-5 or q[central]<1.015e-5 for v,q in qs):continue
            geometric+=1
            he=[q.copy() for q in hs]
            for q,s in zip(he,changed):q[hi]=s[root]
            mc=project_mass(hs,he,hc,m[halo],groups[halo])
            if mc is None:continue
            conservative+=1
            force_errors=[];energy_errors=[]
            for (E,f),s in zip(base_force,changed[1:]):
                e,g,_=elastic(changed[0],s,local,np.arange(len(nodes)),mu,K)
                # Compare virtual work in the original material coordinate space.
                moved_force=g[root].copy()
                g[root]*=1-b.sum()
                g[neighbors]+=b[:,None]*moved_force
                force_errors.append(float(np.linalg.norm(g-f)/max(np.linalg.norm(f),1e-30)))
                energy_errors.append(float(abs(e-E)/max(abs(E),1e-30)))
            score=max(max(force_errors)/.01,max(energy_errors)/.05)
            if best is None or score<best['gate_ratio']:
                best=dict(gate_ratio=score,force_relative=force_errors,energy_relative=energy_errors,
                          reference_position=changed[0][root].tolist(),installed_position=changed[1][root].tolist(),
                          interpolation_nodes=nodes[neighbors].tolist(),interpolation_weights=b.tolist(),
                          densities_kg_m3=[float(np.min(mc/geom(he[0],hc)[0])),float(np.max(mc/geom(he[0],hc)[0]))],
                          new_mass_kg=mc.tolist(),mass_rows=halo.tolist(),source_star_rows=active.tolist(),witness_shape=[float(q[central]) for v,q in qs],
                          maximum_group_mass_drift_kg=float(np.max(abs(np.bincount(groups[halo],weights=mc-m[halo])))))
        results.append(dict(source_row=row,interior_node=int(n),incident_cells=len(active),geometric_candidates=geometric,
                            conservative_candidates=conservative,best=best))
    out.write_text(json.dumps(dict(source=pins,physical_nodes_sha256=sha(er),results=results,
        scope='Reference and installed geometry plus an embedded replay. An accepted candidate still requires a rebuilt physical mass ledger and native qualification.'),indent=2,allow_nan=False)+'\n')

if __name__=='__main__':analyze(*(Path(arg).resolve() for arg in sys.argv[1:]))
