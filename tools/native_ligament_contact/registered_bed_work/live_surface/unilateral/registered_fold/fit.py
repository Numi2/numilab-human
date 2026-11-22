"""Fit the measured skin contact at source-coordinate precision, conserving material mass moments."""
import argparse
import json
from pathlib import Path
import sys
from select_coupled_patch import choose

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from fit_connected_fat_material import response
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha
from registered_density import density_bounds
from route_local_mass import route
from supported_material_admission import installed_topology, material_state, stress_limits
from torso_cell_geometry import geom


def read(p):
    return json.loads(p.read_text())


def fit(case, ledger, source, output, routed=False, connected=False, joint=False,replay=None):
    trial, accepted = case/'limiting-row', case/'two'
    witness = read(source)
    raw = trial/'stderr.log'
    if sha(raw) != witness['capture']:
        raise ValueError('Contact belongs to a different solve')
    record, = [json.loads(line.partition('=')[2]) for line in raw.read_text().splitlines()
               if line.startswith('fem_deformable_limit_trace=')]
    contact, = record['contacts']
    if contact['nodes'][:4] != witness['native_fem_nodes'] or any(int(s['donor_body'])!=int(witness['sources'][0]['donor_body']) for s in witness['sources']):
        raise ValueError('Feature spans different registrations')
    feat = np.array(witness['registered_nodes'])
    X, Y, raw_tets, role, pins = inputs(accepted)
    X, Y = X.astype(float), Y.astype(float)
    end = accepted/'physical-accepted.node'
    if sha(end) != read(accepted/'verified-endpoint.json')['physical_nodes_sha256']:
        raise ValueError('Endpoint identity mismatch')
    Z = np.loadtxt(end,skiprows=1,usecols=(1,2,3))
    env = read(accepted/'launch.json')['environment']
    t, role = installed_topology(raw_tets, role, env)
    # Solve on the four measured vertices only. Neighbours of the complete
    # tetrahedral stars participate in every check, with zero motion.
    feature_stars=np.flatnonzero(np.isin(t,feat).any(1))
    moving=np.unique(t[feature_stars]) if connected else feat.copy()
    stars=np.flatnonzero(np.isin(t,moving).any(1))
    if routed and not connected:
        stars=np.flatnonzero(np.isin(t,np.unique(t[stars])).any(1))
    ids=np.unique(t[stars]);local=np.searchsorted(ids,t[stars])
    center=np.searchsorted(ids,feat);variables=np.searchsorted(ids,moving);nvar=len(variables)
    X,Y,Z = (p[ids] for p in (X,Y,Z))
    role = role[stars]
    path = ledger/'current-density-closed-flow.npz'
    if sha(path) != read(ledger/'mass-transfer.json')['flow_sha256']:
        raise ValueError('Unmatched source mass flow')
    with np.load(env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ']) as remap:
        keep = remap['retained_input_cell_rows']
    if stars.max() >= len(keep):
        raise ValueError('Feature has remeshed mass ancestry')
    with np.load(path) as flow:
        m = flow['element_mass_kg'][keep[stars]]
        group = flow['group_for_cell'][keep[stars]]
    props, hashes = materials(case/'owner')
    limits = {r:density_bounds(case/'owner', r) for r in np.unique(role)}
    cap = stress_limits(hashes)
    def state(q):
        pt,a,b,c = q[center]
        normal = np.cross(b-a,c-a)
        normal /= np.linalg.norm(normal)
        return float(np.dot(pt-a, normal)), normal
    start,n = state(X)
    def energy_force(x,y):
        energy = 0.; force = np.zeros_like(x)
        for r in np.unique(role):
            e,f = response(x,y,local[role==r],*props[r-15])
            energy += e; force += f
        return energy,force
    base = [energy_force(X,q) for q in (Y,Z)]
    normal = np.sign(start)*n
    flags = np.stack([(variables[:,None]==tet).sum(1) for tet in local])
    candidates=[]
    trials=[]
    # All coordinates are emitted Float32 in reference and load. One-axis
    # moves make per-cell volumes exactly affine and preserve first moments
    # whenever the donor-weighted nodal displacements sum to zero.
    for axis in np.argsort(-abs(normal)):
        quantum = np.maximum(abs(np.spacing(X[variables,axis].astype('f4'))),abs(np.spacing(Y[variables,axis].astype('f4')))).astype(float)
        if np.any(quantum==0):
            raise ValueError('Missing coordinate scale')
        G=np.zeros((len(local),nvar)); v0=geom(X,local)[0]
        for k,node in enumerate(variables):
            y1=X.copy();y1[node,axis]+=quantum[k]
            G[:,k]=geom(y1,local)[0]-v0
        lower,upper=np.array([limits[m] for m in role]).T
        if (m/v0 < lower-1e-7).any() or (m/v0 > upper+1e-7).any():
            raise ValueError('Original local density outside the material law')
        rows=[];lo=[];hi=[]
        def constrain(mat,a,b):
            A=np.atleast_2d(mat)
            scale=np.max(abs(A),axis=1)
            low,high=np.broadcast_to(a,(len(A),)),np.broadcast_to(b,(len(A),))
            if np.any((low[scale==0]>0)|(high[scale==0]<0)):
                raise ValueError('Unsatisfied invariant without movable node')
            use=scale>0
            rows.extend(A[use]/scale[use,None]);lo.extend(low[use]/scale[use]);hi.extend(high[use]/scale[use])
        if routed:
            for q in (X,Y,Z):
                volume,quality=geom(q,local)
                jac=np.empty_like(G)
                for k,node in enumerate(variables):
                    test=q.copy();test[node,axis]+=quantum[k]
                    jac[:,k]=geom(test,local)[0]-volume
                edge=np.linalg.norm(q[local[:,[1,2,3,2,3,3]]]-q[local[:,[0,0,0,1,1,2]]],axis=2).max(1)
                threshold=np.maximum(1e-18,edge**3*1e-5/6)
                constrain(jac,threshold-volume,np.inf)
        else:
            constrain(G,np.maximum(1.000001e-18,m/upper)-v0,m/lower-v0)
            for g in np.unique(group):
                moment=((m*(group==g))@flags)*.25*quantum
                constrain(moment,-1e-14,1e-14)
        if connected:
            # Admit motion only if the complete potentially changed part of
            # every source group can carry its original mass inside the law.
            for g in np.unique(group):
                sel=group==g
                constrain(lower[sel]@G[sel],-np.inf,max(0.,m[sel].sum()-lower[sel]@v0[sel]))
                constrain(upper[sel]@G[sel],min(0.,m[sel].sum()-upper[sel]@v0[sel]),np.inf)
        # Source-space normal separation on all four feature nodes.
        deriv=[]
        for k,node in enumerate(variables):
            p=X.copy();p[node,axis]+=quantum[k]
            deriv.append(np.sign(start)*(state(p)[0]-start))
        # A feature whose original gap is smaller than a single representable
        # source displacement needs clearance distinguishable in that source.
        resolved=max(contact['barrier_parameters'][2]*2,2*sum(abs(normal)*abs(np.spacing(X[center].astype('f4'))).max(axis=0)))
        constrain(deriv,resolved-abs(start),np.inf)
        rows=np.asarray(rows)
        A=np.vstack([np.c_[rows,np.zeros_like(rows)],np.c_[np.eye(nvar),-np.eye(nvar)],np.c_[-np.eye(nvar),-np.eye(nvar)]])
        obj=np.r_[np.zeros(nvar),quantum/quantum.max()]
        if replay is not None:
            from types import SimpleNamespace
            saved=read(replay)
            if pins!=saved['source'] or moving.tolist()!=saved['node_rows'] or sha(source)!=saved['feature_sha256']:
                raise ValueError('Incompatible proposal')
            trial=next(u for u in saved['trials'] if u['axis']==axis)
            sol=SimpleNamespace(x=np.rint(np.array(trial['move_m'])/quantum) if 'move_m' in trial else None)
        elif joint:
            sol=choose(rows,lo,hi,(X,Y,Z),local,group,m,v0,G,quantum,flags,axis,lower,upper)
        else:
            sol=milp(obj,integrality=[1]*nvar+[0]*nvar,bounds=Bounds([-48]*nvar+[0]*nvar,[48]*(2*nvar)),
                     constraints=LinearConstraint(A,np.r_[lo,[-np.inf]*(2*nvar)],np.r_[hi,[0]*(2*nvar)]),options={'time_limit':3})
        if sol.x is None:
            trials.append(dict(axis=int(axis),reason='infeasible moment, density or separation constraints'))
            continue
        move=np.rint(sol.x[:nvar])*quantum
        dx=np.zeros_like(X);dx[variables,axis]=move
        xx=(X+dx).astype('f4').astype(float);yy=(Y+dx).astype('f4').astype(float);zz=Z+dx
        if not np.array_equal(xx-X,dx) or not np.array_equal(yy-Y,dx):
            raise ValueError('Displacement changed on Float32 emission')
        margins=[float(np.sign(start)*state(q)[0]) for q in (xx,yy,zz)]
        vols,shapes=zip(*(geom(q,local) for q in (xx,yy,zz)))
        valid_geometry=min(map(np.min,vols))>1e-18 and min(map(np.min,shapes))>=1e-5
        data=dict(axis=int(axis),move_m=move.tolist(),resolved_gap_m=float(resolved),measured_gaps_m=margins,
                  geometry=bool(valid_geometry),minimum_shape=float(min(map(np.min,shapes))))
        if not valid_geometry or min(margins)<resolved*.99:
            trials.append(dict(**data,reason='emitted geometry or separation'));continue
        forces=[];energy=[];phys=True
        for q,(e,f) in zip((yy,zz),base):
            en,fo=energy_force(xx,q)
            forces.append(float(np.linalg.norm(fo-f)/max(np.linalg.norm(f),1e-30)))
            energy.append(float(abs(en-e)/max(abs(e),1e-30)))
            _,s,p=material_state(xx[local],q[local],props[role-15])
            phys=phys and bool((s[:,0]<=1.5).all() and (s[:,-1]>=2/3).all() and (p<=cap[role-15]).all())
        data.update(force_errors=forces,energy_errors=energy,stretch_and_stress=phys)
        transferred=m.copy()
        if routed:
            for g in np.unique(group):
                sel=group==g
                if not np.any(dx[local[sel]]):
                    continue
                material,=np.unique(role[sel])
                res=route((X,Y,Z),(xx,yy,zz),local[sel],m[sel],limits[material])
                if res is None:
                    data['unroutable_group']=int(g)
                    orig=m[sel].sum(); v=geom(xx,local[sel])[0]
                    data['route_failure_details']=dict(cells=int(sel.sum()),source_mass_kg=orig,
                        minimum_mass_kg=limits[material][0]*v.sum(),maximum_mass_kg=limits[material][1]*v.sum(),
                        original_density_range=[float((m[sel]/v0[sel]).min()),float((m[sel]/v0[sel]).max())],
                        largest_move_m=float(abs(dx[local[sel]]).max()))
                    break
                transferred[sel]=res
            if 'unroutable_group' in data:
                trials.append(dict(**data,reason='no conservative density route'));continue
        errors=[]
        for g in np.unique(group):
            sel=group==g
            for old,new in zip((X,Y,Z),(xx,yy,zz)):
                a=old[local[sel]].mean(1);b=new[local[sel]].mean(1);origin=a.mean(0)
                errors.append(np.max(abs(transferred[sel]@(b-origin)-m[sel]@(a-origin))))
        data['max_donor_moment_change_kg_m']=float(max(errors))
        data['max_donor_mass_change_kg']=float(max(abs((transferred-m)[group==g].sum()) for g in np.unique(group)))
        ok=phys and max(forces)<=.01 and max(energy)<=.05 and data['max_donor_moment_change_kg_m']<=1.00001e-14
        trials.append(dict(**data,reason='passed local checks' if ok else 'mechanical fidelity or source moment'))
        if ok:
            candidates.append(data)
            if routed:
                np.savez_compressed(output.parent/f'routed-axis-{axis}.npz',cell_rows=keep[stars],mass_kg=transferred,original_mass_kg=m,node_rows=moving,displacement_m=dx[variables])
    report=dict(source=pins,feature_sha256=sha(source),node_rows=moving.tolist(),feature_nodes=feat.tolist(),source_gap_m=abs(start),
                affected_cells=len(feature_stars),carrier_region_cells=len(stars),allow_mass_routing=routed,connected=connected,joint=joint,source_cell_mass_kg=float(m.sum()),source_donor=int(witness['sources'][0]['donor_body']),
                trials=trials,selected=min(candidates,key=lambda r:np.linalg.norm(r['move_m'])) if candidates else None)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('case','ledger','source','output'):parser.add_argument(name,type=Path)
    parser.add_argument('--route-mass',action='store_true')
    parser.add_argument('--connected',action='store_true')
    parser.add_argument('--joint',action='store_true')
    parser.add_argument('--replay',type=Path)
    args=parser.parse_args()
    fit(args.case.resolve(),args.ledger.resolve(),args.source.resolve(),args.output.resolve(),
        routed=args.route_mass,connected=args.connected,joint=args.joint,replay=args.replay)
