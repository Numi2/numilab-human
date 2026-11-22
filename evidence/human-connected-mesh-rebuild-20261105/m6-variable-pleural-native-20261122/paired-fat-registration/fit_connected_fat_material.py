"""Jointly fit local coordinates and original source-group mass on the Float32 grid."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from fit_dorsal_material import inputs
from material_context import sha
from fit_posterior_equilibrium import materials
import csv
import re
from torso_cell_geometry import geom
from reparameterize_fat_stars import project_mass

EDGES=list(itertools.combinations(range(4),2))

def gradient(points, cells, axis, variables, quantum):
    edges=points[cells[:,1:]]-points[cells[:,:1]]
    a,b,c=edges.transpose(1,0,2)
    grad=np.stack([np.cross(b,c)[:,axis],np.cross(c,a)[:,axis],np.cross(a,b)[:,axis]],axis=1)/6
    grad=np.column_stack((-grad.sum(1),grad))
    out=np.zeros((len(cells),len(variables)))
    for i,n in enumerate(variables):
        rr,cc=np.where(cells==n)
        out[rr,i]=grad[rr,cc]*quantum[i]
    return out


def response(x,y,t,mu,lam):
    # Displacement gradient gives exactly zero strain for a translation.
    R=x[t[:,1:]]-x[t[:,:1]]
    u=y-x
    U=u[t[:,1:]]-u[t[:,:1]]
    D=np.linalg.solve(R,U)
    F=D+np.eye(3);vol=np.linalg.det(R)/6
    lj=np.linalg.slogdet(F)[1]
    FinvT=np.linalg.inv(F).transpose(0,2,1)
    # F - F^-T = (D + D^T + D D^T) F^-T.
    stress=mu*(D+D.transpose(0,2,1)+D@D.transpose(0,2,1))@FinvT+lam*lj[:,None,None]*FinvT
    gradient=np.linalg.solve(R.transpose(0,2,1),stress)*vol[:,None,None]
    force=np.zeros_like(x);np.add.at(force,t[:,0],gradient.sum(1))
    for i in range(3):np.add.at(force,t[:,i+1],-gradient[:,i])
    energy=vol@(.5*mu*(np.sum(D*D,axis=(1,2))+2*(np.trace(D,axis1=1,axis2=2)-lj))+.5*lam*lj**2)
    return energy,force

def compare(states,changes,t,properties):
    measured=[]
    for j in (1,2):
        E,F=response(states[0],states[j],t,*properties)
        e,f=response(changes[0],changes[j],t,*properties)
        measured.append(dict(baseline_force_N=float(np.linalg.norm(F)),baseline_energy_J=E,
                             force_error_N=float(np.linalg.norm(F-f)),energy_error_J=abs(E-e)))
    return measured

def run_patch(states,t,wrow,free,mass,groups,axis,reference,reserve,mu,K):
    # Unknowns are nodal coordinate quanta and fractions of original element mass.
    # solve with unchanged carrier/source allocations, retaining both density limits.
    n=len(free);nc=len(t);N=n+nc
    quant=np.maximum(abs(np.spacing(states[0][free,axis].astype('f4'))),abs(np.spacing(states[1][free,axis].astype('f4')))).astype(float)
    vol=[geom(s,t)[0] for s in states]
    G=[gradient(s,t,axis,free,quant) if reference or j else np.zeros((nc,n)) for j,s in enumerate(states)]
    identity=np.eye(nc);zero=np.zeros_like(identity)
    laws=[];limits=[]
    def add(a,b):
        a=np.atleast_2d(a);b=np.broadcast_to(b,(len(a),))
        scale=np.max(abs(a),axis=1);keep=scale>0
        if np.any((~keep)&(b < -1e-19)):raise ValueError('Fixed condition is invalid')
        laws.append(a[keep]/scale[keep,None]);limits.extend(b[keep]/scale[keep])
    def ge(a,b):add(np.c_[-a,zero],b)
    add(np.c_[850*G[0],-identity*mass],mass-850*vol[0])
    add(np.c_[-950*G[0],identity*mass],950*vol[0]-mass)
    for phase in (1,2):
        lo=.4 if phase==1 else .35
        ge(G[phase]-lo*G[0],vol[phase]-lo*vol[0])
        ge(2.5*G[0]-G[phase],2.5*vol[0]-vol[phase])
    gid,inv=np.unique(groups,return_inverse=True)
    gs=np.eye(len(gid))[inv].T*mass
    eq=[np.c_[np.zeros((len(gid),n)),gs]];eq_rhs=[np.zeros(len(gid))]
    nodeWeights=(t[:,:,None]==free).sum(1)*.25
    fixed_moment=mass@nodeWeights*quant
    # Preserve first moments around the patch. The quadratic term is measured
    # after emission and the fully nonlinear mass equations are re-solved.
    for j,s in enumerate(states):
        c=s[t].mean(1);c-=c.mean(0)
        for k in range(3):
            disp=fixed_moment if k==axis and (reference or j) else np.zeros(n)
            row=np.r_[disp,c[:,k]*mass]
            add(row,3e-14);add(-row,3e-14)
    for j,s in enumerate(states):
        if not reference and j==0:continue
        ge(G[j],vol[j]-1.000001e-18)
        q=np.full(nc,1.0000001e-5);q[wrow]=reserve
        for a,b in EDGES:
            edge=s[t[:,b]]-s[t[:,a]]
            length=np.linalg.norm(edge,axis=1)
            diff=((t[:,b,None]==free).astype(float)-(t[:,a,None]==free))*quant
            ge(G[j]-.5*(q*length*edge[:,axis])[:,None]*diff,vol[j]-q*length**3/6)
    Ae=np.vstack(eq);Be=np.concatenate(eq_rhs)
    Ae/=np.max(np.abs(Ae),axis=1,keepdims=True)
    I=np.eye(n)
    ub=np.vstack((np.c_[np.vstack(laws),np.zeros((len(limits),n))],np.c_[I,np.zeros((n,nc)),-I],np.c_[-I,np.zeros((n,nc)),-I]))
    rhs=np.r_[limits,np.zeros(2*n)]
    eq=np.c_[Ae,np.zeros((len(Ae),n))]
    cap=np.minimum(32000,np.floor(.0004/quant))
    bound=np.r_[cap,np.full(nc,.49),cap]
    lb=-bound.copy();lb[N:]=0
    cost=np.r_[np.zeros(N),quant/quant.min()]
    sol=linprog(cost,A_ub=ub,b_ub=rhs,A_eq=eq,b_eq=Be,bounds=np.c_[lb,bound],method='highs')
    if not sol.success:return dict(status='infeasible')
    center=sol.x[:n]
    candidates=[np.rint(center)]
    for radius in (1,4):
        l=lb.copy();u=bound.copy()
        l[:n]=np.maximum(l[:n],np.floor(center)-radius);u[:n]=np.minimum(u[:n],np.ceil(center)+radius)
        opt=milp(cost,integrality=np.r_[np.ones(n),np.zeros(nc+n)],bounds=Bounds(l,u),
                 constraints=[LinearConstraint(ub,-np.inf,rhs),LinearConstraint(eq,Be,Be)],options=dict(time_limit=2.))
        if opt.x is not None:candidates.append(np.rint(opt.x[:n]));break
    errors=[]
    for bins in candidates:
        changes=[s.copy() for s in states];move=bins*quant
        for j,s in enumerate(changes):
            if j or reference:s[free,axis]+=move
            if j<2:s[free,axis]=s[free,axis].astype('f4')
        vq=[geom(s,t) for s in changes]
        qs=[float(q[wrow]) for v,q in vq]
        if min(q.min() for v,q in vq)<1e-5 or min(v.min() for v,q in vq)<=1e-18:
            errors.append('shape');continue
        if min(vq[1][0]/vq[0][0])<.4 or min(vq[2][0]/vq[0][0])<.35 or max(max(v/vq[0][0]) for v,q in vq[1:])>2.5:
            errors.append('material');continue
        # Screen the linearized mass proposal at the emitted coordinates. The
        # native volume constructor and registered ledger perform the actual allocation.
        m=project_mass(states,changes,t,mass,groups)
        if m is None:
            errors.append('emitted mass');continue
        error=compare(states,changes,t,(mu,K))
        ok=all(r['force_error_N']<=.01*r['baseline_force_N'] and r['energy_error_J']<=.05*abs(r['baseline_energy_J']) for r in error)
        return dict(status='eligible geometry' if ok else 'force or energy',axis=axis,reference_shift=reference,shift_m=move.tolist(),
                    target_shapes=qs,mechanical_checks=error,peak_m=float(max(abs(move))))
    return dict(status='quantized constraints',failures=errors)

def main(native,audit,proposal,ledger,owner,out):
    x,y,t,tag,pins=inputs(native);j=json.loads(audit.read_text())
    er=native/'physical-accepted.node';flow=ledger/'current-density-closed-flow.npz'
    env=json.loads((native/'launch.json').read_text())['environment']
    mp=Path(env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_CSV'])
    if pins!=j['source'] or any(sha(p)!=pin for p,pin in ((proposal,j['proposal_sha256']),(er,j['matched_motion']['physical_nodes_sha256']),
           (flow,j['mass_sha256']),(mp,j['map_sha256']))):raise ValueError('Unmatched native sources')
    z=np.loadtxt(er,skiprows=1,usecols=(1,2,3))
    with np.load(flow) as f:m=f['element_mass_kg'];groups=f['group_for_cell']
    with np.load(proposal) as f:keep=f['retained_input_cell_rows'];rep=f['replacement_tetrahedra']
    witnesses=j['matched_motion']['quality_witnesses']
    protected=np.unique(np.r_[t[np.setdiff1d(np.arange(len(t)),keep)].ravel(),rep.ravel()])
    header=(owner/'NumiHumanLoadedSkinPoseData.inc').read_text()
    from re import search,findall,S
    anchors=[int(a) for a in findall(r'\{\s*\d+u?\s*,\s*(\d+)u?\s*,',search(r'constexpr Anchor anchors\[\]=\{(.*?)\};',header,S)[1])]
    if len(anchors)!=14:raise ValueError('Unmatched physical anchors')
    protected=np.union1d(protected,anchors)
    roles=np.full(len(x),'?',dtype='U1')
    with mp.open() as stream:
        for line in csv.DictReader(stream):
            node=int(line['joint_node_id'])-1
            role=line['registered_surface_layer'] or 'V'
            if roles[node] not in ('?',role):raise ValueError('Conflicting layer labels')
            roles[node]=role
    mu,K=materials(owner)[0][1]
    report=dict(source=pins,accepted_sha256=sha(er),mass_sha256=sha(flow),proposal_sha256=sha(proposal),results=[])
    for w in witnesses:
        row=w['source_row']
        one=np.flatnonzero(np.isin(t,t[row]).any(1))
        available=np.setdiff1d(np.unique(t[one]),protected)
        free=available[np.isin(roles[available],['D','V'])]
        cells=np.flatnonzero(np.isin(t,free).any(1));nodes=np.unique(t[cells]);local=np.searchsorted(nodes,t[cells])
        if not np.all(tag[cells]==16) or not np.isin(cells,keep).all():raise ValueError('Patch is not independent intact adipose')
        part=dict(source_row=int(row),nodes=free.tolist(),incident_cells=len(cells),trials=[])
        states=[s[nodes].astype(float) for s in (x,y,z)]
        for reference in (False,True):
            for axis in (0,1,2):
                for goal in (1.02e-5,1.005e-5):
                    v=run_patch(states,local,int(np.flatnonzero(cells==row)[0]),np.searchsorted(nodes,free),m[cells],groups[cells],axis,reference,goal,mu,K)
                    v.update(axis=axis,reference_shift=reference,target=goal)
                    part['trials'].append(v)
                    print(row,axis,reference,goal,v['status'],v.get('peak_m'),flush=True)
                    if v['status']=='eligible geometry':break
        report['results'].append(part);out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main(*(Path(a).resolve() for a in sys.argv[1:]))
