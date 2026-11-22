"""Fit the smallest mass-weighted registration correction separating measured deep-fat facets."""
from pathlib import Path
import csv
import json
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha
from relax_bonded_interface import Patch
from torso_cell_geometry import geom


def inequalities(a,b):
    shared=set(a)&set(b)
    if len(shared)>1:
        raise ValueError('Expected transverse facets or a single shared vertex')
    if not shared:
        return [(u,v) for u in a for v in b]
    p,=shared
    return [(u,p) for u in a if u!=p]+[(p,v) for v in b if v!=p]


def plane(x,pairs):
    difference=np.stack([x[v]-x[u] for u,v in pairs])
    # The last coordinate is the common separation measured in micrometres.
    A=np.column_stack((-difference*1e6,np.ones(len(pairs))))
    result=linprog([0,0,0,-1],A_ub=A,b_ub=np.zeros(len(pairs)),bounds=[(-1,1)]*3+[(None,None)],method='highs')
    if not result.success or result.x[3]<=0:
        raise ValueError('Accepted control has no strict separating plane')
    return result.x[:3]/np.linalg.norm(result.x[:3])


def fit(native, control, proposal, ledger, owner, out):
    x,y,t,tag,pins=inputs(native)
    start=y.astype(float)
    env=json.loads((native/'launch.json').read_text())['environment']
    p='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    def source(k,h):
        v=Path(env[p+k])
        if sha(v)!=env[p+h]:
            raise ValueError('Changed source '+k)
        return v
    surface=source('SURFACE_D_FACES_CSV','SURFACE_D_FACES_SHA256')
    mapping=source('COMBINED_NODE_MAP_CSV','COMBINED_NODE_MAP_SHA256')
    trials=[]
    for run in (native,control):
        v=run/'physical-accepted.node'
        j=json.loads((run/'accepted-layers.json').read_text())
        if (sha(v)!=j['inputs']['NATIVE_ACCEPTED_NODE_SHA256'] or
            j['inputs']['NATIVE_LOADED_START_NODE_SHA256']!=pins['installed_pose_sha256']):
            raise ValueError('Unmatched physical layer witness')
        trials.append((np.loadtxt(v,skiprows=1,usecols=(1,2,3)),j))
    (actual,report),(clear,original)=trials
    if np.any(original['pair_counts']) or set(report['witnesses'])!={'2-2'}:
        raise ValueError('Expected crossing-free control and isolated deep-fat failure')
    witnesses=report['witnesses']['2-2']
    if len(witnesses)!=report['pair_counts'][2][2]:
        raise ValueError('Unresolved witness identity')
    triangles=np.loadtxt(surface,delimiter=',',skiprows=1,usecols=(1,2,3),dtype=int)
    ids=np.unique(triangles[np.unique(witnesses)])
    table={}
    donors={}
    selected=set(ids.tolist())
    with mapping.open() as f:
        for r in csv.DictReader(f):
            if r['registered_surface_layer']!='D':
                continue
            i=int(r['registered_surface_vertex_row'])
            if i in selected:
                n=int(r['joint_node_id'])-1
                if i in table or r['source_partition']!='fat':
                    raise ValueError('Ambiguous deep-fat owner')
                table[i]=n
                donors[n]=int(r['donor_body'])
    free=np.array(sorted(table.values()))
    if len(free)!=len(ids) or len(set(donors.values()))!=1:
        raise ValueError('Unsupported patch carrier')
    flow=ledger/'current-density-closed-flow.npz'
    if sha(flow)!=json.loads((ledger/'mass-transfer.json').read_text())['flow_sha256'] or sha(proposal)!=env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256']:
        raise ValueError('Unmatched reference mass')
    with np.load(flow) as m, np.load(proposal) as q:
        mass=m['nodal_mass_kg'][free]
        keep=q['retained_input_cell_rows']
        retired=np.setdiff1d(np.arange(len(t)),keep)
        if np.any(np.isin(t[retired],free)) or np.any(np.isin(q['replacement_tetrahedra'],free)):
            raise ValueError('Deep registration intersects an earlier mass transfer')
    cells=np.flatnonzero(np.isin(t,free).any(axis=1))
    if np.any(tag[cells]!=16):
        raise ValueError('Changed interface owner')
    nodes=np.unique(t[cells])
    relative=np.searchsorted(nodes,t[cells])
    patch=Patch(x[nodes].astype(float),relative,np.searchsorted(nodes,free),mass,tag[cells],materials(owner)[0])
    constraints=[]
    low=[]
    reserve=4*np.abs(np.spacing(start[free].astype('f4'))).max()+2*np.linalg.norm(actual[free]-clear[free],axis=1).max()
    for fa,fb in witnesses:
        a=[table[i] for i in triangles[fa]]
        b=[table[i] for i in triangles[fb]]
        pairs=inequalities(a,b)
        for pose in (start,actual):
            n=plane(start if pose is start else clear,pairs)
            for u,v in pairs:
                g=np.zeros((len(free),3))
                g[np.searchsorted(free,u)]=-n
                g[np.searchsorted(free,v)]=n
                constraints.append(g.flatten())
                low.append((reserve-n@(pose[v]-pose[u]))*1e6)
    A=np.array(constraints)
    # Micro-metre variables avoid rounding a sub-micron margin in the optimizer.
    w=mass/mass.mean()
    quantum=np.abs(np.spacing(start[free].astype('f4')).astype(float)).ravel()
    dims=len(quantum)
    step=quantum*1e6
    coordinate_rows=np.searchsorted(nodes,free)
    ix=np.full(len(nodes),-1)
    ix[coordinate_rows]=np.arange(len(free))
    indices=ix[relative]
    def material_rows(pose):
        local=pose[nodes]
        edges=local[relative[:,1:]]-local[relative[:,:1]]
        cof=np.stack((np.cross(edges[:,1],edges[:,2]),np.cross(edges[:,2],edges[:,0]),np.cross(edges[:,0],edges[:,1])),axis=1)
        gradients=np.concatenate((-cof.sum(axis=1,keepdims=True),cof),axis=1)/(6*patch.volume[:,None,None])
        base=(cof[:,0]*edges[:,0]).sum(1)/(6*patch.volume)
        jac=np.zeros((len(relative),len(free),3))
        for c in range(4):
            keep=indices[:,c]>=0
            jac[np.flatnonzero(keep),indices[keep,c]]+=gradients[keep,c]
        vol,quality=geom(local,relative)
        lower=np.maximum.reduce([np.full_like(base,.4),base*1.0001e-5/quality,base*1.0001e-18/vol])
        # Limit the first-order cell-volume change as well as preserving the
        # absolute validity floor. The exact geometry is checked after emission.
        return jac.reshape(len(base),-1)*quantum, np.maximum(lower,.97*base)-base, np.minimum(2.5,1.03*base)-base
    def expand(Q):return np.column_stack((Q,np.zeros_like(Q)))
    signed_constraints=[LinearConstraint(expand(A*step),low,np.inf)]
    for pose in (start,actual):
        Q,lo,hi=material_rows(pose)
        signed_constraints.append(LinearConstraint(expand(Q),lo,hi))
    # Minimise mass-weighted absolute movement on the actual Float32 lattice.
    # Each donor commonly has no exact integer solution for its real-valued
    # masses; report and bound the residual for the rounded coordinates.
    mass_moment=np.kron(mass[None,:],np.eye(3))*quantum
    signed_constraints.append(LinearConstraint(expand(mass_moment*1e12),-1.,1.))
    signed_constraints.append(LinearConstraint(np.block([[np.eye(dims),-np.eye(dims)],[-np.eye(dims),-np.eye(dims)]]),-np.inf,0.))
    cap=np.ceil(np.full(dims,2.)/step)
    result=milp(np.r_[np.zeros(dims),np.repeat(w,3)*step],integrality=np.r_[np.ones(dims),np.zeros(dims)],
                bounds=Bounds(np.r_[-cap,np.zeros(dims)],np.r_[cap,cap]),constraints=signed_constraints,
                options={'time_limit':12.})
    if result.x is None:
        raise ValueError('No grid registration at the required clearance and material limits: '+result.message)
    after=(start[free]+(np.rint(result.x[:dims])*quantum).reshape(-1,3)).astype('f4').astype(float)
    moved=after-start[free]
    signed=(A@moved.ravel()*1e6-np.asarray(low))*1e-6+reserve
    if signed.min()<=0:
        raise ValueError('Float32 emission lost strict separation')
    reg={}
    for label,s in [('installed',start),('replayed',actual)]:
        posed=s[nodes].copy()
        baseline=patch.stats(posed)
        posed[np.searchsorted(nodes,free)]+=moved
        measured=patch.stats(posed)
        if measured['bad_cells'] or measured['outside_material_bounds']:
            raise ValueError('New material quality failure')
        reg[label]=dict(before=baseline,after=measured)
    data=dict(source=pins,proposal_sha256=sha(proposal),map_sha256=sha(mapping),mass_sha256=sha(flow),node_ids=free.tolist(),
              emitted_coordinates=after.tolist(),source_endpoints=[sha(v/'physical-accepted.node') for v in (native,control)],
              deep_face_pairs=witnesses,max_move_m=float(np.linalg.norm(moved,axis=1).max()),
              min_emitted_gap_on_axes_m=float(signed.min()),target_gap_m=float(reserve),moment_change_kg_m=(mass@moved).tolist(),
              mechanical_checks=reg,adjoining_fat_cells=len(cells),donor=next(iter(donors.values())))
    out.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print({k:data[k] for k in ('max_move_m','min_emitted_gap_on_axes_m','moment_change_kg_m','adjoining_fat_cells')})

if __name__=='__main__':fit(*(Path(arg).resolve() for arg in sys.argv[1:]))
