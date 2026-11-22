"""Register a bonded dermis/adipose interface by the two materials' shared force balance."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import Bounds, LinearConstraint, milp, minimize
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha
from torso_cell_geometry import geom


def loaded_mass_moments(points, retired_cells, retired_mass, new_cells, new_mass):
    ids=np.union1d(retired_cells,new_cells)
    removed=np.searchsorted(ids,retired_cells)
    added=np.searchsorted(ids,new_cells)
    dm=np.zeros(len(ids))
    for c in range(4):
        np.add.at(dm,removed[:,c],-retired_mass/4)
        np.add.at(dm,added[:,c],new_mass/4)
    displacement=points[ids]-points[ids[0]]
    total=float(retired_mass.sum())
    radius=float(np.abs(displacement).max())
    signed=dm@displacement
    tol=1e-10*total*radius
    return dict(mass_drift_kg=float(dm.sum()),first_moment_kg_m=signed.tolist(),
                first_moment_limit_kg_m=tol,affected_nodes=len(ids),changed_mass_kg=total,
                passed=bool(abs(dm.sum())<=1.e-10*total and np.all(abs(signed)<=tol)))


def balance_rounded_moments(initial, rounded, dm):
    """Choose nearby binary32 points before certifying the loaded mass moments."""
    out=rounded.copy()
    # Six independently certified cavities share one native first-moment gate.
    # Compensate the arithmetic rounding error without altering any mass.
    weights=np.sum(dm,axis=0)
    for axis in range(3):
        drift=weights@(out[:,axis]-initial[:,axis])
        if abs(drift)<=5e-15:
            continue
        ulp=np.abs(np.spacing(out[:,axis].astype('f4')).astype(float))
        active=np.flatnonzero(abs(weights)*ulp>1e-19)
        if not len(active):
            raise ValueError('No Float32 carrier for the proposed moment change')
        w=weights[active]*ulp[active]*1e15
        n=len(w)
        # Minimize the sum of coordinate shifts measured in metres. Finite
        # registration and material checks are repeated on the emitted points.
        A=np.block([[np.eye(n),-np.eye(n)],[-np.eye(n),-np.eye(n)]])
        solution=milp(np.r_[np.zeros(n),ulp[active]/ulp[active].max()],
                      integrality=np.r_[np.ones(n),np.zeros(n)],
                      bounds=Bounds(np.r_[np.full(n,-16.),np.zeros(n)],np.full(2*n,16.)),
                      constraints=[LinearConstraint(np.r_[w,np.zeros(n)][None,:],-drift*1e15-4.,-drift*1e15+4.),
                                   LinearConstraint(A,-np.inf,0.)],options={'time_limit':4.})
        if solution.x is None:
            raise ValueError('No conservative Float32 rounding on the local lattice')
        out[active,axis]=(out[active,axis]+np.rint(solution.x[:n])*ulp[active]).astype('f4')
    if np.max(abs(weights@(out-initial)))>5e-15:
        raise ValueError('Rounded first moments failed closure')
    return out


class Patch:
    def __init__(self, x, tets, free, rho_mass, role, props):
        self.t = tets
        self.free = free
        self.mass = rho_mass
        self.role = role
        self.mu, self.bulk = props[role - 15].T
        edges = x[tets[:, 1:]] - x[tets[:, :1]]
        self.volume = np.linalg.det(edges) / 6
        d = np.linalg.inv(edges).transpose(0, 2, 1)
        self.G = np.concatenate((-d.sum(1, keepdims=True), d), axis=1)
        if np.min(self.volume) <= 0:
            raise ValueError("Inverted material reference")
        self.scatter = np.full(len(x), -1, dtype=int)
        self.scatter[free] = np.arange(len(free))

    def evaluate(self, y, hessian=False):
        F = self.G.transpose(0, 2, 1) @ (y[self.t] - y[self.t[:, :1]])
        J = np.linalg.det(F)
        if J.min() <= 0:
            return np.inf, None, J, None
        logj = np.log(J)
        p = self.bulk * logj - self.mu
        invt = np.linalg.inv(F).transpose(0, 2, 1)
        V = self.volume
        elastic = V @ (.5*self.mu*((F*F).sum((1, 2))-3-2*logj) + .5*self.bulk*logj**2)
        energy = elastic + 9.81*self.mass @ y[self.free, 2]
        gradient = np.zeros_like(y)
        stress = self.mu[:, None, None]*F + p[:, None, None]*invt
        np.add.at(gradient, self.t, (self.G @ stress)*V[:, None, None])
        grad = gradient[self.free]
        grad[:, 2] += 9.81*self.mass
        if not hessian:
            return energy, grad.ravel(), J, None
        n = len(self.free)
        H = np.zeros((n, 3, n, 3))
        T = self.G @ invt
        slots = self.scatter[self.t]
        for a in range(4):
            for b in range(4):
                mask = (slots[:, a] >= 0) & (slots[:, b] >= 0)
                g = self.G[mask]
                u, v = T[mask, a], T[mask, b]
                block = ((g[:, a] * g[:, b]).sum(1)*self.mu[mask])[:, None, None]*np.eye(3)
                block += self.bulk[mask, None, None]*u[:, :, None]*v[:, None, :]
                block -= p[mask, None, None]*v[:, :, None]*u[:, None, :]
                A = H.transpose(0, 2, 1, 3)
                np.add.at(A, (slots[mask, a], slots[mask, b]), V[mask, None, None]*block)
        return energy, grad.ravel(), J, H.reshape(n*3, n*3)

    def stats(self,y):
        E,g,J,_ = self.evaluate(y)
        vol, q = geom(y,self.t)
        return dict(potential_J=float(E) if g is not None else None, free_force_l2_N=float(np.linalg.norm(g)) if g is not None else None, minimum_J=float(J.min()),
                    maximum_J=float(J.max()), minimum_shape=float(q.min()), min_volume_m3=float(vol.min()),
                    bad_cells=int(np.sum((vol<=1e-18)|(q<1e-5))), outside_material_bounds=int(np.sum((J<.4)|(J>2.5))))


def solve(native, evidence, proposal, ledger, owner, output, predict=False, expand=False, balance_load=False):
    x, y, t, tags, pins = inputs(native)
    y = y.astype(float)
    j = json.loads(evidence.read_text())
    launch = json.loads((native/'launch.json').read_text())
    env = launch['environment']
    mapping = Path(env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_CSV'])
    flow = ledger/'current-density-closed-flow.npz'
    matched = native/'physical-accepted.node'
    if (pins != j['geometry'] or sha(mapping) != j['map_sha256'] or sha(flow) != j['mass_sha256']
            or sha(proposal) != j['mesh_sha256'] or sha(matched) != j['physical_endpoint_sha256']):
        raise ValueError("Unmatched material registration or motion")
    installed = 'installed_witness_rows' in j
    if installed:
        with np.load(proposal) as repair:
            adjacency = np.vstack((t[repair['retained_input_cell_rows']], repair['replacement_tetrahedra']))
        rows = np.asarray(j['installed_witness_rows'], dtype=int)
        if not len(rows) or not np.array_equal(rows, np.unique(rows)) or rows.min()<0 or rows.max()>=len(adjacency):
            raise ValueError("Invalid installed witness rows")
        for row in j['witnesses']:
            if not np.array_equal(adjacency[row['installed_row']], row['node_ids']):
                raise ValueError("Installed witness connectivity changed")
        free = np.unique(adjacency[rows])
    else:
        adjacency = t
        rows = np.array(sorted({w['source_row'] for g in j['joined_interface_rejections']
                                for group in g['failed_components'] for w in group['witnesses']}))
        pockets = sorted({i for g in j['joined_interface_rejections']
                          for group in g['failed_components'] for i in group['retained_source_rows']})
        free = np.unique(t[pockets])
    core = set(free.tolist())
    if expand:
        active_edge = np.isin(adjacency, free).sum(1) >= 2
        free = np.unique(adjacency[active_edge])
    touched = set(free.tolist())
    peers, meta, layers = {}, {}, {}
    with mapping.open() as src:
        for r in csv.DictReader(src):
            i = int(r['joint_node_id'])-1
            if i not in touched:
                continue
            layer = r['registered_surface_layer']
            if expand and i not in core and layer != 'I':
                continue
            partition = r['source_partition']
            donor = int(r['donor_body'])
            if layer == 'I':
                source = int(r['registered_I_source_vertex_row'])
                if source < 0 or not r['merge_peer_partition']:
                    raise ValueError("Invalid bonded anatomical source")
                record = (int(r['source_tetgen_node_id']), r['merge_peer_partition'], int(r['merge_peer_tetgen_node_id']), source)
            elif installed and i in core and layer == 'D' and partition == 'fat' and not r['merge_peer_partition']:
                record = None
            else:
                raise ValueError("Unidentified material layer")
            entries = peers.setdefault(i, {})
            if partition in entries or (i in meta and (donor != meta[i] or layer != layers[i])):
                raise ValueError("Ambiguous anatomical owner")
            entries[partition] = record
            meta[i], layers[i] = donor, layer
    free = np.array(sorted(meta))
    if not core.issubset(meta) or (not expand and set(meta)!=touched):
        raise ValueError("Missing anatomical source")
    for i,entries in peers.items():
        if layers[i]=='D' and set(entries)=={'fat'}:
            continue
        skin, fat = entries['skin'], entries['fat']
        if skin != (fat[2], 'fat', fat[0], fat[3]) or fat[1] != 'skin' or len(entries) != 2:
            raise ValueError('Material nodes do not share one registered physical point')
    with np.load(flow) as f:
        mass = f['nodal_mass_kg'][free].copy()
        em = f['element_mass_kg']
    with np.load(proposal) as p:
        keep = p['retained_input_cell_rows']
        new = p['replacement_tetrahedra']
        retired = np.setdiff1d(np.arange(len(t)),keep)
        parent=p['replacement_parent_cell_rows']
        parent_offsets=p['replacement_parent_offsets']
        new_tags=tags[parent[parent_offsets[:-1]]]
        nm=p['replacement_mass_kg']
        offsets=p['replacement_tet_offsets']
        cavity_delta=[]
        for a,b in zip(offsets[:-1],offsets[1:]):
            src=np.unique(parent[parent_offsets[a]:parent_offsets[b]])
            dm=np.zeros(len(x))
            for k in range(4):
                np.add.at(dm,t[src,k],-em[src]/4)
                np.add.at(dm,new[a:b,k],nm[a:b]/4)
            mass+=dm[free]
            if np.max(abs(dm[free]))>1e-17:
                cavity_delta.append(dm[free].copy())
        mass_map_data=(t[retired],em[retired],new,nm)
        t=np.vstack((t[keep],new))
        em=np.r_[em[keep],nm]
        tags=np.r_[tags[keep],new_tags]
    active = np.flatnonzero(np.isin(t, free).any(axis=1))
    nodes = np.unique(t[active])
    T = np.searchsorted(nodes,t[active])
    local = np.searchsorted(nodes,free)
    roles = tags[active]
    props, material_pins = materials(owner)
    patch = Patch(x[nodes].astype(float), T, local, mass, roles, props)
    # Pin each material's first mass moment for each owning skeleton segment.
    # These are registration constraints. They add zero force rows in the native run.
    constraints = list(cavity_delta)
    donors = np.array([meta[i] for i in free])
    for m in (15,16):
        lump = np.zeros(len(nodes))
        for corner in range(4):
            np.add.at(lump,T[roles==m,corner],em[active[roles==m]]/4)
        for donor in np.unique(donors):
            row = lump[local]*(donors==donor)
            if np.max(row)>0:
                constraints.append(row)
    weights = np.array(constraints)
    basis = np.kron(null_space(weights/np.linalg.norm(weights,axis=1)[:,None]),np.eye(3))
    start = y[nodes].copy()
    end = np.loadtxt(matched,skiprows=1,usecols=(1,2,3))[nodes]
    delta = end-start
    def pose(v):
        p = start.copy()
        p[local] += (basis@v).reshape(-1,3)
        return p
    # An installed-only minimum may push the same bonded face into failure
    # during measured motion. Constrain both load cases in the same coordinates.
    # Preserve the installed determinant envelope. The accepted frame must
    # regain half its loss of margin towards either actual material limit.
    J0=patch.evaluate(start)[2]
    J2=patch.evaluate(end)[2]
    floor0,ceil0=J0.min()-1e-5,J0.max()+1e-5
    floor2=.5*(J2.min()+min(1.,J0.min()))
    ceil2=.5*(J2.max()+max(1.,J0.max()))
    if balance_load:
        # Prior motion came from the unbalanced release. Require the same
        # admissible material envelope on replay while solving for new forces.
        floor0,ceil0=max(.4,.95*J0.min()),min(2.5,1.05*J0.max())
        floor2,ceil2=max(.405, J2.min()+.005),J2.max()-.005
    gram=basis.T@patch.evaluate(start,True)[3]@basis
    val,vec=np.linalg.eigh((gram+gram.T)*.5)
    basis=basis@vec@np.diag(np.sqrt(1e-4/np.maximum(val,val.max()*1e-7)))
    def obj(v):
        E,g,_,_=patch.evaluate(pose(v))
        if g is None:
            return 1e9,np.zeros(len(v))
        return (E-baseline_energy)*1e4,basis.T@g*1e4
    baseline_energy=patch.evaluate(start)[0]
    B=np.zeros((len(nodes),3,basis.shape[1]))
    B[local]=basis.reshape(len(local),3,-1)
    # Analytic derivatives of det(F) use cofactors and remain defined for a
    # singular trial. Coordinate differences avoid cancellation at body scale.
    increments=B[T[:,1:]]-B[T[:,:1]]
    all_touched=np.arange(len(T))
    mapped_rows=rows if installed else np.searchsorted(keep,rows)
    if not installed and not np.array_equal(keep[mapped_rows],rows):
        raise ValueError('Expected certified original witness identities')
    witness_index=np.searchsorted(active,mapped_rows)
    if not np.array_equal(active[witness_index],mapped_rows):
        raise ValueError('Failure witness outside the patch')
    phases=[(start,floor0,ceil0,all_touched),(end,floor2,ceil2,all_touched)]
    prediction_factor=1+1/j["accepted_steps"] if installed else 1.5
    if predict:
        # Secant extrapolation from two equal time steps screens the next step.
        # It neither advances time nor replaces the native third-step certificate.
        phases.append((start+prediction_factor*delta,.43,2.3,witness_index))
        if 'forecast_limits' in j:
            low,high=j['forecast_limits']
            phases.append((start+prediction_factor*delta,low,high,all_touched))
    def limits(v,with_jac=False):
        q=pose(v)
        parts=[]
        for base,lo,hi,subset in phases:
            state=q+(base-start)
            tet=T[subset]
            edges=state[tet[:,1:]]-state[tet[:,:1]]
            scale=6*patch.volume[subset]
            cof=np.stack((np.cross(edges[:,1],edges[:,2]),np.cross(edges[:,2],edges[:,0]),np.cross(edges[:,0],edges[:,1])),axis=1)
            if with_jac:
                d=np.einsum('nka,nkaj->nj',cof,increments[subset])/scale[:,None]
                parts.extend((d,-d))
            else:
                J=(edges[:,0]*cof[:,0]).sum(1)/scale
                parts.extend((J-lo,hi-J))
        return np.concatenate(parts)
    conditions=[dict(type='ineq',fun=limits,jac=lambda v:limits(v,True))]
    if balance_load:
        from bonded_volume_acceleration import volume_rate
        dt=float(launch['argv'][launch['argv'].index('--muscle-step-seconds')+1])
        witness_cells=T[witness_index]
        slots=patch.scatter[witness_cells]
        if slots.min()<0:raise ValueError('Witness force star incomplete')
        refdet=6*patch.volume[witness_index]
        # Scale acceleration to the dimensionless volume change over one
        # native time increment. This is an initialization objective only.
        def acceleration(v,jac=False):
            q=pose(v)
            _,g,_,H=patch.evaluate(q,jac)
            if g is None:raise ValueError('Inverted cell in load-balance trial')
            a=-g.reshape(-1,3)/mass[:,None]
            return dt**2*volume_rate(q,a,witness_cells,slots,refdet,mass,basis,H)
        guard=.005
        def acc_limits(v):
            u=acceleration(v)
            return np.r_[guard-u,guard+u]
        def acc_jac(v):
            k=acceleration(v,True)
            return np.vstack((-k,k))
        energy_objective=obj
        def obj(v):
            E,dE=energy_objective(v)
            if E>=1e9:
                return E,dE
            u=acceleration(v)
            return E+1e5*(u@u),dE+2e5*u@acceleration(v,True)
    result=minimize(obj,np.zeros(basis.shape[1]),jac=True,method='SLSQP',
                    constraints=conditions,
                    options=dict(maxiter=160,ftol=1.e-9))
    if balance_load and np.max(abs(acceleration(result.x)))<guard:
        result=minimize(energy_objective,result.x,jac=True,method='SLSQP',
            constraints=conditions+[dict(type='ineq',fun=acc_limits,jac=acc_jac)],
            options=dict(maxiter=160,ftol=1.e-8))
    v=result.x
    hist=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),
              minimum_constraint=float(limits(v).min()),installed_J_bounds=[floor0,ceil0],
              replay_J_bounds=[floor2,ceil2])
    selected=pose(v)
    selected[local]=selected[local].astype('f4')
    if balance_load and cavity_delta:
        selected[local]=balance_rounded_moments(start[local],selected[local],cavity_delta)
    emitted=selected[local]-start[local]
    result=dict(source=pins, proposal_sha256=sha(proposal), map_sha256=sha(mapping), evidence_sha256=sha(evidence),
                mass_sha256=sha(flow), accepted_pose_sha256=sha(matched), material_sha256=material_pins,
                cavity_moment_constraints=len(cavity_delta), node_roles=[layers[i] for i in free], node_ids=free.tolist(), donors=donors.tolist(),
                emitted_coordinates=selected[local].tolist(), mesh_mass_change_kg=0.,
                max_material_donor_moment_change_kg_m=float(np.max(abs(weights[len(cavity_delta):]@emitted))),
                max_replacement_moment_change_kg_m=float(np.max(abs(np.asarray(cavity_delta)@emitted))) if cavity_delta else 0.,
                maximum_displacement_m=float(np.linalg.norm(emitted,axis=1).max()), history=hist,
                skin_cells=int((roles==15).sum()),fat_cells=int((roles==16).sum()),
                installed_before=patch.stats(start), installed_after=patch.stats(selected),
                reference_motion_before=patch.stats(end), reference_motion_after=patch.stats(selected+delta))
    result['installed_witness_rows' if installed else 'source_witness_rows']=rows.tolist()
    full_pose=y.copy()
    full_pose[free]=selected[local]
    result['loaded_mass_first_moment']=loaded_mass_moments(full_pose,*mass_map_data)
    if balance_load:
        e,gg,_,_=patch.evaluate(selected)
        if gg is not None:
            rounded_acc=volume_rate(selected,-gg.reshape(-1,3)/mass[:,None],witness_cells,slots,refdet,mass,basis)*dt**2
            result['initial_volume_force_balance']=dict(window_s=dt,witness_rows=rows.tolist(),initial_dt2_acceleration=acceleration(np.zeros(basis.shape[1])).tolist(),
                emitted_dt2_acceleration=rounded_acc.tolist(),requested_absolute_bound=guard,
                scope='Material force contribution at zero velocity before contact and skeletal reactions.')
        else:
            result['initial_volume_force_balance']=None
    if predict:
        future=patch.evaluate(selected+prediction_factor*delta)[2]
        result['extrapolated_witness_J']={str(int(k)):float(d) for k,d in zip(rows,future[witness_index])}
        result['extrapolation_scope']='Only the original rejecting cells constrain the secant predictor. Mandatory native checks cover all cells.'
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({a:result[a] for a in ('skin_cells','fat_cells','maximum_displacement_m','max_material_donor_moment_change_kg_m','installed_before','installed_after','reference_motion_after')}))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    for name in ('native','evidence','proposal','ledger','owner','output'):
        parser.add_argument(name,type=lambda p:Path(p).resolve())
    parser.add_argument('--predict',action='store_true')
    parser.add_argument('--expand',action='store_true')
    parser.add_argument('--balance-load',action='store_true')
    solve(**vars(parser.parse_args()))
