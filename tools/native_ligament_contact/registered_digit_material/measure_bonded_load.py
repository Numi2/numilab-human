"""Resolve the initial volume acceleration of each witnessed adipose tetrahedron by material."""
import json
import sys
from pathlib import Path
import numpy as np
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from relax_bonded_interface import Patch
from audit_cavity_response import elastic
from material_context import sha


def measure(run,proposal,ledger,owner,observed,out):
    x,y,t,tag,pins=inputs(run)
    x,y=x.astype(float),y.astype(float)
    job=json.loads((run/'launch.json').read_text())
    motion=json.loads((run/'verified-endpoint.json').read_text())
    recorded=run/'physical-accepted.node'
    j=json.loads(observed.read_text())
    f=ledger/'current-density-closed-flow.npz'
    if (sha(f)!=j['mass_sha256'] or sha(proposal)!=j['mesh_sha256']
        or pins['reference_sha256']!=j['geometry']['reference_sha256'] or pins['elements_sha256']!=j['geometry']['elements_sha256']
        or sha(recorded)!=motion['physical_nodes_sha256'] or sha(proposal)!=job['environment']['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256']):
        raise ValueError('Unmatched material source or accepted coordinates')
    end=np.loadtxt(recorded,skiprows=1,usecols=(1,2,3))
    rows=sorted({w['source_row'] for a in j['joined_interface_rejections'] for c in a['failed_components'] for w in c['witnesses']})
    centers=t[rows].copy()
    with np.load(f) as m, np.load(proposal) as q:
        nodal=m['nodal_mass_kg'].copy()
        retained=q['retained_input_cell_rows']
        removed=np.setdiff1d(np.arange(len(t)),retained)
        new=q['replacement_tetrahedra']
        if not np.isin(rows,retained).all():raise ValueError('Unknown witness')
        for k in range(4):
            np.add.at(nodal,t[removed,k],-m['element_mass_kg'][removed]/4)
            np.add.at(nodal,new[:,k],q['replacement_mass_kg']/4)
        tag=np.r_[tag[retained],tag[q['replacement_parent_cell_rows'][q['replacement_parent_offsets'][:-1]]]]
        t=np.vstack((t[retained],new))
    stars=np.flatnonzero(np.isin(t,np.unique(centers)).any(axis=1))
    nodes=np.unique(t[stars])
    if not np.all(nodal[nodes]>0):raise ValueError('Invalid inertia')
    forces=[]
    for mat,parameters in zip((15,16),materials(owner)[0]):
        _,force,_=elastic(x,y,t[stars[tag[stars]==mat]],nodes,*parameters)
        forces.append(force)
    samples=[]
    for source,cell in zip(rows,centers):
        local=np.searchsorted(nodes,cell)
        xr=x[cell[1:]]-x[cell[0]]
        spatial=y[cell[1:]]-y[cell[0]]
        accepted=end[cell[1:]]-end[cell[0]]
        ref=np.linalg.det(xr)
        v=np.stack([np.cross(spatial[1],spatial[2]),np.cross(spatial[2],spatial[0]),np.cross(spatial[0],spatial[1])])/ref
        dJ=np.vstack((-v.sum(0),v))
        acc=[float(np.sum(dJ*(f[local]/nodal[cell,None]))) for f in forces]
        J0=np.linalg.det(spatial)/ref
        J2=np.linalg.det(accepted)/ref
        samples.append(dict(source_row=source,nodal_masses_kg=nodal[cell].tolist(),registered_J=float(J0),
                            accepted_J=float(J2),actual_delta_J=float(J2-J0),dermis_J_acceleration_per_s2=acc[0],
                            adipose_J_acceleration_per_s2=acc[1],combined_elastic_J_acceleration_per_s2=sum(acc),
                            uniform_gravity_J_acceleration_per_s2=float(-9.81*dJ[:,2].sum())))
    r=dict(source=pins,accepted_s=motion['simulated_s'],probe_nodes=nodes.size,source_witnesses=samples,
           launch_sha256=sha(run/'launch.json'),scope='Reference-coordinate derivative at zero initial velocity. Elastic forces and lumped masses are evaluated on the installed mesh. Native contact, attachment and implicit time integration contribute to subsequent motion.')
    out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':measure(*(Path(arg).resolve() for arg in sys.argv[1:]))
