"""Project first-step native friction through the contemporaneous pressure-weighted point Jacobian."""
import csv
import json
from pathlib import Path
import numpy as np
from run_bed_work import digest


def load(case,q,normal,normal_rows,reference_case,jacobian_case=None):
    def read(p):return json.loads(p.read_text())
    jacobian_case=jacobian_case or case
    capture=jacobian_case.with_name(jacobian_case.name+'-source')/'witness.json.opening.csv'
    observed=read(jacobian_case/'terminal.receipt.json')
    if observed['accepted_step']!=1:raise ValueError('Invalid Jacobian observation')
    contact=case/'resting-com-support-impulses.csv'
    receipt=read(case/'terminal.receipt.json')
    reference=read(reference_case/'terminal.receipt.json')
    logs=[json.loads(x.split('=',1)[1]) for x in (case/'stdout.log').read_text().splitlines() if x.startswith('stand_terminal_state=')]
    if len(logs)!=1 or receipt['accepted_step']!=1 or not np.array_equal(q,np.asarray(logs[0]['initial_q'],dtype='f4')):
        raise ValueError('Friction must be sampled on the exact opening pose')
    if any(r['initial_anatomical_registration']['body_poses']!=reference['initial_anatomical_registration']['body_poses'] for r in (receipt,observed)):
        raise ValueError('Different initial registration')
    if any(digest(p/'registered-bed-columns.csv')!=digest(reference_case/'registered-bed-columns.csv') for p in (case,jacobian_case)):
        raise ValueError('Different bed quadrature')
    data=np.loadtxt(capture,delimiter=',',skiprows=1)
    nr,nv=normal_rows.shape
    if data.shape!=(nr*3,nv+3) or not np.isfinite(data).all():raise ValueError('Incomplete sampled Jacobians')
    data=data.reshape(nr,3,-1)
    if not (np.all(data[:,:,0]==np.arange(nr)[:,None]) and np.all(data[:,:,1]==np.arange(3)) and
            np.array_equal(data[:,2,2:-1],normal_rows) and np.all(data[:,:,-1]==data[:,2:3,-1])):
        raise ValueError('Inconsistent Jacobian or normal-force publication')
    load_N=data[:,2,-1]
    if abs(load_N.sum()-normal)>1e-5:raise ValueError('Unmatched opening normal resultant')
    def rows(out):
        with out.open() as f:return list(csv.DictReader(f))
    ids={}
    for r in rows(case/'registered-bed-columns.csv'):
        i=int(r['row']);geom=int(r['source_geometry_index'])
        if i in ids and ids[i]!=geom:raise ValueError('Ambiguous contact surface')
        ids[i]=geom
    reactions=sorted(rows(contact),key=lambda r:int(r['contact_index']))
    if len(reactions)!=nr:raise ValueError('Missing external contact rows')
    dt=logs[0]['timestep_seconds']
    impulse=[]
    for i,r in enumerate(reactions):
        if (int(r['accepted_step'])!=1 or int(r['contact_index'])!=i or r['row_class']!='external_support_contact' or
            ids.get(i)!=int(r['source_geometry_index']) or [float(r['normal_'+k]) for k in 'xyz']!=[0,0,1]):
            raise ValueError('Mismatched horizontal bed row')
        impulse.append([float(r['impulse_world_'+k+'_ns']) for k in 'xyz'])
    physical=np.asarray(impulse)/dt
    if not np.isfinite(physical).all() or np.any(physical[:,2]<0) or np.any(physical[load_N==0]!=0):
        raise ValueError('Unqualified impulse observation')
    jac=np.zeros_like(data[:,:,2:-1])
    active=load_N>0
    jac[active]=data[active,:,2:-1]/load_N[active,None,None]
    if np.max(abs(jac[active,:,:3]-np.eye(3)))>2e-5:
        raise ValueError('Incorrect translational point Jacobian')
    force=np.einsum('iaj,ia->j',jac[:,:2,:],physical[:,:2])
    if np.max(abs(force[:2]-physical[:,:2].sum(0)))>.001 or abs(force[2])>.001:raise ValueError('Invalid shear resultant')
    return force,dict(impulse_csv_sha256=digest(contact),observation_sha256=digest(capture),source_root=receipt['accepted_root_fingerprint_hex'],
        first_step_bed_force_N=physical.sum(0).tolist(),sum_local_shear_norm_N=float(np.linalg.norm(physical[:,:2],axis=1).sum()),
        generalized_shear=force.tolist(),per_row_shear_N=physical[:,:2].tolist(),
        scope='First-step tangential reaction used as an estimate for initial muscle recruitment. The native solver recomputes contact each physical step.')
