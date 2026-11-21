"""Reconstruct the native candidate residual and bound pressure-only corrections at fixed geometry."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, vstack, hstack
from scipy.optimize import linprog

from material_context import sha

run, assets, out = map(Path,sys.argv[1:])
node_file=run/'candidate-force-nodes.bin'
port_file=run/'candidate-force-ports.bin'
node=np.fromfile(node_file,'<f4').reshape(-1,2,4).astype(float)
port=np.fromfile(port_file,'<f4').reshape(512,4,4).astype(float)
manifest=json.loads((assets/'manifest.json').read_text())
root=int(json.loads((run/'launch.json').read_text())['environment']['NUMI_HUMAN_PELVIC_FORCE_WITNESS_ROOT'])
cert={int(r['side']):r for r in csv.DictReader((run/'fresh-force-certificate.csv').open()) if int(r['candidate_step'])==root and r['endpoint']=='1'}
result=dict(root=root,node_sha256=sha(node_file),port_sha256=sha(port_file),sides={})

def cutting_plane(B,force,directions):
    count=len(force);columns=B.shape[1];h=[]
    def plane(normals):
        blocks=B.multiply(normals.reshape(-1,1)).tocsr()
        sums=blocks[0::3]+blocks[1::3]+blocks[2::3]
        h.append((hstack((-sums,-np.ones((count,1))),format='csr'),-np.einsum('ij,ij->i',normals,force)))
    for axis in np.eye(3):
        for sign in [1,-1]:plane(np.broadcast_to(sign*axis,(count,3)))
    plane(directions)
    for iteration in range(40):
        A=vstack([x[0] for x in h]);b=np.concatenate([x[1] for x in h])
        sol=linprog(np.r_[np.zeros(columns),1.0],A_ub=A,b_ub=b,bounds=(0,None),method='highs',
                    options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        if not sol.success:raise ValueError(sol.message)
        r=force-(B@sol.x[:-1]).reshape(-1,3)
        norm=np.linalg.norm(r,axis=1);upper=norm.max();lower=sol.x[-1]
        if upper-lower<=1e-6:break
        # Add only separating directions at violated nodes to the current LP.
        ids=np.flatnonzero(norm>lower+1e-7);normals=r[ids]/norm[ids,None]
        Bsub=B[np.ravel(3*ids[:,None]+np.arange(3))]
        blocks=Bsub.multiply(normals.reshape(-1,1)).tocsr()
        sums=blocks[0::3]+blocks[1::3]+blocks[2::3]
        h.append((hstack((-sums,-np.ones((len(ids),1))),format='csr'),-np.einsum('ij,ij->i',normals,force[ids])))
    return dict(lower_N=float(lower),upper_N=float(upper),cutting_plane_solves=iteration+1)

for side,part in enumerate(manifest['contact_flexibility']):
    start,n=part['first_node'],part['nodes']
    ids=np.flatnonzero(node[start:start+n,0,3]) + start
    index={int(v):j for j,v in enumerate(ids)}
    rows=[];cols=[];values=[]
    ports=port[256*side:256*side+256]
    live=ports[:,1,0]>=0
    pressure=ports[:,0,3].copy();pressure[~live]=0
    for j in np.flatnonzero(live):
        normals=ports[j,0,:3]
        for k,w in zip(ports[j,2].astype(int),ports[j,3]):
            if k in index and w!=0:
                for axis in range(3):
                    rows.append(3*index[k]+axis);cols.append(j);values.append(w*normals[axis])
    B=coo_matrix((values,(rows,cols)),shape=(3*len(ids),256)).tocsr()
    force=node[ids,0,:3]
    ext=(B@pressure).reshape(-1,3)
    r=force-ext;norm=np.linalg.norm(r,axis=1)
    worst=int(norm.argmax());peak=float(np.linalg.norm(ext,axis=1).max())
    recorded=float(cert[side]['free_residual_N'])
    if abs(norm[worst]-recorded)>3e-6 or abs(peak-float(cert[side]['external_force_N']))>3e-6:
        raise ValueError(('native force parity',side,norm[worst],recorded,peak,cert[side]))
    threshold=.01*max(peak,1e-6)
    allowed=live&((pressure>0)|(ports[:,1,1]<=5e-7))
    B=B[:,allowed];wmap=B[3*worst:3*worst+3].toarray()
    incident=np.flatnonzero(np.linalg.norm(wmap,axis=0)>0)
    sub=wmap[:,incident]
    if len(incident):
        fit=np.linalg.lstsq(sub,force[worst],rcond=1e-12)[0];outside=float(np.linalg.norm(force[worst]-sub@fit))
    else:outside=float(np.linalg.norm(force[worst]))
    directions=np.divide(r,norm[:,None],out=np.zeros_like(r),where=norm[:,None]>0)
    bounded=cutting_plane(B,force,directions)
    result['sides'][str(side)]=dict(free_nodes=len(ids),candidate_ports=int(sum(allowed)),maximum_node=int(ids[worst]),
        actual_residual_N=float(norm[worst]),native_residual_N=recorded,native_scale_N=peak,native_threshold_N=threshold,
        worst_node_incident_pressure_columns=len(incident),worst_node_component_outside_incident_span_N=outside,
        fixed_geometry_pressure_only=bounded)
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result['sides']))
