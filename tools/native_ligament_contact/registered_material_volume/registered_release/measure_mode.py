"""Measure registered internal force and its volume-changing acceleration on the actual material corners."""
from pathlib import Path
import json,sys
import numpy as np
from prepare_indentation import soft_response
from compile_connected_joint import digest
def jac_volume(ref,pos,t):
    E=pos[t[1:]]-pos[t[0]]; V=np.linalg.det(ref[t[1:]]-ref[t[0]])
    G=np.cross(E[[1,2,0]],E[[2,0,1]])/V
    return np.vstack([-G.sum(0),G])
def run(h,v,mat,det,out):
    a=np.load(h/'right-ulna/cavity.npz');rows=a['rows'];sel=int(a['selected']);t=a['local'][np.searchsorted(rows,sel)];nodes=a['ids'][t];x,y,z=a['xyz']
    active=np.searchsorted(rows,a['neighbors']);T=a['local'][active];mass=np.fromfile(v/'node-mass.f64','<f8')[nodes];mu,bulk=np.array([[json.loads(mat.read_bytes())['materials'][str(k)]['registered'][c]for c in ('source_mu_pa','source_logJ_coefficient_pa')]for k in a['labels'][active,1]]).T
    law=soft_response(x,y,T,mu,bulk);f,_=law(y);forces=np.zeros_like(x);np.add.at(forces,T.ravel(),-f.reshape(-1,3))
    g=jac_volume(x,y,t);nv=np.sum(g*(forces[t]/mass[:,None]));dt=json.loads((h/'motion-check/motion.json').read_bytes())['accepted_s']
    look=json.loads(det.read_bytes());r=next(q for q in look['incident_cells']if q['row']==sel and q['mode']=='constant_velocity')
    report=dict(nodes=nodes.tolist(),initial_internal_forces_N=forces[t].tolist(),nodal_masses_kg=mass.tolist(),gradient_per_m=g.tolist(),J_change_explicit_elastic_acceleration=dt**2*nv,actual_measured_J_change=r['accepted_J']-json.loads((h/'right-ulna/region.json').read_bytes())['stages']['initial']['largest_J'],mass_sha=digest(v/'node-mass.f64'))
    out.write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':run(*(Path(s).resolve()for s in sys.argv[1:]))
