"""Compare stored energy, pulled-back physical forces and material mass at a proposed interior reparameterization."""
import argparse,json
from pathlib import Path
import numpy as np
from compile_connected_joint import digest

def response(X,Y,t,mu,bulk):
    e0=X[t[:,1:]]-X[t[:,:1]]
    e=Y[t[:,1:]]-Y[t[:,:1]]
    inv=np.linalg.inv(e0)
    F=np.linalg.solve(e0,e).transpose(0,2,1)
    det=np.linalg.det(F)
    if np.any(det<=0):raise ValueError('Nonpositive material deformation')
    l=np.log(det)
    W=.5*mu*((F*F).sum(axis=(1,2))-3-2*l)+.5*bulk*l**2
    V=np.linalg.det(e0)/6
    P=mu*F+(bulk*l-mu)[:,None,None]*np.linalg.inv(F).transpose(0,2,1)
    # Matrix rows are reference edge vectors; its inverse columns are gradients.
    nodal=(P@inv).transpose(0,2,1)*V[:,None,None]
    forces=np.zeros_like(X)
    np.add.at(forces,t[:,0],nodal.sum(axis=1))
    for j in range(3):np.add.at(forces,t[:,j+1],-nodal[:,j])
    return float(W@V),forces,V

def run(parent,material_source,proposal,out):
    a=parent/'witness-material/volume'
    doc=json.loads(proposal.read_bytes()); materials=json.loads(material_source.read_bytes())
    if any(digest(Path(p))!=s for p,s in materials['material_pins'].items()):raise ValueError('Stale material parameters')
    c=np.loadtxt(a/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype=np.int32)
    mass=np.fromfile(a/'cell-mass.f64',dtype='<f8')
    paths=[a/'mesh.node',a/'current.node',parent/'accepted-material-07/physical-accepted.node',parent/'half-step-material-08/physical-accepted.node']
    if [digest(f) for f in paths]!=doc['poses'] or digest(a/'mesh.ele')!=doc['physical_mesh']:raise ValueError('Geometry mismatch')
    poses=[np.loadtxt(p,skiprows=1,usecols=(1,2,3)) for p in paths];result=[]
    for p in doc['selected']:
        rows=p['affected_cells'];old_t=c[rows,:4]-1;ids,t=np.unique(old_t,return_inverse=True);t=t.reshape(old_t.shape)
        node=np.searchsorted(ids,p['vertex']);neigh=np.searchsorted(ids,p['neighbor_nodes'])
        w=np.array(p['local_weights'])
        B=np.eye(len(ids));B[node,node]=1-w.sum();B[node,neigh]=w
        if abs(B.sum(axis=1)-1).max()>1e-14 or B.min()<0:raise ValueError('Invalid material interpolation')
        role,=set(c[rows,4]);props=materials['materials'][int(role)-15]
        mu,bulk=props['mu_pa'],props['log_volume_coefficient_pa']
        metrics={};X=poses[0][ids];Y=B@X
        Y[node]=Y[node].astype(np.float32)
        for name,y in zip(['registered','accepted','independent_half_step'],poses[1:]):
            e,f,v=response(X,y[ids],t,mu,bulk)
            current=B@y[ids]
            if name=='registered':current[node]=current[node].astype(np.float32)
            ee,ff,vv=response(Y,current,t,mu,bulk)
            # Same kinematically admissible virtual perturbation on each basis.
            df=B.T@ff-f
            metrics[name]=dict(energy_before_J=e,energy_after_J=ee,relative_energy_error=abs(ee-e)/max(abs(e),1e-18),
                               force_before_N=float(np.linalg.norm(f)),force_difference_N=float(np.linalg.norm(df)),
                               relative_force_error=float(np.linalg.norm(df)/max(np.linalg.norm(f),1e-14)))
        rho=mass[rows]/v
        uniform=(rho.max()-rho.min())/rho.mean()<1e-8
        density=float(mass[rows].sum()/v.sum())
        new_m=vv*density
        reached=dict(vertex=p['vertex'],rows=len(t),reference_density_range_kg_m3=[float(rho.min()),float(rho.max())],
                     locally_uniform_density=bool(uniform),conservative_mass_change_kg=float(new_m.sum()-mass[rows].sum()),
                     reference_first_moment_change_kg_m=float(np.linalg.norm(new_m@(Y[t].mean(axis=1))-mass[rows]@(X[t].mean(axis=1)))),
                     states=metrics,heldout_before=p['heldout_before'],heldout_after=p['heldout_after'])
        reached['passed']=bool(uniform and abs(reached['conservative_mass_change_kg'])<1e-17 and reached['reference_first_moment_change_kg_m']<1e-13 and not p['heldout_after'] and not p['heldout_nonpositive'] and
                               all(v['relative_force_error']<=.01 and v['relative_energy_error']<=.05 for v in metrics.values()))
        result.append(reached)
    out.write_text(json.dumps(dict(proposal_sha256=digest(proposal),material_pins=materials['material_pins'],sites=result,
          passed_sites=sum(r['passed'] for r in result), scope='Passive volumetric constitutive forces and energies on shared virtual displacements at unchanged captured poses. Fixed regional boundary and no surface displacement.'),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for a in ('parent','material_source','proposal','out'):p.add_argument(a,type=Path)
    run(**vars(p.parse_args()))
