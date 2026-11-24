"""Join the measured posterior skin/fat tangent to the registered gluteal material and its skeletal donors."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import splu
from prepare_indentation import soft_response
from compile_connected_joint import digest


def read(path):
    return json.loads(path.read_bytes())


def require(ok,msg):
    if not ok:raise ValueError(msg)


def material_patch(j,compartment,case):
    vol=j/'conditioned/witness-material/volume'
    old=read(j/'nonlinear-bed.json')
    require(digest(vol/'mesh.node')==old['reference'] and digest(vol/'current.node')==old['initial'],'Unknown preload mesh')
    X,Y=[np.loadtxt(vol/n,skiprows=1,usecols=(1,2,3)) for n in ('mesh.node','current.node')]
    elems=np.loadtxt(vol/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype=np.int32)
    cells=elems[:,:4]-1; layer=elems[:,4]
    roles=np.load(vol/'material-walls.npz')
    params=np.load(compartment/'parents.npz');fat=params['fat']; donors=params['donors']
    require(np.array_equal(params['reference'][:len(fat)],X[fat]) and np.array_equal(X[fat],Y[fat]),'Changed deep-fat interface reference')
    require(np.all(np.isin(fat,roles['D'])),'Gluteal parents no longer lie on deep fat')
    tmap=np.load(compartment/'reference-tangent.npz')['matrix']
    require(tmap.shape==(3*len(fat)+6*len(donors),)*2,'Incomplete receiver tangent')
    skin=np.unique(roles['O']);normal=np.array(read(j/'support-balance.json')['bed_normal'])
    offset=read(j/'support-balance.json')['source_plane_offset_m'];pot=skin[Y[skin]@normal-offset<.02]
    # The earlier reconstruction specifies its footprint through deep material parents.
    # Include all incident layers, including every local cut node as a free unknown.
    nodes=np.union1d(fat,pot)
    for i in range(2):
        nodes=np.unique(cells[np.isin(cells,nodes).any(1)])
    mask=np.isin(cells,nodes).any(1)
    ids,local=np.unique(cells[mask],return_inverse=True);local=local.reshape(-1,4)
    a=np.tile(local[:,0],3);b=local[:,1:].T.ravel()
    adjacency=sparse.coo_matrix((np.ones(len(a),np.int8),(a,b)),shape=(len(ids),)*2)
    nc,tag=csgraph.connected_components(adjacency,directed=False)
    hits=np.unique(tag[np.searchsorted(ids,fat)])
    keep=np.isin(tag[local[:,0]],hits)
    retained=np.flatnonzero(mask)[keep]
    ids,t=np.unique(cells[retained],return_inverse=True);t=t.reshape(-1,4)
    groups={int(c):int(np.sum(tag==c)) for c in hits}
    sample=case/'physical-posterior-bed-history.csv'
    with sample.open() as f:contact=list(csv.DictReader(f))
    imp={int(r['registered_skin_node']):float(r['terminal_microstep_normal_impulse_Ns']) for r in contact}
    require(len(imp)==len(contact) and min(imp.values())>=0 and all(v in skin for v in imp),'Invalid accepted contact rows')
    loads=np.array([imp.get(int(i),0)for i in ids])
    require(loads.sum()>0,'No native bed load in connected gluteal skin')
    summary=dict(elements=digest(vol/'mesh.ele'),reference=old['reference'],initial=old['initial'],body_donors=donors.tolist(),
        gluteal_parents=digest(compartment/'parents.npz'),gluteal_tangent=digest(compartment/'reference-tangent.npz'),
        all_candidate_components=nc,included_components=groups,material_cells=len(t),free_nodes=len(ids),
        outer_loaded_nodes=ids[loads>0].tolist(),patch_impulse_Ns=float(loads.sum()),native_impulse_Ns=sum(imp.values()),
        native_history=digest(sample),native_launch=digest(case/'launch.json'),outside_loaded_nodes=sorted(set(imp)-set(ids)),
        accepted_load_fraction=loads.sum()/sum(imp.values()))
    return X[ids],Y[ids],t,layer[retained],ids,np.searchsorted(ids,fat),tmap,params,normal,loads,summary


def solve(j,compartment,case,out,configuration="registered",footprint="all"):
    out.mkdir()
    x,y,t,layer,ids,port,S,params,n,imp,data=material_patch(j,compartment,case)
    (out/'domain.json').write_text(json.dumps(data,indent=2)+'\n')
    data['configuration']=configuration;data['footprint']=footprint
    data['registered_to_reference_peak_m']=float(np.linalg.norm(y-x,axis=1).max())
    if configuration=='stress_free':
        y=x.copy()
    if footprint=='registered_gluteal':
        receivers=read(j.parent/'posterior-load-path-077/registered-receivers.json')['receivers']
        mask=np.isin(ids,[r['skin_node'] for r in receivers])
        imp=np.where(mask,imp,0.0)
        require(imp.sum()>0,'No matched gluteal pressure')
    data['applied_fraction_of_native_impulse']=float(imp.sum()/data['native_impulse_Ns'])
    mat=read(j.parent/'registration-strain.json')['materials']
    constants=np.array([[mat[str(i)]['registered'][k] for k in ('source_mu_pa','source_logJ_coefficient_pa')] for i in (15,16)])
    mu,bulk=constants[layer-15].T
    law=soft_response(x,y,t,mu,bulk);force,e=law(y,True)
    dofs=3*t[:,:,None]+np.arange(3)
    row,col=np.broadcast_arrays(dofs[:,:,None,:,None],dofs[:,None,:,None,:])
    shape=(3*len(ids),)*2
    skin=sparse.coo_matrix((e.ravel(),(row.ravel(),col.ravel())),shape=shape).tocsc()
    select=(3*port[:,None]+np.arange(3)).ravel();m=len(select)
    row,col=np.meshgrid(select,select,indexing='ij')
    k=skin+sparse.coo_matrix((S[:m,:m].ravel(),(row.ravel(),col.ravel())),shape=shape).tocsc()
    require(np.max(abs(k-k.T))<1e-9*np.max(abs(k)),'Nonreciprocal skin+gluteal tangent')
    factor=splu(k)
    # Apply the relative pattern in the recorded native terminal impulse. Each
    # solution is the derivative per newton of bed load in that pattern.
    rhs=imp[:,None]*n/imp.sum()
    u=factor.solve(rhs.ravel()).reshape(-1,3)
    for i in range(5):
        r=rhs.ravel()-k@u.ravel()
        if np.linalg.norm(r)<1e-7:break
        u+=factor.solve(r).reshape(-1,3)
    du=u[port].ravel()
    f=(S[m:,:m]@du).reshape(-1,6)
    moment=np.sum(f[:,3:]+np.cross(params['origins'],f[:,:3]),axis=0)
    ext_moment=np.sum(np.cross(y,rhs),axis=0)
    # Use independent complex-step derivatives on relative element edges.
    # This avoids losing a small strain to subtraction of two large coordinates.
    a=x[t[:,1:]]-x[t[:,:1]]
    inv=np.linalg.inv(a);volume=np.linalg.det(a)/6
    g=np.concatenate([-inv.sum(2,keepdims=True),inv],axis=2).transpose(0,2,1)
    h=1.e-20
    F=(inv@(y[t[:,1:]]-y[t[:,:1]]+1j*h*(u[t[:,1:]]-u[t[:,:1]]))).transpose(0,2,1)
    P=mu[:,None,None]*F+(bulk*np.log(np.linalg.det(F))-mu)[:,None,None]*np.linalg.inv(F).transpose(0,2,1)
    derivative=(volume[:,None,None]*(g@P.transpose(0,2,1))).imag/h
    fd=np.zeros_like(y);np.add.at(fd,t.ravel(),derivative.reshape(-1,3))
    analytical=(skin@u.ravel()).reshape(-1,3)
    relative_fd=np.linalg.norm(fd-analytical)/max(np.linalg.norm(analytical),1e-30)
    base=np.zeros_like(y);np.add.at(base,t.ravel(),force.reshape(-1,3))
    moving_moment=np.sum(np.cross(u,base),axis=0)
    total=f[:,:3].sum(axis=0)
    report=dict(data,incremental_force_balance_error_N=float(np.linalg.norm(total+n)),incremental_moment_balance_error_Nm=float(np.linalg.norm(moment+ext_moment+moving_moment)),
        opening_prestress_moment_term_Nm=moving_moment.tolist(),uncompensated_moment_Nm=float(np.linalg.norm(moment+ext_moment)),
        equation_residual_N=float(np.linalg.norm(k@u.ravel()-rhs.ravel())),finite_difference_relative=float(relative_fd),
        probe_N=float(np.linalg.norm(rhs.sum(0))),complex_step=h,maximum_compliance_m_per_N=float(np.max(np.linalg.norm(u,axis=1))),
        work_compliance_m_per_N=float(np.sum(rhs*u)),drive_landmarks=ids[np.flatnonzero(imp)].tolist(),
        skeletal_energy_gradients_N_Nm=f.tolist(),reaction_on_bones_N_Nm=(-f).tolist(),
        bodily_reaction_sum_N=(-total).tolist(),undisplaced_skin_reference_stress_N=float(np.linalg.norm(base)),
        frozen_cut_nodes=0)
    (out/'response.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(out/'response.npz',node_ids=ids,tets=t,delta_position_per_N=u,applied_per_N=rhs,donors=params['donors'],body_reaction_per_N=-f)
    print({k:report[k] for k in ('material_cells','free_nodes','accepted_load_fraction','incremental_force_balance_error_N',
         'incremental_moment_balance_error_Nm','equation_residual_N','finite_difference_relative','maximum_compliance_m_per_N','work_compliance_m_per_N')})
    require(report['equation_residual_N']<1e-5 and report['incremental_force_balance_error_N']<1e-5 and report['incremental_moment_balance_error_Nm']<1e-5 and relative_fd<1e-5
            and report['work_compliance_m_per_N']>0,'Unqualified local static response')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('j','compartment','case','out'):p.add_argument(k,type=lambda v:Path(v).resolve())
    p.add_argument('--configuration',choices=('registered','stress_free'),default='registered')
    p.add_argument('--footprint',choices=('all','registered_gluteal'),default='all')
    solve(**vars(p.parse_args()))
