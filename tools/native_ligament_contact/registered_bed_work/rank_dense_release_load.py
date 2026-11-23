"""Measure the opening constitutive release on every installed skin and fat cell."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from registered_material_stars import response
from supported_material_admission import installed_topology
from material_context import sha

def blocks(n,size=32768):
    for start in range(0,n,size):yield slice(start,min(n,start+size))

def resolve(case,ledger,prior):
    read=lambda p:json.loads(p.read_text())
    cfg=read(case/'launch.json');env=cfg['environment']
    x,y,base,tag,pins=inputs(case)
    props,identity=materials(Path(cfg['argv'][0]).parent)
    t,role=installed_topology(base,tag,env)
    flow=ledger/'fine-mass-routed/current-density-closed-flow.npz'
    cavity=Path(env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ'])
    if sha(flow)!=read(flow.parent/'mass-transfer.json')['flow_sha256'] or sha(cavity)!=env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256']:raise ValueError("unmatched installed mass")
    with np.load(flow) as f,np.load(cavity) as z:
        keep=z['retained_input_cell_rows']
        mass=np.r_[f['element_mass_kg'][keep],z['replacement_mass_kg']]
    if len(t)!=len(mass) or np.any(mass<=0):raise ValueError("invalid cell mass")
    X=x.astype(float);Y=y.astype(float);node_mass=np.bincount(t.ravel(),np.repeat(mass*.25,4),minlength=len(y))
    if node_mass.min()<=0 or abs(mass.sum()-node_mass.sum())>1e-10:raise ValueError("nodal mass closure")
    force=np.zeros((2,len(y),3));energy=np.zeros(2)
    for kind in (15,16):
        rows=np.flatnonzero(role==kind)
        for s in blocks(len(rows)):
            nodes,local=np.unique(t[rows[s]],return_inverse=True)
            en,F=response(X[nodes],Y[nodes],local.reshape(-1,4),*props[kind-15])
            force[kind-15,nodes]+=F;energy[kind-15]+=en
    if np.max(np.abs(force.sum(1)))>1e-6:raise ValueError("internal force balance")
    nr=np.linalg.norm(force.sum(0),axis=1)
    rates=np.zeros(len(t))
    acc=force.sum(0)/node_mass[:,None]
    for s in blocks(len(t)):
        ids=t[s]
        R=Y[ids[:,1:]]-Y[ids[:,:1]]
        A=acc[ids[:,1:]]-acc[ids[:,:1]]
        rates[s]=np.trace(np.linalg.solve(R,A),axis1=1,axis2=2)
    if not np.isfinite(rates).all():raise ValueError("invalid opening current coordinates")
    stages=[];trial=None
    for line in (case/'stderr.log').open():
        if line.startswith('fem_newton_iterate='):
            d=json.loads(line.split('=',1)[1]);stages.append((d['iteration'],d['stage'],d['alpha'][0],d['fgmres'][0]['nonlinear'][1]))
        if line.startswith('fem_backtracking_trials='):trial=json.loads(line.split('=',1)[1])
    proof=read(prior)
    prefix=proof['global_element_offset']
    import re
    log=(case/'stdout.log').read_text()
    fascia=re.findall(r'^resting_fascia_object=source_prepared object=\d+ .*?tetrahedra=(\d+)',log,re.M)
    acl=re.findall(r'^resting_inferred_acl=admitted .*?tetrahedra=(\d+)',log,re.M)
    wall=re.findall(r'^resting_aortic_wall_outer_contact=accepted .*?retained_wall_faces=(\d+)',log,re.M)
    if len(fascia)!=12 or len(acl)!=2 or len(wall)!=1 or sum(map(int,fascia+acl))+3*int(wall[0])!=prefix:raise ValueError("unmatched compiled global offset")
    z=trial['trials'][0]
    if z['first_bad_tet']<prefix or not z['tile_witnesses_complete']:raise ValueError("incomplete failure witness")
    witnesses=[('latest',int(z['first_bad_tet'])-prefix)]+[(w['limit'],w['native_cell']-prefix) for w in proof['before_failure']]
    records=[]
    for label,i in witnesses:
        ids=t[i];R=Y[ids[1:]]-Y[ids[0]]
        a=[]
        for F in force:
            B=F[ids]/node_mass[ids,None]
            a.append(float(np.trace(np.linalg.solve(R,B[1:]-B[0]))))
        records.append(dict(witness=label,installed_row=i,source_row=int(keep[i]) if i<len(keep) else None,
            nodes=ids.tolist(),role=int(role[i]),center_m=Y[ids].mean(0).tolist(),
            material_logJ_acceleration_s2=float(rates[i]),dermis_and_fat_s2=a,
            absolute_rank=int(np.count_nonzero(abs(rates)>abs(rates[i])))+1,node_mass_kg=node_mass[ids].tolist(),elastic_force_magnitudes_N=nr[ids].tolist()))
    q=[0,.5,.9,.99,.999,1.]
    return dict(launch=sha(case/'launch.json'),source=pins,material=identity,installed_cells=len(t),nodes=len(y),mass_kg=float(mass.sum()),
        total_internal_force_N=force.sum(1).tolist(),dermis_fat_elastic_energy_J=energy.tolist(),
        quantiles=q,abs_material_logJ_acceleration_s2=np.quantile(abs(rates),q).tolist(),
        nodal_elastic_acceleration_m_s2=np.quantile(nr/node_mass,q).tolist(),source_witnesses=records,
        limit_trials=dict(iteration=trial['iteration'],selected_alpha=trial['selected_alpha'],first_trial=z),
        stage_history=stages,
        scope="Opening force, excluding contact, muscle, damping and skeletal constraints. Instantaneous material forcing at zero velocity; no time integration. Gravity is uniform and cancels from relative acceleration. Native failed-step evidence is recorded separately.")

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('case','ledger','prior'):p.add_argument(k,type=Path)
    print(json.dumps(resolve(**vars(p.parse_args())),indent=2,allow_nan=False))
