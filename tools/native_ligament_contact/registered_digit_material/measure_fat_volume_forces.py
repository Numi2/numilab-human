"""Resolve constitutive forces on each bonded fat volume mode using complete incident material stars."""
import argparse,json
from pathlib import Path
import numpy as np
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from supported_material_admission import installed_topology
from audit_cavity_response import elastic
from material_context import sha

def resolve(run,witness,ledger):
    read=lambda p:json.loads(p.read_text())
    w=read(witness);cfg=read(run/'launch.json');e=cfg['environment'];X,Y,b,role,pins=inputs(run);t,roles=installed_topology(b,role,e)
    p=run/'physical-accepted.node'; flow=ledger/'fine-mass-routed/current-density-closed-flow.npz'
    if pins!=w['geometry'] or sha(p)!=w['accepted_geometry_sha256'] or sha(flow)!=read(flow.parent/'mass-transfer.json')['flow_sha256']:raise ValueError('Different release or mass map')
    cav=Path(e['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ'])
    if sha(cav)!=e['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256']:raise ValueError('Installed stars mismatch')
    with np.load(flow) as a,np.load(cav) as c:m=np.r_[a['element_mass_kg'][c['retained_input_cell_rows']],c['replacement_mass_kg']]
    if len(m)!=len(t) or np.any(m<=0):raise ValueError('Invalid material mass')
    props,material=materials(Path(cfg['argv'][0]).parent)
    x=X.astype(float);y=Y.astype(float);z=np.loadtxt(p,skiprows=1,usecols=(1,2,3))
    out=[]
    for v in w['before_failure']:
        row=v['native_cell']-w['global_element_offset'];nodes=t[row]
        if nodes.tolist()!=v['nodes']:raise ValueError('Mismatched native cell')
        incident=np.flatnonzero(np.isin(t,nodes).any(1));star,local=np.unique(t[incident],return_inverse=True)
        local=local.reshape(-1,4);where=np.searchsorted(star,nodes)
        mass=np.bincount(local.ravel(),np.repeat(m[incident]*.25,4))[where]
        rec=dict(cell=v['source_cell'],nodes=nodes.tolist(),incident_cells=len(incident),routed_lumped_node_kg=mass.tolist(),poses={})
        for name,pos in [('opening',y),('accepted',z)]:
            A=pos[nodes[1:]]-pos[nodes[0]]
            G=np.linalg.inv(A).T;dlog=np.vstack((-G.sum(0),G))
            # A common translation must produce no volume variation.
            if np.linalg.norm(dlog.sum(0))>1e-8:raise ValueError('Translation invariance failed')
            force={};projected={};potential={};base=x[star];current=pos[star]
            for group,g in [('cell',incident==row),('other_fat',(roles[incident]==16)&(incident!=row)),('skin',roles[incident]==15)]:
                if not g.any():f=np.zeros((4,3));en=0.
                else:
                    kind,=np.unique(roles[incident[g]]);en,allf,_=elastic(base,current,local[g],np.arange(len(star)),*props[kind-15])
                    f=allf[where]
                force[group]=f
                projected[group]=float(np.sum(dlog*(f/mass[:,None])))
                potential[group]=en
            total=sum(force.values());q=dict(material_force_N={k:a.tolist() for k,a in force.items()},norm_total_nodal_force_N=np.linalg.norm(total,axis=1).tolist(),
                material_log_volume_acceleration_s2=projected,sum_material_log_volume_acceleration_s2=sum(projected.values()),material_energy_J=potential)
            # Unit virtual displacement is a material dilatation; the mass-metric
            # mode is normalized so that dlogJ/da = 1.
            inv_mass_norm=float(np.sum(dlog*dlog/mass[:,None]));q['virtual_work_J_per_logJ']={k:v/inv_mass_norm for k,v in projected.items()}
            rec['poses'][name]=q
        rec['observed_log_volume_change']=float(np.log(v['accepted_over_start']))
        out.append(rec)
    return dict(source=pins,accepted_node_sha256=sha(p),cell_mass=sha(flow),composition=sha(cav),materials=material,
        scope='Constitutive elastic contribution to local log-volume acceleration at zero nodal velocity under the mapped lumped masses. Native damping, contact and other constraints remain separate.',witnesses=out)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','witness','ledger'):p.add_argument(k,type=Path)
    print(json.dumps(resolve(**vars(p.parse_args())),indent=2,allow_nan=False))
