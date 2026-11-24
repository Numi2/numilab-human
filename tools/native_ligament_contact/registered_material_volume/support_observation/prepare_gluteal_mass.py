"""Transfer a fitted adipose reference volume onto the existing anatomical mass donors."""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from bind_gluteal_receivers import digest,require


def run(registration, prior, assets, volume, material, output):
    data=np.load(registration/'registered-columns.npz')
    cert=json.loads((registration/'anatomical-intersections.json').read_bytes())
    require(cert['registration']==digest(registration/'registration.json'),'Unmatched volume certificate')
    capture=json.loads((prior/'coupled/native-traces/accepted-geometry/step-1.receipt.json').read_bytes())
    poses={p['body_index']:p for p in capture['initial_anatomical_registration']['body_poses']}
    ref=np.loadtxt(prior/'coupled/physical-reference.node',skiprows=1,usecols=(1,2,3))
    walls=np.load(volume/'material-walls.npz')
    fat_nodes=walls['D'][data['deep_fat_face']]
    need=set(map(int,np.unique(fat_nodes)+1))
    source_ids={}
    cfg=json.loads((prior/'coupled/launch.json').read_bytes())
    node_file=Path(cfg['environment']['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_CSV'])
    require(digest(node_file)==cfg['pins'][str(node_file)],'Invalid source registration')
    with node_file.open() as stream:
        for row in csv.DictReader(stream):
            nid=int(row['joint_node_id'])
            if nid in need:
                require(row['registered_surface_layer']=='D','Incorrect retained interface')
                source_ids[nid]=int(row['donor_body'])
    require(source_ids.keys()==need,'Unregistered fat donor')
    m=json.loads((assets/'manifest.json').read_bytes())
    def table(name,kind,width):
        f=assets/(name+'.bin')
        require(digest(f)==m['files'][f.name],'Modified muscle material ' + name)
        return np.fromfile(f,kind).reshape(-1,width,4)
    muscle_faces=table('closed-material-triangles','<u4',3)[:,0,:3][data['muscle_face']]
    vertices=table('closed-material-vertices','<f4',2)
    boundary=table('constitutive-boundary-reference','<f4',3).astype(float)
    p1=vertices[:,0].astype(float)+boundary[:,2]
    parents=vertices[:,1].copy().view('<u4')
    node=table('material-nodes','<f4',3)
    bodies=node[:,2].copy().view('<u4')[:,1:3]
    weights=node[:,:2,3].astype(float)
    require(np.all(weights>=0) and np.max(abs(weights.sum(1)-1))<1e-7,'Muscle donors lack a partition')
    weights/=weights.sum(1,keepdims=True)
    donors=sorted(set(bodies.ravel()) | set(source_ids.values()))
    local_positions=boundary[:,0,:3]+boundary[:,1,:3]
    posed=lambda p,v: Rotation.from_quat(p['quaternion_xyzw']).apply(v)+p['position_m']
    donor_basis=np.zeros((len(node),len(donors)))
    for k in range(2):
        donor_basis[np.arange(len(node)),np.searchsorted(donors,bodies[:,k])]+=weights[:,k]
    skin_basis=np.eye(len(donors))[np.searchsorted(donors,[[source_ids[int(i)+1]for i in row] for row in fat_nodes])]
    muscle_basis=np.einsum('ij,ijk->ik',p1,donor_basis[parents])
    face_muscle=np.einsum('fij,fjk->fik',data['muscle_shape'],muscle_basis[muscle_faces])
    face_fat=np.einsum('fij,fjk->fik',data['fat_shape'],skin_basis)
    top_xyz=posed(poses[128],local_positions)[muscle_faces]
    top=np.einsum('fij,fjk->fik',data['muscle_shape'],top_xyz)
    bottom=np.einsum('fij,fjk->fik',data['fat_shape'],ref[fat_nodes])
    prism=np.concatenate((bottom,top),axis=1)
    basis=np.concatenate((face_fat,face_muscle),axis=1)
    require(np.min(basis)>-1e-10 and np.max(abs(basis.sum(2)-1)) < 1e-10,'Invalid interpolated donor weights')
    # Pull the same diagonal through each registered and reference prism.
    uv=data['plane_xy']
    order=np.lexsort((uv[:,:,1],uv[:,:,0]),axis=1)
    sort=np.concatenate((order,order+3),axis=1)
    prism=np.take_along_axis(prism,sort[:,:,None],1)
    basis=np.take_along_axis(basis,sort[:,:,None],1)
    geom=np.concatenate((np.dstack([uv,data['fat_z']]),np.dstack([uv,data['muscle_z']])),axis=1)
    geom=np.take_along_axis(geom,sort[:,:,None],1)
    rule=np.array([[0,1,2,5],[0,1,5,4],[0,4,5,3]])
    a=prism[:,rule]
    w=basis[:,rule]
    loaded=geom[:,rule]
    det=lambda ar:np.linalg.det(ar[:,:,1:]-ar[:,:,:1])
    rest_det, loaded_det=det(a),det(loaded)
    jac=loaded_det/rest_det
    require(np.all(jac>0) and abs(abs(loaded_det).sum()/6-cert['volume_m3'])<1e-12,'Invalid reference transfer')
    from assemble_gluteal_material import write_kinematics
    output.mkdir(exist_ok=True)
    live_nodes=table('pelvic-reference-nodes','<f4',3).astype(float)
    live_nodes=posed(poses[128],live_nodes[:,0,:3]+live_nodes[:,1,:3])
    write_kinematics(output, data, fat_nodes, ref, parents, p1, muscle_faces, live_nodes,
                     donor_basis, donors, poses, sort, rule, a)
    # Positive degree-two quadrature integrates mass, centroid and inertia for constant density.
    # The declared, linearly interpolated donor partition is evaluated at the same mass points.
    quad=np.full((4,4),(5-np.sqrt(5))/20)
    np.fill_diagonal(quad,(5+3*np.sqrt(5))/20)
    points=np.einsum('qi,ftij->ftqj',quad,a).reshape(-1,3)
    share=np.einsum('qi,ftij->ftqj',quad,w).reshape(-1,len(donors))
    # Signed roundoff from authored boundary shape functions can be below zero.
    require(share.min()>-1e-10,'Negative material donor share')
    share=np.maximum(share,0);share/=share.sum(1)[:,None]
    text=material.read_text()
    rho,lower,upper=map(float,re.search(r'parameter density\s*:\s*kg/m\^3\s*=\s*([\d.]+)\s+in\s+\[([\d.]+),\s*([\d.]+)\]',text).groups())
    mass=np.repeat(abs(rest_det).ravel()*rho/24,4)
    total=mass.sum()
    local={d:Rotation.from_quat(poses[d]['quaternion_xyzw']).inv().apply(points-poses[d]['position_m']) for d in donors}
    output.mkdir(exist_ok=True)
    ledger=output/'mass-reference.txt'
    with ledger.open('w') as f:
        for i,mass_i in enumerate(mass):
            primary=donors[np.argmax(share[i])]
            print('node',i,primary,mass_i,*local[primary][i],file=f)
            for j,b in enumerate(donors):
                if share[i,j]>0:print('source',i,b,mass_i*share[i,j],*local[b][i],file=f)
    centroid=(mass @ points)/total
    delta=points-centroid
    cov=delta.T @ (mass[:,None]*delta)
    inertia=np.eye(3)*np.trace(cov)-cov
    report={'ledger':digest(ledger),'material':digest(material),'reference_nodes':digest(prior/'coupled/physical-reference.node'),
            'registration':cert['registration'],'donor_map':digest(node_file),'mass_points':len(points),
            'density_kg_m3':rho,'rest_volume_m3':total/rho,'registered_volume_m3':cert['volume_m3'],
            'total_kg':total,'density_sensitivity_kg':[total*lower/rho,total*upper/rho],
            'center_of_mass_world_m':centroid.tolist(),'inertia_about_com_kg_m2':inertia.tolist(),
            'donor_kg':{str(d):float(mass @ share[:,j]) for j,d in enumerate(donors)},
            'min_registration_J':float(jac.min()),'max_registration_J':float(jac.max())}
    require(abs(sum(report['donor_kg'].values())-total)<1e-13 and np.linalg.eigvalsh(inertia).min()>0,
            'Invalid mass or inertia')
    (output/'mass.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(report['total_kg'],report['donor_kg'], report['min_registration_J'],report['max_registration_J'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('registration','prior','assets','volume','material','output'):
        parser.add_argument(k,type=Path)
    run(**vars(parser.parse_args()))
