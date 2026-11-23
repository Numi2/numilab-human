"""Measure the actual three source-boundary heights along each overloaded skin ray."""
import csv,json,sys
from pathlib import Path
import numpy as np
from inspect_bed_load_paths import intersections
from material_context import sha

def measure(case,paths,opening,sample_all=None):
    cfg=json.loads((case/'launch.json').read_text());env=cfg['environment']
    prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    def pin(name,hash_name):
        p=Path(env[prefix+name])
        if sha(p)!=env[prefix+hash_name]:raise ValueError('Registration changed')
        return p
    ref=pin('TETGEN_NODE','TETGEN_NODE_SHA256');coords=np.loadtxt(ref,skiprows=1,usecols=(1,2,3)).astype('f4')
    nodes=pin('COMBINED_NODE_MAP_CSV','COMBINED_NODE_MAP_SHA256')
    rows={k:{} for k in 'OID'}
    with nodes.open() as f:
        for r in csv.DictReader(f):
            k=r['registered_surface_layer']
            if k in rows:
                local=int(r['registered_surface_vertex_row']);global_row=int(r['joint_node_id'])-1
                if local in rows[k] and (k!='I' or rows[k][local]!=global_row):raise ValueError('Ambiguous boundary row')
                rows[k][local]=global_row
    out=json.loads(paths.read_text())
    from scipy.spatial.transform import Rotation
    registration=json.loads((case/'registered-initial-bodies.json').read_text())['all_bodies']
    rc=json.loads((opening/'terminal.receipt.json').read_text())
    if rc['base_pack_file_sha256']!=out['pack']: raise ValueError('Different native anatomical assembly')
    current=sorted(rc['initial_anatomical_registration']['body_poses'],key=lambda row:row['body_index'])
    if [r['body_index'] for r in current]!=list(range(len(registration))):raise ValueError('Incomplete opening poses')
    source_rotation=Rotation.from_quat([r['current']['orientation_world_xyzw'] for r in registration]).as_matrix()
    source_translation=np.array([r['current']['position_world_m'] for r in registration])
    target_rotation=Rotation.from_quat([r['quaternion_xyzw'] for r in current]).as_matrix()
    target_translation=np.array([r['position_m'] for r in current])
    errors={}
    for k in 'OID':
        r=rows[k]
        if sorted(r)!=list(range(len(r))):raise ValueError('Incomplete layer indices')
        v=coords[[r[i] for i in range(len(r))]]
        verts=pin('SURFACE_'+k+'_NODES_CSV','SURFACE_'+k+'_NODES_SHA256')
        table=np.loadtxt(verts,delimiter=',',skiprows=1,usecols=(0,1,2,3,4,8,9,10))
        if not np.array_equal(table[:,0],np.arange(len(v))):raise ValueError('Invalid source surface order')
        owner=table[:,1].astype(int);loc=table[:,5:8]
        loaded=np.einsum('vij,vj->vi',source_rotation[owner],loc)+source_translation[owner]
        # The registered surface CSV can predate a local repair to the TetGen
        # nodes. Preserve that discrepancy as a physical offset in donor space.
        err=np.linalg.norm(loaded-table[:,2:5],axis=1)
        delta=v.astype(float)-loaded
        local_delta=np.einsum('vji,vj->vi',source_rotation[owner],delta)
        v=np.einsum('vij,vj->vi',target_rotation[owner],loc+local_delta)+target_translation[owner]
        if err.max()>1e-6:raise ValueError('Source registration failed to reproduce the native donor frame')
        errors[k]=dict(source_table_vs_dense_donor_max_m=float(err.max()),tetgen_vs_table_max_m=float(np.linalg.norm(delta,axis=1).max()))
        face=pin('SURFACE_'+k+'_FACES_CSV','SURFACE_'+k+'_FACES_SHA256');f=np.loadtxt(face,delimiter=',',skiprows=1,usecols=(1,2,3),dtype='i4')
        if sample_all is not None:sample_all(k,v,f)
        for data in out['witnesses'].values():
            hits=intersections(v,f,np.array(data['point']))
            data.setdefault('full_volume_reference',{})[k]=[dict(dz_m=z,source_face=i) for z,i in hits]
    out['physical_reference']=sha(ref)
    out['source_frame_evaluation']=errors
    out['target_pose_receipt']=sha(opening/'terminal.receipt.json')
    out['registered_boundary_scope']='Unloaded full-volume reference surfaces registered by their anatomical donor into the measured opening skeleton. This is a geometric comparison, not the deformed dense endpoint.'
    return out
if __name__=='__main__':print(json.dumps(measure(*(Path(s) for s in sys.argv[1:])),indent=2))
