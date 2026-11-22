"""Check a rigid-donor construction on every tetrahedron before constructing a loaded full scene."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from evaluate_registered_bed_transfer import digest


def initial_body_targets(donor):
    if donor.is_dir():
        receipt=donor/'terminal.receipt.json'
        return json.loads(receipt.read_text())['initial_anatomical_registration']['body_poses'],dict(target_receipt_sha256=digest(receipt))
    # A rigid first placement is a construction input. Its donor receipt remains the
    # original accepted native reference; the new pose is checked against actual GPU FK.
    cfg=json.loads(donor.read_text());receipt=Path(cfg['placement_reference_receipt'])
    source=Path(cfg['placement_reference_launch'])
    if digest(receipt)!=cfg['placement_reference_receipt_sha256'] or digest(source)!=cfg['placement_reference_launch_sha256']:
        raise ValueError('changed native reference receipt or launch')
    old=json.loads(source.read_text())['argv'];new=cfg['argv']
    i=old.index('--root-pose')+1;j=new.index('--root-pose')+1
    if i!=j or old[:i]!=new[:j] or old[i+3:]!=new[j+3:]:raise ValueError('joint posture changed in rigid placement')
    delta=np.array(new[j:j+3],dtype='f4').astype(float)-np.array(old[i:i+3],dtype='f4').astype(float)
    if not np.array_equal(delta,cfg['construction_translation_m']):raise ValueError('root placement mismatch')
    rows=json.loads(receipt.read_text())['initial_anatomical_registration']['body_poses']
    for r in rows:r['position_m']=(np.asarray(r['position_m'])+delta).tolist()
    return rows,dict(target_receipt_sha256=digest(receipt),target_configuration_sha256=digest(donor),root_translation_m=delta.tolist())


def load_geometry(full, donor):
    source=json.loads((full/'launch.json').read_text())['environment']
    key='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    def input_file(file_key,hash_key):
        p=Path(source[key+file_key])
        if digest(p)!=source[key+hash_key]:raise ValueError('changed dense registration '+file_key)
        return p
    mesh=input_file('TETGEN_NODE','TETGEN_NODE_SHA256')
    elements=input_file('TETGEN_ELEMENTS','TETGEN_ELEMENTS_SHA256')
    mapping=input_file('COMBINED_NODE_MAP_CSV','COMBINED_NODE_MAP_SHA256')
    frames=input_file('DONOR_FRAMES_CSV','DONOR_FRAMES_SHA256')
    with mesh.open() as f:
        n,dim,nattr,nmarkers=map(int,next(f).split())
        if (dim,nattr,nmarkers)!=(3,0,0):raise ValueError('unexpected TetGen node schema')
        coords=np.loadtxt(f,dtype=np.float64)
    if coords.shape!=(n,4) or not np.array_equal(coords[:,0],np.arange(1,n+1)) or not np.isfinite(coords).all():
        raise ValueError('invalid TetGen node identity or coordinates')
    reference=np.ascontiguousarray(coords[:,1:])
    del coords
    with elements.open() as f:
        nt,corners,attrs=map(int,next(f).split())
        if (corners,attrs)!=(4,1):raise ValueError('unexpected TetGen element schema')
        elements_table=np.loadtxt(f,dtype=np.int32)
    if elements_table.shape!=(nt,6) or not np.array_equal(elements_table[:,0],np.arange(1,nt+1)):
        raise ValueError('invalid or missing element identities')
    mat=elements_table[:,5].copy();t=np.ascontiguousarray(elements_table[:,1:5]-1)
    if set(np.unique(mat))!={15,16} or t.min()<0 or t.max()>=n:raise ValueError('invalid tissue or node binding')
    del elements_table
    owner=np.full(n,-1);local=np.empty_like(reference)
    used=np.zeros(n,dtype=np.uint8);outer=[];shared=0
    with mapping.open() as f:
        for row in csv.DictReader(f):
            i=int(row['joint_node_id'])-1
            if not 0<=i<n or row['source_partition'] not in ('skin','fat'):
                raise ValueError('unrecognized skin/fat material node')
            role=1 if row['source_partition']=='skin' else 2
            if used[i]&role:raise ValueError('repeated source material node')
            used[i]|=role
            xyz=[float(row['donor_local_'+a+'_m']) for a in 'xyz']
            k=int(row['donor_body'])
            if owner[i]>=0:
                if row['registered_surface_layer']!='I' or owner[i]!=k or not np.array_equal(local[i],xyz):
                    raise ValueError('inconsistent bonded interface frame at '+str(i))
                shared+=1
                continue
            owner[i]=k
            local[i]=xyz
            if row['registered_surface_layer']=='O':outer.append(i)
    assert np.all(owner>=0)
    target,pose_identity=initial_body_targets(donor)
    poses={row['body_index']:row for row in target}
    with frames.open() as f:original={int(row['donor']):row for row in csv.DictReader(f)}
    posed=np.empty_like(reference);error=0.
    rotations=np.zeros((int(owner.max())+1,3,3))
    for k in np.unique(owner):
        sel=owner==k
        frame=original[k]
        R=np.array([float(frame[f'current_R{i}{j}']) for i in range(3) for j in range(3)]).reshape(3,3)
        p=np.array([float(frame['current_'+a]) for a in 'xyz'])
        rest_error=reference[sel]-(local[sel]@R.T+p)
        error=max(error,float(np.linalg.norm(rest_error,axis=1).max()))
        q=poses[k];R1=Rotation.from_quat(q['quaternion_xyzw']).as_matrix()
        rotations[k]=R1@np.linalg.inv(R)
        posed[sel]=(local[sel]+np.linalg.solve(R,rest_error.T).T)@R1.T+q['position_m']
    assert error<1e-6
    mask=np.zeros(len(reference),bool);mask[outer]=True
    pins=dict(shared_interface_rows=shared,max_source_reconstruction_m=error,reference_sha256=digest(mesh),
              elements_sha256=digest(elements),**pose_identity)
    return reference,posed,owner,t,mat,mask,rotations,pins

def measure(x,owner,t,mat):
    groups={}
    for m in np.unique(mat):
        stat=dict(cells=0,inverted=0,tiny_volume=0,poor_shape=0,mixed_donor=0,inverted_mixed_donor=0,
                  min_signed_volume_m3=1e10,min_normalized_shape=1e10)
        for b in range(0,len(t),32768):
            sl=slice(b,b+32768); f=t[sl][mat[sl]==m]
            if not len(f):continue
            X=x[f]; edge=X[:,1:,:]-X[:,:1,:]
            det=np.einsum('ij,ij->i',edge[:,0],np.cross(edge[:,1],edge[:,2]))
            edge2=np.maximum.reduce([((X[:,i]-X[:,j])**2).sum(1) for i in range(4) for j in range(i)])
            shape=det/(edge2*np.sqrt(edge2))
            if not np.isfinite(shape).all():raise ValueError('nonfinite material')
            mixed=np.any(owner[f]!=owner[f[:,:1]],axis=1)
            stat['mixed_donor']+=int(mixed.sum())
            stat['cells']+=len(f);stat['inverted']+=int((det<=0).sum())
            stat['inverted_mixed_donor']+=int(np.count_nonzero((det<=0)&mixed))
            stat['tiny_volume']+=int((det<=6e-18).sum());stat['poor_shape']+=int((shape<1e-5).sum())
            stat['min_signed_volume_m3']=min(stat['min_signed_volume_m3'],float(det.min()/6))
            stat['min_normalized_shape']=min(stat['min_normalized_shape'],float(shape.min()))
        groups[str(m)]=stat
    return groups

def main(full,donor,output):
    ref,posed,owner,t,mat,mask,_,pins=load_geometry(full,donor)
    groups={stage:measure(x,owner,t,mat) for stage,x in [('reference',ref),('posed',posed)]}
    output.write_text(json.dumps(dict(geometry_only=True,material=groups,**pins,
                      posed_outer_min_gap_m=float(posed[mask,2].min()),posed_outer_vertices_inside_plane=int((posed[mask,2]<0).sum())),indent=2)+'\n')
    print(groups,flush=True)
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
