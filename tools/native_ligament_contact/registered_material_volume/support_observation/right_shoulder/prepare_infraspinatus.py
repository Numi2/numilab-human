"""Construct an anatomically registered candidate volume for the dominant shoulder support site."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation
from pose_glutes import parts
from compile_connected_joint import digest


def dets(x,t):
    return np.linalg.det(x[t[:,1:]]-x[t[:,:1]])


def exterior(t):
    all_faces=np.concatenate([t[:,[0,2,1]],t[:,[0,1,3]],t[:,[1,2,3]],t[:,[2,0,3]]])
    faces,counts=np.unique(np.sort(all_faces,axis=1),return_counts=True,axis=0)
    if np.max(counts)!=2:
        raise ValueError('Nonmanifold cell face')
    return faces[counts==1]


def source_mesh(case):
    cfg=json.loads((case/'launch.json').read_bytes());a=cfg['argv']
    path=Path(a[a.index('--soft-tissue-payload')+1])
    v,w,donors,t,aff=parts(path,83)
    if donors!=[41,34]:raise ValueError('Unexpected MyoSim donors')
    x,ii,inv=np.unique(v,axis=0,return_index=True,return_inverse=True)
    if np.max(abs(w[ii][inv]-w))>1.e-7:
        raise ValueError('Conflicting mechanical weights across source seam')
    return path,x,inv[t],w[ii],aff,donors


def prepare(case,baseline,tool,out):
    out.mkdir(exist_ok=True)
    path,x,f,w,aff,donors=source_mesh(case)
    expected=json.loads(baseline.read_bytes())
    observed=expected['right_infraspinatus']
    if observed['source_payload']!=digest(path) or expected['dominant_skin_node']!=24331:
        raise ValueError('Wrong anatomical registration')
    head=[f'{len(x)} 3 0 0']
    head.extend(f"{i+1} "+' '.join(format(v,'.17g') for v in row) for i,row in enumerate(x))
    head.append(f'{len(f)} 1')
    for i,row in enumerate(f):
        head.extend([f'1 0 {i+1}',f'3 {row[0]+1} {row[1]+1} {row[2]+1}'])
    head+=['0','0']
    (out/'source.poly').write_text('\n'.join(head)+'\n')
    cmd=[str(tool),'-pYq1.4Q','source.poly']
    with (out/'tetgen.log').open('w') as stream:
        subprocess.run(cmd,cwd=out,stdout=stream,stderr=subprocess.STDOUT,check=True)
    nodes=np.loadtxt(out/'source.1.node',skiprows=1);raw=np.loadtxt(out/'source.1.ele',skiprows=1,dtype=np.int32)
    pts=nodes[:,1:4];t=raw[:,1:5]-1
    if (not np.array_equal(pts[:len(x)],x) or not np.array_equal(nodes[:,0],np.arange(1,len(pts)+1))
        or set(map(tuple,exterior(t)))!=set(map(tuple,np.sort(f,axis=1)))):
        raise ValueError('Source coordinates, boundary or winding changed')
    d=dets(pts,t)
    if np.min(d)<=0:raise ValueError('Invalid TetGen cells')
    inv=np.linalg.inv(pts[t[:,1:]]-pts[t[:,:1]])
    grad=np.concatenate([-inv.sum(2,keepdims=True),inv],axis=2).transpose(0,2,1)
    elem=(grad@grad.transpose(0,2,1))*(d/6)[:,None,None]
    row,col=np.broadcast_arrays(t[:,:,None],t[:,None,:])
    L=coo_matrix((elem.ravel(),(row.ravel(),col.ravel())),shape=(len(pts),len(pts))).tocsc()
    weights=np.zeros((len(pts),2));weights[:len(x)]=w
    if len(pts)>len(x):
        weights[len(x):]=spsolve(L[len(x):,len(x):],-L[len(x):,:len(x)]@w)
    if weights.min()< -1e-7 or weights.max()>1.0000001 or np.max(abs(weights.sum(1)-1))>2e-6:
        raise ValueError('Interior donor coordinates invalid')
    outdata={};native_positions={}
    frame=json.loads(next((case/'native-traces/accepted-geometry').glob('*.receipt.json')).read_bytes())
    for name,poses in [('opening',frame['initial_anatomical_registration']['body_poses']),('accepted',frame['accepted_body_poses'])]:
        pose={r['body_index']:r for r in poses}
        moved=np.zeros_like(pts)
        for k,b in enumerate(donors):
            r=Rotation.from_quat(pose[b]['quaternion_xyzw']).as_matrix()
            A=r@aff[k][0];B=pose[b]['position_m']+r@aff[k][1]
            moved+=weights[:,k,None]*(pts@A.T+B)
        J=dets(moved,t)/d
        # A second candidate holds the measured boundary and minimizes the
        # scalar Dirichlet energy of each coordinate component in the interior.
        harm=moved.copy()
        if len(pts)>len(x):
            delta=moved[:len(x)]-pts[:len(x)]
            harm[len(x):]=pts[len(x):]+spsolve(L[len(x):,len(x):],-L[len(x):,:len(x)]@delta)
        hJ=dets(harm,t)/d
        outdata[name]=dict(min_J=float(J.min()),max_J=float(J.max()),inverted=int((J<=0).sum()),
                           harmonic_min_J=float(hJ.min()),harmonic_max_J=float(hJ.max()),harmonic_inverted=int((hJ<=0).sum()),
                           volume_m3=float(dets(moved,t).sum()/6))
        native_positions[name]=moved;native_positions[name+'_harmonic']=harm
    source_volume=d.sum()/6
    if (abs(source_volume-observed['atlas_volume_m3'])>1e-11 or
        abs(outdata['accepted']['volume_m3']-observed['native_display_volume_m3'])>1e-8):
        raise ValueError('Volume does not reproduce source and captured surface')
    np.savez_compressed(out/'registration.npz',ref=pts,tets=t,boundary=f,weights=weights,body_ids=donors,**native_positions)
    report=dict(stable_id=83,source_payload=digest(path),source_row='FJ1500 right infraspinatus',skin_node=int(expected['dominant_skin_node']),
                source_volume_m3=float(source_volume),ref_nodes=len(pts),source_surface_nodes=len(x),source_facets=len(f),cells=len(t),
                fixed_source_facet_count=len(exterior(t)),tetgen_sha256=digest(tool),command=cmd,capture=expected['accepted_receipt'],
                volume_registration=digest(out/'registration.npz'),poses=outdata,
                scope='Source-derived volume candidate. Muscle activation remains with MyoSim. A native force-bearing shoulder interface requires a registered external attachment and a nonoverlapping volume.')
    (out/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(outdata,flush=True)

if __name__=='__main__':prepare(*(Path(v).resolve() for v in sys.argv[1:]))
