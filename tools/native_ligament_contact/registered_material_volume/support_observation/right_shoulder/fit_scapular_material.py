"""Resolve cortical occupancy by moving the connected material while conserving its reference volume."""
from pathlib import Path
import sys,json
import numpy as np
import manifold3d as md
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import factorized
from scipy.spatial.transform import Rotation
from prepare_material_interface import solid
from rebind_ligament_material import banks
from compile_connected_joint import digest
from prepare_infraspinatus import dets


def forces(x,f,clipped,tag):
    """Shape derivative of intersection volume on the moving surface."""
    answer=np.zeros_like(x)
    if clipped.is_empty():return answer
    mesh=clipped.to_mesh64()
    keep=np.repeat(mesh.run_original_id,np.diff(mesh.run_index)//3)==tag
    idx=np.asarray(mesh.face_id)[keep]
    if not len(idx):return answer
    corners=np.asarray(mesh.vert_properties)[:,:3][np.asarray(mesh.tri_verts)[keep]]
    source=x[f[idx]]
    e=source[:,1:]-source[:,:1];d=corners.mean(1)-source[:,0]
    gram=e@e.transpose(0,2,1)
    u=np.linalg.solve(gram,(e@d[:,:,None]))[:,:,0]
    weights=np.column_stack((1-u.sum(1),u))
    normal=np.cross(corners[:,1]-corners[:,0],corners[:,2]-corners[:,0])/2
    np.add.at(answer,f[idx].ravel(),(weights[:,:,None]*normal[:,None,:]).reshape(-1,3))
    return answer


def cortex(case,poses,origin):
    cfg=json.loads((case/'launch.json').read_bytes());bones=Path(cfg['argv'][3]);surfs=banks(bones)
    p={r['body_index']:r for r in poses};result={}
    for b,k in [(34,10),(41,12)]:
        tri,v,_=surfs[b,k];r=p[b]
        bx=(Rotation.from_quat(r['quaternion_xyzw']).apply(v)+r['position_m']).astype('f4').astype(float)
        result[k]=solid((bx-origin)*1000,tri,f'bone{k}')[0]
    return result


def fit(case,record,out):
    out.mkdir(exist_ok=True)
    z=dict(np.load(record));t=z['tets'];f=z['boundary'];origin=z['opening'].mean(0)
    y=(z['opening']-origin)*1000;trial0=y.copy()
    frame=json.loads(next((case/'native-traces/accepted-geometry').glob('*.receipt.json')).read_bytes())
    bone=cortex(case,frame['initial_anatomical_registration']['body_poses'],origin)
    det=dets(y,t);inv=np.linalg.inv(y[t[:,1:]]-y[t[:,:1]])
    g=np.concatenate([-inv.sum(2,keepdims=True),inv],axis=2).transpose(0,2,1)
    ij,ji=np.broadcast_arrays(t[:,:,None],t[:,None,:]);E=(g@g.transpose(0,2,1))*(det/6)[:,None,None]
    mass=np.bincount(t.ravel(),weights=np.repeat(det/24,4))
    # Sobolev regularization is solely a registration metric in millimetres.
    ell=np.median(np.linalg.norm(y[f[:,1]]-y[f[:,0]],axis=1))*2
    L=coo_matrix((E.ravel(),(ij.ravel(),ji.ravel())),shape=(len(y),)*2).tocsc()+diags(mass/(ell*ell))
    solve=factorized(L.tocsc())
    V0=det.sum()/6
    def evaluate(pos,grad=False):
        m,tag,_=solid(pos,f,'muscle83')
        if m.status()!=md.Error.NoError:raise ValueError('Invalid muscle geometry')
        cut=[m^b for b in bone.values()]
        if any(c.status()!=md.Error.NoError for c in cut):raise ValueError('Invalid cortical intersection')
        v=np.array([a.volume() for a in cut]+[m.volume()-V0])
        if not grad:return v
        derivatives=[forces(pos,f,c,tag) for c in cut]
        full=np.zeros_like(pos);face=pos[f]
        normals=np.cross(face[:,1]-face[:,0],face[:,2]-face[:,0])/6
        for c in range(3):np.add.at(full,f[:,c],normals)
        return v,np.array(derivatives+[full])
    startv,initialG=evaluate(y,True)
    direction=np.random.default_rng(83).normal(size=y.shape)
    eps=1.e-5
    measured=(evaluate(y+eps*direction)-evaluate(y-eps*direction))/(2*eps)
    predicted=np.sum(direction*initialG,axis=(1,2))
    if not np.allclose(measured,predicted,rtol=1e-6,atol=5e-7):raise ValueError(('Volume derivative mismatch',measured,predicted))
    log=[]
    for it in range(55):
        v,G=evaluate(y,True)
        log.append(dict(iteration=it,volume_errors_mm3=v.tolist(),min_relative_J=float((dets(y,t)/det).min())))
        if np.max(abs(v))<1e-8:break
        active=(abs(v)>1e-10);active[-1]=True
        A=G[active];H=np.array([solve(g) for g in A])
        gram=A.reshape(len(A),-1)@H.reshape(len(A),-1).T
        mag=np.sqrt(np.maximum(gram.diagonal(),1e-30))
        coeff=np.linalg.lstsq(gram/mag[:,None]/mag[None,:],-v[active]/mag,rcond=1e-11)[0]/mag
        delta=np.einsum('a,aij->ij',coeff,H)
        for k in range(18):
            alpha=2.**(-k);yn=y+alpha*delta
            if np.min(dets(yn,t)/det)<.02:continue
            vv=evaluate(yn)
            if np.linalg.norm(vv)<np.linalg.norm(v):
                y=yn;log[-1]['alpha']=alpha;break
        else:
            log[-1]['stop']='positive-cell or volume decrease failed';break
    disp=(y-trial0)/1000
    observed={}
    for name,pose in [('opening',frame['initial_anatomical_registration']['body_poses']),('accepted',frame['accepted_body_poses'])]:
        x=z[name]+disp
        solid_m,_,_=solid((x-origin)*1000,f,'muscle83')
        b=cortex(case,pose,origin)
        c={str(k):float(abs((solid_m^s).volume())) for k,s in b.items()}
        observed[name]=dict(overlap_mm3=c,measured_volume_mm3=solid_m.volume(),min_source_J=float((dets(x,t)/dets(z['ref'],t)).min()),
                            min_relative_J=float((dets(x,t)/dets(z[name],t)).min()))
    r=dict(parent=digest(record),start_intersection_mm3=startv.tolist(),volume_gradient_relative=float(np.linalg.norm(measured-predicted)/np.linalg.norm(measured)),
           smoothing_length_mm=float(ell),frozen_vertices=0,peak_registration_mm=float(np.linalg.norm(disp,axis=1).max()*1000),
           rms_registration_mm=float(np.sqrt(np.mean(np.sum(disp**2,axis=1)))*1000),stages=observed,log=log)
    np.savez_compressed(out/'fitted.npz',displacement_m=disp,source_cells=t)
    (out/'fit.json').write_text(json.dumps(r,indent=2)+'\n')
    print(observed,log[-1],flush=True)

if __name__=='__main__':fit(*(Path(s).resolve()for s in sys.argv[1:]))
