"""Make a one-sided construction boundary for the finite precision of source registration.

Only the Boolean cutting tool moves. Anatomical and physical cortex coordinates
stay in the source banks. IDs and triangle connectivity stay identical.
"""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from prepare_posterior_muscles import mesh_volume
import manifold3d as m
from prepare_material_interface import read_inputs,solid,close_remote_boundary
from prepare_posterior_muscles import closed
from material_context import sha
def cutting_surface(vertices,faces,roi):
    v,indices=np.unique(vertices.astype(float),axis=0,return_inverse=True);t=indices[faces]
    tris=v[t];n=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]);n/=np.linalg.norm(n,axis=1)[:,None]
    if mesh_volume(v,t)<0:n=-n
    # Cover two independent roundings at each side of a source interface.
    # Precision is measured in the registered world frame on both input surfaces.
    magnitude=np.maximum(abs(v).max(0),abs(roi).max(0)).astype('f4')
    spacing=np.spacing(magnitude).astype(float);guard=2*np.linalg.norm(spacing)
    moving_faces=((tris.max(1)>=roi[0]-guard)&(tris.min(1)<=roi[1]+guard)).all(1)
    active=np.zeros(len(v),bool);active[t[moving_faces].ravel()]=True
    acc=np.zeros_like(v);minimum=np.full(len(v),np.inf)
    for k in range(3):
        a=tris[:,(k+1)%3]-tris[:,k];b=tris[:,(k+2)%3]-tris[:,k]
        angle=np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.einsum('ij,ij->i',a,b))
        np.add.at(acc,t[:,k],angle[:,None]*n)
    length=np.linalg.norm(acc,axis=1);acc/=np.maximum(length,1e-300)[:,None]
    for k in range(3):np.minimum.at(minimum,t[:,k],np.einsum('ij,ij->i',acc[t[:,k]],n))
    displacement=np.zeros_like(v);ordinary=active&(minimum>.1)
    displacement[ordinary]=acc[ordinary]*(guard/minimum[ordinary])[:,None]
    for i in np.flatnonzero(active&~ordinary):
        incident=n[(t==i).any(1)];A=np.vstack((np.column_stack((-incident,np.zeros(len(incident)))),np.column_stack((np.vstack((np.eye(3),-np.eye(3))),-np.ones(6)))))
        opt=linprog([0,0,0,1],A_ub=A,b_ub=np.r_[-np.ones(len(incident)),np.zeros(6)],bounds=[(None,None)]*3+[(0,None)])
        if not opt.success:raise ValueError(('no feasible local cortical envelope',i))
        displacement[i]=guard*opt.x[:3]
    check=np.einsum('fkj,fj->fk',displacement[t[moving_faces]],n[moving_faces])
    if check.min()<guard*(1-1e-8):raise ValueError('one-sided bound lost')
    q=(v+displacement)[indices];res=dict(guard_m=float(guard),source_ulp_xyz_m=spacing.tolist(),
        moving_vertices=int(sum(active)),moving_faces=int(sum(moving_faces)),max_displacement_m=float(np.linalg.norm(displacement,axis=1).max()),
        minimum_projected_offset_m=float(check.min()))
    return q,res
def run(cfg,receipt,capture,base,out):
    out.mkdir(exist_ok=True);data,lineage=read_inputs(cfg,receipt,capture,base)
    glute=np.vstack((data['muscle_29'][0],data['muscle_30'][0]));roi=np.stack((glute.min(0),glute.max(0)))
    objs={};parent={};result={}
    for name,(v,f) in data.items():
        if name.startswith('skin_') and name!='skin_deep':continue
        if name=='skin_deep':v,f,_=close_remote_boundary(v,f,roi[1])
        if name.startswith('bone_'):
            v,bound=cutting_surface(v,f,roi);result[name]=bound
        a,i,_=solid(v,f,name)
        if a.status()!=m.Error.NoError or a.is_empty():raise ValueError(('invalid source solid',name))
        objs[name]=a;parent[i]=name
    reports={}
    for sid,bones in [(29,(1,2,18)),(30,(1,3,19))]:
        name=f'muscle_{sid}';a=objs[name]
        for b in bones:a=a-objs[f'bone_{b}']
        a=a^objs['skin_deep']
        if a.status()!=m.Error.NoError or a.is_empty():raise ValueError(name)
        surface=a.to_mesh64();v=surface.vert_properties[:,:3];f=np.asarray(surface.tri_verts,int)
        labels=np.array([parent[k] for k in np.repeat(surface.run_original_id,np.diff(surface.run_index)//3)])
        ancestry=np.array(surface.face_id);assert not ((labels=='skin_deep')&(ancestry>=len(data['skin_deep'][1]))).any()
        eq=closed(v.astype('f4'),f)
        if any(eq.values()):raise ValueError(('rounded material topology',sid,eq))
        dest=out/(name+'.npz');np.savez_compressed(dest,vertices=v,triangles=f,parent_bank=labels,parent_face=ancestry)
        reports[name]=dict(volume_ml=a.volume()*1e6,parents={str(k):int(sum(labels==k)) for k in np.unique(labels)},topology=eq,sha256=sha(dest))
    (out/'source.json').write_text(json.dumps(dict(lineage=lineage,construction_bounds=result,material=reports),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
