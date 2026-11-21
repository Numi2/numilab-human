"""Bind cut-volume pressure quadrature to registered cortical or deep-adipose material owners."""
import csv,json,sys
from pathlib import Path
import numpy as np
from prepare_material_interface import read_inputs
from material_context import sha
from numilab_human.capsular_material_registration import read_material_banks

def gram(pts,parent):
    a=parent[:,1]-parent[:,0];b=parent[:,2]-parent[:,0];n=np.cross(a,b)
    nn=(n*n).sum(1);r=pts-parent[:,:1]
    if nn.min()<=0:raise ValueError('Degenerate parent facet')
    v=np.einsum('bki,bi->bk',np.cross(r,b[:,None,:]),n)/nn[:,None]
    w=np.einsum('bki,bi->bk',np.cross(a[:,None,:],r),n)/nn[:,None]
    return np.stack((1-v-w,v,w),axis=2)

def load_deep(env,faces,n):
    prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    paths={}
    for role in ('SURFACE_D_FACES','COMBINED_NODE_MAP'):
        p=Path(env[prefix+role+'_CSV']);paths[role]=p
        if sha(p)!=env[prefix+role+'_SHA256']:raise ValueError('Unverified '+role)
    incidence=np.empty_like(faces);i=0
    with paths['SURFACE_D_FACES'].open() as fh:
        for i,row in enumerate(csv.DictReader(fh),start=1):
            if int(row['face'])!=i-1:raise ValueError('Invalid D face sequence')
            incidence[i-1]=[int(row['v'+str(k)]) for k in range(3)]
    if i!=len(faces):raise ValueError('Incorrect full-volume boundary height')
    # Native pack compaction may reindex input vertices. Reconstruct the mapping
    # from every oriented triangle, including triangles outside these muscle ports.
    emitted=faces.flatten();original=incidence.flatten()
    lookup=np.full(n,-1,int);lookup[emitted]=original
    if not np.array_equal(lookup[emitted],original) or len(np.unique(lookup))!=n or min(lookup)<0:
        raise ValueError('Native boundary has different oriented topology')
    ids=np.full(lookup.max()+1,-1,int);xyz=np.full((len(ids),3),np.nan)
    with paths['COMBINED_NODE_MAP'].open() as fh:
        for row in csv.DictReader(fh):
            if row['registered_surface_layer']!='D':continue
            j=int(row['registered_surface_vertex_row'])
            if ids[j]!=-1:raise ValueError('Ambiguous D volume node')
            ids[j]=int(row['joint_node_id']);xyz[j]=[float(row['current_'+k+'_m']) for k in 'xyz']
    if min(ids)<1 or len(np.unique(ids))!=len(ids):raise ValueError('Missing D volume node')
    return ids[lookup],xyz[lookup],{k:sha(p) for k,p in paths.items()}

def run(config,receipt,capture,base,part,volume,out):
    out.mkdir(exist_ok=True);cfg=json.loads(config.read_text());surfs,inputs=read_inputs(config,receipt,capture,base)
    bone=read_material_banks(Path(cfg['argv'][3]));xdeep,fdeep=surfs['skin_deep'];node,rest,info=load_deep(cfg['environment'],fdeep,len(xdeep))
    result=dict(inputs=inputs,registration=info,deep_source_to_emitted_maximum_m=np.linalg.norm(rest-xdeep,axis=1).max(),sides={})
    for sid in (29,30):
        ip=volume/(str(sid)+'.npz');pd=np.load(ip);surface=np.load(part/('muscle_%d.npz'%sid))
        faces=surface['parent_face'][pd['port_partition_face']];names=surface['parent_bank'][pd['port_partition_face']]
        points=pd['port_vertices'];muscle_shape=pd['port_shape'];carrier=pd['cells'][pd['port_owner_cell']];xm=pd['vertices'][carrier].astype(float)
        nr=len(points);bshape=np.empty((nr,3,3));target=np.empty((nr,3,3));nodes=np.zeros((nr,3),int);bodies=np.full(nr,-1,int)
        for name in np.unique(names):
            ix=np.flatnonzero(names==name);x,f=surfs[str(name)];src=f[faces[ix]]
            target[ix]=x[src];bshape[ix]=gram(points[ix],target[ix])
            if name=='skin_deep':nodes[ix]=node[src]
            else:bodies[ix]=bone[2][int(name[5:])][0]
        def interp(b,p):return np.einsum('fki,fij->fkj',b,p)
        receiver=interp(bshape,target);sender=interp(muscle_shape,xm)
        err=lambda a,b:float(np.linalg.norm(a-b,axis=-1).max())
        stats=dict(facets=nr,rigid_body_ids=np.unique(bodies[bodies>=0]).tolist(),deep_fem_nodes=len(np.unique(nodes[bodies<0])),receiver_position_error_m=err(receiver,points),sender_position_error_m=err(sender,points),minimum_receiver_barycentric=bshape.min(),minimum_sender_barycentric=muscle_shape.min())
        if min(bshape.min(),muscle_shape.min()) < -2.e-7 or max(stats['receiver_position_error_m'],stats['sender_position_error_m'])>1.e-8:raise ValueError(stats)
        # Independent nonuniform pressures exercise every facet with no signed cancellation assumption.
        # Integrate linear pressure against the linear test functions exactly on each patch.
        av=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])*.5
        pressure=3000*(1+.17*np.sin(points[:,:,0]*27)+.21*np.cos(points[:,:,1]*33)+.05*np.sin(points[:,:,2]*21))
        coeff=(pressure+pressure.sum(1)[:,None])/12
        fa=np.einsum('fk,fki,fj->fij',coeff,muscle_shape,av)
        fb=-np.einsum('fk,fki,fj->fij',coeff,bshape,av)
        total=np.linalg.norm(fa.sum(1),axis=1).sum()
        force=fa.sum(1)+fb.sum(1);torque=np.cross(xm,fa).sum(1)+np.cross(target,fb).sum(1)
        rng=np.random.default_rng(sid);va=rng.normal(size=xm.shape);vb=rng.normal(size=target.shape)
        dual=(va*fa).sum((1,2))+(vb*fb).sum((1,2))
        direct=np.einsum('fk,fkj,fj->f',coeff,interp(muscle_shape,va)-interp(bshape,vb),av)
        stats.update(force_residual_n=np.linalg.norm(force,axis=1).max(),moment_residual_Nm=np.linalg.norm(torque,axis=1).max(),virtual_power_residual_w=max(abs(dual-direct)),applied_sum_of_facet_force_norms_N=total)
        if max(stats[k] for k in ('force_residual_n','moment_residual_Nm','virtual_power_residual_w'))>1.e-9:raise ValueError(stats)
        output=out/(str(sid)+'.npz')
        np.savez_compressed(output,muscle_nodes=carrier.astype('u4'),target_shape=bshape,target_joint_node_id=nodes.astype('u4'),target_body=bodies.astype('i4'),target_bank=names,target_face=faces.astype('u4'),target_triangle=target.astype('f4'))
        stats.update(input_sha256=sha(ip),output_sha256=sha(output));result['sides'][sid]=stats
        print(sid,stats,flush=True)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
