"""Compare a source-parent embedding to the measured motion of the same native mesh."""
import json
import re
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix,diags
from scipy.sparse.linalg import splu
from compile_connected_joint import digest
from supported_material_admission import geom


def bary_matrix(t,w,size):
    # SciPy compacts indices and values in place when it drops exact zeros.
    # The registration arrays are also used to compute the next pose.
    matrix=csr_matrix((w.ravel().copy(),t.ravel().copy(),np.arange(0,4*len(t)+1,4)),shape=(len(t),size))
    matrix.eliminate_zeros()
    return matrix


def fit_increment(matrix,mass,delta,prior):
    """Fit increments only. Nullspace components stay at the supplied prior."""
    gram=matrix.T @ matrix.multiply(mass[:,None])
    diagonal=gram.diagonal()
    seen=diagonal>0
    if not np.any(seen):
        raise ValueError('No active source nodes')
    scale=1/np.sqrt(diagonal[seen])
    M=diags(scale)
    unit=(M @ gram[seen][:,seen] @ M).tocsc()
    unit.setdiag(unit.diagonal()+1e-12)
    rhs=matrix.T @ (mass[:,None]*(delta-matrix @ prior))
    sol=prior.copy()
    sol[seen]+=scale[:,None]*splu(unit).solve(scale[:,None]*rhs[seen])
    return sol


def audit_field(x,u,y,t,material,regions):
    report={}
    for tissue,tag in [('skin',15),('fat',16)]:
        cells=np.flatnonzero(material==tag)
        by_region=np.zeros((int(regions.max())+1,3))
        inverted=0
        low=0
        for i in range(0,len(cells),16384):
            ids=cells[i:i+16384]
            tc=t[ids]
            ref=x[tc]
            edges=ref[:,1:]-ref[:,:1]
            volume=np.abs(np.linalg.det(edges))/6
            def strain(field):
                d=field[tc]
                H=np.linalg.solve(edges,d[:,1:]-d[:,:1])
                return .5*(H+H.transpose(0,2,1)+H @ H.transpose(0,2,1))
            true=strain(u)
            predicted=strain(y)
            np.add.at(by_region,regions[ids],np.column_stack((volume,volume*np.sum(true*true,axis=(1,2)),volume*np.sum((true-predicted)**2,axis=(1,2)))))
            v,q=geom(x+y,tc)
            inverted+=int((v<=0).sum())
            low+=int((q<1e-5).sum())
        vs,ns,err=by_region.sum(axis=0)
        report[tissue]=dict(inverted=inverted,below_shape=low,relative_incremental_strain_error=float(np.sqrt(err/ns)),rms_reference_incremental_strain=float(np.sqrt(ns/vs)),rms_incremental_strain_error=float(np.sqrt(err/vs)),regions=[dict(region=int(k),rms_reference=float(np.sqrt(a[1]/a[0])),rms_error=float(np.sqrt(a[2]/a[0]))) for k,a in enumerate(by_region) if a[0]>0])
    return report


def run(work):
    stage=work/'witness-material'
    vol=stage/'volume'
    evidence=json.loads((stage/'accepted-motion.json').read_bytes())
    cap=work/'accepted-material-07/physical-accepted.node'
    if digest(cap)!=evidence['file'] or digest(vol/'current.node')!=evidence['initial_file']:
        raise ValueError('Motion input mismatch')
    src=json.loads((stage/'current-p1/current-paired-volume-state.json').read_bytes())['source330_identity']
    def parent(k,cols,dtype='f8'):
        p=Path(src[k]['path'])
        if digest(p)!=src[k]['sha256']:
            raise ValueError('Unbound source parent')
        return np.loadtxt(p,delimiter=',',skiprows=1,usecols=cols,dtype=dtype)
    load=lambda p:np.loadtxt(p,skiprows=1,usecols=(1,2,3))
    x=load(vol/'current.node');u=load(cap)-x
    mass=np.fromfile(vol/'node-mass.f64','<f8')
    tets=np.loadtxt(vol/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    fine=tets[:,:4]-1
    reg=np.loadtxt(vol/'current-child.labels.csv',skiprows=1,delimiter=',',usecols=(3,),dtype='i4')
    table=stage/'binding/tables/combined-node-map.csv'
    names=table.open().readline().strip().split(',')
    cols=[names.index(k) for k in ('joint_node_id','parent_tet_row','parent_bary0','parent_bary1','parent_bary2','parent_bary3')]
    binding=np.loadtxt(table,delimiter=',',skiprows=1,usecols=cols)
    node=binding[:,0].astype('i4')-1
    count=np.bincount(node,minlength=len(x))
    if np.any(count<1) or np.any(count>2):
        raise ValueError('Missing or ambiguous registered source carrier')
    p=parent('source330_nodes',(2,3,4))
    pt=parent('source330_tets',(3,4,5,6),'i4')
    role_tets=binding[:,1].astype('i4');controls=pt[role_tets]
    B=bary_matrix(controls,binding[:,2:],len(p))
    initial_error=x[node]-B @ p
    # A source vertex with tiny observed barycentric mass can take an arbitrary
    # displacement in an unconstrained least-squares registration. Partition
    # unity gives a bounded translational update in the measured body frame.
    row_mass=mass[node]/count[node]
    carried=B.T @ row_mass
    offset=B.T @ (row_mass[:,None]*initial_error)
    q=p+np.divide(offset,carried[:,None],out=np.zeros_like(offset),where=carried[:,None]>0)
    # Establish affine completeness at the actual loaded surface, rather than
    # driving a late repair using an old fixed normal offset.
    src_edges=lambda pts:pts[pt[:,1:]]-pts[pt[:,:1]]
    E=src_edges(q);old=src_edges(p)
    det=np.linalg.det(E)
    sv=np.linalg.svd(E,compute_uv=False)
    good=(det*np.linalg.det(old)>0)&(sv[:,-1]>sv[:,0]*1e-5)
    inv=np.zeros_like(E);inv[good]=np.linalg.inv(E[good])
    w3=np.einsum('ij,ijk->ik',x[node]-q[controls[:,0]],inv[role_tets])
    w=np.column_stack((1-w3.sum(1),w3))
    reconstruction=np.linalg.norm(np.einsum('ij,ijk->ik',w,q[controls])-x[node],axis=1)
    if np.max(reconstruction[good[role_tets]])>1e-8:
        raise ValueError('Parent barycentric coordinates lost affine completeness')
    valid=(reconstruction<=1e-10)&good[role_tets]&(w.min(1)>=-1e-8)&(w.max(1)<=1+1e-8)
    A=bary_matrix(controls,w,len(p))
    lo=np.full(len(x),len(node),'i4');hi=np.full(len(x),-1,'i4')
    seq=np.arange(len(node),dtype='i4');np.minimum.at(lo,node,seq);np.maximum.at(hi,node,seq)
    # When the bonded side maps to incompatible material traces keep the node fine.
    err=A[lo]-A[hi]
    agree=np.asarray(err.multiply(err).sum(axis=1)).ravel()<1e-16
    fit=valid[lo]&valid[hi]&agree
    eligible_count=int(fit.sum())
    bed=json.loads((work.parent/'gravity-contact/geometry.json').read_bytes())
    sep=(x @ np.array(bed['normal']))-bed['normal'][2]*bed['height_m']
    minimum=np.full(len(x),reg.max()+1,'i4'); maximum=np.full(len(x),-1,'i4')
    for c in fine.T:
        np.minimum.at(minimum,c,reg);np.maximum.at(maximum,c,reg)
    app=stage/'binding/app/NumiHumanLoadedSkinPoseData.inc'
    anchors=[int(n)for n in re.findall(r'^\{\d+u,(\d+)u,\d+u,\{',app.read_text(),flags=re.M)]
    if len(anchors)!=json.loads((stage/'binding/build.json').read_bytes())['anchors']:
        raise ValueError('Incomplete physical anchor list')
    coupled=(sep<.002)|(minimum!=maximum)
    coupled[np.load(vol/'material-walls.npz')['C'].ravel()]=True
    coupled[anchors]=True
    fit &= ~coupled
    nodes=np.flatnonzero(fit);local=A[lo[nodes]]
    if not len(nodes):raise ValueError('No eligible source-coordinate rows')
    dc=fit_increment(local,mass[nodes],u[nodes],np.zeros_like(q))
    proposed=u.copy();proposed[nodes]=local @ dc
    gap=np.linalg.norm(local @ q-x[nodes],axis=1)
    if max(gap)>1e-9:
        raise ValueError(f'Embedding cannot reconstruct bed pose at {nodes[gap.argmax()]}: {gap.max():.9g}, largest source coordinate {abs(q).max():.9g}, percentile gaps {np.quantile(gap,[.5,.95,1])}')
    used=np.unique(local.indices)
    from scipy.spatial.transform import Rotation
    rot=Rotation.from_rotvec([.12,-.13,.18]);offset=np.array([.02,.13,-.11])
    rigid_error=np.linalg.norm(local @ (rot.apply(q)+offset)-(rot.apply(x[nodes])+offset),axis=1)
    rng=np.random.default_rng(0)
    f=rng.standard_normal((len(nodes),3));dq=rng.standard_normal(q.shape)
    g=local.T @ f
    torque=np.cross(q,g).sum(0)-np.cross(x[nodes],f).sum(0)
    virtual=abs(np.sum(f*(local @ dq))-np.sum(g*dq))
    if max(rigid_error)>1e-9 or np.linalg.norm(torque)>1e-7 or virtual>1e-8:
        raise ValueError('Work or torque transfer mismatch')
    err=np.linalg.norm(proposed-u,axis=1)
    field=audit_field(x,u,proposed,fine,tets[:,4],reg)
    reply=dict(mesh=digest(vol/'mesh.ele'),position=digest(vol/'current.node'),capture=digest(cap),registrations=digest(table),source330={k:src[k]['sha256'] for k in ('source330_nodes','source330_tets')},native_seconds=evidence['accepted_s'],source_parent_nodes=len(p),source_parents=len(pt),registered_fine_nodes=len(x),rows_measured=len(node),prior_registration_error_quantiles_mm=(np.quantile(np.linalg.norm(initial_error,axis=1),[.5,.95,1])*1000).tolist(),bound_parent_displacement_m=float(np.linalg.norm(q-p,axis=1).max()),unreconstructible_binding_rows=int((reconstruction>1e-10).sum()),refitted_parent_tets_folded=int((det*np.linalg.det(old)<=0).sum()),eligible_fine_rows_before_interface_protection=eligible_count,embedded_nodes=len(nodes),active_parent_nodes=len(used),control_node_count=len(x)-len(nodes)+len(used),coordinate_reduction_fraction=(len(nodes)-len(used))/len(x),eligible_fraction=len(nodes)/len(x),maximum_start_reconstruction_error_m=float(max(gap)),maximum_rigid_reproduction_error_m=float(max(rigid_error)),test_moment_error_Nm=float(np.linalg.norm(torque)),test_virtual_work_error_J=float(virtual),rms_nodal_displacement_error_m=float(np.sqrt(np.average(err**2,weights=mass))),max_nodal_displacement_error_m=float(err.max()),incremental_strain_and_validity=field,admit=all(v['inverted']==v['below_shape']==0 and v['relative_incremental_strain_error']<=.05 for v in field.values()),scope='Authoring-side displacement-fit diagnostic on one accepted full-order frame; source-body ancestry, regional ownership and all retained material boundaries use the registered scene. Material inverse mass and dynamic tangent are not substituted by this test. No sustained or reduced native throughput is measured.')
    from enrich_source_embedding import enrich
    reply['refined']=enrich(stage,x,u,proposed,fit.copy(),local,q,mass,fine,tets[:,4],reg,audit_field)
    (stage/'source-embedding-motion.json').write_text(json.dumps(reply,indent=2)+'\n')
    print({k:r for k,r in reply.items() if k not in ('incremental_strain_and_validity','scope','refined')})
    print({k:{t:v for t,v in r.items() if t!='regions'} for k,r in field.items()})

if __name__=='__main__':
    run(Path(__file__).resolve().parent)
