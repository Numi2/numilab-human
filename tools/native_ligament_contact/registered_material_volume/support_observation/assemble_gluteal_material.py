"""Condense the registered adipose reference tangent through the native muscle compliance."""
import json
import re
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.transform import Rotation
from bind_gluteal_receivers import digest, require


def write_kinematics(out, data, fat, ref, parents, shape, mf, muscle, weights, donors, poses, order, rule, reference_tets):
    fs, fi = np.unique(fat, return_inverse=True)
    ms, mi = np.unique(parents[mf],return_inverse=True)
    fi=fi.reshape(fat.shape)
    mi=mi.reshape(parents[mf].shape)+len(fs)
    n=len(fs)+len(ms)
    point=np.zeros((len(fat),6,n))
    for c in range(3):
        np.add.at(point,(np.arange(len(fat))[:,None],np.arange(3),fi[:,c,None]),data['fat_shape'][:,:,c])
        for j in range(4):
            np.add.at(point,(np.arange(len(fat))[:,None],np.arange(3)+3,mi[:,c,j,None]),
                      data['muscle_shape'][:,:,c]*shape[mf[:,c],j,None])
    point=np.take_along_axis(point,order[:,:,None],axis=1)
    nodes=np.vstack((ref[fs],muscle[ms]))
    elements=point[:,rule].reshape(-1,4,n)
    sample=reference_tets.reshape(-1,4,3)
    require(np.max(abs(elements @ nodes-sample))<2e-12,'Anatomical coordinates do not reproduce reference')
    edge=sample[:,1:]-sample[:,:1]
    gradient=np.linalg.solve(edge,elements[:,1:]-elements[:,:1])
    identity_error=np.max(abs(gradient@(nodes-nodes.mean(0))-np.eye(3)))
    require(identity_error<2e-7 and np.max(abs(gradient.sum(2)))<1e-6,'Linear field reproduction failed')
    directory=out/'material'
    directory.mkdir(exist_ok=True)
    sparse.save_npz(directory/'gradient.npz',sparse.csr_matrix(gradient.reshape(-1,n)))
    np.savez_compressed(directory/'parents.npz',fat=fs,muscle=ms,reference=nodes,
                        volume=np.abs(np.linalg.det(edge))/6,donors=donors,body_weights=weights[ms],
                        origins=[poses[d]['position_m'] for d in donors], bed_normal=data['frame'][:,2],
                        pelvis_rotation=Rotation.from_quat(poses[128]['quaternion_xyzw']).as_matrix(),
                        linear_reproduction_error=identity_error)


def compliance(assets, parents):
    m=json.loads((assets/'manifest.json').read_bytes())
    for f in ('cortical-free-index.bin','cortical-flexibility.bin'):
        require(digest(assets/f)==m['files'][f],'Modified material compliance')
    ids=parents['muscle']
    idx=np.fromfile(assets/'cortical-free-index.bin','<u4')
    vectors=np.memmap(assets/'cortical-flexibility.bin','<f4',mode='r').reshape(-1,4)
    Q=parents['pelvis_rotation']
    blocks=np.zeros((len(ids),3,len(ids),3))
    for r in m['contact_flexibility']:
        off=r['first_node'];n=r['nodes'];k=r['input_nodes']
        select=np.flatnonzero((ids>=off)&(ids<off+n))
        atlas=vectors[r['first_vector']:][:3*n*k].reshape(n,k,3,4)
        col=idx[ids[select]]
        if np.any(col>=k):
            raise ValueError('A required material parent is missing its response')
        local=atlas[(ids[select]-off)[:,None],col[None,:],:,:3]
        world=np.einsum('ax,ijyx,by->iajb',Q,local,Q,optimize=True)
        for i,global_i in enumerate(select):
            blocks[global_i][:,select,:]=world[i]
    H=blocks.reshape(3*len(ids),-1)
    require(np.linalg.norm(H-H.T)<1e-6*np.linalg.norm(H),'Nonreciprocal muscle compliance')
    vals,vec=np.linalg.eigh((H+H.T)*.5)
    require(vals[0]>=-1e-7*vals[-1],'Indefinite muscle compliance')
    active=vals>np.finfo(float).eps*len(vals)*vals[-1]
    return (vec[:,active]*np.sqrt(vals[active])),float(vals[0])


def reference_stiffness(grad,volumes,mu,bulk):
    # P = mu (F-F^-T) + bulk log(J) F^-T. Differentiate at F=I.
    # The six strain rows use engineering shear, in world coordinates.
    g=[grad[i::3] for i in range(3)]
    rows=[]
    for a,b in ((0,0),(1,1),(2,2),(1,2),(0,2),(0,1)):
        components=[g[b] if i==a else g[a] if i==b else sparse.csr_matrix(g[0].shape)
                    for i in range(3)]
        rows.append(sparse.hstack(components,format='csr'))
    B=sparse.vstack(rows,format='csr')
    D=np.diag([2*mu]*3+[mu]*3)
    D[:3,:3]+=bulk
    K=(B.T @ sparse.kron(sparse.csr_matrix(D),sparse.diags(volumes)) @ B).toarray()
    # Convert component-major to node-major.
    order=np.arange(K.shape[0]).reshape(3,-1).T.ravel()
    return K[np.ix_(order,order)]


def assemble(path,assets,material):
    p=dict(np.load(path/'parents.npz'))
    g=sparse.load_npz(path/'gradient.npz')
    source=material.read_text()
    require('0.5 * shear * (I1() - 3 - 2 * log(J()))' in source and
            '0.5 * bulk * pow(log(J()), 2)' in source,'Unrecognized adipose constitutive law')
    mu,bulk=[float(re.search(r'parameter '+v+r'\s*:\s*MPa\s*=\s*([\d.]+)',source)[1])*1e6 for v in ('shear','bulk')]
    K=reference_stiffness(g,p['volume'],mu,bulk)
    split=3*len(p['fat'])
    H,min_H=compliance(assets,p)
    # condense only the existing muscle response; the registered adipose energy
    # is retained, including its shear and volumetric terms.
    C=K[split:,:]
    A=np.eye(H.shape[1])+H.T @ C[:,split:] @ H
    solve=cho_factor((A+A.T)*.5)
    eliminated=H @ cho_solve(solve,H.T @ C)
    local=K-C.T @ eliminated
    donors=p['donors'];origins=p['origins'];weights=p['body_weights']
    T=np.zeros((len(K),split+6*len(donors)));T[:split,:split]=np.eye(split)
    points=p['reference'][split//3:]
    for j,center in enumerate(origins):
        for i in range(len(points)):
            a=split+3*i;b=split+6*j;w=weights[i,j]
            T[a:a+3,b:b+3]=w*np.eye(3)
            T[a:a+3,b+3:b+6]=w*np.cross(np.eye(3),points[i]-center).T
    S=T.T @ local @ T
    rng=np.random.default_rng(78)
    probes=rng.standard_normal((S.shape[0],8))*1e-5
    external=T @ probes
    internal=-eliminated @ external
    physical=external.copy();physical[split:]+=internal
    forces=K @ physical
    condensed_error=np.linalg.norm(S@probes-T.T@forces)/np.linalg.norm(S@probes)
    balance=np.linalg.norm(internal+H @ (H.T @ forces[split:]))/np.linalg.norm(internal)
    # Impose all six common rigid motions. Their infinitesimal strain vanishes.
    null=np.zeros((S.shape[0],6));points=p['reference'][:split//3];center=points.mean(0)
    for i,point in enumerate(points):
        null[3*i:3*i+3,:3]=np.eye(3)
        null[3*i:3*i+3,3:]=np.cross(np.eye(3),point-center).T
    for j,origin in enumerate(origins):
        i=split+6*j;null[i:i+3,:3]=np.eye(3);null[i+3:i+6,3:]=np.eye(3)
        null[i:i+3,3:]=np.cross(np.eye(3),origin-center).T
    motion_error=np.linalg.norm(S@null)/np.linalg.norm(S)
    global_balance=np.linalg.norm(null.T@(S@probes))/np.linalg.norm(S@probes)
    energy=np.linalg.eigvalsh((S+S.T)*.5)
    require(max(condensed_error,balance,motion_error,global_balance)<1e-8 and
            energy[0]>-1e-9*energy[-1], 'Physical projection or equilibrium failed')
    # Evaluate a common virtual bed-normal indentation, with stationary bones.
    test_displacement=p['bed_normal']*1e-4
    u=null[:,:3]@test_displacement;u[split:]=0
    f=(S@u).reshape(-1)
    force=f[:split].reshape(-1,3).sum(0)
    reactions=f[split:].reshape(-1,6)
    out=dict(fat_nodes=split//3,muscle_nodes=len(p['muscle']),rigid_donors=donors.tolist(),
             tangent_freedom=len(S),material=digest(material),material_parameters_Pa=[mu,bulk],
             projected_volume_m3=float(p['volume'].sum()),min_muscle_compliance_eigenvalue=min_H,
             direct_condensed_error=condensed_error,free_muscle_residual=balance,
             common_rigid_motion_error=motion_error,force_moment_balance_error=global_balance,
             unit_test_displacement_world_m=test_displacement.tolist(),test_fat_energy_gradient_N=force.tolist(),
             test_rigid_energy_gradient_N_Nm=reactions.tolist(),
             scope='Stress-free tangent and virtual load. No candidate force is applied to the native simulation.')
    np.savez_compressed(path/'reference-tangent.npz',matrix=S,recovery=-eliminated @ T)
    (path/'reference-tangent.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print({k:out[k] for k in ('fat_nodes','muscle_nodes','tangent_freedom','free_muscle_residual','force_moment_balance_error')})


if __name__=='__main__':
    import sys
    assemble(*(Path(k) for k in sys.argv[1:]))
