"""Restore quadrature credits on descendants of intersecting cells on the same material sheet."""
import subprocess
import numpy as np

def solve(executable, candidate_keys, quota, targets, recipients, nq, seed):
    """Use the existing fixed-point allocator. Keep every source and receiver mass fixed."""
    keys=np.unique(np.concatenate(candidate_keys));Q=keys//len(recipients);C=keys%len(recipients)
    which=np.flatnonzero(quota>0)
    M=len(which);N=len(recipients);s=M+N;t=s+1;nodes=np.arange(M);rows=np.arange(N)
    a=np.r_[np.full(M,s),np.searchsorted(which,Q),M+rows,t].astype('<i4')
    b=np.r_[nodes,M+C,np.full(N,t),s].astype('<i4')
    total=float(quota.sum());lo=np.r_[quota[which],np.zeros(len(keys)),targets,0.]
    hi=np.r_[quota[which],np.minimum(quota[Q],targets[C]),targets,total*1.01]
    initial=np.zeros(len(keys))
    if len(seed):
        raw=np.asarray(seed);jj=np.searchsorted(recipients,raw[:,1].astype('i4'));pos=np.searchsorted(keys,raw[:,0].astype('i8')*N+jj)
        np.add.at(initial,pos,raw[:,2])
    start=np.r_[quota[which],initial,targets,total]
    buf=np.rec.fromarrays([a,b,lo,hi,start],names='a,b,l,h,w')
    p=subprocess.run([str(executable)],input=np.array([t+1,len(a)],'<i4').tobytes()+buf.tobytes(),capture_output=True,check=True)
    v=np.frombuffer(p.stdout,'<f8')
    if len(v)!=1+len(a):raise ValueError('Truncated allocation')
    w=v[1+M:1+M+len(keys)]
    aerr=np.bincount(Q,weights=w,minlength=nq)-quota;berr=np.bincount(C,weights=w,minlength=N)-targets
    fail_a=np.flatnonzero(abs(aerr)>np.maximum(quota*3e-12,3e-19));fail_b=np.flatnonzero(abs(berr)>np.maximum(targets*3e-12,3e-19))
    stats=dict(eligible_pairs=len(keys),fixed_point_deficit_kg=float(v[0]),maximum_carrier_discrepancy_kg=float(max(abs(aerr))),maximum_child_discrepancy_kg=float(max(abs(berr))),carrier_deficits=len(fail_a),child_deficits=len(fail_b),unmatched_carriers=fail_a[:20].tolist(),unmatched_children=recipients[fail_b[:20]].tolist())
    if p.stderr.startswith(b'augmentations='):stats['augmentations']=int(p.stderr.split(b'=')[1])
    positive=w>0
    return np.column_stack((Q[positive],recipients[C[positive]],w[positive])),stats

def neighbor_candidates(tets, geometric_keys, keys, qpos, target_region, xyz, offsets, regions, local, radius, hops=1):
    """Traverse one shared triangular face within the repaired component; never cross an exposed wall."""
    from scipy.sparse import coo_matrix
    n=len(tets)
    faces=np.sort(tets[:,[(1,2,3),(0,2,3),(0,1,3),(0,1,2)]].reshape(-1,3),axis=1).astype('i4')
    order=np.argsort(np.ascontiguousarray(faces).view('V12').ravel(),kind='stable')
    match=np.all(faces[order[:-1]]==faces[order[1:]],axis=1)
    if np.any(match[1:]&match[:-1]):raise ValueError('Nonmanifold material face')
    a=order[:-1][match]//4;b=order[1:][match]//4
    adj=coo_matrix((np.ones(2*len(a),dtype=bool),(np.r_[a,b],np.r_[b,a])),shape=(n,n)).tocsr()
    base=np.unique(np.concatenate(geometric_keys));src=base//n;cell=base%n
    prior=coo_matrix((np.ones(len(base),dtype=bool),(src,cell)),shape=(len(qpos),n)).tocsr()
    seen=prior
    for _ in range(hops):seen=seen.maximum(seen@adj)
    q,c=seen.nonzero();dist2=((qpos[q]-xyz[c])**2).sum(1);eligible=dist2<=radius**2
    target=local[c]
    for rid in np.unique(target_region[q]):
        ix=np.flatnonzero(target_region[q]==rid)
        viable=np.searchsorted(offsets[1:],np.flatnonzero(regions==rid),side='right')
        eligible[ix]&=np.isin(target[ix],viable)
    extra=q[eligible].astype('i8')*n+c[eligible]
    return [np.unique(np.r_[np.concatenate(keys),extra])]
