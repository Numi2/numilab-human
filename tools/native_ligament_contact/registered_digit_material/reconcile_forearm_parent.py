"""Transfer a changed source parent to its existing child's rest-space representation."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from registered_forearms import read,records,sha
from extensor_registration import affines

def transfer(m,original,parent,destination,posefile,emit):
    import struct,tempfile
    rest=read(posefile)['all_bodies']
    pose={r['body_index']:{'position_m':r['rest']['position_world_m'],'quaternion_xyzw':r['rest']['orientation_world_xyzw']} for r in rest}
    if sorted(pose)!=list(range(157)):raise ValueError('Incomplete compiler rest pose')
    def tuples(path):
        data=path.read_bytes();tt,vo,nv=records(data);row,=tt[tt[:,6]==m.bank];bi,nb,vi,n,*_=map(int,row)
        v=np.frombuffer(data,'<f4',14*nv,vo).reshape(-1,14)[vi:vi+n]
        binds=[struct.unpack_from('<I8f',data,64+32*len(tt)+36*i) for i in range(bi,bi+nb)]
        slot=v[:,6:10].copy().view('<u4');w=v[:,10:14].astype(float);slot[(slot==0xffffffff)&(w==0)]=0
        if slot.max()>=nb or abs(w.sum(1)-1).max()>1e-5:raise ValueError('Invalid sparse weights in child')
        A,t=affines(slot,w,binds,pose)
        return v[:,:3].copy(),A,t
    X,Ap,tp=tuples(m.owner);Y,_,_=tuples(parent);u,Ac,tc=tuples(original)
    rows=next(r['vertices'] for r in read(m.receipt)['passive_tissue_parent_bindings']['surfaces'] if r['stable_id']==m.sid)
    raw=[t for v in rows for t in v]
    if any(t[0]!=m.parent_index or t[1]!=m.bank for t in raw):raise ValueError('Unexpected source carrier')
    M=csr_matrix(([t[3] for t in raw],[t[2] for t in raw],np.r_[0,np.cumsum([len(v) for v in rows])]),shape=(len(rows),len(X)))
    def mapped(A,x,t):return (A@x[:,:,None])[:,:,0]+t
    old=M@mapped(Ap,X,tp);child=mapped(Ac,u,tc);diff=M@((Ap@(Y-X)[:,:,None])[:,:,0])
    initial=np.linalg.norm(old-child,axis=1)
    if initial.max()>2e-6:raise ValueError(('Wrong actual material rest pose',initial.max()))
    select=np.any(diff,axis=1);v=u.copy()
    v[select]+=np.linalg.solve(Ac[select],diff[select,:,None])[:,:,0].astype('f4')
    # Existing coincident raw source seams get one emitted coordinate. Check all
    # distinct donor tuples afterwards in the actual articulated rest frame.
    _,inv=np.unique(u,axis=0,return_inverse=True)
    mean=np.zeros((inv.max()+1,3));np.add.at(mean,inv,v.astype(float)-u)
    v=(u+mean[inv]/np.bincount(inv)[inv,None]).astype('f4')
    final=np.linalg.norm(mapped(Ac,v,tc)-(old+diff),axis=1)
    if final.max()>2e-6:raise ValueError(('Child to parent reconstruction failed',final.max()))
    with tempfile.TemporaryDirectory(dir=destination.parent) as folder:
        p=Path(folder)/'material.npz';np.savez(p,positions=v,changed=np.any(v!=u,axis=1))
        record=emit(SimpleNamespace(owner=original,bank=m.sid,x=u),p,destination)
    return dict(**record,named_rest_pose=sha(posefile),initial_reconstruction_error_m=float(initial.max()),final_reconstruction_error_m=float(final.max()))
