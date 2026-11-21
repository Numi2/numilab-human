"""Greedy energy-optimal enrichment of the existing region responses using registered point loads."""
import json,sys
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation
from pelvic_elastic_response import stiffness
from cortical_distance import CorticalSurface
from qualify_material_motion import frame
from pelvic_specimen import properties
from material_context import sha

def build(part,existing,pool,receipt,material,library,out):
    out.mkdir(exist_ok=True);manifest=json.loads((pool/'manifest.json').read_text())
    frames={r['body_index']:frame(r) for r in json.loads(receipt.read_text())['initial_anatomical_registration']['body_poses']}
    tri=np.fromfile(pool/'cortical-triangles.bin','f4').reshape(-1,10,4)[:,:3,:3].astype(float)
    records=np.fromfile(pool/'pool-material-points.bin','f4').reshape(-1,7,4);owners=records[:,3].copy().view('u4')
    laws=properties(material);start=0;metrics={}
    for side,(sid,fem) in enumerate(((29,131),(30,145))):
        reg=np.load(existing/f'{sid}.npz');p=np.load(part/'projected'/f'{sid}.npz');x=p['vertices'].astype(float);t=p['cells'];load=p['cell_volumes']>0
        A,_=stiffness(x,t[load],p['cell_volumes'][load],laws['mu'],laws['lam'])
        free=np.setdiff1d(np.unique(t[load]),reg['fixed']);ids=(3*free[:,None]+np.arange(3)).ravel()
        K=A[ids][:,ids];scale=1/np.sqrt(K.diagonal());D=sparse.diags(scale);fac=splu((D@K@D).tocsc())
        keep=owners[:,1]==fem;rec=records[keep];body=owners[keep,2];banks=rec[:,4].copy().view('u4')
        normals=np.empty((len(rec),3))
        for b,first,n in np.unique(np.column_stack((body,banks[:,:2])),axis=0):
            i=np.flatnonzero((body==b)&(banks[:,0]==first));xx=tri[first:first+n].reshape(-1,3)
            surf=CorticalSurface(library,xx,np.arange(len(xx)).reshape(-1,3));_,normal,*_=surf.query(rec[i,2,:3])
            normals[i]=Rotation.from_quat(frames[b][3:]).apply(normal)
        node=rec[:,5].copy().view('u4').astype(int)-start;w=rec[:,6].astype(float);start+=len(x)
        if node.min()<0 or node.max()>=len(x):raise ValueError('Ancestry escaped local side')
        entries=w[:,:,None]*normals[:,None,:]
        dof=3*node[:,:,None]+np.arange(3)
        L=sparse.csr_matrix((entries.ravel(),(np.repeat(np.arange(len(rec)),12),dof.ravel())),shape=(len(rec),3*len(x)))[:,ids]
        # Only column blocks of the inverse are requested from the factor. Compute each
        # row's compliance once without forming or retaining a dense inverse.
        physical=np.zeros(len(rec));Lt=L.T.tocsc()
        for lo in range(0,len(ids),64):
            hi=min(lo+64,len(ids));n=hi-lo;rhs=np.zeros((len(ids),n));rhs[np.arange(lo,hi),np.arange(n)]=scale[lo:hi]
            ux=scale[:,None]*fac.solve(rhs)
            local=L[:,lo:hi].tocoo()
            idx,which=np.unique(local.row,return_inverse=True)
            if len(idx):
                products=L[idx]@ux
                np.add.at(physical,local.row,local.data*products[which,local.col])
        modes=reg['modes'].reshape(3*len(x),-1);basis=modes[ids].copy();coeff=L@basis
        initial=np.sum(coeff**2,axis=1);remain=physical-initial;eligible=physical>1e-12;metric=lambda z:np.quantile(z[eligible]/physical[eligible],[0,.5,.95,1]).tolist()
        history=[];before=metric(remain)
        for k in range(96-basis.shape[1]):
            score=np.divide(remain,physical,out=np.zeros_like(remain),where=eligible)
            j=int(score.argmax());l=L.getrow(j).toarray().ravel();u=scale*fac.solve(scale*l);candidate=u-basis@coeff[j]
            # Reorthogonalize in K. All previously admitted regional coordinates are retained.
            candidate-=basis@(basis.T@(K@candidate));h=float(candidate@(K@candidate))
            if h<=1.e-16:break
            candidate/=np.sqrt(h);projection=np.asarray(L@candidate);remain-=projection**2
            history.append(dict(pool_index=int(np.flatnonzero(keep)[j]),true_compliance_m_N=physical[j],unrepresented_fraction_before=score[j]))
            coeff=np.column_stack((coeff,projection));basis=np.column_stack((basis,candidate))
        ans=np.zeros((3*len(x),basis.shape[1]));ans[ids]=basis
        check=float(abs(basis.T@(K@basis)-np.eye(basis.shape[1])).max())
        if check>1.e-5 or remain.min() < -max(physical.max()*1.e-8,1.e-13):raise ValueError(('projection failed',sid,check,remain.min()))
        output={k:reg[k] for k in reg.files if k!='modes'};output['modes']=ans.reshape(len(x),3,-1)
        np.savez_compressed(out/f'{sid}.npz',**output)
        metrics[sid]=dict(source_sha256=sha(existing/f'{sid}.npz'),mode_count=basis.shape[1],contact_rows=len(rec),unrepresented_compliance_fraction_before=before,unrepresented_compliance_fraction_after=metric(remain),orthogonality_error=check,selected=history)
        print(sid,before,metric(remain),check,flush=True)
    (out/'enrichment.json').write_text(json.dumps(metrics,indent=2)+'\n')
if __name__=='__main__':build(*map(Path,sys.argv[1:]))
