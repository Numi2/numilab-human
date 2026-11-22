"""Prepare loaded coordinates with physical skin-plane and registered deep boundary conditions."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from screen_registered_dense_posture import load_geometry,measure
from qualify_loaded_pose_registration import anchors
from construct_registered_donor_field import errors
from untangle_registered_donor_field import differentials
from recover_material_thickness import audit

def main(full,donor,seed,out):
    x,tgt,owner,t,mat,outer,_,pins=load_geometry(full,donor)
    if json.loads((seed/'result.json').read_text())['source']!=pins:raise ValueError('unknown dense pose identity')
    y=x+np.load(seed/'displacement.npy');initial=y.copy()
    env=json.loads((full/'launch.json').read_text())['environment'];bound=anchors(env)
    fixed=np.array([a[1] for a in bound.values()])
    bed=outer & (y[:,2]<=0)
    if np.any(tgt[fixed,2]<=0):raise ValueError('inconsistent tissue and skeletal target below plane')
    free=bed.copy();free[fixed]=True
    for i in range(4):free[t[free[t].any(1)].ravel()]=True
    nodes=np.flatnonzero(free);pack=np.full(len(x),-1,np.int32);pack[nodes]=np.arange(len(nodes))
    incident=np.flatnonzero(free[t].any(1));rs=[];cs=[];vs=[]
    for start in range(0,len(incident),16384):
        tet=t[incident[start:start+16384]]
        det,g,_=differentials(x,tet)
        # Reference energy integral of the P1 displacement gradient, used as a
        # construction guess only. All final contact forces remain native-owned.
        K=(g@g.transpose(0,2,1))/(6*det[:,None,None])
        ids=pack[tet];keep=(ids[:,:,None]>=0)&(ids[:,None,:]>=0)
        rs.append(np.broadcast_to(ids[:,:,None],keep.shape)[keep]);cs.append(np.broadcast_to(ids[:,None,:],keep.shape)[keep]);vs.append(K[keep])
    K=coo_matrix((np.concatenate(vs),(np.concatenate(rs),np.concatenate(cs))),shape=(len(nodes),len(nodes))).tocsc()
    scale=1/np.sqrt(K.diagonal());pre=diags(scale);K=(pre@K@pre).tocsc()
    d=np.zeros((len(nodes),3))
    def apply(mask,values,dims):
        idx=np.flatnonzero(mask);remain=np.flatnonzero(~mask)
        u=np.zeros((len(nodes),len(dims)));u[idx]=values/scale[idx,None]
        rhs=-(K[remain][:,idx]@u[idx])
        local=K[remain][:,remain].tocsc()
        solver=splu(local,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,options={'SymmetricMode':True})
        u[remain]=solver.solve(rhs)
        res=np.linalg.norm(local@u[remain]-rhs)/max(np.linalg.norm(rhs),1e-30)
        if res>1.e-7:raise ValueError(('inaccurate harmonic extension',res))
        d[:,dims]=scale[:,None]*u
    hard=np.zeros(len(nodes),bool);hard[pack[fixed]]=True
    target=tgt[nodes]-initial[nodes];apply(hard,target[hard,:2],[0,1])
    for _ in range(12):
        constrained=hard | bed[nodes]
        z=np.where(hard,target[:,2],-initial[nodes,2])
        apply(constrained,z[constrained,None],[2])
        # Set constrained values directly after the solve to avoid roundoff through zero.
        y[nodes]=initial[nodes]+d;y[fixed]=tgt[fixed].astype('f4').astype(float);y[bed,2]=0.
        new=outer & (y[:,2]<0.)
        if not new.any():break
        if np.any(new&~free):raise ValueError('support expansion crossed the material fit neighborhood')
        bed|=new
    else:raise ValueError('unresolved unilateral construction boundary')
    rr=np.concatenate([differentials(x,t[k:k+32768])[0] for k in range(0,len(t),32768)])
    j,h,det,bad=audit(y,t,rr)
    out.mkdir(exist_ok=False);np.save(out/'displacement.npy',y-x)
    locked=np.zeros((len(x),3),bool);locked[fixed]=True
    np.savez_compressed(out/'construction-boundary.npz',node=np.flatnonzero(locked.any(1)),axis=locked[locked.any(1)],value=y[locked.any(1)],trial_contact=np.flatnonzero(bed))
    result=dict(source=pins,free_nodes=len(nodes),incident_cells=len(incident),fixed_attachments=len(fixed),contact_rows=int(bed.sum()),
                min_outer_height_m=float(y[outer,2].min()),min_all_node_height_m=float(y[:,2].min()),failed=int(bad.sum()),
                quality=measure(y,owner,t,mat),correction_to_seed=errors(y,initial),hard_anchor_error=errors(y[fixed],tgt[fixed]),
                scope='Contact and anchor coordinates used only at construction. Native contact pressure, reaction and detached motion must be solved at runtime.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result['failed'],result['quality'],flush=True)
if __name__=='__main__':main(*(Path(v).resolve() for v in sys.argv[1:]))
