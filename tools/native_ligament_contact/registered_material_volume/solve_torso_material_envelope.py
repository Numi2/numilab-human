"""Fit the connected volume to the measured envelope with exact one-axis tetrahedron bounds."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.optimize import linprog
from torso_cell_geometry import geom
from emit_skin_source_registration import sha

def main():
    p=argparse.ArgumentParser()
    for k in ('source','target','out'):p.add_argument(k,type=Path)
    p.add_argument('--amplitude',type=float,default=1.)
    p.add_argument('--previous',type=Path,action='append',default=[])
    p.add_argument('--contact-history',type=Path,action='append',default=[])
    p.add_argument('--cortical',type=Path);p.add_argument('--parent-cap',type=Path);p.add_argument('--witness-set',type=Path);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    s=np.load(args.source);target=np.load(args.target);x=s['x'].astype('f8');t=s['t'];n=len(x)
    v,q=geom(x,t);goal=(target['x'][:n,2].astype('f8')-x[:,2])*1000*args.amplitude
    assert np.array_equal(target['unmoved'][:n],s['x'])
    fixed=np.zeros(n,bool);fixed[s['fix']]=True
    # The warm-start source is a cut with certified physical cells and contacts.
    # Optimize representable vertical displacements in millimetres, with identity
    # always feasible even at cells immediately above the stored quality floor.
    edge=x[t[:,1:]]-x[t[:,:1]];inv=np.linalg.inv(edge)
    G=np.c_[-inv[:,2,:].sum(1),inv[:,2,:]]*.001
    grad=sp.coo_matrix((G.ravel(),(np.repeat(np.arange(len(t)),4),t.ravel())),shape=(len(t),n)).tocsr()
    step=np.spacing(x[:,2].astype('f4')).astype('f8')
    margin=np.sum(abs(G)*(step[t]*500),1)+2e-10
    min_ratio=np.maximum.reduce([np.full(len(t),.83),np.sqrt(1.e-5/q),1.00001e-18/v])
    lower=np.minimum(1.,min_ratio+margin)-1
    upper=np.maximum(0.,.18-margin)
    # Bound edge length so q=6V/Lmax^3 stays above the native floor.
    # Aggregate identical edge rows by taking the strictest interval once.
    local=t[:,[0,0,0,1,1,2,1,2,3,2,3,3]].reshape(len(t),2,6).transpose(0,2,1)
    local=np.sort(local,axis=2);ed,ei=np.unique(local.reshape(-1,2),axis=0,return_inverse=True)
    L=(6*v*min_ratio/1.e-5)**(1/3)
    e=x[local[:,:,1]]-x[local[:,:,0]]
    lim=np.sqrt(np.maximum(0,L[:,None]**2-np.sum(e[:,:,:2]**2,2)))
    high=np.maximum(0.,(lim-e[:,:,2])*1000);low=np.maximum(0.,(lim+e[:,:,2])*1000)
    bh=np.full(len(ed),np.inf);bl=bh.copy();np.minimum.at(bh,ei.ravel(),high.ravel());np.minimum.at(bl,ei.ravel(),low.ravel())
    # A direction bound also prevents gratuitously sharp strain within otherwise
    # well-shaped cells while allowing measured skin contour changes.
    length=np.linalg.norm(x[ed[:,1]]-x[ed[:,0]],axis=1)*1000
    bh=np.minimum(bh,.5*length);bl=np.minimum(bl,.5*length)
    E=sp.coo_matrix((np.tile([-1.,1.],len(ed)),(np.repeat(np.arange(len(ed)),2),ed.ravel())),shape=(len(ed),n)).tocsr()
    for old in args.previous:
        last=np.load(old);xx=last['x'].astype('f8');vp,qp=geom(xx,t)
        bad=(vp<=1e-18)|(qp<1e-5)
        # A complete near-floor element translates exactly when its corners share
        # one native-representable shift. Pin just unresolved rounding witnesses.
        fixed[np.unique(t[bad])]=True
    A=sp.vstack([grad,-grad,E,-E],format='csr');b=np.r_[upper,-lower,bh,bl]
    from torso_projection_order import section_order
    order,room,ordered_pairs=section_order(s,x,args.contact_history)
    if args.witness_set:
        prior=np.load(args.witness_set)
        assert str(prior['source_sha256'])==sha(args.source)
        fixed[prior['fixed_nodes']]=True
        order=sp.csr_matrix((prior['order_data'],prior['order_indices'],prior['order_indptr']),shape=(len(prior['order_rhs']),n))
        room=prior['order_rhs'];ordered_pairs=int(prior['ordered_pairs'])
    A=sp.vstack([A,order],format='csr');b=np.r_[b,room]
    if args.cortical:
        bone=np.load(args.cortical);limits=bone['max_delta_mm'];C=sp.coo_matrix((bone['weight'].ravel(),(np.repeat(np.arange(len(limits)),3),bone['corners'].ravel())),shape=(len(limits),n))
        A=sp.vstack([A,C],format='csr');b=np.r_[b,limits]
    ii=np.flatnonzero(~fixed);A=A[:,ii];keep=np.diff(A.indptr)>0;A=A[keep];b=b[keep]
    I=sp.eye(len(ii),format='csr');Z=sp.csr_matrix(A.shape)
    M=sp.vstack([sp.hstack([A,Z]),sp.hstack([I,-I]),sp.hstack([-I,-I])],format='csc')
    volume_weight=np.bincount(t.ravel(),weights=np.repeat(v*.25,4),minlength=n)
    weight=np.clip(np.sqrt(volume_weight/np.median(volume_weight)),.1,10.)
    upper_bound=np.full(len(x),max(1.,float(np.max(goal))))
    if args.parent_cap:
        limits=np.load(args.parent_cap);assert np.array_equal(limits['nodes'],s['nodes']);upper_bound=np.minimum(upper_bound,limits['limit_mm'])
    bounds=[(0.,h) for h in upper_bound[ii]]+[(0.,None)]*len(ii)
    print('solving',M.shape,M.nnz,flush=True)
    res=linprog(np.r_[np.zeros(len(ii)),weight[ii]],A_ub=M,b_ub=np.r_[b,goal[ii],-goal[ii]],bounds=bounds,method='highs-ipm',options={'time_limit':180,'primal_feasibility_tolerance':1e-9})
    if not res.success:raise RuntimeError(res.message)
    y=x.copy();y[ii,2]=(x[ii,2]+res.x[:len(ii)]*.001).astype('f4');v1,q1=geom(y,t)
    assert np.array_equal(y[s['fix']],s['x'][s['fix']])
    data={k:s[k] for k in ('O','I','D','C15','C16','nodes','cell','role','t')};np.savez_compressed(args.out/'proposal.npz',x=y.astype('f4'),**data)
    r=dict(ordered_pairs=ordered_pairs,ordering_rows=len(room),input_sha256=sha(args.source),target_sha256=sha(args.target),proposal_sha256=sha(args.out/'proposal.npz'),mean_target_error_mm=float(np.average(abs((y[:,2]-x[:,2])*1000-goal),weights=weight)),bad_volume=int(sum(v1<=1e-18)),bad_quality=int(sum(q1<1e-5)),minimum_volume_m3=float(min(v1)),minimum_quality=float(min(q1)),mass_mapping='requires new current-geometry carrier assignment',maximum_shift_mm=float(max(y[:,2]-x[:,2])*1000),moved_nodes=int(sum(y[:,2]!=x[:,2])),fixed_nodes=int(sum(fixed)))
    (args.out/'metrics.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
if __name__=='__main__':main()
