"""Retain measured carrier and cell totals while honoring native transport distance."""
import numpy as np
from scipy.optimize import Bounds,linprog
from scipy.sparse import coo_matrix

def route(aa,ci,mw,newrows,source_points,centers,radius,region,offsets,allowed):
    sel=np.isin(ci,newrows)
    carriers=np.unique(aa[sel]);recipients=np.unique(ci[sel])
    a=np.searchsorted(carriers,aa[sel]);c=np.searchsorted(recipients,ci[sel])
    initial=np.zeros((len(carriers),len(recipients)))
    np.add.at(initial,(a,c),mw[sel])
    dist=np.linalg.norm(source_points[carriers,None]-centers[recipients][None],axis=2)
    allowed_edge=dist<=radius
    for j,r in enumerate(recipients):
        allowed_edge[:,j]&=np.isin(region[carriers],allowed[offsets[r]:offsets[r+1]])
    if not np.any((initial>0)&~allowed_edge):return aa,ci,mw
    # Scale each equality by its own supply/demand so tiny material packets carry
    # the same numerical weight as larger ones. Prefer the geometric overlay.
    i,j=np.where(allowed_edge);amount=initial.sum(1);demand=initial.sum(0)
    weight=np.minimum(amount[i],demand[j])
    ns=len(i);k=np.arange(ns)
    A=coo_matrix((np.r_[weight/amount[i],weight/demand[j]],(np.r_[i,len(carriers)+j],np.r_[k,k])),shape=(len(carriers)+len(recipients),ns)).tocsr()
    cst=(dist[i,j]/radius+4*(initial[i,j]==0))*weight/weight.max()
    sol=linprog(cst,A_eq=A,b_eq=np.ones(A.shape[0]),bounds=(0,1),method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
    if not sol.success:raise ValueError('Could not conserve mass within the neighboring source reach: '+sol.message)
    credits=sol.x*weight;use=credits>0
    return np.r_[aa[~sel],carriers[i[use]]],np.r_[ci[~sel],recipients[j[use]]],np.r_[mw[~sel],credits[use]]
