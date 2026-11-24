"""Compose a tractions-only change in the terminal witness's connected component with a bed fit."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from prepare_indentation import soft_response
from compile_connected_joint import digest
from import_conditioned_interior import write_nodes
from supported_material_admission import geom

def run(source,fit,proposal,control,star,material,out,distributed=False):
    out.mkdir();v=source/'volume'
    raw=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');t=raw[:,:4]-1
    read=lambda f:np.loadtxt(f,skiprows=1,usecols=(1,2,3))
    x=read(v/'mesh.node');src=read(v/'current.node')
    loaded=read(fit/'current.node');p=read(proposal/'current.node')
    d=json.loads((fit/'fit.json').read_bytes());prior=json.loads((proposal/'fit.json').read_bytes())
    for a,mesh in [(d,fit),(prior,proposal)]:
        if a['registered']!=digest(mesh/'current.node') or a['control_loaded']!=digest(v/'current.node') or a['source_mass']!=digest(v/'node-mass.f64') or a['elements']!=digest(v/'mesh.ele'):raise ValueError('Different registration or source mass')
    w=json.loads(star.read_bytes());node=int(w['source']['joint_node_id'])-1
    changed=np.flatnonzero(np.any(p!=src,1));mask=np.isin(t,changed);cell,corner=np.nonzero(mask)
    incidence=coo_matrix((np.ones(len(cell)),(cell,np.searchsorted(changed,t[cell,corner]))),shape=(len(t),len(changed))).tocsr()
    _,labels=connected_components(incidence.T@incidence,directed=False)
    if not np.any(changed==node):raise ValueError('Native witness did not move')
    nodes=changed[labels==labels[np.searchsorted(changed,node)]]
    cells=np.flatnonzero(np.isin(t,nodes).any(1));ids,inv=np.unique(t[cells],return_inverse=True);tt=inv.reshape(-1,4)
    if np.any(src[ids]!=loaded[ids]):raise ValueError('Overlapping patch ownership')
    walls=np.load(v/'material-walls.npz')
    if any(np.intersect1d(nodes,walls[k]).size for k in ('O','D','C')):raise ValueError('Moved exterior surface')
    coeff=json.loads(material.read_bytes())['materials']
    mu,kappa=np.array([[coeff[str(i)]['registered'][s] for s in ('source_mu_pa','source_logJ_coefficient_pa')] for i in raw[cells,4]]).T
    law=soft_response(x[ids],src[ids],tt,mu,kappa)
    n=np.searchsorted(ids,node);ti,tirow=np.nonzero(tt==n)
    end=read(control/'physical-accepted.node')
    qbase=geom(end[ids],tt)[1]
    pre=law(src[ids])[0];base=float(np.linalg.norm(pre[ti,tirow].sum(0)))
    dx=p[ids]-src[ids];dx[~np.isin(ids,nodes)]=0
    attempts=[];winner=None
    history=[]
    if distributed:
        from distribute_registration import clip_registration
        local=np.searchsorted(ids,nodes)
        weights,history=clip_registration(x[ids],src[ids],dx,end[ids],tt,local)
        if weights is not None:dx[local]*=weights[:,None]

    for factor in (1.,.875,.75,.625,.5,.375,.25,.125,.0625,.03125,.015625):
        y=(src[ids]+factor*dx).astype('f4').astype(float);f,j=law(y)
        z=end[ids]+(y-src[ids])  # keep the accepted solver's compensated displacement
        try:_,jz=law(z)
        except ValueError:continue
        q=geom(y,tt)[1];qend=geom(z,tt)[1]
        # Preserve the accepted control where it already misses a shape threshold.
        bad=(j<.4)|(j>2.5)|(q<1e-5)|(jz<np.maximum(.35,.35*j))|(jz>2.5)|(qend+1e-14<np.minimum(1e-5,qbase))
        f=float(np.linalg.norm(f[ti,tirow].sum(0)))
        attempts.append(dict(factor=factor,witness_N=f,unsafe=int(bad.sum()),limiting_cells=cells[bad].tolist()[:20],Jmax_motion=float(jz.max()),Jmin_motion=float(jz.min())))
        if not np.any(bad) and f<base and winner is None:winner=(factor,y)
    (out/'selection.json').write_text(json.dumps(dict(witness=node,component_nodes=len(nodes),cells=len(tt),base_N=base,attempts=attempts,active_set_history=history,proposal=digest(proposal/'fit.json'),bed_fit=digest(fit/'fit.json'),committed_control=digest(control/'physical-accepted.node')),indent=2)+'\n')
    if winner is None:raise ValueError('No admissible connected registration')
    factor,y=winner;registered=loaded.copy();registered[ids]=y
    shifted=np.flatnonzero(np.any(registered!=src,1))
    write_nodes(v/'current.node',out/'current.node',{i:registered[i] for i in shifted})
    d.update(registered=digest(out/'current.node'),changed_nodes=len(shifted),witness_selection=digest(out/'selection.json'),connected_fraction=factor)
    (out/'fit.json').write_text(json.dumps(d,indent=2)+'\n')
    print('selected',factor,'nodes',len(nodes),'initial force',base,'fit',[a['witness_N'] for a in attempts if a['factor']==factor])
if __name__=='__main__':
    import sys
    p=argparse.ArgumentParser()
    for arg in ('source','fit','proposal','control','star','material','out'):p.add_argument(arg,type=lambda s:Path(s).resolve())
    p.add_argument('--distributed',action='store_true')
    run(**vars(p.parse_args()))
