"""Fit the initial deformation of the bonded layers around severe strain, preserving the physical anchors."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import splu
from compile_connected_joint import digest
from certify_transported_material import measure

def run(job,out):
    out.mkdir()
    mat=job/'interval-certified-scene/material'
    x,y=[np.loadtxt(mat/p,skiprows=1,usecols=(1,2,3)).astype('f4').astype(float) for p in ('mesh.node','current.node')]
    g=np.loadtxt(mat/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype='i4')-1
    z=np.load(job/'registered-boundary/registered-boundary.npz');c=np.load(job/'deep-fat-cortex/registered-cortical-facets.npz')
    a=json.loads((job.parent/'ligament-equilibrium-070/scene-registration/registration.json').read_bytes())
    pin=np.unique(np.r_[z['O_node_joint'],c['cell_side_material_bank_nodes'][:,-3:].ravel(),a['first_contact_new'],[e[2] for e in a['anchors']]]).astype(int)
    excluded=np.zeros(len(x),bool);excluded[pin]=True
    bad=np.load(job/'registered-stretch-patch/interior-patch.npz')['severe'];initial=y.copy()
    interior=np.unique(g[bad]);interior=interior[~excluded[interior]]
    # Use a full neighbor ring to avoid moving a failing triangle into an untouched adjacent cell.
    ring=np.unique(g[np.isin(g,interior).any(1)]);nodes=ring[~excluded[ring]]
    ix=np.flatnonzero(np.isin(g,nodes).any(1));t=g[ix]
    mapped=np.full(len(x),-1,dtype='i4');mapped[nodes]=np.arange(len(nodes))
    em=np.stack([x[t[:,k]]-x[t[:,0]] for k in [1,2,3]],axis=1)
    vol=np.linalg.det(em)/6.;risk=np.median(vol);inv=np.linalg.inv(em)
    grad=np.concatenate([-inv.sum(axis=2)[:,:,None],inv],axis=2)
    row=np.broadcast_to(np.arange(len(t)*3).reshape(-1,3,1),grad.shape)
    col=np.broadcast_to(mapped[t][:,None,:],grad.shape);ok=col>=0
    # Use co-rotational registration errors. A floor on the geometrical
    # weight keeps very small bad cells represented in the patch fit.
    weight=np.sqrt(np.maximum(vol,risk*.1)/risk)
    G=coo_matrix(((grad*weight[:,None,None])[ok],(row[ok],col[ok])),shape=(3*len(t),len(nodes))).tocsr()
    A=(G.T@G).tocsc();s=diags(1/np.sqrt(A.diagonal()));factor=splu((s@A@s).tocsc())
    rest_det,scale=measure(x,t)
    def evaluate(p):
        B=p[t[:,1:]]-p[t[:,:1]]
        F=inv@B;u,sv,vt=np.linalg.svd(F);R=u@vt
        d,sc=measure(p,t);J=d/rest_det
        sev=(sv[:,0]>2)|(sv[:,-1]<.5)
        return F,R,dict(severe=int(sev.sum()),worst_stretch_or_compression=float(np.max(np.maximum(sv[:,0],1/sv[:,-1]))),J_min=float(J.min()),J_max=float(J.max()),shape_min=float(np.min(d/sc)),min_six_volume=float(d.min()),fitting_energy=float(np.sum((weight[:,None,None]*(F-R))**2)))
    F,R,summary=evaluate(y);history=[summary]
    for it in range(12):
        rhs=G.T@((R-F)*weight[:,None,None]).reshape(-1,3)
        delta=s@factor.solve(s@rhs);candidate=y.copy()
        for k in range(20):
            candidate[nodes]=(y[nodes]+(2.**-k)*delta).astype('f4').astype(float)
            f,r,v=evaluate(candidate)
            if v['min_six_volume']>6e-18 and v['shape_min']>1e-5 and .35<v['J_min'] and v['J_max']<2.5 and v['fitting_energy']<summary['fitting_energy']*(1-1e-7) and v['worst_stretch_or_compression']<=history[0]['worst_stretch_or_compression'] and v['severe']<=summary['severe']:
                y=candidate;F,R,summary=f,r,v;history.append(v|dict(alpha=2.**-k));break
        else:break
    diff=np.flatnonzero(np.any(y!=initial,axis=1))
    if np.intersect1d(diff,pin).size:raise ValueError('Pinned coordinates changed')
    np.savez_compressed(out/'candidate-positions.npz',nodes=diff,positions=y[diff].astype('f4'))
    report=dict(pose=digest(mat/'current.node'),material=digest(mat/'mesh.node'),elements=digest(mat/'mesh.ele'),source_severe=len(bad),free=len(nodes),patch_cells=len(t),max_move_mm=float(1e3*np.linalg.norm(y-initial,axis=1).max()),moved=len(diff),kept_outer_nodes=len(z['O_node_joint']),history=history)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print({k:report[k] for k in ['free','patch_cells','moved','max_move_mm']},history[0],history[-1],flush=True)
if __name__=='__main__':run(*(Path(p).resolve() for p in sys.argv[1:]))
