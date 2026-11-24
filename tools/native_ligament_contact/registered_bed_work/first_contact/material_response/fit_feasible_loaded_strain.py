"""Refit connected skin and fat with independently feasible updates on nonadjacent nodes."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import splu
from compile_connected_joint import digest
from certify_transported_material import measure

def run(job,out,held=None):
    out.mkdir();mat=job/'interval-certified-scene/material'
    x,y=[np.loadtxt(mat/p,skiprows=1,usecols=(1,2,3)).astype('f4').astype(float) for p in ('mesh.node','current.node')]
    g=np.loadtxt(mat/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype='i4')-1
    z=np.load(job/'registered-boundary/registered-boundary.npz');c=np.load(job/'deep-fat-cortex/registered-cortical-facets.npz');a=json.loads((job.parent/'ligament-equilibrium-070/scene-registration/registration.json').read_bytes())
    excluded=np.zeros(len(x),bool);excluded[z['O_node_joint']]=True;excluded[c['cell_side_material_bank_nodes'][:,-3:].ravel()]=True
    excluded[a['first_contact_new']+[e[2] for e in a['anchors']]]=True
    if held is not None:
        excluded[np.load(held/'nodes.npy')]=True
    severe=np.load(job/'registered-stretch-patch/interior-patch.npz')['severe']
    inner=np.unique(g[severe]);inner=inner[~excluded[inner]]
    ring=np.unique(g[np.isin(g,inner).any(1)]);nodes=ring[~excluded[ring]]
    ix=np.flatnonzero(np.isin(g,nodes).any(1));t=g[ix]
    em=x[t[:,1:]]-x[t[:,:1]];det=np.linalg.det(em);inv=np.linalg.inv(em);weight=np.sqrt(np.maximum(det/np.median(det),.1))
    grad=np.concatenate([-inv.sum(axis=2)[:,:,None],inv],axis=2)
    mapped=np.full(len(x),-1,dtype=int);mapped[nodes]=np.arange(len(nodes));local=mapped[t];ok=local>=0
    row=np.broadcast_to(np.arange(len(t)*3).reshape(-1,3,1),grad.shape);col=np.broadcast_to(local[:,None,:],grad.shape);use=col>=0
    G=coo_matrix(((grad*weight[:,None,None])[use],(row[use],col[use])),shape=(3*len(t),len(nodes))).tocsr()
    A=(G.T@G).tocsc();S=diags(1/np.sqrt(A.diagonal()));factor=splu((S@A@S).tocsc())
    # Color the actual FEM adjacency. Within a color the proposed vertices
    # share no tetrahedron, so each patch objective and constraint is separable.
    adjacency=[set() for _ in nodes]
    for q in local:
        valid=q[q>=0].tolist()
        for j in valid:adjacency[j].update(valid)
    labels=np.full(len(nodes),-1)
    for i in np.argsort([-len(v) for v in adjacency],kind='stable'):
        used={labels[j] for j in adjacency[i]};k=0
        while k in used:k+=1
        labels[i]=k
    slots=np.where(ok);order=labels[local[slots]]
    batches=[]
    for col_id in range(labels.max()+1):
        mask=order==col_id;ii,js=(v[mask] for v in slots);owner=local[ii,js]
        if len(np.unique(ii))!=len(ii):raise ValueError('Conflicting cell updates')
        batches.append((ii,js,owner))
    def metrics(tri,iv,rt):
        F=iv@(tri[:,1:]-tri[:,:1]);u,sv,vt=np.linalg.svd(F);R=u@vt
        e=tri[:,1:]-tri[:,:1];six=np.linalg.det(e);L=np.array([np.sum((tri[:,a]-tri[:,b])**2,axis=1) for a in range(4) for b in range(a)]).max(0)**1.5
        return F,R,np.sum((sv-1)**2,axis=1),(sv[:,0]>2)|(sv[:,-1]<.5),six/rt,six,L,np.maximum(sv[:,0],1/sv[:,-1])
    base=y.copy();history=[]
    _,_,_,_,initial_J,initial_volume,initial_edge,_=metrics(y[t],inv,det)
    # Preserve the existing clearance from the constitutive limits in the
    # initially extreme cells. Improving shear must not purchase a worse
    # starting volume or a thinner already marginal tetrahedron.
    floor_J=np.minimum(.70,initial_J);ceiling_J=np.maximum(1.45,initial_J)
    floor_shape=np.minimum(1e-4,initial_volume/initial_edge)
    for rep in range(7):
        F,R,en,bad,J,vol,edge,worst=metrics(y[t],inv,det)
        history.append(dict(severe=int(bad.sum()),peak=float(worst.max()),energy=float(np.sum(weight**2*en)),J_range=[float(J.min()),float(J.max())],min_shape=float(np.min(vol/edge))))
        if rep==6:break
        delta=S@factor.solve(S@(G.T@((R-F)*weight[:,None,None]).reshape(-1,3)))
        count=0
        for ii,js,owner in batches:
            old=y[t[ii]]
            _,_,energy0,bad0,_,_,_,worst0=metrics(old,inv[ii],det[ii])
            remaining=np.unique(owner)
            for bt in range(18):
                mask=np.isin(owner,remaining)
                if not np.any(mask):break
                I,K,U=ii[mask],js[mask],owner[mask]
                candidate=(y[nodes[U]]+delta[U]*2.**-bt).astype('f4').astype(float)
                vals=old[mask].copy();vals[np.arange(len(I)),K]=candidate
                _,_,energy,bad1,j1,v1,L1,w1=metrics(vals,inv[I],det[I])
                valid=(np.linalg.norm(candidate-base[nodes[U]],axis=1)<.002)&(v1>6e-18)&(v1>1e-5*L1)&(j1>=floor_J[I]-1e-12)&(j1<=ceiling_J[I]+1e-12)&(v1/L1>=floor_shape[I]*(1-1e-12))&(w1<=history[0]['peak']*(1+1e-12))
                count_new=np.bincount(U,weights=bad1.astype(float)-bad0[mask],minlength=len(nodes))
                gain=np.bincount(U,weights=weight[I]**2*(energy0[mask]-energy),minlength=len(nodes))
                reject=np.bincount(U,weights=~valid,minlength=len(nodes))>0
                take=remaining[(~reject[remaining])&(gain[remaining]>1e-14)&(count_new[remaining]<=0)]
                if len(take):
                    y[nodes[take]]=(y[nodes[take]]+2.**-bt*delta[take]).astype('f4').astype(float)
                remaining=np.setdiff1d(remaining,take);count+=len(take)
        if not count:break
    diff=np.flatnonzero(np.any(base!=y,axis=1))
    if excluded[diff].any():raise ValueError('Altered physical anchors or outer surface')
    np.savez_compressed(out/'candidate-positions.npz',nodes=diff,positions=y[diff].astype('f4'))
    report=dict(pose=digest(mat/'current.node'),rest=digest(mat/'mesh.node'),elements=digest(mat/'mesh.ele'),affected_cells=len(ix),unaffected_severe=int(len(severe)-history[0]['severe']),colors=len(batches),moved_nodes=len(diff),maximum_displacement_m=float(np.linalg.norm(y-base,axis=1).max()),history=history)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(history[0],history[-1],flush=True)
if __name__=='__main__':run(*(Path(p).resolve() for p in sys.argv[1:]))
