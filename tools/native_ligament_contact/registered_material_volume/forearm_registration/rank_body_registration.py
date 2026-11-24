"""Attribute the full registered skin/fat internal force to anatomical source regions for initialization."""
from pathlib import Path
import json,sys
import numpy as np
from prepare_indentation import soft_response
from compile_connected_joint import digest

def run(v,material,fit,output):
    read=lambda p:np.loadtxt(p,skiprows=1,usecols=(1,2,3))
    x=read(v/'mesh.node');y=read(v/'current.node');p=read(fit/'current.node')
    raw=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');role=raw[:,4];t=raw[:,:4]-1
    L=np.loadtxt(v/'current-child.labels.csv',delimiter=',',skiprows=1,dtype='i8')
    if not np.array_equal(L[:,0],np.arange(len(t))) or not np.array_equal(L[:,1],role):raise ValueError('Unmatched source labels')
    coeff=json.loads(material.read_bytes())['materials'];dims=[np.zeros(len(t)) for _ in range(2)]
    for m in np.unique(role):
        for f,key in zip(dims,['source_mu_pa','source_logJ_coefficient_pa']):f[role==m]=coeff[str(m)]['registered'][key]
    changed_nodes=np.flatnonzero(np.any(y!=p,1));changed=np.isin(t,changed_nodes).any(1)
    def force(indices,pts):
        return soft_response(x,pts,t[indices],*[f[indices]for f in dims])(pts)[0]
    def scatter(total,indices,pts,scale=1.):
        for start in range(0,len(indices),16384):
            block=indices[start:start+16384];r=force(block,pts)*scale
            np.add.at(total,t[block].ravel(),r.reshape(-1,3))
    f0=np.zeros_like(x);rows=np.arange(len(t));scatter(f0,rows,y)
    f1=f0.copy();idx=np.flatnonzero(changed);scatter(f1,idx,y,-1);scatter(f1,idx,p)
    def summary(f,pts):
        norm=np.linalg.norm(f,axis=1);order=np.argsort(norm)[-12:][::-1];top=[]
        for node in order:
            ci,corner=np.nonzero(t==node);component=force(ci,pts)[np.arange(len(ci)),corner]@(f[node]/norm[node]);i=np.argmax(abs(component))
            top.append(dict(node_zero_based=int(node),net_internal_N=float(norm[node]),dominant_cell=int(ci[i]),dominant_material=int(role[ci[i]]),donor_body=int(L[ci[i],2]),dominant_along_net_N=float(component[i])))
        return dict(total_squared_nodal_force_N2=norm@norm,net_force_N=f.sum(0).tolist(),nodes_above_10_N=int((norm>10).sum()),nodes_above_100_N=int((norm>100).sum()),top=top)
    a,b=summary(f0,y),summary(f1,p)
    d=dict(reference_sha256=digest(v/'mesh.node'),topology_sha256=digest(v/'mesh.ele'),source_coefficients=digest(material),prior_pose=digest(v/'current.node'),candidate_pose=digest(fit/'current.node'),cells=len(t),nodes=len(x),changed_cells=len(idx),before=a,after=b,
           scope='Internal quasistatic elastic forces at initialization. The native coupled residual also includes inertia, support and other interactions.')
    output.write_text(json.dumps(d,indent=2)+'\n');print({k:[z['nodes_above_100_N'],z['nodes_above_10_N'],z['top'][0]]for k,z in [('before',a),('after',b)]})
if __name__=='__main__':run(*(Path(p)for p in sys.argv[1:]))
