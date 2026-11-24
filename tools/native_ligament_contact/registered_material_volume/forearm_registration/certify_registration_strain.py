"""Check deformation modes at every cell incident to an edited material registration."""
from pathlib import Path
import argparse,json
import numpy as np
from compile_connected_joint import digest

def deformation(reference,pose,cells):
    reference_edges=reference[cells[:,1:]]-reference[cells[:,:1]]
    posed_edges=pose[cells[:,1:]]-pose[cells[:,:1]]
    return np.linalg.solve(reference_edges,posed_edges)

def compare(reference,prior,trial,t,global_ids):
    F0,F=deformation(reference,prior,t),deformation(reference,trial,t)
    s0,s=np.linalg.svd(F0,compute_uv=False),np.linalg.svd(F,compute_uv=False)
    # This engineering comparison preserves prior stretch outside a 10 percent
    # near-identity window, allowing 0.5 percent for the emitted candidate.
    # Physical Jacobian and quality gates run independently.
    hi=s[:,0]/np.maximum(s0[:,0],1.1)
    lo=np.minimum(s0[:,-1],.9)/np.maximum(s[:,-1],np.finfo(s.dtype).tiny)
    failed=(hi>1.005)|(lo>1.005)|(np.linalg.det(F)<=0)|~np.isfinite(s).all(1)
    peak=np.argmax(np.maximum(hi,lo))
    return dict(cells=len(t),failed=global_ids[failed].tolist(),worst_cell=int(global_ids[peak]),
                worst_ratio=float(max(hi[peak],lo[peak])),prior_stretches=s0[peak].tolist(),
                candidate_stretches=s[peak].tolist(),min_J=float(np.linalg.det(F).min()))

def run(vol,fit,motion,output):
    read=lambda p:np.loadtxt(p,skiprows=1,usecols=(1,2,3))
    X=read(vol/'mesh.node');Y=read(vol/'current.node');P=read(fit/'current.node')
    rec=json.loads((fit/'fit.json').read_bytes())
    for p,k in [(vol/'mesh.node','source_reference'),(vol/'mesh.ele','elements'),(vol/'current.node','control_loaded'),(fit/'current.node','registered')]:
        if digest(p)!=rec[k]:raise ValueError('Different geometry '+k)
    if Y.shape!=P.shape or not np.all(np.isfinite(P)):raise ValueError('Invalid candidate')
    nodes=np.flatnonzero(np.any(Y!=P,1))
    if not np.array_equal(nodes,np.sort(rec['nodes_0_based'])):raise ValueError('Undeclared coordinate change')
    mesh=np.loadtxt(vol/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    tids=np.flatnonzero(np.isin(mesh[:,:4],nodes+1).any(1))
    if not len(tids):raise ValueError('Changed nodes have no mechanical incidence')
    T=mesh[tids,:4]-1;z=read(motion);results={}
    if z.shape!=X.shape or not np.isfinite(z).all():raise ValueError('Invalid baseline motion')
    for name,a,b in [('registered',Y,P),('retained_displacement',z,z+(P-Y))]:
        roles={}
        for role in np.unique(mesh[tids,4]):
            mask=mesh[tids,4]==role;roles[str(role)]=compare(X,a,b,T[mask],tids[mask])
        results[name]=roles
    failed=sum(len(v['failed']) for r in results.values() for v in r.values())
    d=dict(changed_nodes=len(nodes),affected_cells=len(T),identical_cells=len(mesh)-len(T),material_cells=len(mesh),geometry=digest(fit/'current.node'),measured_control_motion=digest(motion),frames=results,failed_checks=failed)
    output.write_text(json.dumps(d,indent=2)+'\n');print(f'{output.parent.name} {len(T)} affected cells, {failed} failed frame/cell comparisons')
    return failed
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('volume','fit','motion','output'):p.add_argument(n,type=Path)
    a=p.parse_args()
    raise SystemExit(bool(run(a.volume,a.fit,a.motion,a.output)))
