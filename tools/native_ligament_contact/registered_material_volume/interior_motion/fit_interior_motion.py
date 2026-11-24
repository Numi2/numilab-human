"""Fit volume-interior sampling sites over the observed support motion, preserving all material walls."""
import argparse,ctypes,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from compile_connected_joint import digest
from supported_material_admission import geom

def star_geometry(x, cells, vertex, point):
    points=x[cells].copy()
    points[cells==vertex]=point
    e=points[:,1:]-points[:,:1]
    v=np.einsum('ij,ij->i',e[:,0],np.cross(e[:,1],e[:,2]))/6
    length=np.linalg.norm(points[:,[0,0,0,1,1,2]]-points[:,[1,2,3,2,3,3]],axis=2).max(axis=1)
    return v, 6*v/length**3, length

def run(parent,out):
    m=parent/'witness-material'; vol=m/'volume'; report=json.loads((m/'motion-quality.json').read_bytes())
    paths=[vol/'mesh.node',vol/'current.node',parent/'accepted-material-07/physical-accepted.node',parent/'half-step-material-08/physical-accepted.node']
    if digest(paths[2])!=report['geometry'] or digest(vol/'mesh.ele')!=report['elements']:raise ValueError('Mismatched geometry')
    poses=[np.loadtxt(p,skiprows=1,usecols=(1,2,3)) for p in paths]
    cells=np.loadtxt(vol/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype=np.int32)-1
    labels=np.loadtxt(vol/'current-child.labels.csv',skiprows=1,delimiter=',',dtype=np.int64)
    walls=np.load(vol/'material-walls.npz');fixed=np.unique(np.concatenate([a.ravel() for a in walls.values()]))
    bad=np.array([r['cell'] for r in report['witnesses']])
    nodes=np.setdiff1d(np.unique(cells[bad]),fixed);ix=np.flatnonzero(np.isin(cells,nodes).any(axis=1))
    j=parent.parent.parent.parent
    path=Path((j/'storage/interval-checker-locator.txt').read_text().split()[1])
    if digest(path)!=json.loads((j/'verified-connected-material/interval-audit.json').read_bytes())['checker']:
        raise ValueError('Unpinned native interval evaluator')
    dp=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS');ip=np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS')
    check=ctypes.CDLL(str(path)).scanCellIntervals;check.argtypes=[dp,dp,ctypes.c_size_t,ip,ctypes.c_size_t,dp]
    candidates=[]
    for node in nodes:
        rows=ix[(cells[ix]==node).any(axis=1)]
        t=cells[rows]
        if len(np.unique(labels[rows,1]))!=1:raise ValueError('Unprotected material interface')
        baseline=[star_geometry(x,t,node,x[node]) for x in poses[:3]]
        unique,local=np.unique(t,return_inverse=True)
        local=np.ascontiguousarray(local.reshape(t.shape),dtype=np.int32)
        local_i=int(np.searchsorted(unique,node))
        baseline_interval=[np.ascontiguousarray(x[unique]) for x in poses[:2]]
        buffer=np.zeros((len(t),3))
        best=None
        # The candidate stays inside an existing incident tetrahedron. Its
        # barycentric map is shared by the reference and all observed states.
        for c in t:
            neighbors=c[c!=node]
            v0=np.stack([s[0] for s in baseline])
            vside=np.array([[star_geometry(x,t,node,x[j])[0] for j in neighbors] for x in poses[:3]])
            dv=(vside-v0[:,None,:]).transpose(0,2,1)
            N=len(rows)
            lower=[];rhs=[]
            for state,(_,shape,length) in enumerate(baseline):
                scale=length**3*1e-5/6
                lower.extend(np.column_stack((-dv[state]/scale[:,None],np.ones(N))))
                rhs.extend(v0[state]/scale)
            for state in (1,2):
                for sign,factor in [(1,.35),(-1,2.5)]:
                    diff=sign*(v0[state]-factor*v0[0]); d=sign*(dv[state]-factor*dv[0])
                    lower.extend(np.column_stack((-d/v0[0,:,None],np.zeros(N))))
                    rhs.extend(diff/v0[0])
            lower.append([1,1,1,0]);rhs.append(.25)
            result=linprog([0,0,0,-1],A_ub=lower,b_ub=rhs,bounds=[(0,.25)]*3+[(1.02,4)],method='highs')
            if not result.success:continue
            for fraction in (1.,.75,.5,.25):
                w=fraction*result.x[:3]
                pts=[x[node]+w@(x[neighbors]-x[node]) for x in poses[:3]]
                # The native reference loader stores Float32 coordinates. Audit
                # the exact representation of each authored point.
                pts[:2]=[a.astype(np.float32).astype(np.float64) for a in pts[:2]]
                vals=[star_geometry(x,t,node,y) for x,y in zip(poses,pts)]
                if any((v<=1e-18).any() or (s<1e-5).any() for v,s,_ in vals):continue
                if any(np.min(vals[k][0]/vals[0][0])<.35 or np.max(vals[k][0]/vals[0][0])>2.5 for k in (1,2)):continue
                u,v=baseline_interval
                u[local_i],v[local_i]=pts[:2]
                if check(u,v,len(u),local,len(local),buffer):raise ValueError('Invalid interval request')
                if np.any(buffer[:,0]-buffer[:,1]<=float(np.float32(.35))) or np.any(buffer[:,0]+buffer[:,1]>=2.5):continue
                dist=np.linalg.norm(pts[0]-poses[0][node])
                if best is None or dist < best['reference_displacement_m']:
                    best=dict(vertex=int(node), source_cell=int(rows[np.where((t==c).all(axis=1))[0][0]]),
                             neighbor_nodes=neighbors.tolist(), local_weights=w.tolist(), affected_cells=rows.tolist(),
                             reference_displacement_m=float(dist), training_bad_cells_repaired=int((baseline[2][1]<1e-5).sum()))
        if best:candidates.append(best)
    occupied=set();selected=[]
    for c in sorted(candidates,key=lambda r:(-r['training_bad_cells_repaired'],r['reference_displacement_m'],r['vertex'])):
        if occupied.isdisjoint(c['affected_cells']):
            occupied.update(c['affected_cells']);selected.append(c)
    for c in selected:
        node=c['vertex']; ids=c['neighbor_nodes'];w=np.array(c['local_weights']); t=cells[c['affected_cells']]
        x=poses[-1];old=star_geometry(x,t,node,x[node]);new=star_geometry(x,t,node,x[node]+w@(x[ids]-x[node]))
        c['heldout_before']=int((old[1]<1e-5).sum());c['heldout_after']=int((new[1]<1e-5).sum())
        c['heldout_nonpositive']=int((new[0]<=0).sum())
    out.write_text(json.dumps(dict(physical_mesh=report['elements'],poses=[digest(p) for p in paths],boundary_digest=digest(vol/'material-walls.npz'),
        starting_motion_defects=len(bad),all_vertices_on_protected_walls=int(np.all(np.isin(cells[bad],fixed),axis=1).sum()),
        candidates=candidates,selected=selected,scope='Frozen state basis correction; native force and mass comparisons required.'),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('parent','out'):p.add_argument(k,type=Path)
    run(**vars(p.parse_args()))
