"""Find a common interior point of a fat cavity in reference and registered poses."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linprog
from condition_motion_witness import altitude
from compile_connected_joint import digest
from material_cavity_flips import boundary
from supported_material_admission import geom

def fan_point(x,y,old,preserved):
    verts=np.unique(old);f=np.array(boundary(old));interior=np.setdiff1d(verts,f)
    if len(interior)>1 or np.intersect1d(interior,preserved).size:return None
    # Barycentric placement on cavity vertices transports a real material point
    # between registrations and exactly reproduces every common rigid or affine map.
    vertices=np.unique(f);m=len(vertices);constraints=[];bounds=[]
    for p in (x,y):
        z=p[f]-p[vertices[0]]
        unit=np.cross(z[:,1]-z[:,0],z[:,2]-z[:,0]);unit/=np.linalg.norm(unit,axis=1)[:,None]
        local=p[vertices]-p[vertices[0]]
        constraints.append(np.column_stack((unit@local.T,np.ones(len(f)))))
        bounds.append(np.einsum('ij,ij->i',unit,z[:,0]))
    opt=linprog(np.r_[np.zeros(m),-1],A_ub=np.vstack(constraints),b_ub=np.concatenate(bounds),A_eq=np.array([np.r_[np.ones(m),0]]),b_eq=[1],bounds=[(0,1)]*m+[(0,None)],method='highs')
    if not opt.success or opt.x[-1]<=1e-8:return None
    local=np.searchsorted(vertices,f);nt=np.column_stack((np.full(len(f),m),local))
    positions=[];qualities=[]
    for p in (x,y):
        new=(p[vertices].T@opt.x[:m]).astype('f4').astype(float)
        G=np.vstack((p[vertices],new))
        v,s=geom(G,nt);ov,os=geom(p,old)
        if (v<=1e-18).any() or s.min()<1e-5 or abs(v.sum()-ov.sum())>1e-10*ov.sum():return None
        qualities.append(float(altitude(G,nt).min()*1e6));positions.append(new)
    J=geom(np.vstack((y[vertices],positions[1])),nt)[0]/geom(np.vstack((x[vertices],positions[0])),nt)[0]
    if min(J)<.4 or max(J)>2.5:return None
    return dict(altitude_um=min(qualities),apex_reference=positions[0],apex_registered=positions[1],weights=opt.x[:m],boundary_nodes=vertices,faces=f,interior=interior)
def run(work):
    q=work.parent;out=work/'witness-cavity';v=q/'qualified-material'
    get=lambda f:np.loadtxt(f,skiprows=1,usecols=(1,2,3))
    x=get(v/'mesh.node');y=get(work/'selected-pose/current.node')
    t=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype='i4')-1
    z=np.load(v/'material-walls.npz');protected=np.unique(np.concatenate([a.ravel()for a in z.values()]))
    d=json.loads((out/'result.json').read_bytes());attempts=[];best=None
    for a in d['attempts']:
        rows=np.array(a['cells']);p=fan_point(x,y,t[rows],protected)
        attempts.append(dict(rows=rows.tolist(),minimum_altitude_um=p['altitude_um']if p else None))
        if p and(best is None or p['altitude_um']>best['altitude_um']):best=dict(source_cells=rows,**p)
    result=dict(witness=d['witness'],baseline_minimum_altitude_um=d['prior_minimum_altitude_um'],trials=attempts)
    if best:
        f=out/'material-star.npz';np.savez_compressed(f,**best);result.update(candidate=digest(f),minimum_altitude_um=best['altitude_um'],source_cells=best['source_cells'].tolist(),cells=len(best['faces']))
    (out/'star.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':run(Path(__file__).resolve().parent)
