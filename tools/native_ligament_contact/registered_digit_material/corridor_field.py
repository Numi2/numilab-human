"""Fit a single compact volume-preserving field to bilateral digital material landmarks.

The source guide sites and inferred terminal footprints establish the desired
dorsal space. Integrate a smooth divergence-free field on the existing
connected skin/fat material coordinates. Discrete cells still require exact
emitted-volume, mass, contact and native acceptance checks.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from transfer_connected_volume import geometry
from material_context import get_layers, read_inputs, unit
from hand_geometry import Surface

def kernel(diff,radius):
    s=np.linalg.norm(diff,axis=-1)/radius
    h=np.maximum(0,1-s)
    # (-Laplacian I + Hessian) of a radial C4 Wendland function in 3-D.
    anis=15*h**4
    iso=(5*s+1)*h**5-anis*s*s
    u=diff/radius
    return iso[...,None,None]*np.eye(3)+anis[...,None,None]*u[...,None]*u[...,None,:]

def field(x,centers,coeff,radius,dt):
    tree=cKDTree(x)
    stencils=tree.query_ball_point(centers,radius+2*.006)
    def velocity(points):
        v=np.zeros_like(points)
        for p,c,ids in zip(centers,coeff,stencils,strict=True):
            ids=np.asarray(ids)
            if len(ids):v[ids]+=kernel(points[ids]-p,radius)@c
        return v
    y=x.copy()
    for _ in range(8):
        a=velocity(y)
        y+=dt/8*velocity(y+(.5*dt/8)*a)
    if np.linalg.norm(y-x,axis=1).max()>.012:raise ValueError('Trajectory exceeded stencil reserve')
    return y

def fit(w,m,c,material,fp_src,out):
    routes,users,solids,bybody,poses=read_inputs(w/'current-package.json',m,c/'emitted-native/stderr.log')
    surf={k:Surface(*v) for k,v in get_layers(w/'qualified/native-coupled-006/native/resting-human.mrvpack',w/'qualified/motion-008/step-8.receipt.json').items()}
    fp=json.loads(fp_src.read_text());h=.00102
    guides=json.loads(Path('direct-sites.json').read_text())
    centers=[];delta=[];targets=[]
    def prescribe(q,n,gap,needed,label):
        # Place the interpolation condition on the admitted deep-fat material itself.
        I=q+gap*n;centers.append(I);delta.append(max(0.,needed-gap)*n)
        targets.append(dict(label=label,origin=q.tolist(),normal=n.tolist(),before_mm=1000*gap,target_mm=1000*needed))
    for s in guides:
        if s['order']==8:continue
        n=np.array(s['normal']);q=np.array(s['near_bone']);p=np.array(s['world'])
        gap=s['normal_hits_mm']['deep']
        if not gap:raise ValueError('Missing deep scan on assigned phalanx')
        needed=max(h,((p-q)@n)+h/2)
        prescribe(q,n,gap[0]/1000,needed,f"{s['name']}/{s['order']}")
    for fp0 in fp['footprints']:
        surface=Surface(*solids[fp0['bone_bank']])
        for ln in fp0['lines']:
            # The distal end tapers onto the cortex. Reserve dorsal envelope
            # space at the sampled basal section; this is not a gap between
            # the enthesis and its osseous insertion.
            s=ln['samples'][1];q=np.array(s['world_m']);a,b,c0=surface.tri[s['face']];n=unit(np.cross(b-a,c0-a))
            gap=s['layers']['deep']['normal_gap_mm']
            if gap is None or not s['layers']['deep']['inside']:raise ValueError('Invalid base corridor')
            prescribe(q,n,gap/1000,h,fp0['muscle']+'/base/'+str(ln['lateral_fraction']))
    centers=np.asarray(centers);delta=np.asarray(delta);radius=.017
    mats=kernel(centers[:,None]-centers[None,:],radius).transpose(0,2,1,3).reshape(3*len(centers),-1)
    coeff=np.linalg.solve(mats+np.eye(len(mats))*.03,delta.reshape(-1)).reshape(-1,3)
    mat=np.load(material);x=mat['x'];tet=mat['tets'];tags=mat['tags']
    v0,q0=geometry(x,tet);assert v0.min()>1e-18 and q0.min()>1e-5
    local=np.unique(np.concatenate(cKDTree(x).query_ball_point(centers,radius+.012))).astype(int)
    changed=np.zeros(len(x),bool);changed[local]=True
    incident=np.flatnonzero(np.any(changed[tet],axis=1))
    tree=cKDTree(x)
    local_layers={}
    for k,s in surf.items():
        dst,inds=tree.query(s.v)
        assert np.all(dst==0) and len(np.unique(inds))==len(s.v)
        local_layers[k]=(inds,s.f)
    hist=[]
    # Retain the largest connected registration that clears the construction gates.
    # Evaluate each proposal from the identical reference; never accumulate rounding.
    for t in (1.,.5,.25,.125,.0625):
        y=x.copy();y[local]=field(x[local].astype(float),centers,coeff,radius,t).astype('f4')
        vol,quality=geometry(y,tet[incident])
        r=dict(fraction=t,invalid_volume=int((vol<=1e-18).sum()),invalid_quality=int((quality<=1e-5).sum()),incident_cells=len(incident),
               moved_nodes=int(np.any(y!=x,axis=1).sum()),worst_volume_m3=float(vol.min()),worst_quality=float(quality.min()),
               max_displacement_mm=float(np.linalg.norm(y.astype(float)-x,axis=1).max()*1000),
               minimum_local_volume_ratio=float(np.min(vol/v0[incident])),maximum_local_volume_ratio=float(np.max(vol/v0[incident])),
               material_volume_relative={str(tag):float((vol[tags[incident]==tag]-v0[incident][tags[incident]==tag]).sum()/v0[tags==tag].sum()) for tag in np.unique(tags)})
        hist.append(r);print(r,flush=True)
        if t==1.:np.save(out/'full-proposal.npy',y)
        if r['invalid_volume'] or r['invalid_quality']: continue
        np.save(out/'cell-admitted.npy',y);break
    (out/'history.json').write_text(json.dumps(dict(history=hist,targets=targets),indent=2)+'\n')
if __name__=='__main__':fit(*map(Path,sys.argv[1:]))
