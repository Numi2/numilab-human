#!/usr/bin/env python3
"""Offline source-refinement ledger; immutable parent measure is never replaced by registration geometry."""
import numpy as np

def _integer(a, shape, name):
    a=np.asarray(a)
    if a.ndim!=len(shape) or any(n is not None and n!=a.shape[i] for i,n in enumerate(shape)) or not np.issubdtype(a.dtype,np.integer):
        raise ValueError(name+" has invalid shape or noninteger domain")
    return a.astype(np.int64,copy=False)

def _volumes(x,t):
    p=x[t]
    return np.einsum("ij,ij->i",p[:,1]-p[:,0],np.cross(p[:,2]-p[:,0],p[:,3]-p[:,0]))/6

def build_refined_reference_ledger(*,source_points,source_tets,refined_points,refined_tets,
        child_source_tet_rows,child_source_measure_fractions,node_source_offsets,node_source_rows,
        node_source_weights,parent_reference_volumes,parent_mass,source_material,source_donor,
        source_region,source_stable_face):
    x=np.asarray(source_points,dtype=np.float64);y=np.asarray(refined_points,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=3 or y.ndim!=2 or y.shape[1]!=3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("finite source/refined 3D coordinates required")
    t=_integer(source_tets,(None,4),"source tetrahedra")
    u=_integer(refined_tets,(None,4),"refined tetrahedra")
    if not len(t) or not len(u) or t.min()<0 or t.max()>=len(x) or u.min()<0 or u.max()>=len(y):
        raise ValueError("tetrahedron node outside source/refined domain")
    if np.any(np.diff(np.sort(t,axis=1),axis=1)==0) or np.any(np.diff(np.sort(u,axis=1),axis=1)==0):
        raise ValueError("repeated tetrahedron corner")
    if len(np.unique(np.sort(t,axis=1),axis=0))!=len(t) or len(np.unique(np.sort(u,axis=1),axis=0))!=len(u):
        raise ValueError("duplicate source/refined tetrahedron")
    parent=_integer(child_source_tet_rows,(len(u),),"child parent rows")
    if parent.min()<0 or parent.max()>=len(t):raise ValueError("child parent outside source cell domain")
    frac=np.asarray(child_source_measure_fractions,dtype=np.float64)
    if frac.shape!=(len(u),) or not np.isfinite(frac).all() or np.any(frac<=0):
        raise ValueError("source fractions must be positive finite cell rows")
    offsets=_integer(node_source_offsets,(len(y)+1,),"node ancestry offsets")
    rows=_integer(node_source_rows,(None,),"node ancestry rows")
    weights=np.asarray(node_source_weights,dtype=np.float64)
    if weights.shape!=rows.shape or offsets[0]!=0 or offsets[-1]!=len(rows) or np.any(np.diff(offsets)<=0) or rows.min()<0 or rows.max()>=len(x) or not np.isfinite(weights).all() or np.any(weights<=0):
        raise ValueError("invalid nonempty source-node ancestry CSR")
    reconstructed=np.empty_like(y);ancestry=[]
    for i,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
        rr=rows[a:b];ww=weights[a:b]
        if np.any(np.diff(rr)<=0) or abs(ww.sum()-1)>5e-14:raise ValueError("node ancestry unsorted, duplicate or not normalized")
        reconstructed[i]=ww@x[rr];ancestry.append(set(rr.tolist()))
    reconstruction_error=float(np.max(np.linalg.norm(reconstructed-y,axis=1)))
    if reconstruction_error>1e-13:raise ValueError("refined source coordinates do not reconstruct from pinned source")
    for child,original in enumerate(parent):
        allowed=set(t[original].tolist())
        if any(not ancestry[node].issubset(allowed) for node in u[child]):
            raise ValueError("child source-node ancestry escapes original parent tetrahedron")
    sv=_volumes(x,t);rv=_volumes(y,u);fv=_volumes(y.astype(np.float32).astype(np.float64),u)
    if np.any(sv<=0) or np.any(rv<=0) or np.any(fv<=0):raise ValueError("nonpositive source/refined/emitted Float32 cell")
    closure=np.bincount(parent,weights=frac,minlength=len(t))
    if not np.allclose(closure,1,rtol=0,atol=2e-14):raise ValueError("source child fractions do not close original parents")
    volume_error=float(np.max(np.abs(rv/sv[parent]-frac)))
    if volume_error>2e-11:raise ValueError("source fractions disagree with connected geometric subdivision")
    pv=np.asarray(parent_reference_volumes,dtype=np.float64);pm=np.asarray(parent_mass,dtype=np.float64)
    if pv.shape!=(len(t),) or pm.shape!=pv.shape or not np.isfinite(pv).all() or not np.isfinite(pm).all() or np.any(pv<=0) or np.any(pm<=0):
        raise ValueError("invalid immutable parent measure/mass domain")
    labels={}
    for name,value in [('material_index',source_material),('geometric_donor_body',source_donor),
                       ('anatomical_region_id',source_region),('stable_source_face_id',source_stable_face)]:
        value=_integer(value,(len(t),),name)
        if np.any(value<0):raise ValueError("negative typed anatomical identity")
        labels[name]=value[parent]
    if not np.isin(labels['material_index'],[15,16]).all():raise ValueError("unsupported skin/fat material identity")
    measure=pv[parent]*frac
    mass=pm[parent]*frac
    # This is the operational reference geometry, not the immutable material measure.
    density=mass/fv
    lower=np.where(labels['material_index']==15,900.,850.);upper=np.where(labels['material_index']==15,1100.,950.)
    if np.any(density<lower) or np.any(density>upper):raise ValueError("refined canonical Float32 density violates native material bounds")
    mass_closure=float(np.max(np.abs(np.bincount(parent,weights=mass,minlength=len(t))-pm)))
    measure_closure=float(np.max(np.abs(np.bincount(parent,weights=measure,minlength=len(t))-pv)))
    if mass_closure>1e-13 or measure_closure>1e-15:raise ValueError("immutable parent reference ledger does not close")
    return dict(child_parent_offsets=np.arange(len(u)+1,dtype=np.int64),child_parent_rows=parent.copy(),
        child_parent_reference_measure_m3=measure,child_mass_kg=mass,child_reference_volume_m3=fv,
        source_child_fraction=frac.copy(),child_implied_density_kg_m3=density,
        **labels,diagnostics=dict(source_reconstruction_max_m=reconstruction_error,
          source_fraction_geometry_max_error=volume_error,maximum_parent_mass_closure_kg=mass_closure,
          maximum_parent_measure_closure_m3=measure_closure,
          density_min_kg_m3=float(density.min()),density_max_kg_m3=float(density.max())))
