"""Refine the connected bed-bearing skin and fat without changing their reference material."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from certify_interior_motion import response
from compile_connected_joint import digest

EDGES = np.array(list(combinations(range(4), 2)))
FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])


def boundary(t):
    faces, counts = np.unique(np.sort(t[:, FACES].reshape(-1, 3), axis=1), axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('Nonmanifold tetrahedral face')
    return faces[counts == 1]


def close_to(p, center, radius):
    return np.linalg.norm(np.maximum(np.maximum(p.min(1)-center, center-p.max(1)), 0), axis=1) <= radius


def split_rows(t, a, b, m):
    mask = (t == a).any(1) & (t == b).any(1)
    left, right = t[mask].copy(), t[mask].copy()
    left[left == b], right[right == a] = m, m
    return np.concatenate((t[~mask], left, right)), mask


def determinant(p, t):
    return np.linalg.det(p[t[:, 1:]] - p[t[:, :1]])


def quality(x, t):
    diameter = np.linalg.norm(x[t[:, EDGES[:, 1]]] - x[t[:, EDGES[:, 0]]], axis=2).max(1)
    return determinant(x, t)/diameter**3


def refine(w, out, radius, spacing):
    case = w/'native-cavity-bound'
    vol = w/'material-import/volume'
    audit = json.loads((case/'material-check.json').read_bytes())
    bed = json.loads((case/'support-check.json').read_bytes())
    corridor = json.loads((w/'bed-corridor.json').read_bytes())
    paths = [vol/'mesh.node', vol/'current.node', case/'physical-accepted.node']
    pins = [digest(f) for f in paths]
    if pins != [audit[k] for k in ('reference', 'loaded', 'physical_frame')] or corridor['material_endpoint'] != pins[-1]:
        raise ValueError('Skin/fat input does not match captured registers')
    coef = json.loads((w.parent.parent/'registration-strain.json').read_bytes())['materials']
    frames = [np.loadtxt(f, skiprows=1, usecols=(1,2,3)) for f in paths]
    center = frames[-1][corridor['node']]
    raw = np.loadtxt(vol/'mesh.ele', skiprows=1, usecols=(1,2,3,4,5), dtype='i4')
    tet = raw[:, :4] - 1
    groups = [r[close_to(frames[-1][tet[r]], center, radius + .002)] for r in np.array_split(np.arange(len(tet)), 64)]
    rows = np.concatenate(groups)
    ids, inv = np.unique(tet[rows], return_inverse=True)
    base = inv.reshape(-1,4)
    orig = np.stack([s[ids] for s in frames])
    walls = np.load(vol/'material-walls.npz')
    surface, face_ids = {}, {}
    for k in walls.files:
        f = walls[k]
        take = np.flatnonzero(np.isin(f, ids).all(1))
        surface[k] = np.searchsorted(ids, f[take])
        face_ids[k] = take
    original_surface = {k: s.copy() for k, s in surface.items()}
    t = base.copy()
    root = np.arange(len(t))
    portion = np.ones(len(t))
    X = orig.copy()
    rows_B = list(np.eye(len(ids)))
    history = []
    seam = {tuple(f) for f in boundary(t)}
    for s in surface.values():
        seam -= set(map(tuple, np.sort(s, axis=1)))
    seam_nodes = np.asarray(sorted(seam),dtype=int).reshape(-1,3)
    for it in range(8000):
        edge = np.unique(np.sort(t[:, EDGES].reshape(-1,2), axis=1), axis=0)
        start, end = X[-1, edge[:,0]], X[-1, edge[:,1]]
        diff = end-start
        lensq = np.einsum('ij,ij->i', diff,diff)
        u = np.clip(np.einsum('ij,ij->i',center-start,diff)/lensq, 0, 1)
        dist = np.linalg.norm(start+u[:,None]*diff-center,axis=1)
        active = np.flatnonzero((dist<=radius) & (lensq>spacing**2))
        if not len(active):
            break
        a,b = edge[active[lensq[active].argmax()]]
        if len(seam_nodes) and np.any((seam_nodes==a).any(1)&(seam_nodes==b).any(1)):
            raise ValueError('Edge would break the untouched material interface')
        m = len(rows_B)
        midpoint = (X[:,a]+X[:,b])*.5
        # Round the rest vertex in the native precision, then reconstruct its two-word displacement.
        shift = midpoint[0].astype('f4').astype(float)-midpoint[0]
        midpoint += shift
        delta = midpoint[1:]-midpoint[0]
        hi=delta.astype('f4').astype(float)
        midpoint[1:]=midpoint[0]+hi+(delta-hi).astype('f4').astype(float)
        X = np.concatenate((X,midpoint[:,None,:]),axis=1)
        rows_B.append((rows_B[a]+rows_B[b])*.5)
        history.append((a,b,m))
        t,mask = split_rows(t,a,b,m)
        root=np.r_[root[~mask],root[mask],root[mask]]
        portion=np.r_[portion[~mask],portion[mask]*.5,portion[mask]*.5]
        for k,s in surface.items():
            surface[k]=split_rows(s,a,b,m)[0]
    else:
        raise ValueError('Local refinement budget exhausted')
    if not np.array_equal(np.bincount(root,weights=portion,minlength=len(base)),np.ones(len(base))):
        raise ValueError('Nonconservative material ownership')
    # Verify shared and external walls with their actual role incidence.
    def wall_faces(tt, rr, ss):
        result={}
        for role in (15,16):
            shell=set(map(tuple,boundary(tt[rr==role])))
            for k,f in ss.items():
                fset=set(map(tuple,np.sort(f,axis=1)))
                result[role,k]=(len(fset & shell),len(fset))
                shell-=fset
            result[role,'cut']=shell
        return result
    old_role=raw[rows,4]; role=old_role[root]
    before,after=wall_faces(base,old_role,original_surface),wall_faces(t,role,surface)
    for key in before:
        k,v=key
        if v=='cut':
            if before[key]!=after[key]: raise ValueError('Unmatched boundary of connected material')
        elif (before[key][0]==before[key][1]) != (after[key][0]==after[key][1]) or (before[key][0]==0)!=(after[key][0]==0):
            raise ValueError('Material wall incidence changed')
    B=csr_matrix(np.asarray(rows_B))
    old_mass=np.fromfile(vol/'cell-mass.f64',dtype='<f8')[rows]
    # Float32 emission perturbs a few centroids. Preserve each individual
    # parent's reference mass and first moment using its own positive children.
    mass_fraction = portion.copy()
    centers = X[0,t].mean(1)
    moment_residual = 0.
    for i,parent in enumerate(base):
        child = np.flatnonzero(root==i)
        weights = portion[child]
        pos = orig[0,parent]
        centroid = pos.mean(0)
        natural = (centers[child]-centroid) @ np.linalg.inv(pos[1:]-pos[:1])
        A = np.vstack((np.ones(len(child)),natural.T))
        target = np.array([1.,0.,0.,0.])
        scale = np.sqrt(weights)
        delta = np.linalg.lstsq(A*scale,target-A@weights,rcond=1e-11)[0]*scale
        adjusted = weights+delta
        if adjusted.min()<=0 or abs(adjusted.sum()-1)>1e-12 or np.max(abs(delta/weights))>.01:
            raise ValueError('Unrepresentable parent mass map')
        mass_fraction[child] = adjusted
        moment_residual = max(moment_residual,np.linalg.norm(adjusted@centers[child]-centroid))
    if moment_residual>1e-10:
        raise ValueError('Unresolved material first moment')
    mass=old_mass[root]*mass_fraction
    checks={}
    for mat in (15,16):
        src=old_role==mat; dst=role==mat; props=coef[str(mat)]['registered']
        mu,bulk=props['source_mu_pa'],props['source_logJ_coefficient_pa']
        results=[]
        for k in range(len(X)):
            vv=determinant(X[k],t[dst]); old=determinant(orig[k],base)
            e,f,_=response(orig[0],orig[k],base[src],mu,bulk)
            ee,ff,_=response(X[0],X[k],t[dst],mu,bulk)
            q=quality(X[k],t[dst])
            q0=quality(orig[k],base[src])
            ferr=np.linalg.norm(B.T@ff-f)/max(np.linalg.norm(f),1e-8)
            results.append(dict(minimum_shape=float(q.min()),old_minimum_shape=float(q0.min()),bad_shape=int((q<1e-5).sum()),old_bad_shape=int((q0<1e-5).sum()),min_relative_partition=float((vv/(old[root[dst]]*portion[dst])).min()),maximum_volume_fraction_error=float(abs(vv/(old[root[dst]]*portion[dst])-1).max()),energy_J=float(ee),energy_error_J=float(abs(ee-e)),force_error_N=float(np.linalg.norm(B.T@ff-f)),relative_force_error=float(ferr)))
        first_moment=np.linalg.norm(mass[dst]@(X[0,t[dst]].mean(1)-center)-old_mass[src]@(orig[0,base[src]].mean(1)-center))
        checks[mat]=dict(source_cells=int(src.sum()),children=int(dst.sum()),mass_kg=float(mass[dst].sum()),mass_error_kg=float(abs(mass[dst].sum()-old_mass[src].sum())),first_moment_error_kg_m=float(first_moment),frames=results)
    for r in checks.values():
        if r['mass_error_kg']>1e-12 or r['first_moment_error_kg_m']>1e-11:
            raise ValueError('Lost anatomical material mass or first moment')
        for i in r['frames']:
            if (i['bad_shape'] or i['min_relative_partition']<=0 or i['maximum_volume_fraction_error']>.01
                or i['relative_force_error']>.01 and i['force_error_N']>1e-8):
                raise ValueError('Fidelity gate failed')
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'material.npz', ids=ids,source_rows=rows,base=base,t=t,source_parent=root,fraction=portion,mass_fraction=mass_fraction,mass=mass,roles=role,X=X,splits=np.array(history),**surface,**{'original_'+k:v for k,v in original_surface.items()},**{'rows_'+k:v for k,v in face_ids.items()})
    report=dict(input=pins,elements=digest(vol/'mesh.ele'),walls=digest(vol/'material-walls.npz'),mass_source=digest(vol/'cell-mass.f64'),support=digest(case/'support-check.json'),center=center.tolist(),node=corridor['node'],radius_m=radius,target_spacing_m=spacing,maximum_parent_centroid_error_m=moment_residual,maximum_relative_mass_adjustment=float(abs(mass_fraction/portion-1).max()),old_nodes=len(ids),new_nodes=len(rows_B),checks=checks,material=digest(out/'material.npz'),reaction_fraction=corridor['fraction'])
    (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(len(base),len(t),len(history),[(k,r['frames'][-1]['bad_shape'],r['frames'][-1]['relative_force_error'],r['first_moment_error_kg_m'])for k,r in checks.items()],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('w',type=Path);p.add_argument('out',type=Path);p.add_argument('--radius-mm',type=float,default=23.);p.add_argument('--spacing-mm',type=float,default=14.)
    a=p.parse_args();refine(a.w,a.out,a.radius_mm*.001,a.spacing_mm*.001)
