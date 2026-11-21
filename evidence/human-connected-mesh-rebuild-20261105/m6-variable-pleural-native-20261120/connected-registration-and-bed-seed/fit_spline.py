"""Screen smooth, end-pinned registration candidates against the native source banks."""
from scipy.interpolate import BSpline
from scipy.optimize import minimize

import json
import numpy as np
from native_source import banks, clearance, exact, skin, skin_faces

skin_records = clearance._exact_surface_records(skin,skin_faces)
skin_index = exact._prepare_surface_aabb(skin_records)
skin_broad = clearance._float32_skin_aabb_index(skin_records)
shell = skin[skin_faces].astype(float)
normals = np.cross(shell[:,1]-shell[:,0],shell[:,2]-shell[:,0])
lengths = np.linalg.norm(normals,axis=1)
assert (lengths>0).all()
winding = clearance._signed_surface_integral(skin,skin_faces)
normals *= np.sign(winding)/lengths[:,None]

def pairs(points,faces):
    chosen, broad_count = clearance._candidate_target_rows_for_skin_aabb(
        faces,points,skin_broad)
    used = np.unique(faces[chosen])
    bank_records = clearance._exact_surface_records(points[used],
        np.searchsorted(used,faces[chosen]))
    bank_records = [(w,x,y,int(chosen[d]),z) for w,x,y,d,z in bank_records]
    result = exact._audit_pair_prepared_first(skin_index,bank_records)
    assert result['aabb_candidate_pairs'] == broad_count
    return np.asarray(result['triangle_pairs'],dtype=np.int64).reshape(-1,2)

def nearby_pairs(points, faces, neighborhood):
    source = clearance._exact_surface_records(points, faces)
    index = exact._prepare_surface_aabb(source)
    broad = clearance._float32_skin_aabb_index(source)
    found = {}
    for bank, (vertices, triangles) in neighborhood.items():
        chosen, count = clearance._candidate_target_rows_for_skin_aabb(
            triangles, vertices, broad)
        if not len(chosen):
            continue
        used = np.unique(triangles[chosen])
        records = clearance._exact_surface_records(
            vertices[used], np.searchsorted(used, triangles[chosen]))
        records = [(a,b,c,int(chosen[d]),e) for a,b,c,d,e in records]
        audit = exact._audit_pair_prepared_first(index, records)
        assert audit['aabb_candidate_pairs'] == count
        if audit['count']:
            found[bank] = set(map(tuple, audit['triangle_pairs']))
    return found


def surrounds(reference, trial, faces, stable):
    lo = np.minimum(reference.min(axis=0), trial.min(axis=0))
    hi = np.maximum(reference.max(axis=0), trial.max(axis=0))
    near = {key:(points, triangles) for key,(points,triangles) in banks.items()
            if key != (51005,stable) and np.all(points.max(axis=0)>=lo)
            and np.all(points.min(axis=0)<=hi)}
    old = nearby_pairs(reference, faces, near)
    new = nearby_pairs(trial, faces, near)
    keys = sorted(old.keys() | new.keys())
    return {'admitted_target_banks': len(near),
            'new_crossings': sum(len(new.get(k,set()) - old.get(k,set())) for k in keys),
            'largest_changes': sorted([
                {'semantic':k[0], 'stable':k[1], 'before':len(old.get(k,set())),
                 'after':len(new.get(k,set())), 'new_pairs':len(new.get(k,set())-old.get(k,set()))}
                for k in keys], key=lambda r: -r['new_pairs'])[:8]}


def area(points, faces):
    x = points[faces]
    return np.cross(x[:,1]-x[:,0], x[:,2]-x[:,0])

def fit(x, faces, hits, original, basis):
    ns, nm = hits.T
    nodes = faces[nm].ravel()
    n = np.repeat(normals[ns], 3, axis=0)
    offsets = np.einsum('ij,ij->i', n, x[nodes]-1000*shell[np.repeat(ns,3),0])
    operator = (basis[nodes,:,None]*n[:,None,:]).reshape(len(nodes), -1)
    volume = clearance._signed_surface_integral(original, faces)
    tri = original[faces] - original.mean(axis=0)
    gradient = np.zeros_like(original)
    for vertex in range(3):
        np.add.at(gradient,faces[:,vertex],np.cross(tri[:,(vertex+1)%3],tri[:,(vertex+2)%3])/6)
    volume_row = (basis.T @ gradient).ravel()/abs(volume)
    before = clearance._signed_surface_integral(x,faces)/volume-1

    def objective(coeff):
        remaining = np.maximum(operator@coeff + offsets + .025, 0)
        v = volume_row@coeff + before
        return (.5*(remaining@remaining/len(nodes)+.018*(coeff@coeff)+1000*v*v),
                operator.T@remaining/len(nodes)+.018*coeff+1000*v*volume_row)

    k = 3*basis.shape[1]
    result = minimize(objective,np.zeros(k),method='L-BFGS-B',jac=True,
                      bounds=[(-2,2)]*k,options={'maxiter':100,'gtol':1e-7})
    return basis@result.x.reshape(-1,3)

if __name__ == "__main__":
    for stable in (137,138,109,110,73,74):
        xyz, faces = banks[51005,stable]
        # Coordinate aliases must move together. Keep every original face row.
        x, aliases = np.unique(xyz,axis=0,return_inverse=True)
        faces = aliases[faces]
        before_f32 = x.copy()
        original = x.astype(float)*1000
        initial = pairs(x,faces)
        area_before = area(original,faces)
        assert (np.linalg.norm(area_before,axis=1)>0).all()
        # Geometric end bands are only a feasibility approximation for the
        # anatomical attachments. A real source edit also pins exact attachments.
        _,_,basis = np.linalg.svd(original-original.mean(axis=0),full_matrices=False)
        longitudinal = original@basis[0]
        lo,hi = np.quantile(longitudinal,[.05,.95])
        t = np.clip((longitudinal-lo)/(hi-lo),0,1)
        knots = np.r_[np.zeros(4),.2,.4,.6,.8,np.ones(4)]
        basis = BSpline.design_matrix(t,knots,3).toarray()
        basis = basis[:,2:-2]  # enforce value and derivative zero at both ends
        assert np.all(basis[(t==0)|(t==1)] == 0)
        x = original.copy()
        best = initial
        volume = clearance._signed_surface_integral(original,faces)
        best_metrics = None
        failures = []
        for _ in range(2):
            delta = fit(x,faces,best,original,basis)
            choice = None
            for fraction in (1,.5,.25,.125):
                trial = ((x+fraction*delta)*.001).astype('<f4')
                new = trial.astype(float)*1000
                dx = float(np.linalg.norm(new-original,axis=1).max())
                ratio = np.linalg.norm(area(new,faces),axis=1)/np.linalg.norm(area_before,axis=1)
                orientation = np.sum(area(new,faces)*area_before,axis=1)
                v = abs(clearance._signed_surface_integral(new,faces)/volume-1)
                hits = pairs(trial,faces)
                metrics = dict(fraction=fraction, max_shift_mm=dx,
                               minimum_area_ratio=float(ratio.min()),maximum_area_ratio=float(ratio.max()),
                               orientation_failures=int((orientation<=0).sum()),relative_volume_change=float(v),
                               crossings=int(len(hits)))
                if dx>3 or ratio.min()<.8 or ratio.max()>1.25 or (orientation<=0).any() or v>.01:
                    failures.append(metrics)
                    continue
                if len(hits)>=len(best) or (choice is not None and len(hits)>=len(choice[0])):
                    continue
                choice = hits,new,dict(fraction=fraction,max_shift_mm=dx,
                    minimum_area_ratio=float(ratio.min()),maximum_area_ratio=float(ratio.max()),
                    relative_volume_change=float(v))
            if choice is None:
                break
            best,x,best_metrics = choice
            if not len(best):
                break
        surrounding = None
        if best_metrics is not None:
            surrounding = surrounds(before_f32, (x*.001).astype('<f4'), faces, stable)
        print(json.dumps(dict(stable_id=stable,source_pairs=len(initial),fewest_screened_pairs=len(best),
                             retained_proposal=best_metrics,nearby_audit=surrounding,rejected_numerical=failures,stage='outer_skin_feasibility_only',
                             admitted=False)),flush=True)
