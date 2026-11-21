"""Screen witness-local graph displacements against the native source banks."""
from scipy import sparse
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

def area(points, faces):
    x = points[faces]
    return np.cross(x[:,1]-x[:,0], x[:,2]-x[:,0])

def fit(x,faces,hits,locked,original):
    ns,nm = hits.T
    nodes = faces[nm].ravel()
    n = np.repeat(normals[ns],3,axis=0)
    offsets = np.einsum('ij,ij->i',n,x[nodes]-1000*shell[np.repeat(ns,3),0])
    constraints = sparse.csr_matrix(
        (n.ravel(),(np.repeat(np.arange(len(nodes)),3),
                    (3*nodes[:,None]+np.arange(3)).ravel())),
        shape=(len(nodes),3*len(x)))
    edges = np.unique(np.sort(faces[:,[0,1,1,2,2,0]].reshape(-1,2),axis=1),axis=0)
    i,j = edges.T
    weights = 1/np.maximum(.25,np.linalg.norm(x[i]-x[j],axis=1))
    adj = sparse.csr_matrix((np.tile(weights,2),(np.r_[i,j],np.r_[j,i])),
                            shape=(len(x),len(x)))
    active = np.zeros(len(x),bool)
    active[nodes] = True
    for _ in range(5):
        active |= (adj @ active) > 0
    active &= ~locked
    idx = np.flatnonzero(np.repeat(active,3))
    if not len(idx):
        return np.zeros_like(x)
    op = constraints[:,idx]
    degree = np.asarray(adj.sum(axis=1)).ravel()
    lap = sparse.diags(degree)-adj
    volume = clearance._signed_surface_integral(original,faces)
    assert volume != 0
    tri = original[faces] - original.mean(axis=0)
    gradient = np.zeros_like(original)
    for vertex in range(3):
        np.add.at(gradient,faces[:,vertex],np.cross(tri[:,(vertex+1)%3],tri[:,(vertex+2)%3])/6)
    grad = gradient.ravel()/abs(volume)
    before = (clearance._signed_surface_integral(x,faces)-volume)/abs(volume)
    volume_active = grad[idx]
    hold = (x-original).ravel()

    def evaluate(var):
        trial = np.zeros(3*len(x))
        trial[idx] = var
        displacement = (trial + hold).reshape(-1,3)
        regularizer = 4*(lap@displacement)+.05*displacement
        h = np.maximum(op@var+offsets+.025,0)
        relative_volume = volume_active@var+before
        cost = .5*((displacement*regularizer).sum()+12*h@h+1200*relative_volume**2)
        dg = regularizer.ravel()[idx]+12*(op.T@h)+1200*relative_volume*volume_active
        return cost,dg

    bound = [(-3-h,3-h) for h in hold[idx]]
    result = minimize(evaluate,np.zeros(len(idx)),method='L-BFGS-B',jac=True,
                      bounds=bound,options={'maxiter':120,'maxcor':7,'ftol':1e-9})
    delta = np.zeros(x.size)
    delta[idx] = result.x
    return delta.reshape(-1,3)

for stable in (137,138,109,110,73,74):
    xyz, faces = banks[51005,stable]
    # Coordinate aliases must move together. Keep every original face row.
    x, aliases = np.unique(xyz,axis=0,return_inverse=True)
    faces = aliases[faces]
    original = x.astype(float)*1000
    initial = pairs(x,faces)
    area_before = area(original,faces)
    assert (np.linalg.norm(area_before,axis=1)>0).all()
    # Geometric end bands are only a feasibility approximation for the
    # anatomical attachments. A real source edit also pins exact attachments.
    _,_,basis = np.linalg.svd(original-original.mean(axis=0),full_matrices=False)
    longitudinal = original@basis[0]
    lo,hi = np.quantile(longitudinal,[.05,.95])
    locked = (longitudinal<=lo) | (longitudinal>=hi)
    x = original.copy()
    best = initial
    volume = clearance._signed_surface_integral(original,faces)
    best_metrics = None
    for _ in range(2):
        delta = fit(x,faces,best,locked,original)
        choice = None
        for fraction in (1,.5,.25):
            trial = ((x+fraction*delta)*.001).astype('<f4')
            new = trial.astype(float)*1000
            dx = float(np.linalg.norm(new-original,axis=1).max())
            ratio = np.linalg.norm(area(new,faces),axis=1)/np.linalg.norm(area_before,axis=1)
            orientation = np.sum(area(new,faces)*area_before,axis=1)
            v = abs(clearance._signed_surface_integral(new,faces)/volume-1)
            if dx>3 or ratio.min()<.8 or ratio.max()>1.25 or (orientation<=0).any() or v>.01:
                continue
            hits = pairs(trial,faces)
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
    print(json.dumps(dict(stable_id=stable,source_pairs=len(initial),fewest_screened_pairs=len(best),
                         retained_proposal=best_metrics,stage='outer_skin_feasibility_only',
                         admitted=False)),flush=True)
