"""Refit the bonded paired layers with fixed outer geometry and parent ancestry."""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from check_current_parent_volume import (apps, c, coarse, data, density, mc, moved,
                                         p, parent_n, parents, source_points, volumes)

out = Path('paired-skin-fat-refit-001')
layers, rows, _ = moved.shape
old_flat = source_points.reshape(-1,3)
parent_columns = p % parent_n
child_columns = c % rows
child_layers = c // rows
edge0 = np.arange(rows)
edge1 = np.arange(rows)
midpoint_errors = []
for row in range(parent_n,rows):
    owners = np.unique(parents[np.any(child_columns == row,axis=1)])
    endpoints = set(parent_columns[owners[0]])
    for owner in owners[1:]:
        endpoints.intersection_update(parent_columns[owner])
    fits=[]
    for a,b in itertools.combinations(sorted(endpoints),2):
        error = float(np.max(np.abs(source_points[:,row]-(source_points[:,a]+source_points[:,b])/2)))
        if error <= 1e-12:
            fits.append((a,b,error))
    if len(fits)!=1:
        raise ValueError(f'refinement row {row} has no unique recorded midpoint: {fits}')
    edge0[row],edge1[row],err=fits[0]
    midpoint_errors.append(err)


def expand(direct):
    return (direct[:,edge0]+direct[:,edge1])/2


initial = moved[:,:parent_n].copy()
initial_expanded=expand(initial)
old_quality = volumes(old_flat.astype(np.float32).astype(float),c)[1]
initial_quality = volumes(initial_expanded.reshape(-1,3).astype(np.float32).astype(float),c)[1]
need=np.flatnonzero(initial_quality<1e-5)
affected=np.unique(np.concatenate((edge0[child_columns[need]].ravel(),edge1[child_columns[need]].ravel())))
p_ids=np.flatnonzero(np.any(np.isin(parent_columns,affected),axis=1))
# Include the immediate connected layer on both sides of each invalid cell so
# one repaired tetrahedron cannot push a neighboring one through its boundary.
free=np.unique(parent_columns[p_ids])
c_ids=np.flatnonzero(np.any(np.isin(edge0[child_columns],free)|np.isin(edge1[child_columns],free),axis=1))
bodies = data['donor_body']
normals = data['normal_world']
baseline_depth = np.linalg.norm(np.diff(source_points,axis=0),axis=2)


def geometry(x):
    x=x.reshape(2,len(free),3)*1e-3
    trial=initial.copy()
    trial[1:,free]+=x
    return expand(trial)


fitted_rows=np.flatnonzero(np.isin(edge0,free)|np.isin(edge1,free))

def constraints(x):
    points=geometry(x)
    _,quality=volumes(points.reshape(-1,3),c[c_ids])
    offsets=points[:-1]-points[1:]
    depth=np.linalg.norm(offsets[:,fitted_rows],axis=2)*1e3
    original_offsets=(source_points[:-1]-source_points[1:])[:,fitted_rows]
    outward=np.einsum('ijk,ijk->ij',offsets[:,fitted_rows],original_offsets)
    quality_floor=np.minimum(old_quality[c_ids]*0.8,3e-5).clip(min=1.3e-5)
    return np.concatenate(((quality-quality_floor)*1e4,
                           (depth-np.array([[.501],[.501]])).ravel(),
                           (np.array([[2.499],[9.999]])-depth).ravel(),
                           (outward*1e6-0.08).ravel()))


zero=np.zeros(2*len(free)*3)
solution=minimize(lambda x: x@x,zero,jac=lambda x:2*x,
                  method='SLSQP', constraints=[{'type':'ineq','fun':constraints}],
                  bounds=[(-2.5,2.5)]*len(zero),
                  options={'maxiter':120,'ftol':1e-10})
print(json.dumps({'fit_success':bool(solution.success),'message':solution.message,
                  'variables':len(zero),'nearby_child_tets':len(c_ids),'initial_witnesses':need.tolist(),
                  'minimum_constraint':float(constraints(solution.x).min())}),flush=True)
if not solution.success or constraints(solution.x).min()<-1e-7:
    raise SystemExit('paired fit did not converge with the original boundaries')
points=geometry(solution.x)
assert np.array_equal(points[0,:parent_n],initial[0])
xyz=points.reshape(-1,3)
native_xyz=xyz.astype(np.float32).astype(float)
cv,quality=volumes(xyz,c)
cv_f32,quality_f32=volumes(native_xyz,c)
pv,parent_quality=volumes(points[:,:parent_n].reshape(-1,3),p)
pv_f32,parent_quality_f32=volumes(points[:,:parent_n].astype(np.float32).astype(float).reshape(-1,3),p)
rel=np.abs(np.bincount(parents,weights=cv,minlength=len(p))-pv)/pv
depth=np.linalg.norm(np.diff(points,axis=0),axis=2)
assert np.all(cv_f32>0) and np.all(pv_f32>0) and np.all(quality_f32>=1e-5) and np.all(parent_quality_f32>=1e-5)
assert np.max(rel)<1e-6
assert np.all(depth>=0.0005) and depth[0].max()<=0.0025 and depth[1].max()<=0.010
# Nodal mass follows the actual registered material ownership. Contributions
# from the skin and the fat at their shared interface are summed exactly once.
material_mass=cv*density
node_mass=np.bincount(c.ravel(),weights=np.repeat(material_mass*.25,4),minlength=len(xyz))
donor_mass=np.bincount(np.tile(bodies,3),weights=node_mass)
reference=cv_f32*density
assert np.all(node_mass>0) and abs(node_mass.sum()-material_mass.sum())<1e-12
out.mkdir(exist_ok=False)
np.savez_compressed(out/'paired-geometry-and-mass.npz',points_world_m=points,
                    child_tetrahedra=c, parent_tetrahedra=p,child_parent_rows=parents,
                    child_material=mc, thickness_m=depth, node_mass_kg=node_mass,
                    donor_mass_kg=donor_mass)
result={'native_F32_inverted_cells':int(np.count_nonzero(cv_f32<=0)),
        'native_F32_child_quality_minimum':float(quality_f32.min()),
        'native_F32_parent_quality_minimum':float(parent_quality_f32.min()),
        'children':len(c),'parents':len(p),'original_exact_midpoints':len(midpoint_errors),
        'max_source_midpoint_error_m':max(midpoint_errors),'maximum_parent_partition_relative_error':float(rel.max()),
        'skin_mass_kg':float(material_mass[mc==1].sum()),'fat_mass_kg':float(material_mass[mc==2].sum()),
        'nodal_mass_kg':float(node_mass.sum()),'donor_mass_kg':float(donor_mass.sum()),
        'f32_density_volume_mass_change_kg':float(reference.sum()-material_mass.sum()),
        'max_inner_or_deep_local_fit_m':float(np.linalg.norm(points[1:]-initial_expanded[1:],axis=2).max()),
        'max_thickness_change_m':float(np.max(np.abs(depth-baseline_depth))),
        'full_registered_volume_admitted':False,'whole_body_contact_verified':False}
(out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
