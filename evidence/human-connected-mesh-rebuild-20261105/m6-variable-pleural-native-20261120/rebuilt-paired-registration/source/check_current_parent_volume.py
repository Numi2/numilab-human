"""Replay actual paired parent/child cell topology at the rebuilt registration."""
import json
from pathlib import Path
import numpy as np
from rebind_current_skin import read_literal, sha256

apps = Path.home()/'Work/skin-fat-native-integration-209/apps'
coarse = apps/'NumiHumanWholeBodySkinReduction.hpp'
candidate = apps/'NumiHumanFinal128CandidateRegistration.hpp'
topology = apps/'NumiHumanElbowExplicitPairedTopology008.hpp'
data = np.load('skin-source-rebind-002/registration.npz',allow_pickle=False)
new_outer = data['world_m']
old_outer = read_literal(candidate,'kWholeBodySkinCandidateReferenceWorldPoints').reshape(-1,3)
delta = new_outer-old_outer
rows = len(delta)
correction = read_literal(coarse,'kWholeBodySkinMechanicalWorldDeltas').reshape(-1,3)
i = read_literal(coarse,'kWholeBodySkinInnerWorld').reshape(-1,3)
d = read_literal(coarse,'kWholeBodyFatDeepWorld').reshape(-1,3)
source_points=np.stack((old_outer+correction,i,d))
moved=source_points+delta[None]
parent_n=2332
p=read_literal(topology,'kElbowRefinementParentTetrahedra').astype(int).reshape(-1,4)
c=read_literal(topology,'kElbowRefinementChildTetrahedra').astype(int).reshape(-1,4)
mp=read_literal(topology,'kElbowRefinementParentMaterialCodes').astype(int)
mc=read_literal(topology,'kElbowRefinementChildMaterialCodes').astype(int)
parents=read_literal(topology,'kElbowRefinementChildParentTetrahedra').astype(int)
assert np.array_equal(mc,mp[parents])
density=np.take([0.,1000.,900.],mc)


def volumes(points,tets):
    q=points[tets]
    det=np.einsum('ij,ij->i',q[:,1]-q[:,0],np.cross(q[:,2]-q[:,0],q[:,3]-q[:,0]))
    max_edge=max_edges(q)
    return det/6,det/max_edge**3


def max_edges(q):
    return np.maximum.reduce([np.linalg.norm(q[:,a]-q[:,b],axis=1) for a in range(4) for b in range(a+1,4)])


def evaluate(points):
    pv,pq=volumes(points[:,:parent_n].astype(np.float32).astype(float).reshape(-1,3),p)
    xyz=points.astype(np.float32).astype(float).reshape(-1,3)
    cv,cq=volumes(xyz,c)
    mass=cv*density
    balance=np.bincount(parents,weights=cv,minlength=len(p))-pv
    return { 'parent': {'inverted':np.flatnonzero(pv<=0).tolist(),'below_native_quality':np.flatnonzero(pq<1e-5).tolist(),
                        'minimum_oriented_volume_m3':float(pv.min())},
             'child': {'inverted':np.flatnonzero(cv<=0).tolist(),'below_native_quality':np.flatnonzero(cq<1e-5).tolist(),
                       'minimum_normalized_determinant':float(cq.min()),'minimum_oriented_volume_m3':float(cv.min())},
             'max_absolute_parent_partition_error_m3':float(np.abs(balance).max()),
             'invalid_parent_partition_over_1ppm':int(np.count_nonzero(np.abs(balance)>np.abs(pv)*1e-6)),
             'signed_geometric_mass_kg':{'skin':float(mass[mc==1].sum()),'fat':float(mass[mc==2].sum())}}
result={'source_topology_sha256':sha256(topology),'original':evaluate(source_points),'rebound':evaluate(moved)}
Path('paired-parent-volume-screen.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
