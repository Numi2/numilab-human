"""Connect rebuilt paired cells to current landmark and retained face provenance."""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

from rebind_current_skin import oriented, read_literal, sha256

base=Path('skin-source-rebind-002')
paired=Path('paired-skin-fat-refit-001/paired-geometry-and-mass.npz')
source_report=json.loads((base/'receipt.json').read_text())
source_paths=[Path(p) for p in source_report['inputs_sha256'] if p.endswith('.nhskin')]
assert len(source_paths)==2
assert all(sha256(p)==source_report['inputs_sha256'][str(p)] for p in source_paths)
scripts=Path('../resting-clearance-and-atrium-006')
sys.path.insert(0,str(scripts))
from fit_clearance_patches import decode

old,new=[decode(p.read_bytes()) for p in source_paths]
reduction=Path.home()/'Work/skin-fat-native-integration-209/apps/NumiHumanWholeBodySkinReduction.hpp'
parents=read_literal(reduction,'kWholeBodySkinSourceFaces').astype(np.int64)
delta=read_literal(reduction,'kWholeBodySkinMechanicalWorldDeltas')
with np.load(base/'registration.npz',allow_pickle=False) as registration:
    reg={k:registration[k] for k in registration.files}
with np.load(paired,allow_pickle=False) as geometry:
    nodes=geometry['points_world_m']
    node_mass=geometry['node_mass_kg']
    expected_donor_mass=geometry['donor_mass_kg']
assert np.isfinite(nodes).all() and np.isfinite(node_mass).all()
physical_shift=nodes[0]-reg['world_m']
delta=delta.reshape(physical_shift.shape)
projected=reg['source_vertex']<0
residual_shift=physical_shift-delta
assert np.all(np.abs(residual_shift[~projected])<1e-15)
residual=reg['projected_residual_world_m'].copy()
residual[projected]+=residual_shift[projected]
reference_points=nodes[0]-delta
frame=reg['world_to_donor'].reshape(-1,3,3)
local=reg['donor_local_m']+np.einsum('rij,rj->ri',frame,residual_shift)
physical_local=local[None]+np.einsum('rij,lrj->lri',frame,nodes-reference_points)
rigid_comp=np.bincount(np.tile(reg['donor_body'],3),weights=node_mass)
assert np.array_equal(rigid_comp,expected_donor_mass)
by_key=defaultdict(list)
for face, vertices in enumerate(new['inds']):
    by_key[oriented(vertices)].append(face)
current_ids=np.full(len(parents),-1,dtype=np.int64)
for i,face in enumerate(parents):
    matches=by_key[oriented(old['inds'][face])]
    if len(matches)==1:
        current_ids[i]=matches[0]
unresolved=np.flatnonzero(current_ids<0)
out=Path('paired-registration-final')
out.mkdir(exist_ok=False)
reg.update(projected_residual_world_m=residual,world_m=reference_points,donor_local_m=local)
np.savez_compressed(out/'registration.npz',**reg,physical_layers_donor_local_m=physical_local,
                    old_surface_face_parent=parents,current_surface_face_parent=current_ids)
report=dict(input_geometry_sha256=sha256(paired),input_registration_sha256=sha256(base/'registration.npz'),
            registered_source_rows=len(projected),projected_rows=int(projected.sum()),
            max_midpoint_ancestry_correction_m=float(np.linalg.norm(residual_shift[projected],axis=1).max()),
            exact_face_parents=int(np.count_nonzero(current_ids>=0)),
            missing_current_face_parent_count=len(unresolved),
            unresolved_face_rows=[{'coarse_row':int(i),'old_face':int(parents[i]),
                                   'old_vertices':old['inds'][parents[i]].tolist()} for i in unresolved],
            donor_mass_total_kg=float(rigid_comp.sum()),
            registration_sha256=sha256(out/'registration.npz'),runtime_admitted=False)
(out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='unresolved_face_rows'}))
