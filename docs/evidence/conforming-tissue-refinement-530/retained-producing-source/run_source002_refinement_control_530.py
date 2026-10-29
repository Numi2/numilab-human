from pathlib import Path
import numpy as np,json,hashlib,time,csv,sys
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');sys.path.insert(0,str(P))
from refine_conforming_tetra_edges_530 import refine_edges,signed_volumes
out=P/'conforming-refinement-control-530/run-001';out.mkdir(parents=True,exist_ok=False)
base=P/'reduced-physical-boundary-400/attempt-002-parent-ancestry-walk/candidate.npz'
seeds=P/'coarse-boundary-contact-seeds-529/run-001/component-coarse-role-seeds.npz'
with np.load(base) as z:a={k:z[k].copy() for k in z.files}
with np.load(seeds) as z:seed={k:z[k].copy() for k in z.files}
x=a['points_reference_world_m_f64'].astype(np.float32).astype(float);t=a['volume_tetrahedra'];ncol=len(x)//3;assert len(x)==3*ncol
local=[]
for role in ['I','D']:
 faces=a['coarse_boundary_faces_'+role][seed[role+'_unique_coarse_face_rows'].astype(int)]
 local.extend(np.sort(faces[:,[0,1]]%ncol,axis=1));local.extend(np.sort(faces[:,[1,2]]%ncol,axis=1));local.extend(np.sort(faces[:,[2,0]]%ncol,axis=1))
edges0=np.unique(np.asarray(local,dtype=np.int64),axis=0)
edges=np.concatenate([edges0+offset*ncol for offset in range(3)])
start=time.monotonic();r=refine_edges(x,t,edges);elapsed=time.monotonic()-start
params={key:getattr(r,key) for key in r.__dataclass_fields__}
np.savez_compressed(out/'refinement-connectivity.npz',**params)
v0=signed_volumes(x,t);vf=signed_volumes(r.points_reference_f64,r.tetrahedra)
emitted=r.points_reference_f64.astype(np.float32).astype(float);ve=signed_volumes(emitted,r.tetrahedra)
with (P/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv.tets.csv').open() as f:pr=list(csv.DictReader(f))
mass=np.array([float(row['parent_mass_kg']) for row in pr]);cmass=mass[r.child_source_tet_rows]*r.child_source_measure_fractions
closed=np.bincount(r.child_source_tet_rows,weights=cmass,minlength=len(t));volclosed=np.bincount(r.child_source_tet_rows,weights=vf,minlength=len(t))
result={'status':'source002_conforming_three_layer_refinement_control','base_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'seed_sha256':hashlib.sha256(seeds.read_bytes()).hexdigest(),'helper_sha256':hashlib.sha256((P/'refine_conforming_tetra_edges_530.py').read_bytes()).hexdigest(),'control_npz_sha256':hashlib.sha256((out/'refinement-connectivity.npz').read_bytes()).hexdigest(),'input_nodes':len(x),'input_cells':len(t),'role_local_marked_edges':len(edges0),'all_role_marked_edges':len(edges),'output_nodes':len(r.points_reference_f64),'output_cells':len(r.tetrahedra),'original_nodes_unchanged':bool(np.array_equal(x,r.points_reference_f64[:len(x)])),'minimum_source_refined_cell_volume_m3':float(vf.min()),'source_refinement_nonpositive_cells':int(np.count_nonzero(vf<=0)),'canonical_F32_nonpositive_cells':int(np.count_nonzero(ve<=0)),'canonical_F32_nonpositive_cell_rows':np.flatnonzero(ve<=0).tolist(),'canonical_F32_min_cell_volume_m3':float(ve.min()),'parent_mass_kg':float(np.sum(mass,dtype=np.longdouble)),'refined_mass_kg':float(np.sum(cmass,dtype=np.longdouble)),'maximum_original_parent_mass_closure_kg':float(np.max(np.abs(closed-mass))),'maximum_original_parent_relative_volume_sum_error':float(np.max(np.abs(volclosed-v0)/v0)),'authoring_refinement_s':elapsed,'no_target_geometry_projection':True,'physical_admission':False,'runtime':0,'accepted_steps':0}
(out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
