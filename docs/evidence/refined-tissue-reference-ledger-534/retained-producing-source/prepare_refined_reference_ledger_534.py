#!/usr/bin/env python3
from pathlib import Path
import csv,json,hashlib,sys
import numpy as np
from build_refined_reference_ledger_534 import build_refined_reference_ledger
P=Path(__file__).resolve().parent
D=P/'refined-reference-ledger-534/run-001';D.mkdir(parents=True,exist_ok=False)
S=P/'reduced-physical-boundary-400/attempt-002-parent-ancestry-walk/candidate.npz'
R=P/'conforming-refinement-control-530/run-001/refinement-connectivity.npz'
M=P/'native-physical-parent-mass-330/native-source-identity-control-003'
MT=M/'compiled-parent-cell-mass.csv.tets.csv';ML=M/'compiled-parent-cell-mass.csv.labels.csv'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(S)=='2f547b21c6be0f3e21a021d80dad6c63c3a0e6af31d5ebe1780ca1b32c237a54'
assert sha(R)=='bc3831a56663cd561d5465d5943eb7cab275ca82b3c0be92561d62c42e8e8606'
s=np.load(S,allow_pickle=False);r=np.load(R,allow_pickle=False)
mass=list(csv.DictReader(MT.open()));labels=list(csv.DictReader(ML.open()))
assert np.array_equal(np.asarray([[int(v['parent_node'+str(k)]) for k in range(4)] for v in mass]),s['volume_tetrahedra'])
assert np.array_equal(s['child_parent_tet_row'],np.arange(len(mass)))
assert np.array_equal(s['parent_volume_node_row'],np.arange(len(s['points_reference_world_m_f64'])))
for key in ['material_index','geometric_donor_body','anatomical_region_id','stable_source_face_id']:
 assert np.array_equal(np.array([int(v[key]) for v in labels]),s[key])
kwargs=dict(source_points=s['points_reference_world_m_f64'],source_tets=s['volume_tetrahedra'],
 refined_points=r['points_reference_f64'],refined_tets=r['tetrahedra'],
 child_source_tet_rows=r['child_source_tet_rows'],child_source_measure_fractions=r['child_source_measure_fractions'],
 node_source_offsets=r['node_source_offsets'],node_source_rows=r['node_source_rows'],node_source_weights=r['node_source_weights'],
 parent_reference_volumes=np.array([float(v['parent_reference_volume_m3']) for v in mass]),
 parent_mass=np.array([float(v['parent_mass_kg']) for v in mass]),source_material=s['material_index'],
 source_donor=s['geometric_donor_body'],source_region=s['anatomical_region_id'],source_stable_face=s['stable_source_face_id'])
z=build_refined_reference_ledger(**kwargs);diag=z.pop('diagnostics')
np.savez_compressed(D/'child-parent-reference-ledger.npz',**z)
with (D/'child-parent-reference-ledger.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['child_tet_row','parent_tet_row','source_fraction','parent_reference_measure_m3','child_mass_kg',
  'child_reference_volume_m3','material_index','geometric_donor_body','anatomical_region_id','stable_source_face_id'])
 for i in range(len(r['tetrahedra'])):
  w.writerow([i,int(z['child_parent_rows'][i])]+[format(float(z[k][i]),'.17g') for k in
   ['source_child_fraction','child_parent_reference_measure_m3','child_mass_kg','child_reference_volume_m3']]+
   [int(z[k][i]) for k in ['material_index','geometric_donor_body','anatomical_region_id','stable_source_face_id']])
with (D/'source-refined.node').open('w') as f:
 y=r['points_reference_f64'].astype(np.float32).astype(np.float64);f.write(f'{len(y)} 3 0 0\n')
 for i,p in enumerate(y):f.write(str(i+1)+' '+' '.join(format(float(v),'.17g') for v in p)+'\n')
with (D/'source-refined.ele').open('w') as f:
 t=r['tetrahedra'];f.write(f'{len(t)} 4 1\n')
 for i,p in enumerate(t):f.write(str(i+1)+' '+' '.join(str(int(v)+1) for v in p)+f' {int(z["material_index"][i])}\n')
receipt=dict(status='SOURCE_REFINEMENT_IMMUTABLE_PARENT_LEDGER_PREPARATION_PASS',
 source_npz_sha256=sha(S),refinement_connectivity_sha256=sha(R),immutable_parent_files={str(p):sha(p) for p in
 [MT,ML,M/'compiled-parent-cell-mass.csv',M/'compiled-parent-cell-mass.csv.source.csv']},
 source_cell_count=len(mass),refined_cell_count=len(r['tetrahedra']),refined_node_count=len(y),
 original_parent_mass_kg=float(sum(v for v in kwargs['parent_mass'])),refined_mass_kg=float(sum(z['child_mass_kg'])),
 inherited_material_donor_region_stable_face_ids=True,source_coordinate_ancestry_checked=True,
 immutable_measure_formula='parent_330_reference_volume_m3[parent] * exact_source_002_subdivision_fraction',
 unprojected_source_control=True,physical_admission=False,runtime=0,accepted_steps=0,**diag,
 outputs={str(p.name):sha(p) for p in D.iterdir() if p.is_file()})
(D/'prepare-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
