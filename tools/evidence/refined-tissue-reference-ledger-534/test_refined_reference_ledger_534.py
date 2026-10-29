#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent/"conforming-tissue-refinement-530"))
from refine_conforming_tetra_edges_530 import refine_edges,signed_volumes
from build_refined_reference_ledger_534 import build_refined_reference_ledger

def main():
 x=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
 t=np.array([[0,1,2,3]])
 r=refine_edges(x,t,np.array([[0,1],[0,2]]))
 v=signed_volumes(x,t)
 kw=dict(source_points=x,source_tets=t,refined_points=r.points_reference_f64,refined_tets=r.tetrahedra,
  child_source_tet_rows=r.child_source_tet_rows,child_source_measure_fractions=r.child_source_measure_fractions,
  node_source_offsets=r.node_source_offsets,node_source_rows=r.node_source_rows,node_source_weights=r.node_source_weights,
  parent_reference_volumes=v*.99,parent_mass=v*1000,source_material=np.array([15]),source_donor=np.array([23]),
  source_region=np.array([24]),source_stable_face=np.array([777]))
 z=build_refined_reference_ledger(**kw)
 assert abs(z['child_mass_kg'].sum()-v[0]*1000)<1e-12
 assert abs(z['child_parent_reference_measure_m3'].sum()-v[0]*.99)<1e-15
 assert np.array_equal(z['stable_source_face_id'],np.full(len(r.tetrahedra),777))
 controls=[]
 for name,key,value in [
  ('wrong_parent','child_source_tet_rows',np.ones(len(r.tetrahedra),dtype=np.int64)),
  ('missing_fraction','child_source_measure_fractions',r.child_source_measure_fractions*.5),
  ('wrong_geometry','refined_points',r.points_reference_f64+np.array([.01,0,0])),
  ('noninteger_parent','child_source_tet_rows',r.child_source_tet_rows.astype(float)),
  ('density_outside','parent_mass',v*1300),
  ('nongeometric_fractions','child_source_measure_fractions',r.child_source_measure_fractions+np.array([.01,-.01,0])),
  ('duplicate_cells','refined_tets',np.vstack((r.tetrahedra[0],r.tetrahedra[0],r.tetrahedra[2])))]:
  bad={**kw,key:value}
  try:build_refined_reference_ledger(**bad)
  except ValueError:controls.append(name)
  else:raise AssertionError(name+' accepted')
 duplicate_source=np.vstack((x,x[0]))
 rows=r.node_source_rows.copy();rows[rows==0]=4
 bad={**kw,'source_points':duplicate_source,'node_source_rows':rows}
 # Keep CSR row order canonical while attempting ancestry outside the parent.
 orderrows=[];orderweights=[]
 for a,b in zip(r.node_source_offsets[:-1],r.node_source_offsets[1:]):
  ix=np.argsort(rows[a:b]);orderrows.extend(rows[a:b][ix]);orderweights.extend(r.node_source_weights[a:b][ix])
 bad.update(node_source_rows=np.array(orderrows),node_source_weights=np.array(orderweights))
 try:build_refined_reference_ledger(**bad)
 except ValueError:controls.append('ancestry_escapes_parent')
 else:raise AssertionError('ancestry escape accepted')
 print('PASS source measure distinct from operational geometry; inherited labels; nested fractions; '+','.join(controls))
if __name__=='__main__':main()
