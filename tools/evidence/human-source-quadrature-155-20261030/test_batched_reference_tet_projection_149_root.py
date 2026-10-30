"""Compare batched static anchor geometry with existing217 scalar owner."""
from pathlib import Path
import numpy as np,sys,importlib.util,time,hashlib,json
H=Path(__file__).resolve().parent
def imp(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 sys.path.insert(0,str(H/'downstream-003-current-typed-228/native-domain-217-source-005'))
 owner=imp('reg149',H/'current-semantic-atlas-088/export_authoring_registered_surface_sidecars_230_semantic_088.py')
 parent,index,_,pm=owner.load_parent_current(H.parent/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv')
 hp=H/'batched_reference_tet_projection_149_root.py';helper=imp('batch149',hp)
 points=[];rows=[]
 with np.load(H/'skin010-restored-cavity-batch-072-expand1/skin-restored-boundary-proposal.npz')as z:p=z['points_world_m'].astype(float);t=z['tetrahedra']
 with np.load(H/'current-semantic-atlas-088/skin-semantic-atlas-002/output-001/skin-current-semantic-support.npz')as z:co=z['cell_group_offsets'];po=z['cell_group_parent_tet_offsets'];pr=z['cell_group_parent_tet_rows']
 for c in range(1024):
  pool=set(pr[po[co[c]]:po[co[c+1]]].tolist());centroid=p[t[c]].mean(axis=0)
  for pt in sorted(pool):points.append(centroid);rows.append(pt)
 actual=len(rows);rng=np.random.default_rng(149)
 for i in range(512):
  row=int(rng.integers(len(parent['tets'])));b=rng.dirichlet(np.ones(4));q=b@parent['points'][parent['tets'][row]]
  if i%2:q=q+rng.normal(size=3)*.02
  points.append(q);rows.append(row)
 points=np.array(points);rows=np.array(rows,np.int32)
 start=time.perf_counter();out=helper.project_pairs(points,rows,parent['points'],parent['tets'],index['inverses']);batch=time.perf_counter()-start
 start=time.perf_counter();old=[owner.native.closest_point_on_parent_tet(parent,index,int(r),p)for p,r in zip(points,rows)];scalar=time.perf_counter()-start
 q=np.array([x[0]for x in old]);d=np.array([x[2]for x in old])
 maximum_point=float(np.linalg.norm(out['projected']-q,axis=1).max());maximum_distance=float(np.abs(out['distance']-d).max())
 assert maximum_point<2e-9 and maximum_distance<2e-9,(maximum_point,maximum_distance)
 assert helper.project_pairs(np.empty((0,3)),np.empty(0,np.int32),parent['points'],parent['tets'],index['inverses'])['bary'].shape==(0,4)
 for badp,badr in [(np.array([[np.nan,0,0]]),np.array([0])),(np.zeros((1,3)),np.array([-1]))]:
  try:helper.project_pairs(badp,badr,parent['points'],parent['tets'],index['inverses'])
  except ValueError:pass
  else:raise AssertionError('invalid input accepted')
 report={'status':'PASS_BATCHED_STATIC_ANCHOR_PROJECTION_PARITY','physical_admission':False,'native_steps':0,'scope':'Static asset compiler projection of explicitly requested P1 carrier pairs, not simulation/per-step physics or global-nearest proof.','helper_path':str(hp),'helper_sha256':sha(hp),'actual_Skin_P1_pairs':actual,'random_inside_outside_pairs':512,'total_pairs':len(rows),'maximum_projected_point_difference_m':maximum_point,'maximum_distance_difference_m':maximum_distance,'scalar_s':scalar,'batched_s':batch,'measured_projection_speedup':scalar/batch,'controls':['empty pairs','nonfinite points rejected','invalid parent rows rejected'],'source330':pm}
 folder=H/'batched-static-anchor-projection-parity-149-root';folder.mkdir();(folder/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
