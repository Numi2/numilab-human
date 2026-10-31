import hashlib, importlib.util, json, pathlib, time
import numpy as np
H=pathlib.Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
S=H/'fractional-semantic-support-090-sparse-graph'
R=S/'fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001'
TARGET=H/'fine-physical-volume-706/current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz'
M22=R/'attempt-022-source-exhaustion-reserved-batch/run-001/merged-held-centroid-noncentroid-source-exhaustion.npz'
M21=R/'attempt-021-source-exhaustion-batch/run-001/merged-held-centroid-noncentroid-exhaustion.npz'
V22=R/'attempt-022-source-exhaustion-reserved-batch/independent-replay-013.json'
PRED=H/'fine-physical-volume-706/solve_oriented_cavity_milp_081.py'
OUT=R/'attempt-054-exact-f32-direct-density'
DEST=OUT/'exact-f32-direct-density-reconciliation.json'
EXPECTED={'target':'5dcebfd751dedcb1891559352e9d0e673ea3a8e15a8edeae8179c6d04e847211','m22':'69c212a0a11395a7c0584b99fc4cc00fd543754baec4c542266b8a4226957623','m21':'6609bd2720dff820fac1075e42d99f4a4c011f15f3a7e8c2597cc0ac5c918c19','v22':'7ce2d8dc4ab8c57da2b2b3b3692dece7f1fd8fe0901b15e26e7a5cbe1a1ea090','predicate':'3f7c276bef3bb4427f6acb41f6eeedffbeae273c9493a3b7854b244a5a2fa024'}
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
for p,k in [(TARGET,'target'),(M22,'m22'),(M21,'m21'),(V22,'v22'),(PRED,'predicate')]:
 if sha(p)!=EXPECTED[k]: raise RuntimeError('pinned input mismatch '+k)
receipt22=json.loads(V22.read_text())
if receipt22['inputs'][str(TARGET)]!=EXPECTED['target'] or receipt22['inputs'][str(M22)]!=EXPECTED['m22']:
 raise RuntimeError('022 receipt does not bind exact target/M22')
if receipt22['inputs'][str(H/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv.nodes.csv')]!='7044051353d173b94143a4027ba7dcc468b221512b3f41d5819d27025b36d044':raise RuntimeError('source330 node input mismatch')
if receipt22['inputs'][str(H/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv.tets.csv')]!='82e20067454748a9eca91bc6f06ab343677c051a139797ae051c64022bd87992':raise RuntimeError('source330 tet input mismatch')
spec=importlib.util.spec_from_file_location('exact_f32_det_081',PRED); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
with np.load(TARGET,allow_pickle=False) as z:
 points=np.asarray(z['points_world_m']); tets=np.asarray(z['tetrahedra'],dtype=np.int64)
 if points.dtype!=np.float32:raise RuntimeError('target points not canonical F32')
with np.load(M22,allow_pickle=False) as z:
 mass22=np.asarray(z['child_current_mass_kg'],dtype=np.float64)
with np.load(M21,allow_pickle=False) as z:
 mass21=np.asarray(z['child_current_mass_kg'],dtype=np.float64)
if len(mass22)!=len(tets) or len(mass21)!=len(tets):raise RuntimeError('child row domain mismatch')
# Exact dyadic determinant on the exact binary32 coordinates; determinant units are 2^-447 m^3.
used=np.unique(tets.ravel()); coords={int(i):mod.ip(points[int(i)]) for i in used}
det6=np.empty(len(tets),dtype=object)
t0=time.monotonic()
for i,tet in enumerate(tets):det6[i]=int(mod.det(*(coords[int(v)] for v in tet)))
absdet=np.fromiter((abs(int(x)) for x in det6),dtype=np.float64,count=len(det6))
vol=absdet/(6.0*2.0**447)
if np.any(absdet==0):raise RuntimeError('exact F32 zero determinant found')
if np.any(det6<0):raise RuntimeError('exact F32 negative determinant found in canonical orientation')
threshold=950.0+5e-10
# Exact comparison of each float64 mass and threshold against integer det6; volume is det6/(6*2^447).
from fractions import Fraction
thr=Fraction.from_float(threshold); high=Fraction.from_float(950.0); scale=6*(1<<447)
def over_exact(m,d,q):
 fm=Fraction.from_float(float(m)); return fm*scale > q*int(d)
def density_exact_float(m,d):return float(Fraction.from_float(float(m))*scale/int(d))
raw22=[]; tol22=[]; raw21=[]; tol21=[]; max22=(0.0,-1); max21=(0.0,-1)
for i,d in enumerate(absdet):
 f=int(d)
 r22=density_exact_float(mass22[i],f); r21=density_exact_float(mass21[i],f)
 if r22>max22[0]:max22=(r22,i)
 if r21>max21[0]:max21=(r21,i)
 if over_exact(mass22[i],f,high):raw22.append(i)
 if over_exact(mass22[i],f,thr):tol22.append(i)
 if over_exact(mass21[i],f,high):raw21.append(i)
 if over_exact(mass21[i],f,thr):tol21.append(i)
# Independent numerical formula comparisons on same arrays.
xyz=points.astype(np.float64); t=tets
vtr=np.abs(np.einsum('ij,ij->i',xyz[t[:,1]]-xyz[t[:,0]],np.cross(xyz[t[:,2]]-xyz[t[:,0]],xyz[t[:,3]]-xyz[t[:,0]])))/6.0
vdet=np.empty(len(t),np.float64)
for lo in range(0,len(t),100000):
 hi=min(len(t),lo+100000); tt=t[lo:hi]; xx=xyz[tt]
 vdet[lo:hi]=np.abs(np.linalg.det(np.stack([xx[:,1]-xx[:,0],xx[:,2]-xx[:,0],xx[:,3]-xx[:,0]],axis=1)))/6.0
numerical={}
for name,v in [('scalar_triple_f64',vtr),('numpy_linalg_det_f64',vdet)]:
 rr22=mass22/v; rr21=mass21/v
 numerical[name]={'m22_raw_gt950':int(np.count_nonzero(rr22>950.0)),'m22_gt950_plus_tolerance':int(np.count_nonzero(rr22>threshold)),'m21_raw_gt950':int(np.count_nonzero(rr21>950.0)),'m21_gt950_plus_tolerance':int(np.count_nonzero(rr21>threshold)),'m22_max_density':float(rr22.max()),'m21_max_density':float(rr21.max()),'volume_rows_bitwise_different_from_exact_rounded':int(np.count_nonzero(v!=vol))}
report={'schema':'numi.human.current406-022-exact-f32-density-reconciliation.v1','status':'PASS_RECONCILED_EXACT_F32_VOLUME_AND_RECEIPT_METHOD','physical_admission':False,'native_steps':0,'complete_allocation':False,'inputs':{str(p):sha(p) for p in [TARGET,M21,M22,V22,PRED]},'source330_pins':{'nodes_sha256':'7044051353d173b94143a4027ba7dcc468b221512b3f41d5819d27025b36d044','tets_sha256':'82e20067454748a9eca91bc6f06ab343677c051a139797ae051c64022bd87992'},'geometry':{'canonical_point_dtype':str(points.dtype),'point_count':int(len(points)),'cell_count':int(len(tets)),'target_sha256':sha(TARGET),'determinant_owner_sha256':sha(PRED),'exact_determinant_unit':'2^-447 m^3 det6','exact_zero_count':0,'exact_negative_count':0},'density':{'limit_kg_m3':950.0,'unchanged_tolerance_kg_m3':5e-10,'m22_raw_gt950':len(raw22),'m22_gt950_plus_tolerance':len(tol22),'m22_ids_gt950_plus_tolerance':tol22,'m21_raw_gt950':len(raw21),'m21_gt950_plus_tolerance':len(tol21),'m21_ids_gt950_plus_tolerance':tol21,'m22_max_density_exact_det':max22[0],'m22_max_density_exact_det_child':max22[1],'m21_max_density_exact_det':max21[0],'m21_max_density_exact_det_child':max21[1],'m22_m21_child_mass_array_equal':bool(np.array_equal(mass22,mass21)),'numerical_evaluation_comparisons':numerical},'reconciliation':{'022_independent_receipt_formula':'widen exact canonical F32 coordinates to binary64; compute abs(dot(x1-x0,cross(x2-x0,x3-x0)))/6','052_diagnostic_formula':'numpy.linalg.det of the 3x3 edge matrix','statement':'Do not confuse a greater-than-950 raw count with overages beyond the existing 5e-10 kg/m^3 tolerance. Numerically evaluated determinants differ on near-degenerate cells. Exact dyadic predicate and independent 013 formula are separately reported; do not change the gate tolerance.'},'elapsed_seconds':time.monotonic()-t0}
DEST.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':report['status'],'density':report['density'],'report':str(DEST),'report_sha256':sha(DEST),'elapsed_seconds':report['elapsed_seconds']},indent=2,sort_keys=True))
