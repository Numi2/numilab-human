import hashlib,importlib.util,json,pathlib
from fractions import Fraction
import numpy as np
H=pathlib.Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
R=H/'fractional-semantic-support-090-sparse-graph/fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001'
T=H/'fine-physical-volume-706/current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz'
M21=R/'attempt-021-source-exhaustion-batch/run-001/merged-held-centroid-noncentroid-exhaustion.npz'
M22=R/'attempt-022-source-exhaustion-reserved-batch/run-001/merged-held-centroid-noncentroid-source-exhaustion.npz'
V22=R/'attempt-022-source-exhaustion-reserved-batch/independent-replay-013.json'
PREV=R/'attempt-054-exact-f32-direct-density/exact-f32-direct-density-reconciliation.json'
PRED=H/'fine-physical-volume-706/solve_oriented_cavity_milp_081.py'
OUT=R/'attempt-054-exact-f32-direct-density/exact-overage-row-replay-055.json'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for q in iter(lambda:f.read(1<<20),b''):h.update(q)
 return h.hexdigest()
expected={T:'5dcebfd751dedcb1891559352e9d0e673ea3a8e15a8edeae8179c6d04e847211',M21:'6609bd2720dff820fac1075e42d99f4a4c011f15f3a7e8c2597cc0ac5c918c19',M22:'69c212a0a11395a7c0584b99fc4cc00fd543754baec4c542266b8a4226957623',V22:'7ce2d8dc4ab8c57da2b2b3b3692dece7f1fd8fe0901b15e26a7e5cbe1a1ea090',PREV:'d10302d30e44b7388cf9f4a71195a5fbd928b08c36fdeeae9b8e96a242eb701d',PRED:'3f7c276bef3bb4427f6acb41f6eeedffbeae273c9493a3b7854b244a5a2fa024'}
for p,d in expected.items():
 if sha(p)!=d:raise RuntimeError('input SHA mismatch: '+str(p))
v22=json.loads(V22.read_text()); prev=json.loads(PREV.read_text())
if v22['inputs'][str(T)]!=expected[T] or v22['inputs'][str(M22)]!=expected[M22]:raise RuntimeError('022 receipt input scope mismatch')
ids=np.asarray(prev['density']['m22_ids_gt950_plus_tolerance'],dtype=np.int64)
expected_ids=np.array([198545,233459,233636,288320,288360,539645,752551,804459],dtype=np.int64)
if not np.array_equal(ids,expected_ids):raise RuntimeError('exact overage witness roster differs')
with np.load(T,allow_pickle=False) as z:points=z['points_world_m'];tets=z['tetrahedra']
if points.dtype!=np.float32:raise RuntimeError('target coordinate dtype is not F32')
with np.load(M21,allow_pickle=False) as z:m21=z['child_current_mass_kg'].astype(np.float64)
with np.load(M22,allow_pickle=False) as z:m22=z['child_current_mass_kg'].astype(np.float64)
if len(m21)!=len(tets) or len(m22)!=len(tets):raise RuntimeError('mass and target domains differ')
spec=importlib.util.spec_from_file_location('exact_det_081_replay',PRED);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
limit=Fraction.from_float(950.0+5e-10);rows=[]
for i in ids:
 tet=tets[int(i)];d=abs(int(mod.det(*(mod.ip(points[int(v)]) for v in tet))))
 vol=Fraction(d,6*(1<<447));rho21=Fraction.from_float(float(m21[i]))/vol;rho22=Fraction.from_float(float(m22[i]))/vol
 if not rho21>limit or not rho22>limit:raise RuntimeError('row no longer crosses exact threshold')
 rows.append({'child_row':int(i),'det6_abs':d,'mass_m21_kg':float(m21[i]),'mass_m22_kg':float(m22[i]),'mass_delta_kg':float(m22[i]-m21[i]),'density_m21_kg_m3':float(rho21),'density_m22_kg_m3':float(rho22),'density_delta_kg_m3':float(rho22-rho21),'over_950_plus_5e-10_before':True,'over_950_plus_5e-10_after':True})
if not np.array_equal(m21[ids],m22[ids]):raise RuntimeError('022 batch changed a preexisting exact overage row')
report={'schema':'numi.human.current406-exact-f32-density-inherited-overage-row-replay.v1','status':'PASS_EXACT_F32_OVERAGE_ROWS_INHERITED_UNCHANGED','physical_admission':False,'native_steps':0,'complete_allocation':False,'inputs':{str(p):sha(p) for p in [T,M21,M22,V22,PREV,PRED]},'formula':{'coordinate_source':'points_world_m canonical float32 from target 406; interpreted exactly as dyadic binary32','determinant_owner':'081 exact dyadic predicate; det6 integer units 2^-447 m^3','volume':'abs(det6)/(6*2^447) m^3','density_gate_kg_m3':950.0,'unchanged_tolerance_kg_m3':5e-10},'counts':{'exact_m21_over_tolerance':len(ids),'exact_m22_over_tolerance':len(ids),'rows_changed_by_022_batch':0},'overage_rows':rows,'metrics':{'max_exact_density_m21_kg_m3':max(x['density_m21_kg_m3'] for x in rows),'max_exact_density_m22_kg_m3':max(x['density_m22_kg_m3'] for x in rows),'max_mass_delta_kg':max(abs(x['mass_delta_kg']) for x in rows),'max_density_delta_kg_m3':max(abs(x['density_delta_kg_m3']) for x in rows)},'interpretation':'Exact dyadic evaluation identifies eight inherited rows above the unchanged density tolerance. Their M21 and M22 mass values are identical, so the 022 source-exhaustion additions did not worsen these overages. The independent 013 cross-product calculation reports three; the 052 numpy.linalg.det calculation reports twenty. These counts differ from determinant evaluation, not target geometry/source330/022 ledger pins.'}
OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':report['status'],'rows':len(rows),'report':str(OUT),'sha256':sha(OUT)},indent=2))
