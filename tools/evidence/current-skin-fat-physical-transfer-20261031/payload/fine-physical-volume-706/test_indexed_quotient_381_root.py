"""Actual-owner parity and incidence replay for indexed209 quotient lookups."""
from pathlib import Path
import json,time,hashlib,importlib.util,numpy as np
H=Path(__file__).resolve().parent
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
A=load('dense209control381',H/'transfer_surface_contractions_to_volume_209_root.py')
B=load('indexed379control381',H/'transfer_surface_contractions_indexed_379_root.py')
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
rpath=H/'paired-I-interface-377-root/run002/report.json';r=json.loads(rpath.read_text())
with np.load(r['source_I']) as z:sp=z['points_world_m_f32'];sf=z['faces']
selected=np.unique(np.linspace(0,len(r['operations'])-1,64,dtype=np.int64))
proofs=[];started=time.monotonic()
for rr in r['roles']:
 with np.load(rr['input']) as z:p=z['points_world_m'];t=z['tetrahedra'];mat=z['material_indices']
 boundary=A.boundary(t);look={A.key(x):i for i,x in enumerate(p)};keys={tuple(sorted(map(int,f))) for f in boundary}
 def state():return {'p':p,'t':t.copy(),'mat':mat,'active':np.ones(len(t),bool),'map':np.arange(len(p)),'look':look,'boundary_keys':keys}
 dense=state();fast=state()
 flat=t.ravel();order=np.argsort(flat,kind='stable');offset=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(p)))]
 fast.update(node_incidence_order=order,node_incidence_offsets=offset,node_incidence_added={})
 timing={'dense_s':0.,'indexed_s':0.};keeps=[]
 for k in selected:
  o=r['operations'][int(k)]
  tic=time.monotonic();a,wa=A.try_op(dense,sp,sf,o);timing['dense_s']+=time.monotonic()-tic
  tic=time.monotonic();b,wb=B.try_op(fast,sp,sf,o);timing['indexed_s']+=time.monotonic()-tic
  assert wa==wb=='PASS' and a.keys()==b.keys()
  for key in a:
   if isinstance(a[key],np.ndarray):assert np.array_equal(a[key],b[key]),key
   else:assert a[key]==b[key],key
  for v,x in [(dense,a),(fast,b)]:
   v['t'][x['rows']]=x['new'];v['active'][x['rows'][x['deleted']]]=False;v['map'][x['retired_volume_node']]=x['kept_volume_node']
  fast['node_incidence_added'].setdefault(b['kept_volume_node'],set()).update(map(int,b['rows'][~b['deleted']]))
  keeps.append(b['kept_volume_node'])
 # Independently reconstruct every dynamically extended keep star from the full current body.
 for n in keeps:
  base=order[offset[n]:offset[n+1]]//4;added=np.asarray(sorted(fast['node_incidence_added'].get(n,())),np.int64)
  rows=np.unique(np.r_[base,added]);rows=rows[fast['active'][rows]&np.any(fast['t'][rows]==n,axis=1)]
  full=np.flatnonzero(fast['active']&np.any(fast['t']==n,axis=1));assert np.array_equal(rows,full)
 assert np.array_equal(dense['t'],fast['t']) and np.array_equal(dense['active'],fast['active'])
 # Repeated retired endpoint must preserve the same explicit failure, not become a stale-star proposal.
 for k in selected[[0,-1]]:
  o=r['operations'][int(k)];a,wa=A.try_op(dense,sp,sf,o);b,wb=B.try_op(fast,sp,sf,o)
  assert a is b is None and wa==wb=='dependent_contraction'
 proofs.append({'role':rr['role'],'input_sha256':sha(rr['input']),'actual_passing_operations_compared':len(selected),'current_full_body_dynamic_stars_compared':len(keeps),'repeated_retired_endpoints_rejected':True,'timing':timing,'lookup_path_speedup':timing['dense_s']/max(timing['indexed_s'],1e-30)})
out=H/'indexed-interface-owner-381-root';out.mkdir(exist_ok=False)
receipt={'status':'PASS_ACTUAL_OWNER_INDEXED_QUOTIENT_PARITY','physical_admission':False,'native_steps':0,'input_operation_report_sha256':sha(rpath),'dense_owner_sha256':sha(H/'transfer_surface_contractions_to_volume_209_root.py'),'indexed_owner_sha256':sha(H/'transfer_surface_contractions_indexed_379_root.py'),'batch_producer_sha256':sha(H/'repair_paired_I_interfaces_indexed_380_root.py'),'same_exact_quality_sign_boundary_and_rejection_predicates':True,'role_controls':proofs,'verifier_sha256':sha(__file__),'elapsed_s':time.monotonic()-started,'limitation':'construction lookup microbenchmark only, not integrated simulation throughput'}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
