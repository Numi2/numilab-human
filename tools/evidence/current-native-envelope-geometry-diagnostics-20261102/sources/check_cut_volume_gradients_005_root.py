from pathlib import Path
import numpy as np, json, hashlib, time
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
R=H/'joint-local-i-patch-preflight-002/run-005-root-inventory'
start=time.monotonic()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((R/'report.json').read_text());sp=Path(report['sensitivities']['path'])
assert sha(sp)==report['sensitivities']['sha256']
tp=Path(report['typed']['path']);assert sha(tp)==report['typed']['sha256']
with np.load(sp) as z:
 j=z['active_I_joint_nodes'];g=z['gradients'];rows=[z['component_%02d_rows'%i] for i in range(11)]
with np.load(tp) as z:p=z['points_reference_world_m_f64'];t=z['volume_tetrahedra']
assert np.array_equal(p,p.astype(np.float32).astype(np.float64)), 'Inventory must match emitted F32 rest coordinates exactly'
rng=np.random.default_rng(20261102);checks=[]
for trial in range(3):
 d=np.zeros_like(p);d[j]=rng.normal(size=(len(j),3));d[j]/=np.maximum(np.linalg.norm(d[j],axis=1)[:,None],1e-30)
 h=1e-6
 for i,rr in enumerate(rows):
  tt=t[rr];x=p[tt];v=d[tt]
  def volume(a):return np.sum(np.einsum('ij,ij->i',a[:,1]-a[:,0],np.cross(a[:,2]-a[:,0],a[:,3]-a[:,0])),dtype=np.float64)/6
  plus=volume(x+h*v);minus=volume(x-h*v);fd=(plus-minus)/(2*h);analytic=float(np.sum(g[i]*d[j]));scale=max(abs(analytic),abs(fd),1e-12);rel=abs(fd-analytic)/scale
  checks.append({'trial':trial,'component':i,'analytic_m2':analytic,'finite_difference_m2':float(fd),'relative_error':float(rel),'pass':bool(abs(fd-analytic)<2e-10+1e-5*scale)})
out={'schema':report['schema'],'status':'PASS_RANDOM_DIRECTION_VOLUME_DERIVATIVE_CHECKS' if all(x['pass'] for x in checks) else 'FAIL_VOLUME_DERIVATIVE_CHECKS','physical_admission':False,'native_steps':0,'checks':checks,'maximum_relative_error':max(x['relative_error'] for x in checks),'elapsed_seconds':time.monotonic()-start,'source':{'path':str(Path(__file__)),'sha256':sha(Path(__file__))},'inventory':{'path':str(R/'report.json'),'sha256':sha(R/'report.json')}}
(R/'gradient-controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='checks'}));assert all(x['pass'] for x in checks)
