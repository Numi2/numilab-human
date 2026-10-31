from pathlib import Path
import numpy as np,json,hashlib,tempfile,time
from source_semantic_support_191_root import load_support,check_edges
H=Path(__file__).resolve().parent;P=H.parent;D=P/'fractional-semantic-support-090-sparse-graph'
out=H/'semantic-flow-verifier-191-tests-root';out.mkdir(exist_ok=False)
rows=[]
for role,mat,support_dir,partial_dir in [
 ('Skin',15,'skin-743497-run-002','skin-743497-carrier-run-014-partial-flow-root/partial-flow-skin-014'),
 ('Fat',16,'fat-853406-run-003-root','fat-853406-carrier-run-009-partial-flow-export/partial-flow-008')]:
 start=time.monotonic();sp=D/support_dir/'semantic-support.bin';pd=D/partial_dir
 parents=np.loadtxt(pd/'transport-source-parent.csv',delimiter=',',skiprows=1,ndmin=2)
 child_count=743497 if role=='Skin' else 853406
 d,sh=load_support(sp,mat,child_count,parents[:,3].astype(np.uint32),parents[:,4].astype(np.uint32),parents[:,5].astype(np.uint32))
 flow=np.loadtxt(pd/'transport-flow.csv',delimiter=',',skiprows=1,ndmin=2)
 nonpreferred=check_edges(d,flow[:,0].astype(np.uint32),flow[:,1].astype(np.uint32),parents[:,4].astype(np.uint32))
 rejects=[]
 region=int(parents[0,4]);good_children=set((d['child_region_keys'][(d['child_region_keys']&np.uint64(0xffffffff))==region]>>np.uint64(32)).tolist())
 wrong=next(c for c in range(child_count) if c not in good_children)
 try:check_edges(d,np.array([wrong]),np.array([0]),parents[:,4].astype(np.uint32))
 except ValueError as e:rejects.append(str(e))
 else:raise AssertionError('out-of-onehop flow silently accepted')
 with tempfile.TemporaryDirectory(dir=out) as t:
  bad=Path(t)/'bad.bin';bad.write_bytes(sp.read_bytes()+b'x')
  try:load_support(bad,mat,child_count,parents[:,3].astype(np.uint32),parents[:,4].astype(np.uint32),parents[:,5].astype(np.uint32))
  except ValueError as e:rejects.append(str(e))
  else:raise AssertionError('trailing support byte accepted')
 rows.append({'role':role,'source_parent_rows':len(parents),'partial_flow_edges_checked':len(flow),'children':child_count,'groups':len(d['group_materials']),'nonpreferred_p1_but_onehop_permitted_edges':nonpreferred,'support_sha256':sh,'status':'PASS_PARTIAL_FLOW_SEMANTIC_PERMISSIONS_ONLY','rejections':rejects,'elapsed_s':time.monotonic()-start})
report={'status':'PASS_FULL_DOMAIN_SUPPORT_AND_PARTIAL_FLOW_PERMISSION_CHECKS','physical_admission':False,'native_steps':0,'complete_mass_allocation':False,'roles':rows,'support_verifier_sha256':hashlib.sha256((H/'source_semantic_support_191_root.py').read_bytes()).hexdigest(),'final_allocation_verifier_sha256':hashlib.sha256((H/'verify_recorded_carrier_allocation_191_root.py').read_bytes()).hexdigest()}
(out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
