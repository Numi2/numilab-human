from pathlib import Path
import sys, json, hashlib, struct, time
import numpy as np
root=Path('/Users/n/numi-human-carrier-storage-1372');sys.path.insert(0,str(root/'src'))
from numilab_human.passive_tissue_parent_carriers import CarrierRow,encode_carriers
b=Path('/Users/n/numi-human-back-m6-1349')
files=[b/'attempt-412-descending-cap-self-crossing-flip-screen-003/full_candidate.json',b/'attempt-424-loop-triangulated-cap-interface-153-154-001/full_candidate.json']
out=root/'tools/evidence/passive-carrier-storage-m6-1372'
out.mkdir(parents=True,exist_ok=True)
result={'scope':'Actual carrier encoder storage gate on retained candidates; synthetic unit normals and one-body field isolate triangle storage. Not anatomy or native admission.','inputs':{},'rows':[]}
for path in files:
 result['inputs'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 c=json.loads(path.read_text())
 for sid in range(153,159):
  points=np.asarray([[float(int(n)/int(d)) for n,d in p] for p in c['points_exact'][str(sid)]])
  v=np.column_stack((points,np.tile([0.,0.,1.],(len(points),1))))
  fs=np.asarray([f for f in c['faces'][str(sid)] if f is not None])
  row=CarrierRow(sid,v,fs,struct.pack('<I8f',20,0,0,0,0,0,0,1,1),np.ones((len(points),1)))
  started=time.monotonic(); error=None
  try: encode_carriers([row],registration_fingerprint=1,source_digest=b'x'*32)
  except ValueError as e:error=str(e)
  expect_reject='424-' in str(path) and sid<=156
  assert (error is not None)==expect_reject,(sid,error)
  if error:assert 'exactly degenerate triangle' in error,error
  result['rows'].append(dict(candidate=path.parent.name,row=sid,accepted=error is None,error=error,elapsed_s=time.monotonic()-started))
result['passed']=True
result['source_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'src/numilab_human/passive_tissue_parent_carriers.py',root/'src/numilab_human/cardiac_cavity_intersections.py']}
p=out/'real-candidate-regression.json';assert not p.exists();p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result))
