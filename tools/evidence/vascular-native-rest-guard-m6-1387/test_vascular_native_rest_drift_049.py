#!/usr/bin/env python3
"""Reject inconsistent carrier coordinates before creating a native scene."""
from pathlib import Path
import hashlib,json,subprocess,sys
import numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
C=A/'forearm-vascular-carrier-cpu-048-001'
O=A/'vascular-native-rest-drift-control-049-001'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not O.exists();O.mkdir()
r=json.loads((C/'report.json').read_text())
fields=[Path(p) for p in r['field_sha256']]
original=next(p for p in fields if int(np.load(p)['stable_surface_id'][0])==483)
z=np.load(original);data={k:z[k].copy() for k in z.files}
i=int(np.flatnonzero(data['weight_body91']==np.float32(1))[0])
data['body91_local_positions'][i,0]+=np.float32(1e-4)
bad=O/'inconsistent-cached-rest-stable483.npz';np.savez(bad,**data)
fields=[bad if p==original else p for p in fields]
constructor=A/'prepare_combined_anatomy_native_047.py'
out=O/'rejected-scene'
args=[sys.executable,str(constructor),'--parent',str(A/'liver-island-native-027-002'),'--source-preparation',str(A/'vascular-source-no-op-control-038-001/prepared'),'--output',str(out)]
for f in fields:args+=['--field',str(f)]
pins={str(p.resolve()):sha(p) for p in [constructor,Path(__file__),original,*fields]}
with (O/'constructor.log').open('w') as log:
 done=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
message=(O/'constructor.log').read_text()
assert done.returncode!=0 and 'fixed native vessel registered-rest gate failed' in message and not out.exists()
assert all(sha(p)==h for p,h in pins.items())
result={'status':'native_registered_rest_drift_rejected_before_output','input_sha256':pins,'stable_id':483,'vertex':i,'deliberate_local_carrier_x_change_m':1e-4,'cached_registered_rest_unchanged':True,'returncode':done.returncode,'output_created':out.exists(),'gpu_used':False,'native_run_performed':False,'anatomical_admission':False}
(O/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'report_sha256':sha(O/'report.json')}))
