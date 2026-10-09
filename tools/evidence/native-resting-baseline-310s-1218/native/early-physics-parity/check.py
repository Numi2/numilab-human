from pathlib import Path
import hashlib,json,datetime
base=Path('/Users/n/numi-human-retained-delivery-20261009')
prior=base/'muscle-conforming-refinement-1216/native-refinement-1217-attempt2/native-run'
current=base/'skin-resting-multipose-clearance-1218/native-baseline-310s-preparation/native-run'
out=current.parent/'early-physics-parity-001'
out.mkdir(exist_ok=False)
def sha(b):return hashlib.sha256(b).hexdigest()
rows=[]
for name in ('resting-coupled.csv','resting-com-momentum-diagnostic.csv','resting-com-support-impulses.csv'):
 old=(prior/name).read_bytes()
 with (current/name).open('rb') as f:prefix=f.read(len(old))
 rows.append({'file':name,'prior_path':str(prior/name),'current_path':str(current/name),'prior_bytes':len(old),'prior_data_rows':len(old.splitlines())-1,'prior_sha256':sha(old),'captured_current_prefix_sha256':sha(prefix),'prefix_byte_exact':prefix==old})
r={'status':'all_retained_40s_trace_prefixes_byte_exact' if all(x['prefix_byte_exact'] for x in rows) else 'trace_prefix_mismatch','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prior_horizon_s':40.0000018999,'checks':rows,'qualification':'Comparison of the first closed 40-second prior traces with equal-length prefixes observed during the active 310-second run. The active run is not closed or qualified by this check. Geometry, later physical state and complete root identity are not assessed.'}
(out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
