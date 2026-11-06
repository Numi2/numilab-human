from pathlib import Path
import sys,json,hashlib,io
import numpy as np
w=Path('/Users/numi/Work/human-integrated-gaps-20261104');h=Path('/Users/numi/Work/human-checkpoint-owner-20261105');o=Path(__file__).resolve().parent
sys.path.insert(0,str(h/'tools'))
from build_human_conditioned_cavities import cavity_payload
original=json.loads((w/'current-higher-degree-cavity-641/proposal/report.json').read_text());plan=json.loads((w/'higher-degree-native-642/install-motion.plan.json').read_text());execution=json.loads((w/'higher-degree-native-642/install-motion.execution.json').read_text())
assert execution['exit_code']==1 and execution['resource_stop_reason'] is None
assert plan['stages'][0]['environment']['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256']==original['output']['sha256']
log=w/'higher-degree-native-642/install-motion.stderr.log'
with log.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==execution['stderr_sha256']
witness=[json.loads(s.split('=',1)[1]) for s in log.read_text().splitlines() if s.startswith('skin_fat_cavity_transport_failures=')]
assert len(witness)==1 and witness[0]['complete'] is True and witness[0]['maximum_transport_distance_m']==.187444
indices=witness[0]['cavity_indices'];assert len(indices)==len(set(indices)) and all(type(i) is int and 0<=i<len(original['patches']) for i in indices)
assert 0<len(indices)<len(original['patches'])
excluded=set(indices);patches=[p for i,p in enumerate(original['patches']) if i not in excluded]
with np.load(original['input']['volume']['path']) as z:count=len(z['volume_tetrahedra'])
payload=cavity_payload(original['input']['volume']['sha256'],count,patches);b=io.BytesIO();np.savez_compressed(b,**payload)
actual=o/'proposal/fixed-boundary-volume-proposal.npz'
with actual.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
assert hashlib.sha256(b.getvalue()).hexdigest()==digest
(o/'filter-replay.json').write_text(json.dumps(dict(status='PASS_BYTE_EXACT_AUTHENTICATED_FAILURE_FILTER_REPLAY',proposal_sha256=digest,retained_cavities=len(patches),excluded=indices,transport_radius_unchanged=True,physical_admission=False),indent=2)+'\n')
