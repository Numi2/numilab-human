import gzip
import hashlib
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent


def audit(log, workload):
    tokens = [tuple(int(x, 16) for x in words.split(',')) for words in
              re.findall(r'^GateB physical-token words=(.+)$', log, re.M)]
    if len(tokens) != 9 or any(len(t) != 8 for t in tokens):
        raise ValueError('missing physical candidates')
    if [t[4] for t in tokens] != [1,2,3,4,4,5,6,7,8] or tokens[3] != tokens[4]:
        raise ValueError('generation or exact retry drift')
    if [t[3] for t in tokens] != [1000 + 10*g for g in [1,2,3,4,4,5,6,7,8]]:
        raise ValueError('clock drift')
    for owner in ['human','matter']:
        codes = re.findall(r'^mrnx_' + owner + r'_status code=(\d+)', log, re.M)
        if codes != ['0'] * 9:
            raise ValueError('failed or missing physical outcome')
    if 'Executed 1 test, with 0 failures' not in log and 'Executed 1 tests, with 0 failures' not in log:
        raise ValueError('test completion missing')
    if len(workload) < 2 or any(r['competing_gpu'] for r in workload):
        raise ValueError('competing workload or missing monitoring')
    times = [r['elapsed_seconds'] for r in workload]
    if not all(b > a and b-a < 5 for a,b in zip([0] + times, times)):
        raise ValueError('workload monitoring gap')
    if times[-1] < 270:
        raise ValueError('workload monitoring truncated')
    return {'attempted_roots':9, 'accepted_roots':8, 'accepted_seconds':0.00008,
            'root_timestep_microseconds':10, 'retry':'bitwise',
            'physical_failed_candidates':0, 'numivivo_jobs_detected':0}


def verify(base=BASE):
    r = json.loads((base/'receipt.json').read_text())
    for a in r['artifacts']:
        raw = (base/a['path']).read_bytes()
        if len(raw) != a['bytes'] or hashlib.sha256(raw).hexdigest() != a['sha256']:
            raise ValueError('artifact drift')
    launch = json.loads((base/'costal-idle-clean-launch.json').read_text())
    if launch['returncode'] != 0 or launch.get('interrupted_reason') or launch['competing_workloads']:
        raise ValueError('run failed or was interrupted')
    for owner in ['native', 'brain']:
        state = launch['source_state'][owner]
        if state != {'revision':r[owner+'_commit'], 'status':''}:
            raise ValueError('source drift')
    if hashlib.sha256((base/'run_costal.executed.py').read_bytes()).hexdigest() != launch['runner_sha256']:
        raise ValueError('executed runner drift')
    expected = {'bounded_costal_callback_deadline':True, 'exact_retry':True,
                'timeout_cause_established':False, 'prepared_anatomical_tissue':False,
                'sustained_loading':False, 'standing':False, 'walking':False,
                'calibration':False, 'performance':False, 'full_release':False}
    if r['qualification'] != expected:
        raise ValueError('qualification drift')
    result = audit(gzip.decompress((base/'costal-idle-clean.log.gz').read_bytes()).decode(),
                   json.loads((base/'costal-idle-clean-workload.json').read_text()))
    if result != r['measurements']:
        raise ValueError('measurement drift')
    return {'status':'bounded_costal_deadline_verified', **result}


if __name__ == '__main__':
    print(json.dumps(verify(),indent=2))
