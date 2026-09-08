"""Verify bounded curved-support evidence without promoting dynamic failures."""
from pathlib import Path
import hashlib,json,math,re,sys
root=Path(__file__).resolve().parent
repo=root.parents[2]
sys.path.insert(0,str(repo/'src'))
from numilab_human.support_stance import assess
receipt=json.loads((root/'receipt.json').read_text())
for name,expected in receipt['files'].items():
    data=(root/name).read_bytes()
    assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256'],name
qualified=json.loads((root/'qualified-certificate/receipt.json').read_text())
assert qualified['runtime_revision']==receipt['native_revision'] and qualified['runtime_worktree_status']==''
assert qualified['status']=='wrench_only_passed' and qualified['failures']==[] and qualified['exit_code']==0
metrics,failures=assess((root/'qualified-certificate/stdout.log').read_text(),0,18)
assert not failures and metrics['support_payload']=='NHCNT2' and metrics['internal_balanced']=='false'
assert metrics['active_support_contacts']=='6'
oracle=json.loads((root/'source-oracle.json').read_text())
assert oracle['certificate_sha256']==hashlib.sha256((root/'qualified-certificate/stdout.log').read_bytes()).hexdigest()
assert oracle['contact_payload_sha256']==hashlib.sha256((root/'myosim-fullbody-support-primitives.nhcnt').read_bytes()).hexdigest()
assert len(oracle['contacts'])==18 and len(oracle['primitives'])==10
assert oracle['minimum_full_primitive_gap_m']>=-1e-6 and oracle['maximum_lowering_error_m']<=2e-7
assert all(math.isfinite(x) and abs(x)<=1e-3 for x in oracle['force_residual_n']+oracle['moment_residual_nm'])
assert oracle['source_files_verified']==57
conformance=json.loads((root/'verified-conformance.json').read_text())
assert conformance['native_revision']==receipt['native_revision'] and conformance['native_worktree']==''
assert len(conformance['runs'])==3 and all(r['exit_code']==0 for r in conformance['runs'])
assert '100% tests passed out of 2' in (root/'verified-ctest.log').read_text()
assert 'sphere surface:' in (root/'verified-articulated-gpu.log').read_text()
assert 'ellipsoid surface:' in (root/'verified-articulated-gpu.log').read_text()
assert 'shapes=point,sphere,ellipsoid rollback=exact' in (root/'verified-matter-gpu.log').read_text()
assert 'Ran 191 tests' in (root/'python-tests-verified.log').read_text() and 'OK (skipped=7)' in (root/'python-tests-verified.log').read_text()
text=(root/'native-64-steps.log').read_text()
assert 'stand_deterministic_replay=bitwise' in text
state=next(json.loads(l.split('=',1)[1]) for l in text.splitlines() if l.startswith('stand_terminal_state='))
assert state['step_count']==64 and state['timestep_seconds']==1e-4 and state['root_assistance'] is False
assert len(state['q'])==129 and len(state['v'])==128
negative=json.loads((root/'dynamic-source-oracle.json').read_text())
assert negative['native_state_sha256']==hashlib.sha256((root/'native-64-steps.log').read_bytes()).hexdigest()
assert negative['primitive_geometry_passed'] is False and negative['joint_limit_violations']
assert negative['wrench_passed'] is None
for key in ('internal_equilibrium','dynamic_contact','dynamic_joint_limits','v5_anatomical_acceptance',
            'costal_requalification','standing','walking','calibration','performance','full_release'):
    assert receipt['qualification'][key] is False,key
print(json.dumps({'status':'bounded_curved_geometry_passed','native_revision':receipt['native_revision'],
    'full_primitive_gap_m':oracle['minimum_full_primitive_gap_m'],
    'dynamic_contact_qualified':False,'standing_qualified':False,'walking_qualified':False,'full_release':False},indent=2))
