"""Compare bounded costal outcomes without treating them as balance evidence."""
import gzip
import json
import re
from pathlib import Path

base = Path(__file__).parent
prior = base.parent / 'costal-idle-20260908'
before = json.loads((prior / 'costal-idle-clean-launch.json').read_text())
after = json.loads((base / 'bvh-clean-launch.json').read_text())
assert before['returncode'] == after['returncode'] == 0
assert after['source_state']['native']['revision'] == '2ca77bbcb0b7976b1706f3a9c8e5b0274fc08a75'
assert all(not item['status'] for item in after['source_state'].values())
assert before['environment'] == after['environment']
assert before['brain_test_binary_sha256'] == after['brain_test_binary_sha256']
for key, digest in before['artifact_sha256'].items():
    if key not in ['NUMANX_METALROBO_LIBRARY', 'NUMANX_MATTER_METALLIB']:
        assert after['artifact_sha256'][key] == digest, key
assert after['environment']['MTL_DEBUG_LAYER'] == '1'
workload = json.loads((base / 'bvh-clean-workload.json').read_text())
assert workload and all(not row['competing_gpu'] for row in workload)
assert workload[-1]['elapsed_seconds'] >= after['elapsed_seconds'] - 3
old = gzip.open(prior / 'costal-idle-clean.log.gz', 'rt').read()
new = gzip.open(base / 'bvh-clean.log.gz', 'rt').read()
expected_test = 'testAuthoredMatterBrainProposalApplyAndJointPublication'
assert f'{expected_test}]\' passed (' in old
assert f'{expected_test}]\' passed (' in new
assert 'Executed 1 test, with 0 failures' in new
comparisons = {}
for prefix, expected_count in [('mrnx_matter_status', 9), ('mrnx_human_status', 9),
                                ('GateB risk-diagnostics', 9), ('GateB policy-head', 28)]:
    left = [row for row in old.splitlines() if row.startswith(prefix)]
    right = [row for row in new.splitlines() if row.startswith(prefix)]
    assert len(left) == len(right) == expected_count, prefix
    assert left == right, prefix
    comparisons[prefix] = {'rows': len(right), 'exact': True}
checks = gzip.open(base / 'native-checks.log.gz', 'rt').read()
assert re.search(r'100% tests passed(?:, 0 tests failed)? out of 7', checks)
assert checks.count('surface_bvh fixture=') == 13
result = {'native_revision': after['source_state']['native']['revision'],
          'brain_revision': after['source_state']['brain']['revision'],
          'before_runner_seconds': before['elapsed_seconds'],
          'after_runner_seconds': after['elapsed_seconds'],
          'runner_speedup': before['elapsed_seconds'] / after['elapsed_seconds'],
          'xctest_seconds': float(re.search(r'passed \(([0-9.]+) seconds\)', new)[1]),
          'compared_diagnostics': comparisons,
          'accepted_roots': 8, 'physical_seconds': 0.00008,
          'native_tests_passed': 7,
          'scope': 'source-default costal transaction and implementation throughput; not standing, anatomical equilibrium, whole-state cross-revision replay, or a production performance envelope'}
print(json.dumps(result, indent=2))
