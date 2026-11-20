"""Re-run the native coupled scene from the measured registered first-touch height."""
from pathlib import Path

source = Path('run_gravity_continuation.py').read_text()
source = source.replace("run = root / 'gravity-continuation'", "run = root / 'registered-first-touch-trial'")
needle = "launch['argv'].extend(('--resting-gravity-ramp-steps','3'))"
setup = needle + """
import re
release = root / 'gravity-continuation/stdout.log'
log = release.read_text()
root_match = re.search(r'executed_root_xyz_m=\\[([^,]+),([^,]+),([^\\]]+)\\]',log)
gap_match = re.search(r'static_support_witness_source=.*?initial_gap_min_m=(\\S+)',log)
assert root_match and gap_match, release
aligned_z = float(root_match[3])
observed_gap = float(gap_match[1])
first_touch_z = aligned_z - observed_gap + 1.0e-5
assert 0.04 <= observed_gap <= 0.045
app=root/'registered-first-touch-build/numi-human-native'
with app.open('rb') as stream:
    launch['pins'][str(app)] = hashlib.file_digest(stream,'sha256').hexdigest()
launch['argv'][0] = str(app)
launch['argv'][launch['argv'].index('--root-pose')+3] = format(first_touch_z,'.17g')
launch['argv'].append('--mechanics-only')
launch['environment']['NUMI_HUMAN_RESTING_REGISTERED_FIRST_TOUCH'] = '1'
launch['first_touch'] = {'prior_root_z_m': aligned_z, 'prior_registered_gap_m': observed_gap,
                        'requested_root_z_m': first_touch_z, 'requested_gap_m': 1.0e-5,
                        'prior': str(release)}
"""
assert source.count(needle) == 1
exec(compile(source.replace(needle, setup), 'registered_first_touch_trial', 'exec'))
