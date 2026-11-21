"""Select the root pose at which the recorded fine material volume was registered."""
import json,sys,re
from pathlib import Path
from emit_skin_source_registration import sha
src,log,out=map(Path,sys.argv[1:])
d=json.loads(src.read_text());line=next(l for l in log.read_text().splitlines() if l.startswith('resting_initial_support_alignment=applied'))
def triple(k):return json.loads(re.search(k+r'=(\[[^\]]+\])',line).group(1))
args=d['argv'];i=args.index('--root-pose')+1
if list(map(float,args[i:i+3]))!=triple('authored_root_xyz_m') or list(map(float,args[i+3:i+7]))!=triple('authored_root_quaternion_xyzw'):raise ValueError('Different source root')
assert 'limb_q_unchanged=1' in line and 'per_step_root_adjustment=none' in line
args[i:i+3]=list(map(str,triple('executed_root_xyz_m')))
d['physical_volume_release_reference']={'native_log_sha256':sha(log),'launch_sha256':sha(src),'launch_path':str(src),'provenance':line}
# Keep historical output roots discoverable by the replay harness.
out.write_text(json.dumps(d,indent=2)+'\n')
