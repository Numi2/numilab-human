"""Use the completed native articulation as a fresh construction pose."""
import json
import sys
from pathlib import Path
from material_context import sha

base, posture, out = map(Path, sys.argv[1:])
posture = posture.resolve(strict=True)
config = json.loads((base / 'config.json').read_text())
args = config['argv']
for key, width in (('--root-pose', 8), ('--pose-q', 3)):
    while key in args:
        offset = args.index(key)
        del args[offset:offset + width]
assert '--resting-release-initialization' in args and '--resting-initial-posture' not in args
args.extend(('--resting-initial-posture', str(posture)))
config['pins'][str(posture)] = sha(posture)
config['construction_posture'] = {'source': str(posture), 'sha256': sha(posture), 'prior_velocity_and_forces_imported': False}
out.mkdir(exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
(out / 'run_supported.py').hardlink_to(base / 'run_supported.py')
