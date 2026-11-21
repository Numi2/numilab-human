from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import os

home = Path.home()
w = home / 'Work/human-integrated-gaps-20261104'
p = w / 'resting-clearance-and-atrium-017'
here = p / 'extended-equilibrium-027/full-volume-bed-placement-030'
spec = importlib.util.spec_from_file_location('compression', home / 'storage-cleanup-20261101-transparent-compression-001/compress_authorized_inputs.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.OUT = here / 'trial-storage-receipts'
current = p / 'floor-native-live-parent-026'
helper.CUR_MAN = current / 'launch.json'
helper.CUR_SPEC = current / 'compact-motion-evidence.json'
helper.PIN = current / 'exact-skin-surface.json'
protected = '\n'.join(path.read_text() for path in (helper.CUR_MAN, helper.CUR_SPEC, helper.PIN))
old_root = home / 'Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125'
groups = defaultdict(list)
for trial in ('routine-control-prefix-226','native-scene-view-229','native-scene-view-227','contact-cache-control-prefix-221'):
    native = old_root / trial / 'install-motion'
    for path in (native/'resting-human.mrvpack',native/'accepted-geometry/step-0.mrvpack',native/'accepted-geometry/step-32.mrvpack'):
        if path.is_file():
            s = path.lstat()
            if not path.is_symlink() and str(path) not in protected and s.st_blocks*512 > 240*1024**2:
                groups[s.st_dev,s.st_ino].append(path)
for index, paths in enumerate(groups.values()):
    if len(paths) != paths[0].stat().st_nlink:
        continue
    result = helper.compress_alias_group(f'historical-accepted-geometry-{index}', paths, helper.sha(paths[0]))
    print(json.dumps({'paths':list(map(str, paths)),'saved':result.get('allocated_bytes_saved',0),'available':helper.df()['available_bytes']}), flush=True)
paths = sorted(path for trial in (w/'coarse-hand-cavity-024').glob('*') for path in trial.glob('native/resting-human.mrvpack'))
paths += [
    w/'loaded-initialization-023/activation-010/native/accepted-geometry/step-16.mrvpack',
    w/'tissue-registration-native-022/native-verified/native/accepted-geometry/step-16.mrvpack',
]
for index,path in enumerate(paths):
    if helper.df()['available_bytes'] >= 3*1024**3:
        break
    if not path.is_file() or path.is_symlink() or str(path) in protected:
        continue
    s = path.lstat()
    if s.st_nlink != 1 or s.st_blocks*512 < 120*1024**2:
        continue
    result = helper.compress_single(f'historical-native-capture-{index}', path, helper.sha(path))
    print(json.dumps({'path':str(path),'saved':result.get('allocated_bytes_saved',0),'available':helper.df()['available_bytes']}), flush=True)
