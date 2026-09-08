import hashlib
import json
import pathlib
import subprocess
import sys
import time

root = pathlib.Path('/Users/n/human-coupled-recruitment-20260908')
repo = pathlib.Path('/Users/n/MetalRobo-human-completion-20260907')
label = sys.argv[1]
sweeps = int(sys.argv[2])
command = json.loads((root / 'baseline-command.json').read_text())
command[4] = str(root / label)
command[-1] = str(sweeps)
if len(sys.argv) > 3:
    command.extend(['--whole-body-pose-sweeps', sys.argv[3]])
metadata = {
    'command': command,
    'native_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
    'native_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True),
    'source_sha256': hashlib.sha256((repo / 'src/core/NumiHumanMuscleEquilibrium.cpp').read_bytes()).hexdigest(),
    'binary_sha256': hashlib.sha256(pathlib.Path(command[0]).read_bytes()).hexdigest(),
    'workloads': subprocess.check_output(['ps', '-axo', 'pid,pcpu,etime,command'], text=True),
    'authority': 'offline_native_initialization_only',
}
metadata['artifacts_sha256'] = {}
for path in [pathlib.Path(command[0]), pathlib.Path('/Users/n/MetalRobo-human-completion-build-20260907/lib/libmetalrobo.dylib')] + [repo / relative for relative in ['src/core/NumiHumanMuscleEquilibrium.cpp', 'include/metalrobo/NumiHumanMuscleEquilibrium.hpp', 'apps/numilab_human_myosim_visual_probe.mm', 'tests/numi_human_static_support_test.cpp']] + [pathlib.Path(argument) for argument in command[1:] if pathlib.Path(argument).is_file()]:
    metadata['artifacts_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
patch = subprocess.check_output(['git', 'diff', '--binary'], cwd=repo)
(root / f'{label}-source.patch').write_bytes(patch)
metadata['patch_sha256'] = hashlib.sha256(patch).hexdigest()
(root / f'{label}-launch.json').write_text(json.dumps(metadata, indent=2) + '\n')
start = time.monotonic()
with (root / f'{label}-stdout.log').open('w') as stdout, (root / f'{label}-stderr.log').open('w') as stderr:
    run = subprocess.run(command, stdout=stdout, stderr=stderr)
metadata['returncode'] = run.returncode
metadata['elapsed_seconds'] = time.monotonic() - start
(root / f'{label}-completion.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps({'label': label, 'returncode': run.returncode, 'elapsed_seconds': metadata['elapsed_seconds']}), flush=True)
