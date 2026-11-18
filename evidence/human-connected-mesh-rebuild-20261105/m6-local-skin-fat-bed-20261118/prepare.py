import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

publication = Path(__file__).resolve().parent
destination = Path(sys.argv[1]).resolve()
pins = json.loads((publication / 'source-pins.json').read_text())

def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()

for filename, entry in pins['source_files'].items():
    if entry['base_file'] is not None:
        base = Path(entry['base_file'])
        if digest(base) != entry['base_sha256']:
            raise ValueError(f'Overlay base has changed: {base}')

destination.mkdir()
for filename, entry in pins['source_files'].items():
    target = destination / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if entry['base_file'] is not None:
        shutil.copyfile(entry['base_file'], target)
subprocess.run(['git', 'apply', '--check', '--unidiff-zero',
                str(publication / 'local-bed-contact.patch')], cwd=destination, check=True)
subprocess.run(['git', 'apply', '--unidiff-zero',
                str(publication / 'local-bed-contact.patch')], cwd=destination, check=True)
for filename, entry in pins['source_files'].items():
    changed = destination / filename
    if digest(changed) != entry['tested_sha256']:
        raise ValueError(f'Overlay result differs from the tested source: {changed}')

for name, entry in pins['unchanged_app_headers'].items():
    if digest(Path(entry['file'])) != entry['sha256']:
        raise ValueError(f'App include has changed: {entry["file"]}')
    shutil.copyfile(entry['file'], destination / name)
for filename in ('build.py', 'build_skin.py', 'build_shaders.py', 'run.py'):
    shutil.copyfile(publication / filename, destination / filename)
print(f'Exact tested source overlays prepared in {destination}', flush=True)
if '--stage-only' not in sys.argv[2:]:
    # The pinned owner build reads adjacent includes and keeps the original
    # checkout untouched. Supply the flattened input names its recipes use.
    for filename in pins['source_files']:
        source = destination / filename
        shutil.copyfile(source, destination / source.name)
    for name in ('build_shaders.py', 'build_skin.py', 'build.py'):
        subprocess.run([sys.executable, str(destination / name)], check=True)
