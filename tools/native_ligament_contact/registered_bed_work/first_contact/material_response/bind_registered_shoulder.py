"""Install the registered NHMYO/NHTEND pair in an existing native Human launch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bind(launch, folder):
    registration = folder / 'registration.json'
    info = json.loads(registration.read_bytes())
    args = list(launch['argv'])
    if args.count('--tendon-payload') != 1:
        raise ValueError('Expected one tendon input')
    tendon_slot = args.index('--tendon-payload') + 1
    old_muscle, bone, old_tendon = map(Path, (args[2], args[3], args[tendon_slot]))
    muscle, tendon = folder / old_muscle.name, folder / old_tendon.name
    manifest = tendon.with_suffix('.manifest.json')
    pins = dict(launch['pins'])
    active = (old_muscle, old_tendon, bone)
    hashes = tuple(sha(p) for p in active)
    if any(pins.get(str(p), actual) != actual for p, actual in zip(active, hashes)):
        raise ValueError('Active launch input differs from its pinned digest')
    if hashes[2] != info['source_bone'] or hashes[:2] not in (
        (info['parent_muscle'], info['parent_tendon']), (info['muscle'], info['tendon'])
    ):
        raise ValueError('Incompatible muscle, tendon or bone source')
    if sha(muscle) != info['muscle'] or sha(tendon) != info['tendon']:
        raise ValueError('Registered payload hash mismatch')
    meta = json.loads(manifest.read_bytes())
    if (meta['payload']['sha256'] != info['tendon'] or
        meta['source']['bodyparts3d_bone_payload']['sha256'] != info['source_bone']):
        raise ValueError('Registration and tendon manifest do not match')
    args[2], args[tendon_slot] = str(muscle), str(tendon)
    pins.update({str(p): sha(p) for p in (muscle, tendon, manifest)})
    result = dict(launch, argv=args, pins=pins, registered_shoulder=sha(registration))
    if 'regional_tendon_manifest' in result:
        result['regional_tendon_manifest'] = pins[str(manifest)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('launch', 'registration_directory', 'output'):
        parser.add_argument(name, type=Path)
    paths = parser.parse_args()
    result = bind(json.loads(paths.launch.read_bytes()), paths.registration_directory.resolve())
    data = (json.dumps(result, indent=2) + '\n').encode()
    with tempfile.NamedTemporaryFile(dir=paths.output.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.link(temp, paths.output)
        except FileExistsError:
            if paths.output.read_bytes() != data:
                raise ValueError('Output already contains a different launch')
        finally:
            temp.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
