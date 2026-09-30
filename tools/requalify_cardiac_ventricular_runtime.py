"""Recheck the retained full-source ventricular step on the current Matter runtime.

This checks the exact 1 us active/zero fixture after runtime changes. It does
not turn the prescribed tension or synthetic supports into a heartbeat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'Docs/media/cardiac-ventricular-native-step-20260930'
PACKAGE = ROOT / 'Build/cardiac-active-tension-ingress-20260930/full-ventricular-cooked.nmpkg'
TENSION = OLD / 'cooked-tension-100ms.f32le'
DEFAULT_RECEIPT = ROOT / 'Docs/media/cardiac-current-runtime-20260930/receipt.json'
SCHEMA = 'numi.human.ventricular-current-runtime-requalification.v1'


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError('ventricular current-runtime requalification: ' + message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    require(not result.stderr and len(result.stdout.strip().splitlines()) == 1,
            'native step did not return one clean JSON row')
    return result.stdout


def native_revision(root: Path) -> str:
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                              check=True, capture_output=True,
                              text=True).stdout.strip()
    status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain=v1',
                             '--untracked-files=no'], check=True,
                            capture_output=True, text=True).stdout
    require(len(revision) == 40 and not status, 'Matter source revision is dirty')
    return revision


def requalify(native_root: Path, build_dir: Path, receipt_path: Path,
              *, execute: bool, capture: bool) -> dict:
    native_root, build_dir, receipt_path = (Path(p).resolve() for p in
                                             (native_root, build_dir, receipt_path))
    binary = build_dir / 'numi-matter-ventricular-source-step'
    metallib = build_dir / 'shaders/NumiMatter.metallib'
    prior_path = OLD / 'receipt.json'
    prior = json.loads(prior_path.read_text())
    require(prior['schema'] ==
            'numi.human.ventricular-native-source-step-publication.v1'
            and prior['accepted_native_steps_per_run'] == 1
            and prior['full_four_chamber_accepted_steps'] == 0
            and prior['heartbeat_qualified'] is False,
            'prior bounded result changed')
    require(sha(PACKAGE) == prior['cooked_package_sha256']
            and sha(TENSION) == prior['files'][TENSION.name]['sha256'],
            'source cooked package or active tension changed')
    expected = {
        'active': json.loads((OLD / 'active-run-1.json').read_text()),
        'zero': json.loads((OLD / 'zero-run.json').read_text()),
    }
    require(expected['active'] == json.loads((OLD / 'active-run-2.json').read_text())
            and expected['active']['status_code'] == 0
            and expected['zero']['status_code'] == 0,
            'prior active/zero evidence changed')
    expected_hashes = {
        'active': prior['files']['active-nodes.bin']['sha256'],
        'replay': prior['active_replay_state_sha256'],
        'zero': prior['files']['zero-nodes.bin']['sha256'],
    }
    require(expected_hashes['active'] == expected_hashes['replay']
            and sha(OLD / 'active-nodes.bin') == expected_hashes['active']
            and sha(OLD / 'zero-nodes.bin') == expected_hashes['zero'],
            'prior retained states changed')
    require(binary.is_file() and metallib.is_file(), 'current native build absent')
    common = {
        'schema': SCHEMA,
        'status': 'current_runtime_requalified_identical_bounded_ventricular_fixture',
        'native_revision': native_revision(native_root),
        'native_step_binary_sha256': sha(binary),
        'native_metallib_sha256': sha(metallib),
        'prior_receipt_sha256': sha(prior_path),
        'cooked_package_sha256': sha(PACKAGE),
        'source_tension_sha256': sha(TENSION),
        'verifier_source_sha256': sha(Path(__file__)),
        'state_sha256': expected_hashes,
        'active_vs_zero_nodes_different': prior['active_vs_zero_nodes_different'],
        'maximum_active_minus_zero_displacement_m':
            prior['maximum_active_minus_zero_displacement_m'],
        'ventricular_tetrahedra': prior['ventricular_tetrahedra'],
        'ventricular_nodes': prior['ventricular_nodes'],
        'timestep_seconds': prior['timestep_seconds'],
        'accepted_native_steps_per_run': 1,
        'full_four_chamber_accepted_steps': 0,
        'heartbeat_qualified': False,
        'boundary': ('Same prescribed source-ordered ventricular tension, synthetic '
                     'density, three synthetic fixed nodes and one 1 us step as the '
                     'prior fixture. Identical accepted states requalify the bounded '
                     'runtime result only; no four-chamber mechanics, hydraulic wall '
                     'coupling, native electrical propagation or heartbeat follows.'),
    }
    if execute:
        with tempfile.TemporaryDirectory(prefix='ventricular-current-',
                                         dir=str(ROOT / 'Build')) as location:
            cases = {}
            for name, zero in (('active', False), ('replay', False), ('zero', True)):
                state = Path(location) / f'{name}.bin'
                command = [str(binary), str(PACKAGE), str(TENSION), str(state)]
                if zero:
                    command.append('--zero')
                measured = json.loads(run(command))
                require(measured == expected['zero' if zero else 'active']
                        and sha(state) == expected_hashes[name],
                        f'current native {name} differs from retained result')
                cases[name] = measured
        require(cases['active'] == cases['replay'], 'active replay differs')
        common['cases'] = cases
    if capture:
        require(execute and not receipt_path.exists(),
                'capture needs execution and a new receipt path')
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(common, indent=2, sort_keys=True) + '\n').encode()
        pending = receipt_path.with_name(receipt_path.name + f'.{os.getpid()}.pending')
        with pending.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, receipt_path)
    else:
        saved = json.loads(receipt_path.read_text())
        require(saved['cases']['active'] == expected['active']
                and saved['cases']['replay'] == expected['active']
                and saved['cases']['zero'] == expected['zero'],
                'retained native status rows differ')
        if not execute:
            common['cases'] = saved['cases']
        require(saved == common, 'current source, binary, or evidence identity drift')
    return common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path)
    parser.add_argument('--receipt', type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    native = args.native_root.resolve()
    build = (args.build_dir or native / 'Build/cardiac-active-native').resolve()
    result = requalify(native, build, args.receipt,
                       execute=args.execute or args.capture, capture=args.capture)
    print(json.dumps({'status': result['status'],
                      'native_revision': result['native_revision'],
                      'active_replay_bitwise': True,
                      'heartbeat_qualified': False}, sort_keys=True))


if __name__ == '__main__':
    main()
