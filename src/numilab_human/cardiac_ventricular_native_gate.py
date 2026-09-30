"""Audit the bounded full-source ventricular native step and optionally rerun it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from . import cardiac_active_force_gate as reference
from . import cardiac_active_tension_gate as tension_gate
from . import cardiac_material_frames as frames


def require(condition: bool, message: str) -> None:
    reference.require(condition, 'ventricular native gate: ' + message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1 << 20):
            value.update(chunk)
    return value.hexdigest()


def checked(path: Path, record: dict) -> None:
    require(path.is_file() and not path.is_symlink() and
            path.stat().st_size == record['bytes'] and
            digest(path) == record['sha256'], f'hash/size of {path.name}')


def command(arguments: list[str]) -> str:
    result = subprocess.run(arguments, capture_output=True, text=True,
                            check=True)
    require(not result.stderr and len(result.stdout.strip().splitlines()) == 1,
            f'native command output: {arguments[0]}')
    return result.stdout


def audit(asset: Path, activation: Path, tension: Path, evidence: Path,
          native_repo: Path, cooked_package: Path, cook_binary: Path,
          step_binary: Path, *, execute: bool = False) -> dict:
    asset, activation, tension, evidence, native_repo, cooked_package, \
        cook_binary, step_binary = (Path(path).resolve() for path in
            (asset, activation, tension, evidence, native_repo,
             cooked_package, cook_binary, step_binary))
    receipt = json.loads((evidence / 'receipt.json').read_text())
    require(receipt['schema'] ==
            'numi.human.ventricular-native-source-step-publication.v1' and
            receipt['status'] ==
            'accepted_one_synthetic_boundary_full_ventricular_source_step' and
            receipt['ventricular_tetrahedra'] == 1097534 and
            receipt['ventricular_nodes'] == 218077 and
            receipt['synthetic_density_kg_m3'] == 1050 and
            receipt['synthetic_fixed_nodes'] == 3 and
            receipt['timestep_seconds'] == 1e-6 and
            receipt['accepted_native_steps_per_run'] == 1 and
            receipt['full_four_chamber_accepted_steps'] == 0 and
            receipt['heartbeat_qualified'] is False,
            'scope and physical boundary')
    require(receipt['independent_gate_source_sha256'] == digest(Path(__file__)),
            'independent verifier source identity')
    require(digest(asset / 'manifest.json') ==
            receipt['source_asset_manifest_sha256'], 'source asset manifest')
    for name, (expected_sha, _shape, expected_bytes) in frames.SOURCE_BUFFERS.items():
        checked(asset / name, {'sha256': expected_sha, 'bytes': expected_bytes})
    expected_files = {
        'material-frame-manifest.json', 'material-frame-rotations.f64le',
        'cooked-tension-100ms.f32le', 'active-nodes.bin', 'zero-nodes.bin',
        'full-ventricular-cook.json', 'active-run-1.json',
        'active-run-2.json', 'zero-run.json', 'comparison.json',
        'independent-gate.json',
    }
    require(set(receipt['files']) == expected_files, 'exact published file set')
    for name, record in receipt['files'].items():
        checked(evidence / name, record)
    frame_manifest = json.loads((evidence / 'material-frame-manifest.json').read_text())
    require(frame_manifest['source_identity_record']['asset_manifest_sha256'] ==
            receipt['source_asset_manifest_sha256'] and
            frame_manifest['buffer']['sha256'] ==
            receipt['files']['material-frame-rotations.f64le']['sha256'] and
            frame_manifest['buffer']['shape'] == [1470083, 4] and
            frame_manifest['qualification']['physical_steps'] == 0,
            'full-source derived material frames')
    require(digest(tension / 'summary.json') ==
            receipt['source_tension_summary_sha256'] and
            digest(tension / 'active-tension-100ms.f32le') ==
            receipt['source_tension_100ms_sha256'],
            'source-ordered candidate tension')
    prior_gate = tension_gate.audit(asset, activation, tension)
    require((tension / 'independent-gate.json').read_text() ==
            json.dumps(prior_gate, sort_keys=True, indent=2) + '\n',
            'independent full-source tension parity')
    checked(cooked_package, {'bytes': receipt['cooked_package_bytes'],
                             'sha256': receipt['cooked_package_sha256']})
    revision = subprocess.run(['git', '-C', str(native_repo), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(['git', '-C', str(native_repo), 'status',
                             '--porcelain=v1', '--untracked-files=no'],
                            capture_output=True, text=True, check=True).stdout
    require(revision == receipt['native_revision'] and not status,
            'clean native source revision')
    require(digest(cook_binary) == receipt['cook_binary_sha256'] and
            digest(step_binary) == receipt['step_binary_sha256'],
            'exact native binaries')
    comparison = json.loads((evidence / 'comparison.json').read_text())
    active = json.loads((evidence / 'active-run-1.json').read_text())
    replay = json.loads((evidence / 'active-run-2.json').read_text())
    zero = json.loads((evidence / 'zero-run.json').read_text())
    cooked = json.loads((evidence / 'full-ventricular-cook.json').read_text())
    require(active == replay and active['device'] == 'Apple M4' and
            active['abi'] == 35 and active['status_code'] == 0 and
            active['completed_microsteps'] == 1 and
            active['accepted_native_steps'] == 1 and
            active['zero_tension_input'] is False and
            zero['status_code'] == 0 and
            zero['accepted_native_steps'] == 1 and
            zero['zero_tension_input'] is True and
            cooked['source_to_cooked_float32_bitwise'] is True and
            cooked['ventricular_tetrahedra'] == 1097534,
            'native cook and accepted-step results')
    active_state = np.fromfile(evidence / 'active-nodes.bin', '<f4').reshape(-1, 16)
    zero_state = np.fromfile(evidence / 'zero-nodes.bin', '<f4').reshape(-1, 16)
    require(len(active_state) == 218077 and len(zero_state) == 218077 and
            np.isfinite(active_state).all() and np.isfinite(zero_state).all() and
            np.array_equal(active_state[:, 3], zero_state[:, 3]) and
            np.array_equal(active_state[:3, :3], zero_state[:3, :3]),
            'finite node states, mass conservation and synthetic supports')
    difference = (active_state[:, :3].astype(np.float64) -
                  zero_state[:, :3].astype(np.float64))
    lengths = np.linalg.norm(difference, axis=1)
    require(comparison['active_state_sha256'] ==
            receipt['files']['active-nodes.bin']['sha256'] ==
            receipt['active_replay_state_sha256'] and
            comparison['zero_state_sha256'] ==
            receipt['files']['zero-nodes.bin']['sha256'] and
            comparison['nodes_different_from_zero'] ==
            receipt['active_vs_zero_nodes_different'] ==
            int(np.count_nonzero(lengths)) and
            comparison['maximum_active_minus_zero_displacement_m'] ==
            receipt['maximum_active_minus_zero_displacement_m'] ==
            float(lengths.max()),
            'full-state replay and active-versus-zero comparison')
    if execute:
        with tempfile.TemporaryDirectory(prefix='ventricular-native-gate-') as location:
            temporary = Path(location)
            package = temporary / 'recooked.nmpkg'
            field = temporary / 'cooked-tension.f32le'
            recooked = command([str(cook_binary), str(asset),
                                str(evidence / 'material-frame-rotations.f64le'),
                                str(tension / 'active-tension-100ms.f32le'),
                                frame_manifest['buffer']['sha256'],
                                str(package), str(field)])
            require(json.loads(recooked) == cooked and
                    digest(package) == receipt['cooked_package_sha256'] and
                    digest(field) ==
                    receipt['files']['cooked-tension-100ms.f32le']['sha256'],
                    'recomputed complete ventricular cook')
            for name, expected, expected_hash, zero_input in (
                ('active-1', active, receipt['files']['active-nodes.bin']['sha256'], False),
                ('active-2', replay, receipt['active_replay_state_sha256'], False),
                ('zero', zero, receipt['files']['zero-nodes.bin']['sha256'], True),
            ):
                nodes = temporary / f'{name}.bin'
                arguments = [str(step_binary), str(package), str(field), str(nodes)]
                if zero_input:
                    arguments.append('--zero')
                measured = json.loads(command(arguments))
                require(measured == expected and digest(nodes) == expected_hash,
                        f'replayed native {name} transaction')
    return {
        'schema': 'numi.human.ventricular-native-step-gate.v1',
        'status': 'passed_exact_reexecution' if execute else 'passed_retained_evidence',
        'native_revision': revision,
        'ventricular_tetrahedra': 1097534,
        'active_replay_bitwise': True,
        'nodes_different_from_zero': int(np.count_nonzero(lengths)),
        'maximum_active_minus_zero_displacement_m': float(lengths.max()),
        'accepted_native_steps_per_run': 1,
        'full_four_chamber_accepted_steps': 0,
        'heartbeat_qualified': False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('asset', 'activation', 'tension', 'evidence', 'native_repo',
                 'cooked_package', 'cook_binary', 'step_binary', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path,
                            required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    result = audit(args.asset, args.activation, args.tension, args.evidence,
                   args.native_repo, args.cooked_package, args.cook_binary,
                   args.step_binary, execute=args.execute)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
