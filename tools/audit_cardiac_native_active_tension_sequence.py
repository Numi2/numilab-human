"""Audit retained evidence for the short native ventricular tension sequence.

This validates hashes, source-field lineage, native statuses, replay identity,
and recorded claim boundaries. It does not qualify a heartbeat or physiology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'Docs/media/cardiac-native-tension-sequence-20261002'


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError('cardiac native tension sequence audit: ' + message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def native_identity(repo: Path, build: Path, expected: dict) -> dict:
    repo = repo.resolve()
    build = build.resolve()
    revision = subprocess.run(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'], check=True,
        text=True, capture_output=True).stdout.strip()
    dirty = subprocess.run(
        ['git', '-C', str(repo), 'status', '--porcelain=v1',
         '--untracked-files=no'], check=True, text=True,
        capture_output=True).stdout
    require(revision == expected['matter_revision'] and not dirty,
            'Matter source revision or tracked worktree differs')
    binaries = {
        'cook': 'numi-matter-ventricular-source-cook',
        'step': 'numi-matter-ventricular-source-step',
        'metallib': 'shaders/NumiMatter.metallib',
    }
    for key, name in binaries.items():
        require(sha(build / name) == expected['binary_sha256'][key],
                f'Matter {key} binary identity')
    sources = {
        'ventricular_source_cook_cpp': 'matter/tools/ventricular_source_cook.cpp',
        'ventricular_source_step_mm': 'matter/tools/ventricular_source_step.mm',
        'runtime_mm': 'matter/src/runtime.mm',
    }
    for key, name in sources.items():
        require(sha(repo / name) == expected['source_sha256'][key],
                f'Matter source identity: {name}')
    return {'matter_revision': revision, 'matter_binary_and_source_hashes_match': True}


def audit(evidence: Path, *, matter_repo: Path | None = None,
          build_dir: Path | None = None, package: Path | None = None,
          asset: Path | None = None) -> dict:
    evidence = evidence.resolve()
    receipt = json.loads((evidence / 'receipt.json').read_text())
    require(receipt['schema'] ==
            'numi.human.cardiac-native-active-tension-sequence.v1'
            and receipt['status'] ==
            'eight_consecutive_native_ventricular_steps_accepted_bitwise_replay_and_zero_control',
            'receipt schema or status')

    for name, expected in receipt['retained_files_sha256'].items():
        path = evidence / name
        require(path.is_file() and sha(path) == expected,
                f'retained file hash: {name}')

    sequence = json.loads((evidence / 'sequence-manifest.json').read_text())
    frames = sequence['frames']
    require(sequence['frame_count'] == 8 and len(frames) == 8
            and sequence['cells_per_frame'] == 1097534,
            'source sequence dimensions')
    gate = json.loads((evidence / 'independent-gate.json').read_text())
    require(gate['status'] == 'passed_full_source_float32_tension_to_residual'
            and gate['accepted_native_anatomical_steps'] == 0
            and len(gate['frames']) == len(frames),
            'source residual gate claim boundary')
    gate_metrics = receipt['source']['source_tension_gate']
    require(sha(ROOT / 'src/numilab_human/cardiac_active_tension_gate.py')
            == receipt['native']['source_sha256']['cardiac_active_tension_gate_py']
            and gate_metrics['source_sha256'] == gate['gate_source_sha256']
            and gate_metrics['maximum_component_difference_n'] == max(
                frame['maximum_component_difference_n'] for frame in gate['frames'])
            and gate_metrics['maximum_relative_l2_difference'] == max(
                frame['relative_l2_difference'] for frame in gate['frames'])
            and gate_metrics['maximum_net_internal_residual_n'] == max(
                frame['net_internal_residual_norm_n'] for frame in gate['frames']),
            'source residual implementation or maxima')
    for index, frame in enumerate(frames):
        source_file = evidence / f"source-active-tension-{frame['time_label']}.f32le"
        require(source_file.stat().st_size == 1470083 * 4
                and sha(source_file) == frame['source_sha256']
                and frame['source_sha256'] == gate['frames'][index]['tension_sha256'],
                f"source tension frame {frame['time_label']}")
    cooked = evidence / 'cooked-tension-sequence.f32le'
    require(cooked.stat().st_size == 8 * 1097534 * 4
            and sha(cooked) == sequence['sequence_sha256']
            == receipt['source']['cooked_tension_sequence_sha256'],
            'cooked tension sequence identity')

    native = receipt['native']
    active_json = json.loads((evidence / 'active.json').read_text())
    replay_json = json.loads((evidence / 'replay.json').read_text())
    zero_json = json.loads((evidence / 'zero.json').read_text())
    for label, run in (('active', active_json), ('replay', replay_json),
                       ('zero', zero_json)):
        require(run['status_code'] == 0 and run['sequence_count'] == 8
                and run['accepted_native_steps'] == 8
                and run['step_status_codes'] == [0] * 8
                and run['step_completed_microsteps'] == [1] * 8
                and run['step_failing_indices'] == [4294967295] * 8,
                f'{label} accepted-step receipt')
        require(run['zero_tension_input'] is (label == 'zero'),
                f'{label} active/zero identity')
    active_path = evidence / 'active-accepted-nodes.bin'
    replay_path = evidence / 'replay-accepted-nodes.bin'
    zero_path = evidence / 'zero-accepted-nodes.bin'
    require(active_path.read_bytes() == replay_path.read_bytes()
            and sha(active_path) == sha(replay_path)
            == native['output_sha256']['active']
            and sha(zero_path) == native['output_sha256']['zero'],
            'complete native state replay hashes')
    active = np.fromfile(active_path, dtype='<f4').reshape(218080, 16)
    zero = np.fromfile(zero_path, dtype='<f4').reshape(218080, 16)
    require(bool(np.isfinite(active).all()) and bool(np.isfinite(zero).all())
            and np.array_equal(active[:, 3].view('<u4'),
                               zero[:, 3].view('<u4')),
            'finite node states and matched mass field')

    state_audit = native['state_audit']
    require(state_audit['fixed_nodes_unchanged_from_source'] is True
            and state_audit['active_zero_mass_fields_bitwise_equal'] is True
            and state_audit['active_state_sha256'] == state_audit['replay_state_sha256']
            and state_audit['split_point_contact_pairs'],
            'state audit boundaries')
    mapping = np.fromfile(evidence / 'cooked-source-nodes.u32le', dtype='<u4')
    require(mapping.shape == (218080,), 'cooked source-node mapping size')
    pair_distances = []
    for pair in state_audit['split_point_contact_pairs']:
        lv, rv, source = (pair['lv_cooked_node'], pair['rv_cooked_node'],
                          pair['source_node'])
        require(mapping[lv] == source and mapping[rv] == source,
                'split contact source-node identity')
        active_distance = float(np.linalg.norm(active[lv, :3] - active[rv, :3]))
        zero_distance = float(np.linalg.norm(zero[lv, :3] - zero[rv, :3]))
        require(abs(active_distance - pair['active_separation_m']) <= 1e-12
                and abs(zero_distance - pair['zero_separation_m']) <= 1e-12,
                'split contact measured separation')
        pair_distances.append({'source_node': source,
                               'active_separation_m': active_distance,
                               'zero_separation_m': zero_distance})

    cook_checks = json.loads((evidence / 'frame-cook-checks.json').read_text())
    require(cook_checks['frames_independently_cpp_cook_checked'] == 7
            and all(item['source_to_cooked_float32_bitwise'] is True
                    for item in cook_checks['results']),
            'all source-to-cooked field checks')

    result = {
        'schema': 'numi.human.cardiac-native-active-tension-sequence-audit.v1',
        'status': 'passed',
        'evidence_receipt_sha256': sha(evidence / 'receipt.json'),
        'source_frames_checked': len(frames),
        'source_cpp_projection_checks': 1 + cook_checks[
            'frames_independently_cpp_cook_checked'],
        'accepted_native_steps_active_replay_zero': [8, 8, 8],
        'active_replay_bitwise_identical': True,
        'active_state_sha256': sha(active_path),
        'zero_state_sha256': sha(zero_path),
        'split_contact_distances': pair_distances,
        'heartbeat_qualified': False,
    }
    if package is not None:
        package = package.resolve()
        require(sha(package) == receipt['source']['cooked_package_sha256'],
                'cooked package hash')
        result['cooked_package_sha256'] = sha(package)
    if asset is not None:
        asset = asset.resolve()
        manifest = asset / 'manifest.json'
        require(sha(manifest) == receipt['source']['source_asset_manifest_sha256'],
                'source asset manifest hash')
        result['source_asset_manifest_sha256'] = sha(manifest)
        source_positions = np.fromfile(asset / 'nodes.f64le', dtype='<f8').reshape(-1, 3)
        initial = np.asarray(source_positions[mapping], dtype='<f4').astype(np.float64)
        moved_counts = {}
        maximum_displacements = {}
        for label, state, native_run in (
                ('active', active, active_json), ('zero', zero, zero_json)):
            delta = state[:, :3].astype(np.float64) - initial
            distances = np.sqrt(np.einsum('ij,ij->i', delta, delta))
            moved_counts[label] = int(np.count_nonzero(distances > 0.0))
            maximum_displacements[label] = float(distances.max(initial=0.0))
            require(np.array_equal(state[:3, :3], initial[:3].astype('<f4'))
                    and moved_counts[label] == native_run['moved_nodes']
                    and abs(maximum_displacements[label] -
                            native_run['maximum_displacement_m']) <= 1e-12,
                    f'{label} source-rest displacement and fixed-node metrics')
        result['movement_recomputed_from_source_positions'] = {
            'moved_nodes': moved_counts,
            'maximum_displacement_m': maximum_displacements,
        }
    if matter_repo is not None:
        native_result = native_identity(
            matter_repo,
            build_dir or matter_repo / 'Build/cardiac-active-native',
            native)
        result.update(native_result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, default=EVIDENCE / 'receipt.json')
    parser.add_argument('--matter-repo', type=Path,
                        help='optionally verify the exact committed Matter source and binaries')
    parser.add_argument('--build-dir', type=Path,
                        help='Matter native build directory; defaults under --matter-repo')
    parser.add_argument('--package', type=Path,
                        help='optionally verify the recooked native FEM package')
    parser.add_argument('--asset', type=Path,
                        help='optionally verify the local case18 source manifest')
    args = parser.parse_args()
    result = audit(args.receipt.resolve().parent,
                   matter_repo=args.matter_repo,
                   build_dir=args.build_dir,
                   package=args.package,
                   asset=args.asset)
    print(json.dumps(result, sort_keys=True, separators=(',', ':')))


if __name__ == '__main__':
    main()
