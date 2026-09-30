"""Verify the full-source ventricular point-contact mechanics correction.

The corrected native cook separates three LV/RV point-only source contacts.
This is a bounded synthetic-support FEM step, not a heartbeat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / 'Build/cardiac-electrical-source-20260930/asset'
SOURCE = ROOT / 'Docs/media/cardiac-source-activation-20260930'
PRIOR = ROOT / 'Docs/media/cardiac-ventricular-native-step-20260930'
TENSION = ROOT / 'Docs/media/cardiac-active-tension-ingress-20260930/active-tension-100ms.f32le'
FRAMES = PRIOR / 'material-frame-rotations.f64le'
NATIVE = Path('/Users/home/numi-lab-cardiac-native-20260930')
BUILD = NATIVE / 'Build/cardiac-active-native'
PACKAGE = ROOT / 'Build/cardiac-point-contact-split-20260930/ventricular.nmpkg'
EVIDENCE = ROOT / 'Docs/media/cardiac-point-contact-split-20260930'
POINT_ONLY = np.array([17565, 170947, 235754], dtype='<u4')


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError('cardiac point-contact split: ' + message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def command(args: list[str]) -> str:
    result = subprocess.run(args, check=True, text=True, capture_output=True)
    require(not result.stderr and len(result.stdout.strip().splitlines()) == 1,
            'native command did not return one clean receipt')
    return result.stdout


def native_identity(native: Path, build: Path) -> dict:
    revision = command(['git', '-C', str(native), 'rev-parse', 'HEAD']).strip()
    status = subprocess.run(['git', '-C', str(native), 'status', '--porcelain=v1',
                             '--untracked-files=no'], check=True, text=True,
                            capture_output=True).stdout
    require(len(revision) == 40 and not status, 'native source revision is dirty')
    result = {'native_revision': revision}
    for name, binary in (
        ('cook_binary_sha256', 'numi-matter-ventricular-source-cook'),
        ('step_binary_sha256', 'numi-matter-ventricular-source-step'),
        ('topology_binary_sha256', 'numi-matter-ventricular-source-topology-check'),
    ):
        result[name] = sha(build / binary)
    result['metallib_sha256'] = sha(build / 'shaders/NumiMatter.metallib')
    result['native_cook_source_sha256'] = sha(native / 'matter/tools/ventricular_source_cook.cpp')
    result['native_step_source_sha256'] = sha(native / 'matter/tools/ventricular_source_step.mm')
    result['native_topology_source_sha256'] = sha(
        native / 'matter/tools/ventricular_source_topology_check.cpp')
    return result


def audit(asset: Path, source: Path, prior: Path, evidence: Path,
          native: Path, build: Path, package: Path,
          *, execute: bool = False) -> dict:
    asset, source, prior, evidence, native, build, package = (
        Path(p).resolve() for p in
        (asset, source, prior, evidence, native, build, package))
    prior_receipt = json.loads((prior / 'receipt.json').read_text())
    require(prior_receipt['ventricular_nodes'] == 218077
            and prior_receipt['ventricular_tetrahedra'] == 1097534
            and prior_receipt['heartbeat_qualified'] is False,
            'prior merged-node baseline')
    manifest = json.loads((asset / 'manifest.json').read_text())
    require(sha(asset / 'manifest.json') ==
            prior_receipt['source_asset_manifest_sha256'],
            'pinned source asset identity')
    arrays = {}
    for name, kind, shape in (
        ('nodes.f64le', '<f8', (-1, 3)),
        ('tetrahedra.u32le', '<u4', (-1, 4)),
        ('labels.u32le', '<u4', (-1,)),
    ):
        path = asset / name
        require(path.stat().st_size == manifest['buffers'][name]['bytes']
                and sha(path) == manifest['buffers'][name]['sha256'],
                f'pinned source buffer {name}')
        arrays[name] = np.fromfile(path, kind).reshape(shape)
    xyz, tets, labels = (arrays[name] for name in
                         ('nodes.f64le', 'tetrahedra.u32le', 'labels.u32le'))
    require(xyz.shape == (300965, 3) and tets.shape == (1470083, 4)
            and labels.shape == (1470083,), 'source dimensions')
    source_gate_path = source / 'independent-gate.json'
    source_gate = json.loads(source_gate_path.read_text())
    require(source_gate['asset_manifest_sha256'] == sha(asset / 'manifest.json')
            and source_gate['exact_face_couplings'] == 4494
            and source_gate['point_only_contacts_kept_separate'] ==
            POINT_ONLY.tolist()
            and source_gate['source_model_reproduction_qualified'] is False,
            'independent complete-face source topology')
    selected = (labels == 1) | (labels == 2)
    cells = tets[selected]
    regions = labels[selected]
    unique, first_index = np.unique(cells.ravel(), return_index=True)
    first_seen = unique[np.argsort(first_index)]
    mapping_path = evidence / 'cooked-source-nodes.u32le'
    mapping = np.fromfile(mapping_path, '<u4')
    require(len(cells) == 1097534 and len(mapping) == 218080
            and np.array_equal(mapping[:218077], first_seen)
            and np.array_equal(mapping[-3:], POINT_ONLY)
            and len(np.unique(mapping)) == 218077,
            'source-to-cooked node ordering and point-only duplicates')
    published = np.fromfile(source / 'ventricular-source-nodes.u32le', '<u4')
    published_regions = np.fromfile(source / 'ventricular-dof-regions.u32le', '<u4')
    require(np.array_equal(published, np.concatenate([unique, POINT_ONLY]))
            and np.array_equal(published_regions,
                np.concatenate([np.zeros(len(unique), '<u4'),
                                np.full(3, 2, '<u4')])),
            'published ventricular electrical/region quotient')

    native_info = native_identity(native, build)
    require(sha(FRAMES) == prior_receipt['files'][FRAMES.name]['sha256']
            and sha(TENSION) == prior_receipt['source_tension_100ms_sha256'],
            'source-derived frame and tension identity')
    source_tension = np.fromfile(TENSION, '<f4')
    cooked_tension = np.fromfile(evidence / 'cooked-tension.f32le', '<f4')
    require(len(source_tension) == len(labels)
            and len(cooked_tension) == len(cells)
            and np.array_equal(cooked_tension, source_tension[selected]),
            'every source-to-cooked active-tension cell identity')
    cook = json.loads((evidence / 'cook.json').read_text())
    topology = json.loads((evidence / 'package-topology.json').read_text())
    require(cook['status'] == 'full_source_ventricular_cook_pass'
            and cook['ventricular_nodes'] == 218080
            and cook['compiled_tetra_node_mapping_checked'] is True
            and cook['point_only_lv_rv_nodes_split'] == 3
            and cook['source_to_cooked_float32_bitwise'] is True
            and topology['status'] ==
            'full_source_package_point_contact_split_verified'
            and topology['ventricular_nodes'] == 218080
            and topology['source_to_cooked_tetrahedra_checked'] == 1097534
            and topology['point_only_lv_rv_nodes_split'] == 3
            and cook['synthetic_assembled_mass_kg'] ==
            topology['synthetic_assembled_mass_kg'],
            'native cook and separate package reader')
    package_sha = sha(package)
    active = json.loads((evidence / 'active.json').read_text())
    replay = json.loads((evidence / 'replay.json').read_text())
    zero = json.loads((evidence / 'zero.json').read_text())
    require(active == replay and active['device'].startswith('Apple M')
            and active['status_code'] == zero['status_code'] == 0
            and active['accepted_native_steps'] ==
            zero['accepted_native_steps'] == 1
            and active['point_only_lv_rv_nodes_split'] ==
            zero['point_only_lv_rv_nodes_split'] == 3
            and active['zero_tension_input'] is False
            and zero['zero_tension_input'] is True
            and active['heartbeat_qualified'] is False
            and zero['heartbeat_qualified'] is False,
            'native accepted/replay/zero step claims')
    states = {}
    hashes = {}
    for name in ('active', 'replay', 'zero'):
        path = evidence / f'{name}-nodes.bin'
        values = np.fromfile(path, '<f4').reshape(-1, 16)
        require(values.shape == (218080, 16)
                and bool(np.isfinite(values).all())
                and np.array_equal(values[:, 8:11], xyz[mapping].astype('<f4')),
                f'{name} accepted state and source reference geometry')
        states[name] = values
        hashes[name] = sha(path)
    require(hashes['active'] == hashes['replay']
            and np.array_equal(states['active'][:, 3], states['zero'][:, 3])
            and np.array_equal(states['active'][:3, :3], states['zero'][:3, :3]),
            'bitwise replay, mass and fixed-node controls')
    old_active = np.fromfile(prior / 'active-nodes.bin', '<f4').reshape(-1, 16)
    old_zero = np.fromfile(prior / 'zero-nodes.bin', '<f4').reshape(-1, 16)
    require(old_active.shape == old_zero.shape == (218077, 16)
            and sha(prior / 'active-nodes.bin') ==
            prior_receipt['files']['active-nodes.bin']['sha256']
            and sha(prior / 'zero-nodes.bin') ==
            prior_receipt['files']['zero-nodes.bin']['sha256']
            and np.array_equal(states['zero'][:218077, :3], old_zero[:, :3]),
            'matched prior native mechanical control')

    # Independent Float64 source-volume assembly checks that the three new
    # states own the RV portions of the original lumped mass, not extra mass.
    base = np.full(len(xyz), -1, np.int32)
    base[first_seen] = np.arange(218077, dtype=np.int32)
    mapped = base[cells]
    for offset, source_node in enumerate(POINT_ONLY):
        mapped[(regions == 2)[:, None] & (cells == source_node)] = 218077+offset
    reference_mass = np.zeros(218080, np.float64)
    for start in range(0, len(cells), 50000):
        end = start+50000
        p = xyz[cells[start:end]]
        a, b, c = p[:, 1]-p[:, 0], p[:, 2]-p[:, 0], p[:, 3]-p[:, 0]
        volume = np.einsum('ij,ij->i', a, np.cross(b, c))/6.0
        require(bool((volume > 0).all()), 'source ventricular orientation')
        local = mapped[start:end]
        reference_mass += np.bincount(local.ravel(),
            weights=np.broadcast_to((1050.0*volume/4)[:, None], local.shape).ravel(),
            minlength=218080)
    accepted_mass = states['active'][:, 3].astype(np.float64)
    max_mass_relative_error = float(np.max(
        np.abs(accepted_mass-reference_mass)/reference_mass))
    require(max_mass_relative_error < 2e-5 and
            abs(float(accepted_mass.sum())-
                float(old_active[:, 3].sum(dtype=np.float64))) < 1e-12,
            'source-lumped and prior total mass closure')
    point_checks = []
    for offset, source_node in enumerate(POINT_ONLY):
        left_index = int(base[source_node])
        right_index = 218077+offset
        old_mass = float(old_active[left_index, 3])
        split_mass = float(accepted_mass[left_index]+accepted_mass[right_index])
        separation = float(np.linalg.norm(
            states['active'][left_index, :3].astype(np.float64)-
            states['active'][right_index, :3].astype(np.float64)))
        require(abs(split_mass-old_mass) < 1e-12
                and accepted_mass[right_index] > 0
                and separation > 0,
                f'point-only LV/RV independent state {int(source_node)}')
        point_checks.append({'source_node': int(source_node),
                             'lv_cooked_node': left_index,
                             'rv_cooked_node': right_index,
                             'old_shared_mass_kg': old_mass,
                             'new_split_mass_sum_kg': split_mass,
                             'accepted_lv_rv_separation_m': separation})
    old_difference = (states['active'][:218077, :3].astype(np.float64)-
                      old_active[:, :3].astype(np.float64))
    baseline_changed = int(np.count_nonzero(np.any(old_difference != 0, axis=1)))
    baseline_max = float(np.linalg.norm(old_difference, axis=1).max())
    require(baseline_changed > 0 and baseline_max > 0,
            'corrected topology has no causal difference from merged baseline')

    result = {
        'schema': 'numi.human.cardiac-ventricular-point-contact-split.v1',
        'status': 'full_source_native_point_contacts_split_and_step_accepted',
        'source_asset_manifest_sha256': sha(asset / 'manifest.json'),
        'source_activation_gate_sha256': sha(source_gate_path),
        'published_ventricular_mapping_sha256': sha(
            source / 'ventricular-source-nodes.u32le'),
        'prior_merged_receipt_sha256': sha(prior / 'receipt.json'),
        'prior_active_state_sha256': sha(prior / 'active-nodes.bin'),
        'cooked_package_sha256': package_sha,
        'cooked_package_bytes': package.stat().st_size,
        'source_tension_sha256': sha(TENSION),
        'source_frames_sha256': sha(FRAMES),
        'auditor_source_sha256': sha(Path(__file__)),
        'native': native_info,
        'files_sha256': {name: sha(evidence / name) for name in (
            'cook.json', 'package-topology.json', 'active.json', 'replay.json',
            'zero.json', 'active-nodes.bin', 'replay-nodes.bin',
            'zero-nodes.bin', 'cooked-source-nodes.u32le',
            'cooked-tension.f32le')},
        'accepted_state_sha256': hashes,
        'source_cells': 1470083,
        'ventricular_tetrahedra': 1097534,
        'ventricular_nodes': 218080,
        'point_only_lv_rv_nodes_split': point_checks,
        'maximum_source_mass_relative_error': max_mass_relative_error,
        'baseline_shared_nodes_with_changed_position': baseline_changed,
        'maximum_split_minus_merged_displacement_m': baseline_max,
        'native_accepted_steps_per_run': 1,
        'four_chamber_accepted_steps': 0,
        'native_bitwise_replay': True,
        'heartbeat_qualified': False,
        'boundary': ('Exact source ventricular subset; synthetic 1050 kg/m3 inertia, '
                     'three fixed nodes, one 1 us active-tension FEM microstep. '
                     'No anatomical support, unloaded reference, chamber loading, '
                     'electrical-to-mechanical native coupling or heartbeat.'),
    }
    if execute:
        with tempfile.TemporaryDirectory(prefix='cardiac-point-split-') as directory:
            tmp = Path(directory)
            cooked = json.loads(command([str(build / 'numi-matter-ventricular-source-cook'),
                str(asset), str(FRAMES), str(TENSION),
                prior_receipt['files'][FRAMES.name]['sha256'],
                str(source / 'ventricular-source-nodes.u32le'),
                str(source / 'ventricular-dof-regions.u32le'),
                str(tmp / 'recooked.nmpkg'), str(tmp / 'tension.f32le'),
                str(tmp / 'mapping.u32le')]))
            require(cooked == cook and sha(tmp / 'recooked.nmpkg') == package_sha
                    and sha(tmp / 'tension.f32le') == sha(evidence / 'cooked-tension.f32le')
                    and sha(tmp / 'mapping.u32le') == sha(mapping_path),
                    'native full-source recook')
            checked = json.loads(command([
                str(build / 'numi-matter-ventricular-source-topology-check'),
                str(tmp / 'recooked.nmpkg'), str(asset), str(tmp / 'mapping.u32le')]))
            require(checked == topology, 'rechecked package topology')
            for name in ('active', 'replay', 'zero'):
                arguments = [str(build / 'numi-matter-ventricular-source-step'),
                             str(tmp / 'recooked.nmpkg'), str(tmp / 'tension.f32le'),
                             str(tmp / f'{name}.bin')]
                if name == 'zero':
                    arguments.append('--zero')
                measured = json.loads(command(arguments))
                require(measured == (zero if name == 'zero' else active)
                        and sha(tmp / f'{name}.bin') == hashes[name],
                        f'exact native {name} replay')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, default=ASSET)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--prior', type=Path, default=PRIOR)
    parser.add_argument('--evidence', type=Path, default=EVIDENCE)
    parser.add_argument('--native', type=Path, default=NATIVE)
    parser.add_argument('--build', type=Path, default=BUILD)
    parser.add_argument('--package', type=Path, default=PACKAGE)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    result = audit(args.asset, args.source, args.prior, args.evidence,
                   args.native, args.build, args.package, execute=args.execute)
    encoded = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.receipt.exists():
        require(args.receipt.read_text() == encoded, 'retained receipt drift')
    else:
        args.receipt.write_text(encoded)
    print(json.dumps({'status': result['status'],
                      'point_only_lv_rv_nodes_split': len(
                          result['point_only_lv_rv_nodes_split']),
                      'baseline_shared_nodes_with_changed_position':
                      result['baseline_shared_nodes_with_changed_position'],
                      'heartbeat_qualified': False}))


if __name__ == '__main__':
    main()
