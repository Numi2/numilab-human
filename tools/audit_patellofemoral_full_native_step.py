"""Capture and independently verify the bounded full-source cartilage step.

This is an internal reproducibility receipt, not a loaded-knee qualification.
The four compressed accepted states are retained so the exact full-boundary
geometry check can be rerun without the original GPU or Matter checkout.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from numilab_human.abdominal_organ_separation import exact_integer_meshes
from numilab_human.open_knee import parse_source
from tools.audit_patellofemoral_pose_clearance import count_intersections, run as pose_run
from tools.export_patellofemoral_matter_input import run as export_run


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'Docs/media/patellofemoral-full-native-20260930'
WORK = ROOT / 'Build/full-patellofemoral-matter-20260930'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return digest(path.read_bytes())


def accepted_geometry(side: str, positions: bytes, source) -> dict:
    assert len(positions) == 50991 * 3 * 4
    nodes = np.frombuffer(positions, dtype='<f4').reshape((50991, 3))
    meshes = {}
    offset = 0
    for region in ('PTC', 'FMC'):
        source_region = source.regions[region]
        local = {identifier: index for index, identifier in
                 enumerate(source_region.node_ids)}
        faces = np.asarray([
            [local[identifier] for identifier in face]
            for face in source.surfaces[f'{region}_All_Faces'].faces
        ], dtype=np.int64)
        size = len(source_region.node_ids)
        meshes[region] = (nodes[offset:offset + size].astype(np.float64), faces)
        offset += size
    assert offset == 50991
    _, exact = exact_integer_meshes(meshes)
    return count_intersections(exact)


def native_case(binary: Path, side: str, label: str,
                *, pose_um: int = 20, baseline: bool = False,
                contact_off: bool = False) -> tuple[dict, bytes]:
    input_name = f'open-knee-{side}-ptc-fmc'
    if pose_um != 20:
        input_name += f'-{pose_um}um'
    output = WORK / f'{side}-receipt-{label}.f32le'
    command = [str(binary), str(WORK / f'{input_name}.nhcar'), str(output)]
    if baseline:
        command.append('--baseline')
    else:
        command += ['--contact-slop-m', '0.00002',
                    '--approach-speed-mps', '0.1']
    if contact_off:
        command.append('--disable-contact')
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    row = json.loads(result.stdout)
    assert row['side'] == (0 if side == 'left' else 1)
    assert row['source_nodes'] == 50991
    assert row['source_tetrahedra'] == 208177
    return row, output.read_bytes()


def capture(matter_root: Path, build_dir: Path) -> dict:
    binary = build_dir / 'numi-matter-patellofemoral-full-surface-step'
    metallib = build_dir / 'shaders/NumiMatter.metallib'
    material = matter_root / 'matter/materials/open_knee_cartilage_isotropic_preflight.nmatter'
    for path in (binary, metallib, material):
        assert path.is_file(), path
    pose14 = pose_run(14)
    assert pose14['sides']['left']['candidate']['segment_or_polygon_crossing_pairs'] == 0
    inputs20 = export_run()
    inputs14 = export_run(WORK / 'pose-clearance-14um.json')
    source = parse_source(ROOT / 'Sources/open-knee-oks003')
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    cases = {}
    geometry = {}
    for side in ('left', 'right'):
        cases[side] = {}
        baseline, _ = native_case(binary, side, 'baseline', baseline=True)
        near, _ = native_case(binary, side, '14um', pose_um=14)
        assert baseline['status_code'] == 6 and baseline['completed_microsteps'] == 0
        assert baseline['rollback_bitwise'] and baseline['diagnostics'][2] == -1
        assert near['status_code'] == 6 and near['completed_microsteps'] == 0
        assert near['rollback_bitwise'] and near['diagnostics'][2] == -4
        cases[side]['baseline'] = baseline
        cases[side]['near_plane_14um'] = near
        for mode in ('on', 'off'):
            row, positions = native_case(binary, side, mode,
                                         contact_off=(mode == 'off'))
            assert row['status_code'] == 0 and row['completed_microsteps'] == 1
            assert not row['rollback_bitwise']
            assert row['active_deformable_histories'] == (11 if mode == 'on' else 0)
            cases[side][mode] = row
            geometry[f'{side}_{mode}'] = accepted_geometry(side, positions, source)
            assert geometry[f'{side}_{mode}']['segment_or_polygon_crossing_pairs'] == 0
            assert geometry[f'{side}_{mode}']['point_contact_pairs'] == 0
            (EVIDENCE / f'{side}-{mode}.f32le.gz').write_bytes(
                gzip.compress(positions, compresslevel=9, mtime=0))
            cases[side][f'{mode}_positions_sha256'] = digest(positions)
        on = gzip.decompress((EVIDENCE / f'{side}-on.f32le.gz').read_bytes())
        off = gzip.decompress((EVIDENCE / f'{side}-off.f32le.gz').read_bytes())
        a = np.frombuffer(on, dtype='<f4').reshape((-1, 3))
        b = np.frombuffer(off, dtype='<f4').reshape((-1, 3))
        delta = np.linalg.norm(a.astype(np.float64) - b.astype(np.float64), axis=1)
        cases[side]['contact_ab'] = {
            'changed_ptc_nodes': int(np.count_nonzero(delta[:26121])),
            'changed_fmc_nodes': int(np.count_nonzero(delta[26121:])),
            'maximum_position_delta_m': float(delta.max()),
        }
    replay, replay_positions = native_case(binary, 'left', 'replay')
    assert replay == cases['left']['on']
    assert digest(replay_positions) == cases['left']['on_positions_sha256']
    receipt = {
        'schema': 'numi.human.patellofemoral-full-native-step.v1',
        'status': 'bounded_source_cartilage_contact_preflight',
        'pose_status': 'unadopted_candidate',
        'native_source_revision': subprocess.run(
            ['git', '-C', str(matter_root), 'rev-parse', 'HEAD'],
            check=True, capture_output=True, text=True).stdout.strip(),
        'source_inputs': {'20um': inputs20, '14um': inputs14},
        'native_identity_sha256': {
            'executable': sha(binary), 'metallib': sha(metallib),
            'material': sha(material),
            'contact_shader_source': sha(matter_root / 'matter/src/metal/contact.metalinc'),
            'runtime_source': sha(matter_root / 'matter/src/runtime.mm'),
            'probe_source': sha(matter_root / 'matter/tools/patellofemoral_full_surface_step.mm'),
        },
        'cases': cases,
        'accepted_full_boundary_exact_face_intersections': geometry,
        'left_contact_on_replay_byte_identical': True,
        'source_fmc_vertex_link_manifold_defect_node_id': 233523,
        'limits': [
            'FMC source boundary has a two-fan vertex link; zero face crossings does not qualify disjoint volumes.',
            'PTC current pose is an unadopted 20 micrometer candidate; patellar bone, PTB, QAT and PTL attachments are absent.',
            'The material uses a synthetic density and an unvalidated isotropic energy; no source FEBio material equivalence was established.',
            'One 1 microsecond zero-gravity step with 0.1 m/s prescribed approach is not sustained or physiological load.',
            'Active history count is not pressure or force; accepted barrier impulse field was zero.',
            'Only kinetic energy is reported; strain, contact work and whole-system energy closure are not established.',
            'The conservative FP32 final-state guard rejects unresolved near-plane vertices; it does not establish general clinical nonpenetration.',
        ],
        'loaded_knee_qualified': False,
        'clinical_anatomy_qualified': False,
    }
    path = EVIDENCE / 'receipt.json'
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt


def verify() -> dict:
    receipt = json.loads((EVIDENCE / 'receipt.json').read_text())
    assert receipt['schema'] == 'numi.human.patellofemoral-full-native-step.v1'
    source = parse_source(ROOT / 'Sources/open-knee-oks003')
    pose_path = ROOT / 'Docs/media/patellofemoral-pose-clearance-20260930/receipt.json'
    pose = json.loads(pose_path.read_text())
    assert pose['prescribed_translation_magnitude_m'] == 20.0e-6
    for file, expected in pose['source_files_sha256'].items():
        assert sha(ROOT / 'Sources/open-knee-oks003' / file) == expected
    for side in ('left', 'right'):
        row = receipt['source_inputs']['20um'][side]
        assert row['pose_receipt_sha256'] == sha(pose_path)
        assert row['nhknee_payload_sha256'] == pose['sides'][side]['payload_sha256']
    for side in ('left', 'right'):
        for mode in ('on', 'off'):
            positions = gzip.decompress((EVIDENCE / f'{side}-{mode}.f32le.gz').read_bytes())
            assert digest(positions) == receipt['cases'][side][f'{mode}_positions_sha256']
            assert accepted_geometry(side, positions, source) == \
                receipt['accepted_full_boundary_exact_face_intersections'][f'{side}_{mode}']
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--matter-root', type=Path)
    parser.add_argument('--build-dir', type=Path)
    args = parser.parse_args()
    if args.capture:
        assert args.matter_root and args.build_dir
        output = capture(args.matter_root.resolve(), args.build_dir.resolve())
    else:
        output = verify()
    print(json.dumps({'schema': output['schema'], 'status': output['status'],
                      'loaded_knee_qualified': output['loaded_knee_qualified']},
                     sort_keys=True))
