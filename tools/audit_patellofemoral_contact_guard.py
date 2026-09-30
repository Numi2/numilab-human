"""Verify bilateral native rejection of an authored cartilage crossing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from tools.audit_patellofemoral_crossing_native import FOLDER, crossing, sha


ROOT = Path(__file__).resolve().parents[1]


def cells(path: Path) -> list[list[float]]:
    nodes = [[float(value) for value in line.split()]
             for line in path.read_text().splitlines()]
    assert len(nodes) == 8 and all(len(node) == 3 for node in nodes)
    return nodes


def source_face_indices(side: str) -> list[list[int]]:
    receipt_path = FOLDER / ('receipt.json' if side == 'left' else 'right-receipt.json')
    receipt = json.loads(receipt_path.read_text())
    fixture = FOLDER / f'{side}-crossing-0.txt'
    assert receipt['side'] == side and receipt['fixture_sha256'] == sha(fixture)
    loop_path = ROOT / 'Docs/media/patellofemoral-loop-20260930/receipt.json'
    assert receipt['intersection_loop_receipt_sha256'] == sha(loop_path)
    loop = json.loads(loop_path.read_text())
    segment = loop['compiled'][side]['segments'][receipt['crossing_segment_index']]
    assert receipt['crossing_face_indices'] == [segment['patellar_surface_face_index'],
                                                segment['femoral_surface_face_index']]
    return [[receipt['tetrahedra'][region]['global_node_indices'].index(index)
             + 4 * side_index for index in segment[key]]
            for side_index, (region, key) in enumerate((
                ('PTC', 'patellar_node_indices'),
                ('FMC', 'femoral_node_indices')))]


def audit(lab_root: Path) -> dict:
    lab_root = lab_root.resolve()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=lab_root,
                                     text=True).strip()
    assert commit.startswith('5a09376'), 'native source revision is not the guarded owner'
    binary = lab_root / 'Build/cardiac-active-native/numi-matter-patellofemoral-crossing-probe'
    metallib = lab_root / 'Build/cardiac-active-native/shaders/NumiMatter.metallib'
    assert sha(binary) == 'ab9d2d59109aaaab4bc46046ced9115208da0f3ab7db8cd166b77db78d68f989'
    assert sha(metallib) == 'ff3937eacff88872b8a8adf0990ab5369bf2565570190ee160c5c712fd0ce01d'

    names = ('native-guard-left', 'native-guard-left-replay',
             'native-guard-right', 'native-guard-close-contact',
             'native-guard-separated')
    outputs = {name: json.loads((FOLDER / f'{name}.json').read_text())
               for name in names}
    assert outputs['native-guard-left'] == outputs['native-guard-left-replay'], \
        'guarded left rejection does not replay exactly'

    source_rows = {}
    for side in ('left', 'right'):
        human_fixture = FOLDER / f'{side}-crossing-0.txt'
        lab_fixture = lab_root / f'matter/tests/fixtures/open_knee_{side}_crossing_cell.txt'
        assert sha(human_fixture) == sha(lab_fixture), 'native fixture drift'
        faces = source_face_indices(side)
        input_crossing = crossing(cells(human_fixture), faces)
        output = outputs[f'native-guard-{side}']
        retained_crossing = crossing(output['accepted_positions_m'], faces)
        assert input_crossing['crossing'] and retained_crossing['crossing'], \
            'source crossing disappeared without an accepted correction'
        assert output['status_code'] == 6 and output['completed_microsteps'] == 0
        assert output['rollback_bitwise'] and output['diagnostics'][2] == -1
        assert output['active_deformable_histories'] == 0
        source_rows[side] = {
            'fixture_sha256': sha(human_fixture),
            'input': input_crossing,
            'retained_after_rejection': retained_crossing,
            'status_code': output['status_code'],
            'completed_microsteps': output['completed_microsteps'],
            'rollback_bitwise': output['rollback_bitwise'],
        }

    near_fixture = lab_root / 'matter/tests/fixtures/fem_close_contact_cells.txt'
    near = outputs['native-guard-close-contact']
    assert near['status_code'] == 0 and near['completed_microsteps'] == 1
    assert near['active_deformable_histories'] > 0
    assert not crossing(cells(near_fixture), [[0, 1, 2], [4, 5, 6]])['crossing']
    assert not crossing(near['accepted_positions_m'], [[0, 1, 2], [4, 5, 6]])['crossing']
    separated = outputs['native-guard-separated']
    assert separated['status_code'] == 0 and separated['completed_microsteps'] == 1
    assert separated['active_deformable_histories'] == 0

    result = {
        'schema': 'numi.human.patellofemoral-contact-guard-audit.v1',
        'status': 'bilateral_source_cell_initial_crossings_rejected',
        'boundary': 'Two source tetrahedra per knee only. Native fail-closed admission and exact rollback do not resolve the authored cartilage intersection or qualify loaded knee contact.',
        'lab_coupled_commit': commit,
        'probe_binary_sha256': sha(binary),
        'metallib_sha256': sha(metallib),
        'near_contact_fixture_sha256': sha(near_fixture),
        'source_cells': source_rows,
        'close_noncrossing_contact_accepted': True,
        'close_noncrossing_active_deformable_histories': near['active_deformable_histories'],
        'left_rejection_replay_exact': True,
        'native_outputs_sha256': {name: sha(FOLDER / f'{name}.json') for name in names},
        'loaded_contact_qualified': False,
        'clinical_anatomy_qualified': False,
    }
    (FOLDER / 'native-guard-audit.json').write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lab-root', type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().lab_root), sort_keys=True))
