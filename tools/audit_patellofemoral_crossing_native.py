"""Audit a native Matter two-cell crossing step against exact face geometry."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from numilab_human.cardiac_cavity_intersections import triangle_intersection_points


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'Docs/media/patellofemoral-crossing-fixture-20260930'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crossing(nodes: list[list[float]], faces: list[list[int]]) -> dict:
    vertices = [[tuple(Fraction.from_float(float(value)) for value in nodes[index])
                 for index in face] for face in faces]
    denominator = math.lcm(*(value.denominator for triangle in vertices
                             for point in triangle for value in point))
    integers = [tuple(tuple(value.numerator * (denominator // value.denominator)
                            for value in point) for point in triangle)
                for triangle in vertices]
    points = sorted(set(triangle_intersection_points(*integers)))
    length = (math.dist(*(tuple(float(value) / denominator for value in point)
                          for point in points)) if len(points) == 2 else 0.0)
    return {'intersection_point_count': len(points),
            'segment_length_m': length,
            'crossing': len(points) == 2 and length > 0.0}


def run() -> dict:
    fixture_path = FOLDER / 'left-crossing-0.txt'
    receipt_path = FOLDER / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['fixture_sha256'] == sha(fixture_path)
    loop = json.loads((ROOT / 'Docs/media/patellofemoral-loop-20260930/receipt.json').read_text())
    assert receipt['intersection_loop_receipt_sha256'] == sha(
        ROOT / 'Docs/media/patellofemoral-loop-20260930/receipt.json')
    segment = loop['compiled']['left']['segments'][receipt['crossing_segment_index']]
    assert receipt['crossing_face_indices'] == [segment['patellar_surface_face_index'],
                                                segment['femoral_surface_face_index']]
    faces = [[receipt['tetrahedra'][region]['global_node_indices'].index(index)
              + 4 * side for index in segment[key]]
             for side, (region, key) in enumerate((
                 ('PTC', 'patellar_node_indices'),
                 ('FMC', 'femoral_node_indices')))]
    initial = [[float(value) for value in line.split()]
               for line in fixture_path.read_text().splitlines()]
    assert len(initial) == 8 and all(len(node) == 3 for node in initial)
    outputs = {name: json.loads((FOLDER / f'{name}.json').read_text())
               for name in ('native-overlap-step', 'native-overlap-replay',
                            'native-separated-control')}
    assert outputs['native-overlap-step'] == outputs['native-overlap-replay'], \
        'native result does not replay exactly'
    assert all(row['status_code'] == 0 and row['completed_microsteps'] == 1
               and len(row['accepted_positions_m']) == 8 for row in outputs.values())
    first = crossing(initial, faces)
    after = crossing(outputs['native-overlap-step']['accepted_positions_m'], faces)
    separated = crossing(outputs['native-separated-control']['accepted_positions_m'], faces)
    assert first['crossing'] and after['crossing'] and not separated['crossing'], \
        'native crossing/control classification changed'
    assert outputs['native-overlap-step']['active_deformable_histories'] > 0
    assert outputs['native-separated-control']['active_deformable_histories'] == 0
    result = {
        'schema': 'numi.human.patellofemoral-native-crossing-audit.v1',
        'status': 'failed_initial_crossing_clearance',
        'fixture_receipt_sha256': sha(receipt_path),
        'input_face_node_indices': faces,
        'native_outputs_sha256': {name: sha(FOLDER / f'{name}.json')
                                  for name in outputs},
        'initial': first,
        'accepted_after_one_microsecond': after,
        'separated_control': separated,
        'accepted_overlap_step_status': outputs['native-overlap-step']['status_code'],
        'accepted_overlap_active_deformable_histories':
            outputs['native-overlap-step']['active_deformable_histories'],
        'exact_native_replay': True,
        'gap': 'A native accepted contact step leaves the authored patella-femur surface crossing in place. Contact activation and solver acceptance do not establish noninterpenetration.',
        'loaded_contact_qualified': False,
        'clinical_anatomy_qualified': False,
    }
    (FOLDER / 'native-step-audit.json').write_text(json.dumps(result, indent=2,
                                                              sort_keys=True) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(run(), sort_keys=True))
