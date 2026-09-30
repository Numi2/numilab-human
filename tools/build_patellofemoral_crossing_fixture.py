"""Extract two exact source tetrahedra around the first compiled knee crossing.

This is a minimal native-kernel diagnostic, not a loaded-knee model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from numilab_human.open_knee import parse_source
from tools.verify_patellofemoral_surface import payload


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'Build/patellofemoral-surface-20260930/left'
OUTPUT = ROOT / 'Docs/media/patellofemoral-crossing-fixture-20260930'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    source = parse_source(ROOT / 'Sources/open-knee-oks003')
    manifest_path = BUILD / 'open-knee-oks003-left.manifest.json'
    payload_path = BUILD / 'open-knee-oks003-left.nhknee'
    decoded = payload(payload_path, json.loads(manifest_path.read_text()), source)
    loop_path = ROOT / 'Docs/media/patellofemoral-loop-20260930/receipt.json'
    loop = json.loads(loop_path.read_text())
    assert sha(payload_path) == loop['compiled']['left']['payload_sha256']
    segment = loop['compiled']['left']['segments'][0]
    selected = []
    owners = {}
    for region, face_key in (('PTC', 'patellar_node_indices'),
                             ('FMC', 'femoral_node_indices')):
        face = set(segment[face_key])
        row = decoded['regions'][region]
        tets = decoded['tetrahedra'][row['first_tet']:row['first_tet'] + row['tet_count']]
        matching = [(row['first_tet'] + index, [int(node) for node in tet])
                    for index, tet in enumerate(tets)
                    if face.issubset(set(int(node) for node in tet))]
        assert len(matching) == 1, f'{region} crossing face must have one exterior owner'
        tet_index, nodes = matching[0]
        owners[region] = {'global_tet_index': tet_index, 'global_node_indices': nodes}
        selected.extend(decoded['positions'][nodes].tolist())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fixture_path = OUTPUT / 'left-crossing-0.txt'
    fixture_path.write_text(''.join(' '.join(format(value, '.17g') for value in node) + '\n'
                                    for node in selected))
    result = {
        'schema': 'numi.human.patellofemoral-crossing-cell-fixture.v1',
        'boundary': 'Two exact source tetrahedra selected at one of 18 crossing triangle pairs. Diagnostic kernel input only; synthetic material and no physiological load.',
        'payload_sha256': sha(payload_path),
        'manifest_sha256': sha(manifest_path),
        'intersection_loop_receipt_sha256': sha(loop_path),
        'source_files_sha256': loop['source_file_sha256'],
        'crossing_segment_index': 0,
        'crossing_face_indices': [segment['patellar_surface_face_index'],
                                  segment['femoral_surface_face_index']],
        'tetrahedra': owners,
        'fixture_sha256': sha(fixture_path),
        'loaded_contact_qualified': False,
        'clinical_anatomy_qualified': False,
    }
    (OUTPUT / 'receipt.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(build(), sort_keys=True))
