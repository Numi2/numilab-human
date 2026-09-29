"""Independent FP64 transverse witnesses for each failed eight-organ pair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from numilab_human import lung_envelope as lung


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PAYLOAD = ROOT/'Docs/media/right-eye-laterality-20260929/candidate/right-eye-laterality-candidates.nhanatomy'
RECEIPT = HERE/'receipt-v2.json'


def strict_edge_plane_hit(first: np.ndarray, second: np.ndarray):
    normal = np.cross(second[1]-second[0], second[2]-second[0])
    for edge in range(3):
        start, end = first[edge], first[(edge+1) % 3]
        d0 = float(normal @ (start-second[0]))
        d1 = float(normal @ (end-second[0]))
        if d0*d1 >= 0:
            continue
        point = start + d0/(d0-d1)*(end-start)
        basis = np.stack((second[1]-second[0], second[2]-second[0]), axis=1)
        bary = np.linalg.lstsq(basis, point-second[0], rcond=None)[0]
        margin = min(float(bary[0]), float(bary[1]), float(1-bary.sum()))
        if margin > 1e-9:
            return {'point_owner_local_m': point.tolist(), 'source_edge': edge,
                    'signed_plane_endpoints': [d0, d1],
                    'minimum_target_triangle_barycentric': margin}
    return None


def main() -> None:
    primary = json.loads(RECEIPT.read_text())
    assert hashlib.sha256(PAYLOAD.read_bytes()).hexdigest() == primary['selected_payload_sha256']
    _, header, records, vertices, indices = lung.decode(PAYLOAD)
    assert header[:2] == (5, 602)
    ids = {row['label']: row['source_stable_id'] for row in primary['organs']}

    def face(label: str, index: int):
        record = records[ids[label]-1]
        fi = int(record[3]) + 3*index
        return vertices[indices[fi:fi+3], :3].astype(np.float64)

    witnesses = []
    for pair in primary['pairs']:
        if pair['status'] != 'surface_crossing':
            continue
        found = None
        for first_id, second_id in pair['first_intersecting_triangle_pairs']:
            a, b = face(pair['first'], first_id), face(pair['second'], second_id)
            hit = strict_edge_plane_hit(a, b)
            direction = 'first_edge_through_second_face'
            if hit is None:
                hit = strict_edge_plane_hit(b, a)
                direction = 'second_edge_through_first_face'
            if hit is not None:
                found = {'first': pair['first'], 'second': pair['second'],
                         'first_triangle_id': first_id,
                         'second_triangle_id': second_id,
                         'direction': direction, **hit}
                break
        assert found is not None, (pair['first'], pair['second'])
        witnesses.append(found)
    assert len(witnesses) == 7
    (HERE/'transverse-witness-v2.json').write_text(json.dumps({
        'schema': 'numi.human.abdominal-organ-transverse-witness.v2',
        'primary_receipt_sha256': hashlib.sha256(RECEIPT.read_bytes()).hexdigest(),
        'selected_payload_sha256': primary['selected_payload_sha256'],
        'witnesses': witnesses,
        'boundary': ('Independent double-precision strict edge/plane and barycentric '
                     'calculations confirm one transverse crossing per failed organ pair. '
                     'Complete counts come from the exact-rational primary audit.'),
    }, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'strict_transverse_witness_count': len(witnesses)}))


if __name__ == '__main__':
    main()
