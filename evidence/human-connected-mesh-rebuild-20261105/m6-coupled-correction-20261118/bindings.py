"""Trace exact intersection witnesses back to original NHSKIN records."""
import collections
import json
import mmap
from pathlib import Path
import struct
import sys

import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / 'resting-surface-clearance-001'))
import audit

caps = root / 'regional-bed-layers-001/capture-4s/native/accepted-geometry'


def read_skin(pack, receipt):
    with pack.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as raw:
        header, directory = audit.clearance._PACK_HEADER, audit.clearance._PACK_DIRECTORY
        sec = {}
        for i in range(header.unpack_from(raw)[2]):
            record = directory.unpack_from(raw, header.size + i * directory.size)
            sec[record[0]] = record
        index, primitive = sec[3], sec[4]
        instance = [row[1] for row in audit.primitives(raw, {}, skin_only=True)][0]
        first, count = struct.unpack_from('<2I', raw, primitive[2] + instance * 64)
        _, _, pos, _ = next(audit.primitives(raw, {}, skin_only=True))
        records = np.ndarray(index[4], '<u4', raw, index[2])[first:first+count].copy().reshape(-1, 3)
        start = receipt['skin_source_mapping']['pack_first_vertex']
        assert (records >= start).all()
        ids = records - start
        points = np.zeros((receipt['skin_source_mapping']['source_vertex_count'], 3), dtype='f')
        points[np.unique(ids)] = pos
        return points, ids


def rank(values):
    return collections.Counter(values).most_common(18)


def main(end, report_path, output_path):
    begin = caps / 'step-0.mrvpack'
    receipt = json.loads(end.with_suffix('.receipt.json').read_text())
    initial = json.loads(begin.with_suffix('.receipt.json').read_text())
    p, faces = read_skin(end, receipt)
    p0, f0 = read_skin(begin, initial)
    assert np.array_equal(faces, f0)
    vi = receipt['skin_source_mapping']
    vertex_map = np.memmap(vi['vertex_map']['path'], dtype='<u4', mode='r').reshape(-1, 12)
    assert (vertex_map[:, 2] == 3).all()
    influences = np.memmap(vi['influences']['path'], dtype='<u4', mode='r').reshape(-1, 12)
    offsets = vertex_map[:, 0] - vi['influence_first_index']
    weight = influences[:, 3].view('<f4')
    body = influences[:, 8]
    assert all(vi[k]["sha256"] == initial["skin_source_mapping"][k]["sha256"]
               for k in ("vertex_map", "influences"))
    owner = np.array([body[o + np.argmax(weight[o:o+c])] for o, c in vertex_map[:, :2] - [vi['influence_first_index'], 0]])
    body_weights = np.array([weight[o:o+c].sum(dtype='d') for o, c in zip(offsets, vertex_map[:, 1])])
    assert np.max(abs(body_weights-1)) < 1.e-4
    report = json.loads(report_path.read_text())
    pairs = np.asarray(report['self_intersections']['triangle_pairs'])
    pair_ids = faces[pairs]
    owner_map = owner[faces].astype(int)
    median_owner = {}
    for group in owner_map[pairs]:
        key = tuple(sorted(tuple(sorted(set(map(int, t)))) for t in group))
        median_owner[key] = median_owner.get(key, 0) + 1
    distinct = np.unique(pairs)
    groups = collections.defaultdict(list)
    for group in owner_map[distinct]:
        groups[tuple(sorted(set(group)))].append(1)
    output = dict(
        total=len(pairs),
        dominant_owner_pairs=[(k,v) for k,v in sorted(median_owner.items(),key=lambda row:-row[1])[:30]],
        skin_face_owners=rank(owner_map[distinct].reshape(-1).tolist()),
        source_pose_max_displacement_m=float(np.linalg.norm(p-p0,axis=-1).max()),
        semantic_owners={
            semantic:rank(owner_map[np.concatenate([
                np.asarray(row['triangle_pairs'])[:,0] for row in report['surfaces']
                if row['semantic']==semantic and row['count']
            ])].reshape(-1).tolist()) for semantic in (51004,51005,51011)
        }
    )
    print(json.dumps(output, indent=2))
    output_path.write_text(json.dumps(output,indent=2)+'\n')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.capture, args.audit, args.output)
