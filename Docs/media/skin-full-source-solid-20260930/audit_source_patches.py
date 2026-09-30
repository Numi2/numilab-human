"""Classify the 98 minor FJ2810 index components by exact sheet contact."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from numilab_human import model as human


ROOT = human.REPOSITORY_ROOT
HERE = Path(__file__).resolve().parent


def main() -> None:
    archive, member, raw = human._bodyparts_obj_member(ROOT/'Sources', 'is_a', 'FJ2810')
    visual = human.read_json(ROOT/'Build/skin-seam-continuity-20260929/production/payload/bodyparts3d-myosim-skinned-shell.manifest.json')
    pinned = visual['source']['skin']
    assert member == pinned['member']
    assert human.sha256(archive) == pinned['archive_sha256']
    assert hashlib.sha256(raw).hexdigest() == pinned['member_sha256']
    vertices, faces = human._bodyparts_obj_triangles(raw, member)
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    a = np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2]))
    b = np.concatenate((faces[:, 1], faces[:, 2], faces[:, 0]))
    graph = coo_matrix((np.ones(len(a), dtype=np.uint8), (a, b)),
                       shape=(len(vertices), len(vertices))).tocsr()
    count, labels = connected_components(graph, directed=False)
    assert count == 100
    ranking = [component for component, _ in Counter(labels).most_common()]
    _, quotient = np.unique(vertices, axis=0, return_inverse=True)
    outer = set(quotient[labels == ranking[0]])
    inner = set(quotient[labels == ranking[1]])
    groups = {'both_sheets': 0, 'outer_only': 0, 'inner_only': 0,
              'neither_sheet': 0}
    for component in ranking[2:]:
        ids = set(quotient[labels == component])
        has_outer, has_inner = bool(ids & outer), bool(ids & inner)
        name = ('both_sheets' if has_outer and has_inner else
                'outer_only' if has_outer else
                'inner_only' if has_inner else 'neither_sheet')
        groups[name] += 1
    assert groups == {'both_sheets': 68, 'outer_only': 19,
                      'inner_only': 11, 'neither_sheet': 0}
    result = {
        'schema': 'numi.human.fj2810-minor-source-patch-interfaces.v1',
        'source_member_sha256': pinned['member_sha256'],
        'archive_sha256': pinned['archive_sha256'],
        'script_sha256': human.sha256(Path(__file__)),
        'source_index_component_count': count,
        'outer_index_vertex_count': int(np.count_nonzero(labels == ranking[0])),
        'inner_index_vertex_count': int(np.count_nonzero(labels == ranking[1])),
        'minor_index_vertex_count': int(len(vertices)
                                        - np.count_nonzero(labels == ranking[0])
                                        - np.count_nonzero(labels == ranking[1])),
        'minor_patch_exact_coordinate_interfaces': groups,
        'boundary': ('Index-component contact classification only. Shared exact '
                     'positions do not assign material or mechanical attachments.'),
    }
    (HERE/'source-component-interfaces.json').write_text(
        json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(groups, sort_keys=True))


if __name__ == '__main__':
    main()
