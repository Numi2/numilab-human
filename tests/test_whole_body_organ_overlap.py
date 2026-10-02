"""Source hierarchy annotations remain separate from geometric conclusions."""

import json

import numpy as np
import pytest

from numilab_human.model import ImportError, sha256
from numilab_human.whole_body_organ_overlap import (
    SourceHierarchy,
    _bounds_disjoint,
)


def write_source_files(root):
    root.mkdir()
    content = {
        'partof_element_parts.txt': (
            'concept id\tname\telement file id\n'
            'FMA0001\torgan\tFJ0001\n'
            'FMA0001\torgan\tFJ0002\n'
            'FMA0002\tpancreatic tissue\tFJ0002\n'
        ),
        'partof_inclusion_relation_list.txt': (
            'parent id\tparent name\tchild id\tchild name\n'
            'FMA0001\torgan\tFMA0002\tpancreatic tissue\n'
        ),
        'isa_element_parts.txt': (
            'concept id\tname\telement file id\n'
            'FMA0010\tanatomical structure\tFJ0001\n'
            'FMA0011\tgland\tFJ0002\n'
        ),
        'isa_inclusion_relation_list.txt': (
            'parent id\tparent name\tchild id\tchild name\n'
            'FMA0010\tanatomical structure\tFMA0011\tgland\n'
        ),
    }
    files = {}
    for name, value in content.items():
        path = root / name
        path.write_text(value, encoding='utf-8')
        files[name] = {'sha256': sha256(path)}
    lock = root.parent / 'sources.lock.json'
    lock.write_text(json.dumps({'sources': {'bodyparts3d_4': {'files': files}}}),
                    encoding='utf-8')
    return lock


def test_hierarchy_annotations_return_shared_terms_and_directional_ancestry(tmp_path):
    sources = tmp_path / 'Sources'
    lock = write_source_files(sources)
    hierarchy = SourceHierarchy.load(sources, lock)

    result = hierarchy.relation_annotation('FJ0001', 'FJ0002')

    assert result['shared_direct_concepts']['part_of'] == [
        {'concept_id': 'FMA0001', 'name': 'organ'},
    ]
    assert result['direct_concept_ancestry'] == [
        {'hierarchy': 'part_of', 'ancestor_member': 'first',
         'ancestor_concept_id': 'FMA0001', 'ancestor_name': 'organ',
         'descendant_member': 'second', 'descendant_concept_id': 'FMA0002',
         'descendant_name': 'pancreatic tissue'},
        {'hierarchy': 'is_a', 'ancestor_member': 'first',
         'ancestor_concept_id': 'FMA0010', 'ancestor_name': 'anatomical structure',
         'descendant_member': 'second', 'descendant_concept_id': 'FMA0011',
         'descendant_name': 'gland'},
    ]


def test_hierarchy_hash_drift_fails_closed(tmp_path):
    sources = tmp_path / 'Sources'
    lock = write_source_files(sources)
    (sources / 'partof_element_parts.txt').write_text('changed', encoding='utf-8')

    with pytest.raises(ImportError, match='pinned hierarchy source hash'):
        SourceHierarchy.load(sources, lock)


def test_aabb_disjointness_requires_strict_axis_separation():
    first = np.array([[0., 0., 0.], [1., 1., 1.]])

    assert _bounds_disjoint(first, first + np.array([1.1, 0., 0.]))
    assert not _bounds_disjoint(first, first + np.array([1., 0., 0.]))
