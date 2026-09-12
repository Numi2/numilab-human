#!/usr/bin/env python3
"""Bind the independent numerical reference to the exact admitted source mesh."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def triangle_multiset(mesh):
    triangles = []
    for face in mesh['triangles']:
        tri = tuple(tuple(mesh['vertices_m'][i]) for i in face)
        triangles.append(min(tri, tri[1:]+tri[:1], tri[2:]+tri[:2]))
    return sorted(triangles)


def main():
    reference = json.loads((HERE / 'manifold-first.json').read_text())
    artifact = json.loads((HERE.parent / 'partition.json').read_text())
    matched = {}
    for chamber in artifact['source_geometry']['chambers'][:2]:
        name = chamber['source_id']
        matched[name] = triangle_multiset(chamber['exact_coordinate_quotient']) == triangle_multiset(reference[name])
        if not matched[name]:
            raise AssertionError('numerical reference used different source triangles: '+name)
    encoded = artifact['exact_certificate']['intersection']['moments']['volume_m3']
    exact = Fraction(int(encoded[0], 16), int(encoded[1], 16))
    numerical = reference['intersection']['volume_m3']
    print(json.dumps({'schema': 'HumanPack.cardiac-partition-numerical-reference-comparison.v1',
        'source_oriented_triangle_multisets_match': matched, 'exact_volume_m3_display': float(exact),
        'manifold_volume_m3': numerical, 'manifold_minus_exact_volume_m3': float(Fraction.from_float(numerical)-exact),
        'reference_sha256': hashlib.sha256((HERE / 'manifold-first.json').read_bytes()).hexdigest(),
        'partition_sha256': hashlib.sha256((HERE.parent / 'partition.json').read_bytes()).hexdigest(),
        'qualification': 'independent_numerical_geometry_comparison_only', 'physical_stepping': False},
        sort_keys=True, separators=(',', ':')))


if __name__ == '__main__':
    main()
