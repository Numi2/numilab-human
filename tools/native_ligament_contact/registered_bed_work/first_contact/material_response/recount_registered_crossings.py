"""Reconstruct the full registered-surface crossing counts from exact native triangle witnesses."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def crossings(path):
    result = set()
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['ocular_source_id'] or row['semantic'] == 51010:
                continue
            prefix = (row['semantic'], row['stable_id'], row['primitive'])
            result.update((*prefix, skin, organ) for skin, organ in row['triangle_pairs'])
    return result


def main(parent, initial, replacement, output):
    reference = parent / 'native-input.npz'
    if digest(reference) != read(parent / 'harmonic/result.json')['source']:
        raise ValueError('Mismatched reference tissue mesh')
    comparison = read(replacement / 'native-bank-comparison.json')
    patch = replacement / 'registered-positions.npz'
    manifest = read(initial / 'transverse-banks.json')
    if comparison['source_pack'] != manifest['release'] or digest(patch) != comparison['patch']:
        raise ValueError('Different native subjects or repair coordinates')
    with np.load(reference) as mesh, np.load(patch) as update:
        moved = np.zeros(len(mesh['x']), dtype=bool)
        moved[update['nodes']] = np.any(mesh['x'][update['nodes']] != update['x'], axis=1)
        faces = np.flatnonzero(moved[mesh['node_O']][mesh['face_O']].any(axis=1))
    release = crossings(initial / 'target-witnesses.jsonl')
    new = {(s, b, p, int(faces[i]), j) for s,b,p,i,j in crossings(replacement / 'target-witnesses.jsonl')}
    changed_faces = set(faces)
    candidate = {pair for pair in release if pair[3] not in changed_faces} | new
    checks = (len(release), len(candidate), len(faces), len(release-candidate), len(candidate-release))
    if checks != tuple(comparison[k] for k in ('baseline_pairs','candidate_pairs','tested_faces','removed','new')):
        raise ValueError('Inconsistent complete crossing reconstruction')
    identities = {(r['semantic'],r['stable_id'],r['primitive']):r for r in manifest['banks']}
    before = Counter(pair[:3] for pair in release)
    after = Counter(pair[:3] for pair in candidate)
    banks = []
    for key, count in after.most_common():
        row = identities[key]
        banks.append(dict(semantic=key[0], stable_id=key[1], primitive=key[2],
                          label=row.get('label'), member=row.get('member_id'),
                          reference=before[key], candidate=count))
    banks.sort(key=lambda r: (-r['candidate'], r['primitive']))
    output.write_text(json.dumps(dict(
        native_pack=comparison['source_pack'], replacement=digest(patch),
        starting_witnesses=digest(initial / 'target-witnesses.jsonl'),
        replacement_witnesses=digest(replacement / 'target-witnesses.jsonl'),
        candidate=len(candidate), affected_banks=len(banks),
        by_semantic=dict(sorted(Counter(pair[0] for pair in candidate).items())),
        banks=banks), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('parent','initial','replacement','output'):
        parser.add_argument(arg, type=Path)
    main(**vars(parser.parse_args()))
