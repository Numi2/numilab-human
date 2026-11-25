"""Transfer a compiled cortical binding to a new material pose with identical contact geometry."""
from pathlib import Path
import hashlib
import re

import numpy as np


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_nodes(path):
    with path.open() as stream:
        count, dimension, attributes, markers = map(int, next(stream).split())
        nodes = np.loadtxt(stream)
    if (dimension, attributes, markers) != (3, 0, 0) or nodes.shape != (count, 4):
        raise ValueError('Unexpected material node table')
    if not np.array_equal(nodes[:, 0], np.arange(1, count + 1)) or not np.isfinite(nodes).all():
        raise ValueError('Invalid material node IDs or coordinates')
    return nodes[:, 1:]


def rebind(source, reference, candidate, reference_pose, candidate_pose, destination):
    text = source.read_text()
    pin = 'inline constexpr char loaded[]="' + sha256(reference) + '";'
    if text.count(pin) != 1:
        raise ValueError('Compiled contact binding and starting material disagree')
    faces = text.split('inline constexpr Facet faces[]={')[1].split('};')[0].splitlines()
    samples = text.split('inline constexpr Sample samples[]={')[1].split('};')[0].splitlines()
    nodes = set()
    owners = set()
    face_count = sample_count = 0
    for line in filter(str.strip, faces):
        row = re.fullmatch(r'\{\d+,\d+,\d+,\d+,(\d+),\{(\d+),(\d+),(\d+)\},\{\d+,\d+,\d+\}\},', line.strip())
        if row is None:
            raise ValueError('Unrecognized cortical facet')
        owners.add(int(row[1]))
        nodes.update(map(int, row.groups()[1:]))
        face_count += 1
    for line in filter(str.strip, samples):
        row = re.fullmatch(r'\{(\d+),\d+,\d+,(\d+),[^,{}]+,[^,{}]+\},', line.strip())
        if row is None:
            raise ValueError('Unrecognized cortical sample')
        nodes.add(int(row[1]))
        owners.add(int(row[2]))
        sample_count += 1
    old, new = read_nodes(reference), read_nodes(candidate)
    ids, bones = sorted(nodes), sorted(owners)
    if not face_count or not sample_count or old.shape != new.shape or not np.array_equal(old[ids], new[ids]):
        raise ValueError('Cortical facet or sample changed in the new pose')
    if not np.array_equal(reference_pose[bones], candidate_pose[bones]):
        raise ValueError('Cortical body frame changed in the new pose')
    next_pin = 'inline constexpr char loaded[]="' + sha256(candidate) + '";'
    destination.write_text(text.replace(pin, next_pin))
    return dict(faces=face_count, samples=sample_count, contact_nodes=len(ids), cortical_bodies=bones,
                reference=sha256(reference), candidate=sha256(candidate), binding=sha256(destination))
