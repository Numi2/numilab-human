"""Join the actual limiting Newton cells to mass, material labels and saved physical geometry."""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from compile_connected_joint import digest
from supported_material_admission import geom


def load(path):
    return json.loads(path.read_bytes())


def indexed_rows(path, indices, skip_header=True):
    found = {}
    with path.open() as source:
        if skip_header:
            next(source)
        for i, line in enumerate(source):
            if i in indices:
                fields = line.split()
                if int(fields[0]) != i + 1:
                    raise ValueError(f"Unexpected one-based row index in {path}")
                found[i] = fields[1:]
                if len(found) == len(indices):
                    break
    if found.keys() != indices:
        raise ValueError(f"Incomplete material records in {path}")
    return found


def report(case, search, saved_prefix, out):
    config = load(case / 'launch.json')['environment']
    vol = Path(config['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE']).parent
    imported = load(vol / 'native-import.json')
    certificate = load(saved_prefix / 'material-check.json')
    material = vol / 'mesh.ele'
    labels = vol / 'current-child.labels.csv'
    if digest(material) != imported['element_sha256'] or digest(labels) != imported['label_sha256']:
        raise ValueError('Native material input mismatch')
    spans = set(re.findall(r'registered_material_span object=(\d+) element_offset=(\d+) element_count=(\d+)', (case / 'stderr.log').read_text()))
    if len(spans) != 1:
        raise ValueError('Ambiguous native material span')
    owner, offset, count = map(int, spans.pop())
    final = load(search)['cell_backtracking'][-1]
    if final['object'] != owner or count != imported['cells']:
        raise ValueError('Material witness outside registered object')
    indices = {i - offset for i in final['unit_trial_witnesses']}
    if min(indices) < 0 or max(indices) >= count:
        raise ValueError('Witness outside native range')
    elements = indexed_rows(material, indices)
    nodes = {int(n) - 1 for row in elements.values() for n in row[:4]}
    label_rows = {}
    with labels.open() as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            if i in indices:
                if int(row['child_tet_row']) != i:
                    raise ValueError('Unmatched child label')
                label_rows[i] = {k: int(v) for k, v in row.items()}
            if len(label_rows) == len(indices):
                break
    xyz = []
    frame_files = [vol / 'mesh.node', vol / 'current.node', saved_prefix / 'physical-accepted.node']
    for path, key in zip(frame_files, ['reference', 'loaded', 'physical_frame'], strict=True):
        if digest(path) != certificate[key]:
            raise ValueError('Unmatched physical material frame')
        xyz.append(indexed_rows(path, nodes))
    mass_file = vol / 'cell-mass.f64'
    mass = np.memmap(mass_file, dtype='<f8', mode='r', shape=(count,))
    failures = {i for role in certificate['roles'].values() for i in role['failing_cells']}
    cells = []
    for i, row in sorted(elements.items()):
        node_ids = [int(v) - 1 for v in row[:4]]
        label = label_rows[i]
        if label['material_index'] != int(row[4]):
            raise ValueError('Inconsistent material identity')
        geometry = []
        for frame in xyz:
            x = np.array([frame[n][:3] for n in node_ids], dtype=np.float64)
            volumes, qualities = geom(x, np.array([[0, 1, 2, 3]]))
            geometry.append((float(volumes[0]), float(qualities[0])))
        cells.append(dict(global_tet=i+offset, **label, material_nodes=node_ids, reference_kg=float(mass[i]),
                          reference_volume_m3=geometry[0][0], current_J=[x[0]/geometry[0][0] for x in geometry],
                          shape=[x[1] for x in geometry], reference_prefix_shape_failure=i in failures))
    doc = dict(native_search=digest(search), physical_material_receipt=digest(saved_prefix/'material-check.json'),
               material_mass_sha256=digest(mass_file), input_import=imported, limiting=final['global_cell'],
               cells=cells, frames=['reference','loaded','matched_two_step_prefix'],
               scope='The final Newton proposal is a rejected iterate. The three material frames are the input reference, its registered loaded frame and the independently saved two-step physical control.')
    out.write_text(json.dumps(doc, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    report(*(Path(v).resolve() for v in sys.argv[1:]))
